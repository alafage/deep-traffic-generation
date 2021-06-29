# TODO: TCN Autoencoder
# fmt: off
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F
from traffic.core import Traffic
from traffic.core.projection import EuroPP

from deep_traffic_generation.core import TCN
from deep_traffic_generation.core.builders import (
    CollectionBuilder, IdentifierBuilder, TimestampBuilder
)
from deep_traffic_generation.core.datasets import (
    TrafficDataset, TransformerProtocol
)
from deep_traffic_generation.core.utils import (
    get_dataloaders, traffic_from_data
)


# fmt: on
class TCEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            TCN(
                input_dim,
                h_dims[-2],
                h_dims[:-2],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            nn.Conv1d(h_dims[-2], h_dims[-1], kernel_size=1),
            nn.AvgPool1d(sampling_factor),
            # We might want to add a non-linear activation
        )

        self.z_loc = nn.Linear(
            h_dims[-1] * (int(seq_len / sampling_factor)), out_dim
        )
        self.z_log_var = nn.Linear(
            h_dims[-1] * (int(seq_len / sampling_factor)), out_dim
        )

    def forward(self, x):
        z = self.encoder(x)
        _, c, length = z.size()
        z = z.view(-1, c * length)
        return self.z_loc(z), self.z_log_var(z)


class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.sampling_factor = sampling_factor

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=sampling_factor),
            TCN(
                h_dims[0],
                h_dims[-1],
                h_dims[1:-1],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            nn.Conv1d(h_dims[-1], out_dim, kernel_size=1),
        )

        self.decode_entry = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )

    def forward(self, x):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


class TCVAE(LightningModule):
    """Temporal Convolutional Variational Autoencoder

    Source: http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
    """

    _required_hparams = [
        "learning_rate",
        "step_size",
        "gamma",
        "encoding_dim",
        "sampling_factor",
        "h_dims",
        "kernel_size",
        "dilation_base",
        "dropout",
    ]

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ):
        super().__init__()

        self._check_hparams(config)

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.scaler = scaler
        self.config = config
        self.save_hyperparameters(self.config)

        # non-linear activations
        h_activ: Optional[nn.Module] = None

        self.example_input_array = torch.rand(
            (self.input_dim, self.seq_len)
        ).unsqueeze(0)

        self.encoder = TCEncoder(
            input_dim=input_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            seq_len=self.seq_len,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

        self.decoder = TCDecoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=input_dim,
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.seq_len,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

        self.scale = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        loc, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()
        x_hat = self.decoder(z)
        return z, x_hat

    def configure_optimizers(self) -> dict:
        # optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.hparams.learning_rate, momentum=0.9
        # )
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams, {"hp/valid_loss": 1, "hp/test_loss": 1}
        )

    def gaussian_likelihood(self, x_hat, x):
        mean = x_hat
        dist = torch.distributions.Normal(mean, self.scale)

        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, loc, std):
        """Monte carlo KL divergence

        Parameters:
        -----------
        z: torch.Tensor
            embbeding tensor
        loc: torch.Tensor
            location parameter for q.
        std: torch.Tensor
            standard deviation for q.
        """
        # define the first two probabilities
        p = torch.distributions.Normal(
            torch.zeros_like(loc), torch.ones_like(std)
        )
        q = torch.distributions.Normal(loc, std)

        # get q(z|x)
        log_qzx = q.log_prob(z)
        # get p(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, y = batch
        # encode x to get the location and log variance parameters
        loc, log_var = self.encoder(x)

        # sample z from q(z|x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        # decode z
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, x)

        # kullback-leibler divergence
        kl = self.kl_divergence(z, loc, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        self.log("train_loss", elbo)
        return elbo

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loc, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loc, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        self.log("hp/test_loss", loss)
        return x, x_hat

    def test_epoch_end(self, outputs) -> None:
        """TODO: replace build traffic part by a function."""
        idx = 0
        original = (
            outputs[0][0][idx].unsqueeze(0).cpu().numpy().transpose(0, 2, 1)
        )
        reconstructed = (
            outputs[0][1][idx].unsqueeze(0).cpu().numpy().transpose(0, 2, 1)
        )
        data = np.concatenate((original, reconstructed))
        n_samples = data.shape[0]
        # build traffic
        data = data.reshape((n_samples, -1))
        if self.scaler is not None:
            data = self.scaler.inverse_transform(data)
        n_obs = int(data.shape[1] / len(self.hparams.features))
        builder = CollectionBuilder(
            [
                IdentifierBuilder(n_samples, n_obs),
                TimestampBuilder(),
                # FIXME: requires first values for lat/lon
                # LatLonBuilder(build_from="azgs"),
            ]
        )
        traffic = traffic_from_data(
            data, self.hparams.features, builder=builder
        )
        # generate plot then send it to logger
        self.logger.experiment.add_figure(
            "original vs reconstructed", self.plot_traffic(traffic)
        )

    def plot_traffic(self, traffic: Traffic) -> Figure:
        with plt.style.context("traffic"):
            fig, ax = plt.subplots(
                1, figsize=(5, 5), subplot_kw=dict(projection=EuroPP())
            )
            traffic[1].plot(ax, c="orange", label="reconstructed")
            traffic[0].plot(ax, c="purple", label="original")
            ax.legend()

        return fig

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group("TCVAE")
        parser.add_argument(
            "--name",
            dest="network_name",
            default="TCVAE",
            type=str,
            help="network name",
        )
        parser.add_argument(
            "--lr",
            dest="learning_rate",
            default=1e-3,
            type=float,
            help="learning rate",
        )
        parser.add_argument(
            "--lrstep",
            dest="step_size",
            default=100,
            type=int,
            help="period of learning rate decay (in epochs)",
        )
        parser.add_argument(
            "--lrgamma",
            dest="gamma",
            default=1.0,
            type=float,
            help="multiplicative factor of learning rate decay",
        )
        parser.add_argument(
            "--encoding_dim",
            dest="encoding_dim",
            type=int,
            default=64,
        )
        parser.add_argument(
            "--sampling_factor",
            dest="sampling_factor",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--h_dims",
            dest="h_dims",
            nargs="+",
            type=int,
            default=[16, 16, 16],
        )
        parser.add_argument(
            "--kernel",
            dest="kernel_size",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--dilation",
            dest="dilation_base",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--dropout",
            dest="dropout",
            type=float,
            default=0,
        )

        return parent_parser

    def _check_hparams(self, hparams: Union[Dict, Namespace]):
        for hparam in self._required_hparams:
            if isinstance(hparams, Namespace):
                if hparam not in vars(hparams).keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            elif isinstance(hparams, dict):
                if hparam not in hparams.keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            else:
                raise TypeError(f"Invalid type for hparams: {type(hparams)}.")


def cli_main() -> None:
    pl.seed_everything(42, workers=True)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=Path,
        default=Path("./data/denoised_v3.pkl").absolute(),
    )
    parser.add_argument(
        "--features",
        dest="features",
        nargs="+",
        default=["latitude", "longitude", "altitude", "timedelta"],
    )
    parser.add_argument(
        "--train_ratio", dest="train_ratio", type=float, default=0.8
    )
    parser.add_argument(
        "--val_ratio", dest="val_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1000
    )
    parser.add_argument(
        "--test_batch_size",
        dest="test_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--early_stop", dest="early_stop", action="store_true")
    parser.add_argument(
        "--no-early-stop", dest="early_stop", action="store_false"
    )
    parser.set_defaults(early_stop=False)
    parser.add_argument(
        "--show_latent", dest="show_latent", action="store_true"
    )
    parser.add_argument(
        "--no_show_latent", dest="show_latent", action="store_false"
    )
    parser.set_defaults(show_latent=False)
    parser = Trainer.add_argparse_args(parser)
    parser = TCVAE.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = TrafficDataset(
        args.data_path,
        features=args.features,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        mode="image",
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        args.train_ratio,
        args.val_ratio,
        args.batch_size,
        args.test_batch_size,
    )

    # ------------
    # logger
    # ------------
    tb_logger = TensorBoardLogger(
        "lightning_logs/",
        name="tc_vae",
        default_hp_metric=False,
        log_graph=False,
    )

    # ------------
    # model
    # ------------
    model = TCVAE(
        input_dim=dataset.data.shape[1],
        seq_len=dataset.data.shape[2],
        scaler=dataset.scaler,
        config=args,
    )

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(monitor="hp/valid_loss")
    if args.early_stop:
        print("hey")
        early_stopping = EarlyStopping("hp/valid_loss")
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
        )
    else:
        trainer = Trainer.from_argparse_args(
            args, callbacks=[checkpoint_callback], logger=tb_logger
        )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)

    # ------------
    # visualization
    # ------------
    # TODO: if show_latent then use tensorboard to display the data in the
    # latent space.


if __name__ == "__main__":
    cli_main()
