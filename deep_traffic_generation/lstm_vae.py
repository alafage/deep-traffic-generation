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

from deep_traffic_generation.core.builders import (
    CollectionBuilder, IdentifierBuilder, TimestampBuilder
)
from deep_traffic_generation.core.datasets import (
    TrafficDataset, TransformerProtocol
)
from deep_traffic_generation.core.utils import (
    get_dataloaders, traffic_from_data
)

"""
    Based on sequitur library LSTM_AE (https://github.com/shobrook/sequitur)
    Adapted to handle batch of sequences
"""


# fmt: on
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        h_activ: Optional[nn.Module],
        out_activ: Optional[nn.Module],
    ) -> None:
        super().__init__()

        layer_dims = [input_dim] + h_dims
        self.n_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()

        for index in range(self.n_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.dropout = nn.Dropout()

        self.h_activ = h_activ
        self.out_activ = out_activ

        self.z_loc = nn.Linear(layer_dims[-1], out_dim)
        self.z_log_var = nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.n_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.n_layers - 1:
                h_n = self.out_activ(h_n)

        z = h_n.squeeze(0)
        return self.z_loc(z), self.z_log_var(z)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        h_activ: Optional[nn.Module],
    ) -> None:
        super(Decoder, self).__init__()

        layer_dims = [
            input_dim,
            input_dim,
        ] + h_dims
        self.n_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.n_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.dropout = nn.Dropout()

        self.h_activ = h_activ
        self.fc = nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.n_layers - 1:
                x = self.h_activ(x)

        return self.fc(x)


class LstmVAE(LightningModule):
    """LSTM Autoencoder"""

    _required_hparams = [
        "learning_rate",
        "step_size",
        "gamma",
        "encoding_dim",
        "h_dims",
    ]

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__()

        self._check_hparams(config)

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.scaler = scaler
        self.config = config
        self.save_hyperparameters(self.config)

        # non-linear activations
        self.h_activ: Optional[nn.Module] = None
        self.out_activ: Optional[nn.Module] = None

        self.example_input_array = torch.rand(
            (self.seq_len, self.input_dim)
        ).unsqueeze(0)

        self.encoder = Encoder(
            input_dim=self.input_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            h_activ=self.h_activ,
            out_activ=self.out_activ,
        )

        self.decoder = Decoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.input_dim,
            h_dims=self.hparams.h_dims[::-1],
            h_activ=self.h_activ,
        )

        self.scale = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        loc, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()
        x_hat = self.decoder(z, self.seq_len)
        return z, x_hat

    def configure_optimizers(self) -> dict:
        # optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
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
        x_hat = self.decoder(z, self.seq_len)

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

        x_hat = self.decoder(z, self.seq_len)

        loss = F.mse_loss(x_hat, x)

        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loc, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.decoder(z, self.seq_len)

        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat

    def test_epoch_end(self, outputs) -> None:
        """TODO: replace build traffic part by a function."""
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu().numpy()
        reconstructed = outputs[0][1][idx].unsqueeze(0).cpu().numpy()
        data = np.concatenate((original, reconstructed))
        n_samples = data.shape[0]
        # build traffic
        data = data.reshape((n_samples, -1))
        if self.scaler is not None:
            data = self.scaler.inverse_transform(data)
        n_obs = int(data.shape[1] / len(self.hparams.features))
        builder = CollectionBuilder(
            [IdentifierBuilder(n_samples, n_obs), TimestampBuilder()]
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
        parser = parent_parser.add_argument_group("LstmVAE")
        parser.add_argument(
            "--name",
            dest="network_name",
            default="LstmVAE",
            type=str,
            help="network name",
        )
        parser.add_argument(
            "--lr",
            dest="learning_rate",
            default=1e-2,
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
            default=20,
        )
        parser.add_argument(
            "--h_dims",
            dest="h_dims",
            nargs="+",
            type=int,
            default=[400],
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
    parser = LstmVAE.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = TrafficDataset(
        args.data_path,
        features=args.features,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        mode="sequence",
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
        name="lstm_vae",
        default_hp_metric=False,
        log_graph=False,  # FIXME: Tracer Warning if True
    )

    # ------------
    # model
    # ------------
    model = LstmVAE(
        input_dim=dataset.data.shape[2],
        seq_len=dataset.data.shape[1],
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
