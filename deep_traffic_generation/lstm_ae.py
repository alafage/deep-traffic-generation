from argparse import ArgumentParser, Namespace
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F
from traffic.core import Traffic
from traffic.core.projection import EuroPP
from typing import Optional

from deep_traffic_generation.core.datasets import (
    TrafficDataset,
    TransformerProtocol,
)
from deep_traffic_generation.core.utils import (
    get_dataloaders,
    traffic_from_data,
)
from deep_traffic_generation.core.builders import (
    CollectionBuilder,
    IdentifierBuilder,
    TimestampBuilder,
)


class Encoder(nn.Module):
    def __init__(
        self, seq_len: int, n_features: int, embedding_dim: int = 64
    ) -> None:
        super().__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x, (hidden_n, _) = self.rnn(x)
        return hidden_n.view(-1, self.embedding_dim)


class Decoder(nn.Module):
    def __init__(
        self, seq_len: int, input_dim: int = 64, n_features: int = 1
    ) -> None:
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features


        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.view(-1, self.seq_len, self.input_dim)

        x, (_, _) = self.rnn(x)
        x = self.output_layer(x)
        return x


class LSTMAE(LightningModule):
    """LSTM Autoencoder"""

    _required_hparams = ["learning_rate", "step_size", "gamma"]

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        scaler: Optional[TransformerProtocol],
        config: Namespace,
    ) -> None:
        super().__init__()

        self._check_hparams(config)

        self.config = config
        self.save_hyperparameters(self.config)

        self.seq_len = seq_len
        self.n_features = n_features
        self.scaler = scaler
        self.embedding_dim = self.hparams.embedding_dim

        self.encoder = Encoder(
            self.seq_len, self.n_features, self.embedding_dim
        )
        self.decoder = Decoder(
            self.seq_len, self.embedding_dim, self.n_features
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch == 1:
            sample = torch.rand((1, self.seq_len, self.n_features))
            self.logger.experiment.add_graph(
                LSTMAE(self.seq_len, self.n_features, self.scaler, self.config),
                sample,
            )

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat

    def test_epoch_end(self, outputs) -> None:
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu().numpy()
        reconstructed = outputs[0][1][idx].unsqueeze(0).cpu().numpy()
        data = np.concatenate((original, reconstructed))
        n_samples = 2
        data = data.reshape((2, -1))
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
        parser = parent_parser.add_argument_group("LSTMAE")
        parser.add_argument(
            "--name",
            dest="network_name",
            default="LSTMAE",
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
            "--embedding",
            dest="embedding_dim",
            type=int,
            default=64,
        )

        return parent_parser

    def _check_hparams(self, hparams: Namespace):
        for hparam in self._required_hparams:
            if hparam not in vars(hparams).keys():
                raise AttributeError(
                    f"Can't set up network, {hparam} is missing."
                )


def cli_main() -> None:
    pl.seed_everything(42, workers=True)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
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
        "--train-ratio", dest="train_ratio", type=float, default=0.8
    )
    parser.add_argument(
        "--val-ratio", dest="val_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=1000
    )
    parser.add_argument(
        "--test-batch-size",
        dest="test_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--early-stop", dest="early_stop", action="store_true")
    parser.add_argument(
        "--no-early-stop", dest="early_stop", action="store_false"
    )
    parser.set_defaults(early_stop=False)
    parser.add_argument(
        "--show-latent", dest="show_latent", action="store_true"
    )
    parser.add_argument(
        "--no-show-latent", dest="show_latent", action="store_false"
    )
    parser.set_defaults(show_latent=False)
    parser = Trainer.add_argparse_args(parser)
    parser = LSTMAE.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = TrafficDataset(
        args.data_path,
        features=args.features,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        seq_mode=True,
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
    tb_logger = TensorBoardLogger("lightning_logs/", default_hp_metric=False)

    # ------------
    # model
    # ------------
    model = LSTMAE(
        seq_len=dataset.data.shape[1],
        n_features=dataset.data.shape[2],
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
