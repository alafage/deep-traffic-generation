from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F

from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.utils import get_dataloaders


class LinearAE(LightningModule):
    """Linear Autoencoder"""

    _required_hparams = ["learning_rate", "step_size", "gamma"]

    def __init__(self, x_dim: int, config: Namespace) -> None:
        super().__init__()

        self._check_hparams(config)

        self.x_dim = x_dim
        self.config = config
        self.save_hyperparameters(self.config)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.x_dim),
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
            sample = torch.rand((1, self.x_dim))
            self.logger.experiment.add_graph(
                LinearAE(self.x_dim, self.config), sample
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

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LinearAE")
        parser.add_argument(
            "--name",
            dest="network_name",
            default="LinearAE",
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

        return parent_parser

    def _check_hparams(self, hparams: Namespace):
        for hparam in self._required_hparams:
            if hparam not in vars(hparams).keys():
                raise AttributeError(
                    f"Can't set up network, {hparam} is missing."
                )


def cli_main() -> None:
    pl.seed_everything(42)
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
    parser = LinearAE.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = TrafficDataset(
        args.data_path,
        features=args.features,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
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
    model = LinearAE(x_dim=dataset.data.shape[-1], config=args)

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