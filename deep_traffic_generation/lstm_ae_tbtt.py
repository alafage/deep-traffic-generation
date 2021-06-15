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

class LSTMAE_TBTT(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.truncated_bptt_steps = 2

    def training_step(self, batch, batch_idx, hiddens):
        x, y = batch
        out, hiddens = 
        return {
            "loss": ...,
            "hiddens": hiddens
        }


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
        # seq_mode=True,
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
        "lightning_logs/", name="lstm_ae", default_hp_metric=False
    )

    # ------------
    # model
    # ------------
    model = LSTMAE(
        seq_len=10,
        n_features=1,
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
