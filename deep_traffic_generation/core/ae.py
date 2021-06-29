# fmt: off
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from deep_traffic_generation.core.builders import (
    CollectionBuilder, IdentifierBuilder, LatLonBuilder, TimestampBuilder
)
from deep_traffic_generation.core.datasets import TransformerProtocol
from deep_traffic_generation.core.utils import plot_traffic, traffic_from_data


# fmt: on
class AE(LightningModule):
    """Abstract class for Autoencoder"""

    _required_hparams = [
        "learning_rate",
        "step_size",
        "gamma",
        "encoding_dim",
        "h_dims",
        "dropout",
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
        self.save_hyperparameters(config)

        self.encoder: nn.Module
        self.decoder: nn.Module

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

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """FIXME: too messy."""
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu().numpy()
        reconstructed = outputs[0][1][idx].unsqueeze(0).cpu().numpy()
        info = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
        data = np.concatenate((original, reconstructed))
        data = data.reshape((data.shape[0], -1))
        if len(self.hparams.init_features) > 0:
            data = np.concatenate((info, data), axis=1)
        builder = self.get_builder(nb_samples=2)
        traffic = traffic_from_data(
            data,
            self.hparams.features,
            self.hparams.init_features,
            builder=builder,
        )
        # generate plot then send it to logger
        self.logger.experiment.add_figure(
            "original vs reconstructed", plot_traffic(traffic)
        )

    def get_builder(self, nb_samples: int) -> CollectionBuilder:
        builder = CollectionBuilder(
            [IdentifierBuilder(nb_samples, self.seq_len), TimestampBuilder()]
        )
        if "latitude" not in self.hparams.features:
            if "x" in self.hparams.features:
                builder.append(LatLonBuilder(build_from="xy"))
            elif "track_unwrapped" in self.hparams.features:
                builder.append(LatLonBuilder(build_from="azgs"))
        return builder

    @classmethod
    def network_name(cls) -> str:
        return "ae"

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(f"{cls.network_name()}")
        parser.add_argument(
            "--name",
            dest="network_name",
            default=f"{cls.network_name()}",
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
            "--h_dims",
            dest="h_dims",
            nargs="+",
            type=int,
            default=[],
        )
        parser.add_argument(
            "--dropout", dest="dropout", type=float, default=0.0
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
