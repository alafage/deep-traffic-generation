# fmt: off
from argparse import Namespace
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import AE, FCN
from deep_traffic_generation.core.datasets import (
    TrafficDataset, TransformerProtocol
)
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class LinearAE(AE):
    """Linear Autoencoder"""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(input_dim, seq_len, scaler, config)

        # FIXME: should be in config
        self.h_activ: Optional[nn.Module] = None

        self.example_input_array = torch.rand((1, self.input_dim))

        # FIXME: encoder and decoder should be separate classes
        # encoder
        self.encoder = FCN(
            input_dim=input_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            h_activ=self.h_activ,
            dropout=self.hparams.dropout,
        )
        # decoder
        self.decoder = FCN(
            input_dim=self.hparams.encoding_dim,
            out_dim=input_dim,
            h_dims=self.hparams.h_dims[::-1],
            h_activ=self.h_activ,
            dropout=self.hparams.dropout,
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, _, info = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat, info

    @classmethod
    def network_name(cls) -> str:
        return "linear_ae_copy"


if __name__ == "__main__":
    cli_main(LinearAE, TrafficDataset, "linear", 42)
