# fmt: off
from argparse import Namespace
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import AE, FCN
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.protocols import TransformerProtocol
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class LinearAE(AE):
    """Linear Autoencoder"""

    def __init__(
        self,
        x_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(x_dim, seq_len, scaler, config)

        self.example_input_array = torch.rand((1, self.input_dim))

        self.encoder = FCN(
            input_dim=x_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )
        self.decoder = FCN(
            input_dim=self.hparams.encoding_dim,
            out_dim=x_dim,
            h_dims=self.hparams.h_dims[::-1],
            h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        self.out_activ = nn.Tanh()

    @classmethod
    def network_name(cls) -> str:
        return "linear_ae"


if __name__ == "__main__":
    cli_main(LinearAE, TrafficDataset, "linear", seed=42)
