# fmt: off
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import FCN, VAE
from deep_traffic_generation.core.datasets import (
    TrafficDataset, TransformerProtocol
)
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class _Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = FCN(
            input_dim=input_dim,
            out_dim=h_dims[-1],
            h_dims=h_dims[:-1],
            h_activ=h_activ,
            dropout=dropout,
        )

        self.z_loc = nn.Linear(h_dims[-1], out_dim)
        self.z_log_var = nn.Linear(h_dims[-1], out_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.z_loc(z), self.z_log_var(z)


class LinearVAE(VAE):
    """Linear Variational Autoencoder"""

    _required_hparams = ["learning_rate", "step_size", "gamma"]

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

        # encoder
        self.encoder = _Encoder(
            input_dim=input_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            h_activ=self.h_activ,
            dropout=self.hparams.dropout,
        )
        # decoder
        self.decoder = FCN(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.input_dim,
            h_dims=self.hparams.h_dims[::-1],
            h_activ=self.h_activ,
            dropout=self.hparams.dropout,
        )

        self.scale = nn.Parameter(torch.Tensor([1.0]))

    @classmethod
    def network_name(cls) -> str:
        return "linear_vae"


if __name__ == "__main__":
    cli_main(LinearVAE, TrafficDataset, "linear", seed=42)
