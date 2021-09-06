# fmt: off
from argparse import Namespace
from typing import Dict, List, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import AE, RNN
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.utils import cli_main

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
        dropout: float,
        num_layers: int,
        batch_first: bool,
    ) -> None:
        super().__init__()

        self.encoder = RNN(
            input_dim=input_dim,
            out_dim=out_dim,
            h_dims=h_dims,
            dropout=dropout,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(self, x, lengths):
        _, (h, _) = self.encoder(x)
        return h.squeeze(0)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        seq_len: int,
        dropout: float,
        num_layers: int,
        batch_first: bool,
    ) -> None:
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.decoder = RNN(
            input_dim=input_dim,
            out_dim=h_dims[-1],
            h_dims=[input_dim] + h_dims[:-1],
            dropout=dropout,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.fc = nn.Linear(h_dims[-1], out_dim)

    def forward(self, x, lengths):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (_, _) = self.decoder(x)
        return self.fc(x)


class LSTMAE(AE):
    """LSTM Autoencoder"""

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.example_input_array = [
            torch.rand(
                (
                    1,
                    self.dataset_params["seq_len"],
                    self.dataset_params["input_dim"],
                )
            ),
            torch.Tensor([self.dataset_params["seq_len"]]),
        ]

        self.encoder = Encoder(
            input_dim=self.dataset_params["input_dim"],
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = Decoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.dataset_params["input_dim"],
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.dataset_params["seq_len"],
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        self.out_activ = nn.Tanh()

    @classmethod
    def network_name(cls) -> str:
        return "lstm_ae"


if __name__ == "__main__":
    cli_main(LSTMAE, TrafficDataset, "sequence", seed=42)
