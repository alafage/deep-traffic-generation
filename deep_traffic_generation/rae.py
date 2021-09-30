# fmt: off
from argparse import Namespace
from typing import Dict, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import AE, RNN, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset


# fmt: on
class RAE(AE):
    """Recurrent Autoencoder"""

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.example_input_array = torch.rand(
            (
                1,
                self.dataset_params["seq_len"],
                self.dataset_params["input_dim"],
            )
        )

        self.encoder = RNN(
            input_dim=self.dataset_params["input_dim"],
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = RNN(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.dataset_params["input_dim"],
            h_dims=[self.hparams.encoding_dim] + self.hparams.h_dims[::-1],
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        self.out_activ = nn.Tanh()

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = h.squeeze(0)
        y = z.unsqueeze(1).repeat(1, self.dataset_params["seq_len"], 1)
        x_hat, _ = self.decoder(y)
        x_hat = self.out_activ(x_hat)
        return z, x_hat

    @classmethod
    def network_name(cls) -> str:
        return "rae"


if __name__ == "__main__":
    cli_main(RAE, TrafficDataset, "sequence", seed=42)
