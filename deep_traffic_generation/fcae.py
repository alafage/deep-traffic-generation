# fmt: off
from argparse import Namespace
# from deep_traffic_generation.core.losses import sdtw_loss
from typing import Dict, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import AE, FCN, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset


# fmt: on
class FCAE(AE):
    """Fully-Connected Autoencoder"""

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.example_input_array = torch.rand(
            (1, self.dataset_params["input_dim"])
        )

        self.encoder = FCN(
            input_dim=self.dataset_params["input_dim"],
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )
        self.decoder = FCN(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.dataset_params["input_dim"],
            h_dims=self.hparams.h_dims[::-1],
            h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        self.out_activ = nn.Tanh()

    @classmethod
    def network_name(cls) -> str:
        return "fcae"


if __name__ == "__main__":
    cli_main(FCAE, TrafficDataset, "linear", seed=42)
