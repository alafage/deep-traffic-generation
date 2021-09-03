# fmt: off
from argparse import Namespace
# from deep_traffic_generation.core.losses import sdtw_loss
from typing import Dict, Union

import torch.nn as nn

from deep_traffic_generation.core import AE, FCN
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class LinearAE(AE):
    """Linear Autoencoder"""

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        # self.example_input_array = torch.rand(
        #     (1, self.dataset_params["input_dim"])
        # )

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

    # FIXME
    # def training_step(self, batch, batch_idx):
    #     x, l, _ = batch
    #     _, x_hat = self.forward(x, l)
    #     loss = sdtw_loss(x_hat, x)
    #     self.log("train_loss", loss)
    #     return loss

    @classmethod
    def network_name(cls) -> str:
        return "linear_ae"


if __name__ == "__main__":
    cli_main(LinearAE, TrafficDataset, "linear", seed=42)
