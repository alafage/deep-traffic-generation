# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import AE, TCN
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.protocols import TransformerProtocol
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class TCEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            TCN(
                input_dim,
                h_dims[-1],
                h_dims[:-1],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            nn.Conv1d(h_dims[-1], out_dim, kernel_size=1),
            nn.AvgPool1d(sampling_factor),
            # We might want to add a non-linear activation
        )

    def forward(self, x):
        # print(f"x: {x.size()}")
        z = self.encoder(x)
        # print(f"z: {z.size()}")
        return z


class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=sampling_factor),
            TCN(
                input_dim,
                h_dims[-1],
                h_dims[:-1],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            nn.Conv1d(h_dims[-1], out_dim, kernel_size=1),
        )

    def forward(self, x):
        x_hat = self.decoder(x)
        return x_hat


class TCAE(AE):
    """Temporal Convolutional Autoencoder

    Source: http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
    """

    _required_hparams = AE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
    ]

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(input_dim, seq_len, scaler, config)

        # non-linear activations
        h_activ: Optional[nn.Module] = None

        self.example_input_array = torch.rand(
            (self.input_dim, self.seq_len)
        ).unsqueeze(0)

        self.encoder = TCEncoder(
            input_dim=input_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

        self.decoder = TCDecoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=input_dim,
            h_dims=self.hparams.h_dims[::-1],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

    def test_step(self, batch, batch_idx):
        x, _, info = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "tc_ae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--sampling_factor",
            dest="sampling_factor",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--kernel",
            dest="kernel_size",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--dilation",
            dest="dilation_base",
            type=int,
            default=2,
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(TCAE, TrafficDataset, "image", seed=42)
