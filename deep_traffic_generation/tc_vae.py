# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import TCN, VAE
from deep_traffic_generation.core.datasets import (
    DatasetParams, TrafficDatasetOld
)
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class LinearAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TCEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        seq_len: int,
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
            nn.AvgPool1d(sampling_factor),
            # We might want to add a non-linear activation
            # nn.Tanh(),
        )

        self.z_loc = nn.Linear(
            h_dims[-1] * (int(seq_len / sampling_factor)), out_dim
        )
        self.z_log_var = nn.Linear(
            h_dims[-1] * (int(seq_len / sampling_factor)), out_dim
        )

    def forward(self, x, lengths):
        z = self.encoder(x)
        _, c, length = z.size()
        z = z.view(-1, c * length)
        return self.z_loc(z), self.z_log_var(z)


class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.sampling_factor = sampling_factor

        self.decode_entry = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=sampling_factor),
            TCN(
                h_dims[0],
                out_dim,
                h_dims[1:],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            # nn.Tanh(),
        )

    def forward(self, x, lengths):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


class TCVAE(VAE):
    """Temporal Convolutional Variational Autoencoder

    Source: http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
    """

    _required_hparams = VAE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
    ]

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
                    self.dataset_params["input_dim"],
                    self.dataset_params["seq_len"],
                )
            ),
            torch.Tensor([self.dataset_params["seq_len"]]),
        ]

        self.encoder = TCEncoder(
            input_dim=self.dataset_params["input_dim"],
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            seq_len=self.dataset_params["seq_len"],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            # h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        self.decoder = TCDecoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.dataset_params["input_dim"],
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.dataset_params["seq_len"],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            # h_activ=nn.ReLU(),
            dropout=self.hparams.dropout,
        )

        # non-linear activations
        self.out_activ = LinearAct()

    def test_step(self, batch, batch_idx):
        x, l, info = batch
        loc, log_var = self.encoder(x, l)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.out_activ(self.decoder(z, l))

        loss = F.mse_loss(x_hat, x)

        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), l, torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "tc_vae"

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
    cli_main(TCVAE, TrafficDatasetOld, "image", seed=42)
