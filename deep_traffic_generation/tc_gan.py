# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import torch.nn as nn

from deep_traffic_generation.core import GAN, TCN
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.protocols import TransformerProtocol
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class TCGenerator(nn.Module):
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

        self.generate_from_entry = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )

        self.generator = nn.Sequential(
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
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.generate_from_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.generator(x)
        return x_hat


class TCDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.discriminator = nn.Sequential(
            TCN(
                input_dim,
                1,
                h_dims,
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)


class TCGAN(GAN):
    """Temporal Convolutional Generative Adversarial Network"""

    _required_hparams = GAN._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
    ]

    def __init__(
        self,
        x_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(x_dim, seq_len, scaler, config)

        # non-linear activations
        h_activ: Optional[nn.Module] = None

        self.generator = TCGenerator(
            input_dim=self.hparams.latent_dim,
            out_dim=self.out_dim,
            h_dims=self.hparams.h_dims,
            seq_len=self.seq_len,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

        self.discriminator = TCDiscriminator(
            input_dim=self.out_dim,
            h_dims=self.hparams.h_dims,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

    @classmethod
    def network_name(cls) -> str:
        return "tc_gan"

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
    cli_main(TCGAN, TrafficDataset, "image", seed=42)
