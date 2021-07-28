# TODO: TCN Autoencoder
# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from traffic.data import navaids

from deep_traffic_generation.core import TCN, VAE
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.losses import npa_loss
from deep_traffic_generation.core.protocols import TransformerProtocol
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class TCEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
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
        )

        self.z_loc = nn.Linear(
            h_dims[-1] * (int(seq_len / sampling_factor)), out_dim
        )
        self.z_log_var = nn.Linear(
            h_dims[-1] * (int(seq_len / sampling_factor)), out_dim
        )

    def forward(self, x):
        z = self.encoder(x)
        _, c, length = z.size()
        z = z.view(-1, c * length)
        return self.z_loc(z), self.z_log_var(z)


class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
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
        )

    def forward(self, x):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


class TCVAENPA(VAE):
    """Temporal Convolutional Variational Autoencoder

    Source: http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
    """

    _required_hparams = VAE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
        "align_coef",
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
            seq_len=self.seq_len,
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
            seq_len=self.seq_len,
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            h_activ=h_activ,
            dropout=self.hparams.dropout,
        )

    def training_step(self, batch, batch_idx):
        x, labels, _ = batch
        # encode x to get the location and log variance parameters
        loc, log_var = self.encoder(x)

        # sample z from q(z|x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        # decode z
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, x)
        # recon_loss = F.mse_loss(x, x_hat)

        # kullback-leibler divergence
        c_max = torch.Tensor([self.hparams.c_max])
        C_max = nn.Parameter(c_max).to(self.device)
        C = torch.clamp(
            (C_max / self.hparams.c_stop_iter) * self.current_epoch,
            0,
            self.hparams.c_max,
        )
        kl = self.kl_divergence(z, loc, std)

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = self.hparams.gamma * (kl - C).abs() - recon_loss
        elbo = elbo.mean()

        # navigational point alignment loss
        navids = set(
            [navid for navids in labels for navid in navids.split(",")]
        )
        navpts = torch.Tensor(
            [[navaids[navid].lat, navaids[navid].lon] for navid in navids]
        ).to(self.device)

        npa = npa_loss(x_hat, navpts)

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "npa": npa,
            }
        )
        return elbo + self.hparams.align_coef * npa

    def test_step(self, batch, batch_idx):
        x, _, info = batch
        loc, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "tc_vae_npa"

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
        parser.add_argument(
            "--align_coef",
            dest="align_coef",
            type=float,
            default=1.0,
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(TCVAENPA, TrafficDataset, "image", seed=42)
