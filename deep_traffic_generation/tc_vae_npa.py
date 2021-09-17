# FIXME
# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import TCN, VAE
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.losses import npa_loss
from deep_traffic_generation.core.transforms import PyTMinMaxScaler
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
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:

        super().__init__(dataset_params, config)

        # navpoints coordinates have been projected using EuroPP projection
        with open("./data/navpoints_15.npy", "rb") as f:
            navpts = torch.from_numpy(np.load(f))

        # navpoints coordinates should be scaled according to dataset scaler
        if isinstance(self.dataset_params["scaler"], PyTMinMaxScaler):
            navpts = self.dataset_params["scaler"].partial_transform(
                navpts,
                idxs=np.where(
                    np.isin(self.dataset_params["features"], ["x", "y"])
                )[0],
            )

        self.navpts = navpts

        self.features_idxs = np.where(
            np.isin(
                self.dataset_params["features"],
                ["track", "groundspeed", "altitude", "timedelta"],
            )
        )[0]

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
            dropout=self.hparams.dropout,
        )

        # non-linear activations
        self.out_activ: Optional[nn.Module] = LinearAct()

    def training_step(self, batch, batch_idx):
        x, l, _ = batch
        z, (loc, std), x_hat = self.forward(x, l)

        # reconstruction loss
        # recon_loss = self.gaussian_likelihood(
        #     x_hat[:, self.features_idxs], x[:, self.features_idxs]
        # )
        recon_loss = F.mse_loss(
            x[:, self.features_idxs],
            x_hat[:, self.features_idxs],
            reduction="none",
        ).sum(dim=(1, 2))

        mse_coef = 1e4

        # kullback-leibler divergence
        c_max = torch.Tensor([self.hparams.c_max])
        C_max = nn.Parameter(c_max).to(self.device)
        C = torch.clamp(
            (C_max / self.hparams.c_stop_iter) * self.current_epoch,
            0,
            self.hparams.c_max,
        )
        kl = self.kl_divergence(z, loc, std)

        # navigational point alignment loss
        # unscale track_unwrapped
        track_idx = self.dataset_params["features"].index("track")
        if isinstance(self.dataset_params["scaler"], PyTMinMaxScaler):
            tracks = self.dataset_params["scaler"].partial_inverse(
                x_hat[:, track_idx], idxs=track_idx
            )
        npa = npa_loss(
            x[
                :,
                np.where(np.isin(self.dataset_params["features"], ["x", "y"]))[
                    0
                ],
            ],
            tracks,
            self.navpts.to(self.device),
            reduction="none",
        )

        npa_coef = self.hparams.align_coef * self.hparams.align_gamma ** (
            self.current_epoch // self.hparams.align_step
        )

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = (
            self.hparams.gamma * (kl - C).abs()
            + mse_coef * recon_loss
            + npa_coef * npa
        )
        elbo = elbo.mean()

        # print(elbo)

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "npa": npa.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x, l, _ = batch
        _, _, x_hat = self.forward(x, l)
        loss = F.mse_loss(
            x_hat[:, self.features_idxs], x[:, self.features_idxs]
        )
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, l, info = batch
        loc, log_var = self.encoder(x, l)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.out_activ(self.decoder(z, l))

        loss = F.mse_loss(
            x_hat[:, self.features_idxs], x[:, self.features_idxs]
        )

        self.log("hp/test_loss", loss)
        return (
            torch.transpose(x, 1, 2),
            l,
            torch.transpose(x_hat, 1, 2),
            info,
        )

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
            default=0.1,
        )
        parser.add_argument(
            "--align_step",
            dest="align_step",
            type=int,
            default=500,
        )
        parser.add_argument(
            "--align_gamma", dest="align_gamma", type=float, default=1
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(TCVAENPA, TrafficDataset, "image", seed=42)
