# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F

from .builders import (
    CollectionBuilder, IdentifierBuilder, LatLonBuilder, TimestampBuilder
)
from .protocols import TransformerProtocol
from .utils import plot_traffic, traffic_from_data


# fmt: on
class AE(LightningModule):
    """Abstract class for Autoencoder"""

    _required_hparams = [
        "learning_rate",
        # "step_size",
        # "gamma",
        "encoding_dim",
        "h_dims",
        "dropout",
    ]

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__()

        self._check_hparams(config)

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.scaler = scaler
        self.save_hyperparameters(config)

        self.encoder: nn.Module
        self.decoder: nn.Module

    def configure_optimizers(self) -> dict:
        # optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        # learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=self.hparams.step_size,
        #     gamma=self.hparams.gamma,
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams, {"hp/valid_loss": 1, "hp/test_loss": 1}
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, _, info = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat, info

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """FIXME: too messy."""
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu().numpy()
        reconstructed = outputs[0][1][idx].unsqueeze(0).cpu().numpy()
        info = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
        data = np.concatenate((original, reconstructed), axis=0)
        data = data.reshape((data.shape[0], -1))
        # unscale the data
        if self.scaler is not None:
            data = self.scaler.inverse_transform(data)
        # add info if needed (init_features)
        if len(self.hparams.init_features) > 0:
            info = np.repeat(info, data.shape[0], axis=0)
            data = np.concatenate((info, data), axis=1)
        # get builder
        builder = self.get_builder(nb_samples=2)
        features = [
            "track" if "track" in f else f for f in self.hparams.features
        ]
        # build traffic
        traffic = traffic_from_data(
            data,
            features,
            self.hparams.init_features,
            builder=builder,
        )
        # generate plot then send it to logger
        self.logger.experiment.add_figure(
            "original vs reconstructed", plot_traffic(traffic)
        )

    def get_builder(self, nb_samples: int) -> CollectionBuilder:
        builder = CollectionBuilder(
            [IdentifierBuilder(nb_samples, self.seq_len), TimestampBuilder()]
        )
        if "latitude" not in self.hparams.features:
            if "x" in self.hparams.features:
                builder.append(LatLonBuilder(build_from="xy"))
            elif "track_unwrapped" in self.hparams.features:
                builder.append(LatLonBuilder(build_from="azgs"))
        return builder

    @classmethod
    def network_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        parser = parent_parser.add_argument_group(f"{cls.network_name()}")
        parser.add_argument(
            "--name",
            dest="network_name",
            default=f"{cls.network_name()}",
            type=str,
            help="network name",
        )
        parser.add_argument(
            "--lr",
            dest="learning_rate",
            default=1e-3,
            type=float,
            help="learning rate",
        )
        # parser.add_argument(
        #     "--lrstep",
        #     dest="step_size",
        #     default=100,
        #     type=int,
        #     help="period of learning rate decay (in epochs)",
        # )
        # parser.add_argument(
        #     "--lrgamma",
        #     dest="gamma",
        #     default=1.0,
        #     type=float,
        #     help="multiplicative factor of learning rate decay",
        # )
        parser.add_argument(
            "--encoding_dim",
            dest="encoding_dim",
            type=int,
            default=64,
        )
        parser.add_argument(
            "--h_dims",
            dest="h_dims",
            nargs="+",
            type=int,
            default=[],
        )
        parser.add_argument(
            "--dropout", dest="dropout", type=float, default=0.0
        )

        return parent_parser, parser

    def _check_hparams(self, hparams: Union[Dict, Namespace]):
        for hparam in self._required_hparams:
            if isinstance(hparams, Namespace):
                if hparam not in vars(hparams).keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            elif isinstance(hparams, dict):
                if hparam not in hparams.keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            else:
                raise TypeError(f"Invalid type for hparams: {type(hparams)}.")


class VAE(AE):
    """Abstract class for Beta Variational Autoencoder"""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(input_dim, seq_len, scaler, config)

        self.scale = nn.Parameter(torch.Tensor([self.hparams.scale]))

    def forward(self, x):
        loc, log_var = self.encoder(x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()
        x_hat = self.decoder(z)
        return z, x_hat

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
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
        C_max = nn.Parameter(
            torch.Tensor([self.hparams.c_max], device=self.device)
        )
        C = torch.clamp(
            (C_max / self.hparams.c_stop_iter) * self.current_epoch,
            0,
            self.hparams.c_max,
        )
        kl = self.kl_divergence(z, loc, std)
        # kl = (
        #     -0.5 * (1 + log_var - loc ** 2 - torch.exp(log_var)).sum(dim=1)
        # ).mean(dim=0)

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = self.hparams.gamma * (kl - C).abs() - recon_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kl.mean(),
                "recon_loss": recon_loss.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        loc, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, _, info = batch
        loc, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()

        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        self.log("hp/test_loss", loss)
        return x, x_hat, info

    def gaussian_likelihood(self, x_hat, x):
        mean = x_hat
        dist = torch.distributions.Normal(mean, self.scale)

        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(self, z, loc, std):
        """Monte carlo KL divergence

        Parameters:
        -----------
        z: torch.Tensor
            embbeding tensor
        loc: torch.Tensor
            location parameter for q.
        std: torch.Tensor
            standard deviation for q.
        """
        # define the first two probabilities
        p = torch.distributions.Normal(
            torch.zeros_like(loc), torch.ones_like(std)
        )
        q = torch.distributions.Normal(loc, std)

        # get q(z|x)
        log_qzx = q.log_prob(z)
        # get p(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--gamma",
            dest="gamma",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--c_max",
            dest="c_max",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--c_stop_iter",
            dest="c_stop_iter",
            type=int,
            default=1,
        )
        parser.add_argument("--scale", dest="scale", type=float, default=1.0)

        return parent_parser, parser


class GAN(nn.Module):
    ...
