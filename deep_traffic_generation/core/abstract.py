# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F
from traffic.core.projection import EuroPP

from deep_traffic_generation.core.datasets import DatasetParams

from .builders import (
    CollectionBuilder, IdentifierBuilder, LatLonBuilder, TimestampBuilder
)
from .protocols import TransformerProtocol
from .utils import plot_traffic, traffic_from_data


# fmt: on
class Abstract(LightningModule):
    """Abstract class for deep models"""

    _required_hparams = [
        "lr",
        "lr_step_size",
        "lr_gamma",
        "dropout",
    ]

    def __init__(
        self, dataset_params: DatasetParams, config: Union[Dict, Namespace]
    ) -> None:
        super().__init__()

        self._check_hparams(config)
        self.save_hyperparameters(config)

        self.dataset_params = dataset_params

        # self.criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

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
            dest="lr",
            default=1e-3,
            type=float,
            help="learning rate",
        )
        parser.add_argument(
            "--lrstep",
            dest="lr_step_size",
            default=100,
            type=int,
            help="period of learning rate decay (in epochs)",
        )
        parser.add_argument(
            "--lrgamma",
            dest="lr_gamma",
            default=1.0,
            type=float,
            help="multiplicative factor of learning rate decay",
        )
        parser.add_argument(
            "--dropout", dest="dropout", type=float, default=0.0
        )

        return parent_parser, parser

    def get_builder(self, nb_samples: int, length: int) -> CollectionBuilder:
        builder = CollectionBuilder(
            [
                IdentifierBuilder(nb_samples, length),
                TimestampBuilder(),
            ]
        )
        if "track_unwrapped" in self.dataset_params["features"]:
            if self.dataset_params["info_params"]["index"] == 0:
                builder.append(LatLonBuilder(build_from="azgs"))
            elif self.dataset_params["info_params"]["index"] == -1:
                builder.append(LatLonBuilder(build_from="azgs_r"))
        elif "track" in self.dataset_params["features"]:
            if self.dataset_params["info_params"]["index"] == 0:
                builder.append(LatLonBuilder(build_from="azgs"))
            elif self.dataset_params["info_params"]["index"] == -1:
                builder.append(LatLonBuilder(build_from="azgs_r"))
        elif "x" in self.dataset_params["features"]:
            builder.append(LatLonBuilder(build_from="xy", projection=EuroPP()))

        return builder

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


class AE(Abstract):
    """Abstract class for Autoencoders"""

    _required_hparams = Abstract._required_hparams + [
        "encoding_dim",
        "h_dims",
    ]

    def __init__(
        self, dataset_params: DatasetParams, config: Union[Dict, Namespace]
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder: nn.Module
        self.decoder: nn.Module
        self.out_activ: nn.Module

    def configure_optimizers(self) -> dict:
        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams, {"hp/valid_loss": 1, "hp/test_loss": 1}
        )

    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        x_hat = self.out_activ(self.decoder(z, lengths))
        return z, x_hat

    def training_step(self, batch, batch_idx):
        x, l, _ = batch
        _, x_hat = self.forward(x, l)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, l, _ = batch
        _, x_hat = self.forward(x, l)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, l, info = batch
        _, x_hat = self.forward(x, l)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, l, x_hat, info

    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     """FIXME: too messy."""
    #     idx = 0
    #     original = outputs[0][0][idx].unsqueeze(0).cpu().numpy()
    #     length = int(outputs[0][1][idx].cpu().item())
    #     reconstruct = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
    #     data = np.concatenate((original, reconstruct), axis=0)
    #     n_samples = data.shape[0]
    #     data = data.reshape((n_samples, -1))
    #     # remove padding
    #     data = data[:, : length * len(self.dataset_params["features"])]
    #     # reshape for unscaling
    #     data = data.reshape((-1, len(self.dataset_params["features"])))
    #     # unscale the data
    #     if self.dataset_params["scaler"] is not None:
    #         data = self.dataset_params["scaler"].inverse_transform(data)

    #     data = data.reshape((n_samples, -1))
    #     # add info if needed (init_features)
    #     info_len = len(self.dataset_params["info_params"]["features"])
    #     if info_len > 0:
    #         info = outputs[0][3][idx].unsqueeze(0).cpu().numpy()
    #         info = np.repeat(info, data.shape[0], axis=0)
    #         data = np.concatenate((info, data), axis=1)
    #     # get builder
    #     builder = self.get_builder(
    #         nb_samples=n_samples,
    #         length=length,
    #     )
    #     features = [
    #         "track" if "track" in f else f
    #         for f in self.dataset_params["features"]
    #     ]
    #     # build traffic
    #     traffic = traffic_from_data(
    #         data,
    #         features,
    #         self.dataset_params["info_params"]["features"],
    #         builder=builder,
    #     )
    #     # generate plot then send it to logger
    #     self.logger.experiment.add_figure(
    #         "original vs reconstructed", plot_traffic(traffic)
    #     )

    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     """FIXME: too messy."""
    #     idx = 0
    #     original = outputs[0][0][idx].unsqueeze(0).cpu()
    #     length = int(outputs[0][1][idx].cpu().item())
    #     reconstruct = outputs[0][2][idx].unsqueeze(0).cpu()
    #     data = torch.cat((original, reconstruct), axis=0)
    #     n_samples = data.size(0)
    #     data = data.view((n_samples, -1))
    #     # remove padding
    #     data = data[:, : length * len(self.dataset_params["features"])]
    #     # reshape for unscaling
    #     data = data.reshape((-1, len(self.dataset_params["features"])))
    #     # unscale the data
    #     if self.dataset_params["scaler"] is not None:
    #         data = self.dataset_params["scaler"].inverse_transform(data)

    #     data = data.numpy().reshape((n_samples, -1))
    #     # add info if needed (init_features)
    #     info_len = len(self.dataset_params["info_params"]["features"])
    #     if info_len > 0:
    #         info = outputs[0][3][idx].unsqueeze(0).cpu().numpy()
    #         info = np.repeat(info, data.shape[0], axis=0)
    #         data = np.concatenate((info, data), axis=1)
    #     # get builder
    #     builder = self.get_builder(
    #         nb_samples=n_samples,
    #         length=length,
    #     )
    #     features = [
    #         "track" if "track" in f else f
    #         for f in self.dataset_params["features"]
    #     ]
    #     # build traffic
    #     traffic = traffic_from_data(
    #         data,
    #         features,
    #         self.dataset_params["info_params"]["features"],
    #         builder=builder,
    #     )
    #     # generate plot then send it to logger
    #     self.logger.experiment.add_figure(
    #         "original vs reconstructed", plot_traffic(traffic)
    #     )

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """FIXME: too messy."""
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu()
        reconstructed = outputs[0][2][idx].unsqueeze(0).cpu()
        data = torch.cat((original, reconstructed), dim=0)
        data = data.reshape((data.shape[0], -1))
        # unscale the data
        if self.dataset_params["scaler"] is not None:
            data = self.dataset_params["scaler"].inverse_transform(data)

        data = data.numpy()
        # add info if needed (init_features)
        if len(self.dataset_params["info_params"]["features"]) > 0:
            info = outputs[0][3][idx].unsqueeze(0).cpu().numpy()
            info = np.repeat(info, data.shape[0], axis=0)
            data = np.concatenate((info, data), axis=1)
        # get builder
        builder = self.get_builder(nb_samples=2, length=200)
        features = [
            "track" if "track" in f else f for f in self.hparams.features
        ]
        # build traffic
        traffic = traffic_from_data(
            data,
            features,
            self.dataset_params["info_params"]["features"],
            builder=builder,
        )
        # generate plot then send it to logger
        self.logger.experiment.add_figure(
            "original vs reconstructed", plot_traffic(traffic)
        )

    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     """FIXME: too messy."""
    #     idx = 0
    #     original = outputs[0][0][idx].unsqueeze(0).cpu().numpy()
    #     reconstructed = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
    #     data = np.concatenate((original, reconstructed), axis=0)
    #     data = data.reshape((data.shape[0], -1))
    #     # unscale the data
    #     if self.dataset_params["scaler"] is not None:
    #         data = self.dataset_params["scaler"].inverse_transform(data)
    #     # add info if needed (init_features)
    #     if len(self.dataset_params["info_params"]["features"]) > 0:
    #         info = outputs[0][3][idx].unsqueeze(0).cpu().numpy()
    #         info = np.repeat(info, data.shape[0], axis=0)
    #         data = np.concatenate((info, data), axis=1)
    #     # get builder
    #     builder = self.get_builder(nb_samples=2, length=200)
    #     features = [
    #         "track" if "track" in f else f for f in self.hparams.features
    #     ]
    #     # build traffic
    #     traffic = traffic_from_data(
    #         data,
    #         features,
    #         self.dataset_params["info_params"]["features"],
    #         builder=builder,
    #     )
    #     # generate plot then send it to logger
    #     self.logger.experiment.add_figure(
    #         "original vs reconstructed", plot_traffic(traffic)
    #     )

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        _, parser = super().add_model_specific_args(parent_parser)
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

        return parent_parser, parser


class VAE(AE):
    """Abstract class for Variational Autoencoder"""

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.scale = nn.Parameter(torch.Tensor([self.hparams.scale]))

    def forward(self, x, lengths):
        # encode x to get the location and log variance parameters
        loc, log_var = self.encoder(x, lengths)
        # sample z from q(z|x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(loc, std)
        z = q.rsample()
        # decode z
        x_hat = self.out_activ(self.decoder(z, lengths))
        return z, (loc, std), x_hat

    def training_step(self, batch, batch_idx):
        x, l, _ = batch
        z, (loc, std), x_hat = self.forward(x, l)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, x)
        # recon_loss = -F.mse_loss(x, x_hat, reduce=False).sum(dim=(1, 2))

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

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kl.mean(),
                "recon_loss": recon_loss.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x, l, _ = batch
        _, _, x_hat = self.forward(x, l)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, l, info = batch
        _, _, x_hat = self.forward(x, l)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, l, x_hat, info

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


class GAN(LightningModule):
    """Abstract class for Generative Adversarial Network"""

    _required_hparams = [
        "learning_rate",
        "latent_dim",
        "h_dims",
        "dropout",
    ]

    def __init__(
        self,
        x_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__()

        self._check_hparams(config)

        self.out_dim = x_dim
        self.seq_len = seq_len
        self.scaler = scaler
        self.save_hyperparameters(config)

        self.generator: nn.Module
        self.discriminator: nn.Module

        self.example_input_array = torch.randn(1, self.hparams.latent_dim)

    def configure_optimizers(self) -> Tuple[list, list]:
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"hp/valid_loss": 1})

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _, _ = batch

        # sample noise
        z = torch.randn(x.size(0), self.hparams.latent_dim)
        z = z.type_as(x)

        # train generator
        if optimizer_idx == 0:
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training loop
            valid = torch.ones(x.size(0), 1, self.seq_len)
            valid = valid.type_as(x)

            # adversarial loss is binary cross-entropy
            # print(valid.size(), self.discriminator(self(z)).size())
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            # tqdm_dict = {"g_loss": g_loss}
            # output = OrderedDict(
            #     {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            # )
            self.log("hp/valid_loss", g_loss)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated
            # samples

            # how well can it label as real?
            valid = torch.ones(x.size(0), 1, self.seq_len)
            valid = valid.type_as(x)

            real_loss = self.adversarial_loss(self.discriminator(x), valid)

            # how well can it label as fake?
            fake = torch.zeros(x.size(0), 1, self.seq_len)
            fake = fake.type_as(x)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            # tqdm_dict = {"d_loss": d_loss}
            # output = OrderedDict(
            #     {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            # )
            self.log("d_loss", d_loss)
            return d_loss

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

    def get_builder(self, nb_samples: int) -> CollectionBuilder:
        builder = CollectionBuilder(
            [IdentifierBuilder(nb_samples, self.seq_len), TimestampBuilder()]
        )
        if "track_unwrapped" in self.hparams.features:
            builder.append(LatLonBuilder(build_from="azgs"))
        elif "x" in self.hparams.features:
            builder.append(LatLonBuilder(build_from="xy", projection=EuroPP()))

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
        parser.add_argument(
            "--latent_dim",
            dest="latent_dim",
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
