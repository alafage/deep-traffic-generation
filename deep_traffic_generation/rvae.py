# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from deep_traffic_generation.core import RNN, VAE, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.lsr import GaussianMixtureLSR

# fmt: on
# class Encoder(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         out_dim: int,
#         h_dims: List[int],
#         dropout: float,
#         num_layers: int,
#         batch_first: bool,
#     ) -> None:
#         super().__init__()

#         self.encoder = RNN(
#             input_dim=input_dim,
#             out_dim=h_dims[-1],
#             h_dims=h_dims[:-1],
#             dropout=dropout,
#             num_layers=num_layers,
#             batch_first=batch_first,
#         )

#         self.z_loc = nn.Linear(h_dims[-1], out_dim)
#         self.z_log_var = nn.Linear(h_dims[-1], out_dim)

#     def forward(self, x, lengths):
#         _, (h, _) = self.encoder(x)
#         h = h.squeeze(0)
#         return self.z_loc(h), self.z_log_var(h)


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         out_dim: int,
#         h_dims: List[int],
#         seq_len: int,
#         dropout: float,
#         num_layers: int,
#         batch_first: bool,
#     ) -> None:
#         super(Decoder, self).__init__()

#         self.seq_len = seq_len
#         self.decoder = RNN(
#             input_dim=input_dim,
#             out_dim=h_dims[-1],
#             h_dims=[input_dim] + h_dims[:-1],
#             dropout=dropout,
#             num_layers=num_layers,
#             batch_first=batch_first,
#         )

#         self.fc = nn.Linear(h_dims[-1], out_dim)

#     def forward(self, x, lengths):
#         x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
#         x, (_, _) = self.decoder(x)
#         return self.fc(x)


class RVAE(VAE):
    """Recurrent Variational Autoencoder"""

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

        # self.encoder = Encoder(
        #     input_dim=self.dataset_params["input_dim"],
        #     out_dim=self.hparams.encoding_dim,
        #     h_dims=self.hparams.h_dims,
        #     dropout=self.hparams.dropout,
        #     num_layers=1,
        #     batch_first=True,
        # )

        # self.decoder = Decoder(
        #     input_dim=self.hparams.encoding_dim,
        #     out_dim=self.dataset_params["input_dim"],
        #     h_dims=self.hparams.h_dims[::-1],
        #     seq_len=self.dataset_params["seq_len"],
        #     dropout=self.hparams.dropout,
        #     num_layers=1,
        #     batch_first=True,
        # )
        self.encoder = RNN(
            input_dim=self.dataset_params["input_dim"],
            out_dim=self.hparams.h_dims[-1],
            h_dims=self.hparams.h_dims[:-1],
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        # Latent Space Regularization
        self.lsr = GaussianMixtureLSR(
            input_dim=self.hparams.h_dims[-1],
            out_dim=self.hparams.encoding_dim,
            n_components=self.hparams.n_components,
            fix_prior=self.hparams.fix_prior,
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
        q = self.lsr(h.squeeze(0))
        z = q.rsample()
        y = z.unsqueeze(1).repeat(1, self.dataset_params["seq_len"], 1)
        x_hat, _ = self.decoder(y)
        x_hat = self.out_activ(x_hat)
        return self.lsr.dist_params(q), z, x_hat

    @classmethod
    def network_name(cls) -> str:
        return "rvae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds RVAE arguments to ArgumentParser.

        List of arguments:

            * ``--n_components``: Number of components for the Gaussian Mixture
              modelling the latent space. Default to :math:`1`.

        .. note::
            It adds also the argument of the inherited class `VAE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--n_components", dest="n_components", type=int, default=1
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(RVAE, TrafficDataset, "sequence", seed=42)
