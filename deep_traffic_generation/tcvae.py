# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import TCN, VAE, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDataset
from deep_traffic_generation.core.lsr import GaussianMixtureLSR


# fmt: on
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
        )

    def forward(self, x):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


class TCVAE(VAE):
    """Temporal Convolutional Variational Autoencoder

    Source:
        Inspired from the architecture proposed in the paper `Time Series
        Encodings with Temporal Convolutional Network
        <http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf>`_
        by Markus Thill, Wolfgang Konen and Thomas Bäck.

    Usage example:
        A good model was achieved using this command to train it:

        .. code-block:: console

            python tcvae.py --encoding_dim 32 --h_dims 16 16 16 --features \
track groundspeed altitude timedelta --info_features latitude \
longitude --info_index -1
    """

    _required_hparams = VAE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
        "n_components",
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.example_input_array = torch.rand(
            (
                1,
                self.dataset_params["input_dim"],
                self.dataset_params["seq_len"],
            )
        )

        self.encoder = nn.Sequential(
            TCN(
                input_dim=self.dataset_params["input_dim"],
                out_dim=self.hparams.h_dims[-1],
                h_dims=self.hparams.h_dims[:-1],
                kernel_size=self.hparams.kernel_size,
                dilation_base=self.hparams.dilation_base,
                dropout=self.hparams.dropout,
            ),
            nn.AvgPool1d(self.hparams.sampling_factor),
            nn.Flatten(),
        )

        h_dim = self.hparams.h_dims[-1] * (
            int(self.dataset_params["seq_len"] / self.hparams.sampling_factor)
        )
        # Latent Space Regularization
        self.lsr = GaussianMixtureLSR(
            input_dim=h_dim,
            out_dim=self.hparams.encoding_dim,
            n_components=self.hparams.n_components,
            fix_prior=self.hparams.fix_prior,
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
        self.out_activ = nn.Identity()

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "tcvae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds TCVAE arguments to ArgumentParser.

        List of arguments:

            * ``--dilation``: Dilation base. Default to :math:`2`.
            * ``--kernel``: Size of the kernel to use in Temporal Convolutional
              layers. Default to :math:`16`.
            * ``--n_components``: Number of components for the Gaussian Mixture
              modelling the latent space. Default to :math:`1`.
            * ``--sampling_factor``: Sampling factor to reduce the sequence
              length after Temporal Convolutional layers. Default to
              :math:`10`.

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
            "--n_components", dest="n_components", type=int, default=1
        )

        return parent_parser, parser


if __name__ == "__main__":
    cli_main(TCVAE, TrafficDataset, "image", seed=42)
