# fmt: off
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from deep_traffic_generation.core import RNN, VAE
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.protocols import TransformerProtocol
from deep_traffic_generation.core.utils import cli_main


# fmt: on
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        dropout: float,
        num_layers: int,
        batch_first: bool,
    ) -> None:
        super().__init__()

        self.batch_first = batch_first

        self.encoder = RNN(
            input_dim=input_dim,
            out_dim=h_dims[-1],
            h_dims=h_dims[:-1],
            dropout=dropout,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.z_loc = nn.Linear(h_dims[-1], out_dim)
        self.z_log_var = nn.Linear(h_dims[-1], out_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0)
        return self.z_loc(h), self.z_log_var(h)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        seq_len: int,
        dropout: float,
        num_layers: int,
        batch_first: bool,
    ) -> None:
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.decoder = RNN(
            input_dim=input_dim,
            out_dim=h_dims[-1],
            h_dims=[input_dim] + h_dims[:-1],
            dropout=dropout,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.fc = nn.Linear(h_dims[-1], out_dim)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (_, _) = self.decoder(x)
        return self.fc(x)


# packed_x = pack_padded_sequence(
#             x,
#             lengths,
#             batch_first=self.batch_first,
#             enforce_sorted=False,
#         )


class LstmVAEVL(VAE):
    """LSTM Variational Autoencoder"""

    def __init__(
        self,
        x_dim: int,
        seq_len: int,
        scaler: Optional[TransformerProtocol],
        navpts: Optional[torch.Tensor],
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(x_dim, seq_len, scaler, navpts, config)

        self.scale = nn.Parameter(torch.Tensor([1.0]))

        self.example_input_array = torch.rand(
            (self.seq_len, self.input_dim)
        ).unsqueeze(0)

        self.encoder = Encoder(
            input_dim=self.input_dim,
            out_dim=self.hparams.encoding_dim,
            h_dims=self.hparams.h_dims,
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = Decoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.input_dim,
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.seq_len,
            dropout=self.hparams.dropout,
            num_layers=1,
            batch_first=True,
        )

        self.out_activ = nn.Tanh()

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed_x = pack_padded_sequence(
                x,
                lengths,
                batch_first=self.batch_first,
                enforce_sorted=False,
            )
            z, (loc, std), packed_x_hat = super().forward(packed_x)
            x_hat = pad_sequence(packed_x_hat, batch_first=True)
            return z, (loc, std), x_hat
        else:
            return super().forward(x)

    @classmethod
    def network_name(cls) -> str:
        return "lstm_vae_vl"


if __name__ == "__main__":
    cli_main(LstmVAEVL, TrafficDataset, "sequence", seed=42)
