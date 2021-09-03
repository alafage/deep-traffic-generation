# fmt: off
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# fmt: on
class FCN(nn.Module):
    """Fully Connected Network"""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        layers = []

        for index in range(self.n_layers):
            layer = nn.Linear(
                in_features=layer_dims[index],
                out_features=layer_dims[index + 1],
            )
            layers.append(layer)
            if (index != self.n_layers - 1) and h_activ is not None:
                layers.append(h_activ)

            if (index != self.n_layers - 1) and (dropout > 0):
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, lengths):
        return self.layers(x)


class RNN(nn.Module):
    """Recurrent Network (LSTM based)"""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        dropout: float = 0.0,
        num_layers: int = 1,
        batch_first: bool = False,
    ) -> None:
        super().__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()

        for index in range(self.n_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=num_layers,
                batch_first=batch_first,
                # bidirectional=True,
            )
            self.layers.append(layer)

        self.dropout = dropout

    def forward(self, x):
        """FIXME: enable dropout"""
        for layer in self.layers:
            x, (h_n, c_n) = layer(x)

        return x, (h_n, c_n)


class TemporalBlock(nn.Module):
    """Temporal Block, a causal and dilated convolution with activation
    and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        out_activ: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
            )
        )

        self.out_activ = out_activ
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0), "constant", 0)
        x = self.conv(x)
        x = self.out_activ(x) if self.out_activ is not None else x
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        h_activ: Optional[nn.Module] = None,
        is_last: bool = False,
    ) -> None:
        super().__init__()

        self.is_last = is_last

        self.tmp_block1 = TemporalBlock(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ,
        )

        self.tmp_block2 = TemporalBlock(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            h_activ if not is_last else None,  # deactivate last activation
        )

        # Optional convolution for matching in_channels and out_channels sizes
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        # self.out_activ = h_activ
        self.init_weights()

    def init_weights(self) -> None:
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.tmp_block1(x)
        y = self.tmp_block2(y)
        r = x if self.downsample is None else self.downsample(x)
        return y + r


class TCN(nn.Module):
    """Temporal Convolutional Network.

    Handle data shaped (N, C, L) with N the batch size, C the number
    of channels and L the length of signal sequence.

    Parameters
    ----------
    input_dim: int
        Number of channels in the input sequence.
    out_dim: int
        Number of channels produced by the network.
    h_dims:  List[int]
        Number of channels outputed by the hidden layers.
    kernel_size: int
        Size of the convolving kernel
    dilation_base: int
        Size of the dilation base to compute the dilation of a specific layer.
    activation: torch.nn.Module
        Activation function to use within temporal blocks.
    dropout: float (default=0.2)
        probability of an element to be zeroed.

    Source
    ------
    Adaptation from the code source of the paper `An Empirical Evaluation of
    Generic Convolutional and Recurrent Networks for Sequence Modeling` by
    Shaojie Bai, J. Zico Kolter and Vladlen Koltun.
    See https://arxiv.org/abs/1803.01271

    Notes
    -----
    Keep the kernel size at least as big as the dilation base to avoid any
    holes in your receptive field.
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        kernel_size: int,
        dilation_base: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer_dims = [input_dim] + h_dims + [out_dim]
        self.n_layers = len(layer_dims) - 1
        layers = []

        for index in range(self.n_layers):
            dilation = dilation_base ** index
            in_channels = layer_dims[index]
            out_channels = layer_dims[index + 1]
            is_last = index == (self.n_layers - 1)
            layer = ResidualBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
                h_activ,
                is_last,
            )
            layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
