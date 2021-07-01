from typing import List

from torch import nn


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
            )
            self.layers.append(layer)

        self.dropout = dropout

    def forward(self, x):
        """FIXME: enable dropout"""
        for layer in self.layers:
            x, (h_n, c_n) = layer(x)

        return x, (h_n, c_n)
