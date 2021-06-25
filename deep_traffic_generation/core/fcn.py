# fmt: off
from typing import List, Optional

import torch.nn as nn


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

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
