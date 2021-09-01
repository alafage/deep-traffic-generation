from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from traffic.core import Traffic
from traffic.data import navaids

from .protocols import TransformerProtocol
from .utils import extract_features


class TrafficDataset(Dataset):
    """Traffic Dataset

    Parameters
    ----------
    shape: str (default='linear')
        Define the shape of the data: `linear` (N, HxL), `sequence` (N, L, H)
        and `image` (N, H, L).
        With N the number of instances, H the number of features and L the
        sequence length.
    """

    _repr_indent = 4
    _available_shapes = ["linear", "sequence", "image"]

    def __init__(
        self,
        traffic: Traffic,
        features: List[str],
        init_features: List[str] = [],
        scaler: Optional[TransformerProtocol] = None,
        label: Optional[str] = None,
        navpts: bool = False,
        shape: str = "linear",
    ) -> None:

        assert shape in self._available_shapes, (
            f"{shape} shape is not available. "
            + f"Available shapes are: {self._available_shapes}"
        )

        self.features = features
        self.init_features = init_features
        # self.target_transform = target_transform

        # extract features
        data = extract_features(traffic, features, init_features)
        # get labels
        self.label = label
        self.labels: Optional[np.ndarray] = None

        if label is not None:
            self.labels = np.array([f._get_unique(label) for f in traffic])

        # navpoint coordinates
        self.navpts: Optional[torch.Tensor] = None
        if navpts:
            navids = np.array([f._get_unique("navpoints") for f in traffic])
            self.navids = set(
                [navid for navids_ in navids for navid in navids_.split(",")]
            )
            self.navpts = torch.Tensor(
                [
                    [navaids[navid].lat, navaids[navid].lon]
                    for navid in self.navids
                ]
            )

        # fmt: off
        # separate features from init_features
        self.dense = data[:, len(init_features):]
        self.sparse = data[:, :len(init_features)]

        # fmt: on
        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = self.scaler.fit(self.dense)
            self.dense = self.scaler.transform(self.dense)

        self.shape = shape

        if shape in ["sequence", "image"]:
            self.dense = self.dense.reshape(
                self.dense.shape[0], -1, len(self.features)
            )
            if shape == "image":
                self.dense = self.dense.transpose(0, 2, 1)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        features: List[str],
        init_features: List[str] = [],
        scaler: Optional[TransformerProtocol] = None,
        label: Optional[str] = None,
        navpts: bool = False,
        shape: str = "linear",
    ) -> "TrafficDataset":
        file_path = (
            file_path if isinstance(file_path, Path) else Path(file_path)
        )
        traffic = Traffic.from_file(file_path)
        return cls(
            traffic, features, init_features, scaler, label, navpts, shape
        )

    def __len__(self) -> int:
        return len(self.dense)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List, torch.Tensor]:
        trajectory = torch.Tensor(self.dense[idx])
        info = torch.Tensor(self.sparse[idx])
        label = []
        if self.label is not None:
            label = self.labels[idx]

        return trajectory, label, info

    @property
    def input_dim(self) -> int:
        if self.shape in ["linear", "sequence"]:
            return self.dense.shape[-1]
        elif self.shape == "image":
            return self.dense.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.dense.shape[1]
        elif self.shape == "image":
            return self.dense.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        # if self.file_path is not None:
        #     body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
