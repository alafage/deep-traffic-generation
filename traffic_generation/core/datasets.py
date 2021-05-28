from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from traffic.core import Traffic
from typing_extensions import Protocol

from .utils import extract_features


class TransformerProtocol(Protocol):
    def fit(self, X: np.ndarray) -> "TransformerProtocol":
        return self.fit(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        ...


class TrafficDataset(Dataset):
    """TODO: description"""

    _repr_indent = 4

    def __init__(
        self,
        file_path: Union[str, Path],
        features: List[str],
        scaler: Optional[TransformerProtocol] = None,
    ) -> None:
        self.file_path = (
            file_path if isinstance(file_path, Path) else Path(file_path)
        )
        self.features = features
        # self.target_transform = target_transform

        traffic = Traffic.from_file(self.file_path).query(f"cluster != {-1}")
        # extract features
        self.data = extract_features(traffic, self.features)

        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """TODO"""
        trajectory = torch.Tensor(self.data[idx])
        return trajectory

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.file_path is not None:
            body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
