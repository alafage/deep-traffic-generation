from pathlib import Path
from typing import List, Optional, Tuple, Union

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
        label: Optional[str] = None,
        seq_mode: bool = False,
    ) -> None:
        self.file_path = (
            file_path if isinstance(file_path, Path) else Path(file_path)
        )
        self.features = features
        # self.target_transform = target_transform

        traffic = Traffic.from_file(self.file_path)
        # extract features
        self.data = extract_features(traffic, self.features)
        self.labels: Optional[np.ndarray] = None

        if label is not None:
            self.labels = np.array([f._get_unique(label) for f in traffic])

        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)

        if seq_mode:
            self.data = self.data.reshape(self.data.shape[0], -1, len(self.features))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        trajectory = torch.Tensor(self.data[idx])
        label = 0
        if self.labels is not None:
            label = self.labels[idx]

        return trajectory, label

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.file_path is not None:
            body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
