from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from traffic.core import Traffic

from .protocols import TransformerProtocol


class Infos(TypedDict):
    features: List[str]
    index: Optional[int]


class DatasetParams(TypedDict):
    features: List[str]
    file_path: Optional[Path]
    info_params: Infos
    input_dim: int
    scaler: Optional[TransformerProtocol]
    shape: str


class TrafficDataset(Dataset):
    """Traffic Dataset

    Note: handles variable trajectory lengths.

    Parameters
    ----------
    shape: str (default='sequence')
        Define the shape of the data: `linear` (N, HxL), `image` (N, H, L) and
        `sequence` (N, L, H).
        With N the number of instances, H the number of features and L the
        sequence length.
    """

    _repr_indent = 4
    _available_shapes = ["linear", "sequence", "image"]

    def __init__(
        self,
        traffic: Traffic,
        features: List[str],
        shape: str = "sequence",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> None:
        super().__init__()

        assert shape in self._available_shapes, (
            f"{shape} shape is not available. "
            + f"Available shapes are: {self._available_shapes}"
        )

        self.file_path: Optional[Path] = None
        self.features = features
        self.shape = shape
        self.scaler = scaler
        self.info_params = info_params

        self.data: torch.Tensor
        self.lengths: List[int]
        self.infos: List[Any]

        # fit scaler
        tmp = traffic.data[features].values
        self.scaler = self.scaler.fit(tmp)

        # extract features and lengths
        data = [
            [
                torch.FloatTensor(
                    self.scaler.transform(
                        f.data[features]
                        .values.ravel()
                        .reshape(-1, len(features))
                    )
                ),
                len(f),
            ]
            for f in traffic
        ]

        feats, self.lengths = zip(*data)
        self.data = pad_sequence(feats, batch_first=True)

        if self.shape == "linear":
            self.data = self.data.view(self.__len__(), -1)
        elif self.shape == "image":
            self.data = torch.transpose(self.data, 1, 2)

        # extract infos if needed
        self.infos = []
        # TODO: change condition (if not is_empty(self.info_params))
        if self.info_params["index"] is not None:
            self.infos = [
                f.data[self.info_params["features"]]
                .iloc[self.info_params["index"]]
                .values.ravel()
                for f in traffic
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, List[Any]]:
        infos = []
        if self.info_params["index"] is not None:
            infos = self.infos[index]
        return self.data[index], self.lengths[index], infos

    @property
    def input_dim(self) -> int:
        if self.shape in ["linear", "sequence"]:
            return self.data.shape[-1]
        elif self.shape == "image":
            return self.data.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        """Returns sequence length (i.e. maximum sequence length)."""
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.data.shape[1]
        elif self.shape == "image":
            return self.data.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def parameters(self) -> DatasetParams:
        return DatasetParams(
            features=self.features,
            file_path=self.file_path,
            info_params=self.info_params,
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            shape=self.shape,
        )

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> None:
        file_path = (
            file_path if isinstance(file_path, Path) else Path(file_path)
        )
        traffic = Traffic.from_file(file_path)
        dataset = cls(traffic, features, shape, scaler, info_params)
        dataset.file_path = file_path
        return dataset

    def __repr__(self) -> str:
        """TODO: add shape details"""
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.file_path is not None:
            body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
