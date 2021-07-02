from typing import Protocol

import numpy as np
import pandas as pd


class BuilderProtocol(Protocol):
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


class TransformerProtocol(Protocol):
    def fit(self, X: np.ndarray) -> "TransformerProtocol":
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        ...
