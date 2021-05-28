from typing import List, Optional, Protocol, Tuple

import numpy as np
from torch.utils.data import DataLoader, random_split
from traffic.core import Traffic


class DatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...


def extract_features(
    traffic: Traffic,
    features: List[str],
) -> np.ndarray:
    """Extract features from Traffic data according to the feature list.

    Parameters
    ----------
    traffic: Traffic
    features: str-list
        Labels of the columns to extract from the underlying dataframe of
        Traffic object.

    Returns
    -------
    np.ndarray
        Feature vector.
    """
    X = np.stack(list(f.data[features].values.ravel() for f in traffic))

    return X


def get_dataloaders(
    dataset: DatasetProtocol,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    test_batch_size: Optional[int],
    num_workers: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    val_size = int(train_size * val_ratio)
    train_size -= val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=test_batch_size
        if test_batch_size is not None
        else len(val_dataset),
        shuffle=True,
        num_workers=num_workers,
    )

    if test_size > 0:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size
            if test_batch_size is not None
            else len(val_dataset),
            shuffle=True,
            num_workers=num_workers,
        )
    else:
        test_loader = val_loader

    return train_loader, val_loader, test_loader
