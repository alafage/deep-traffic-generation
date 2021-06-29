from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from traffic.core import Traffic
from traffic.core.projection import EuroPP

from .builders import BuilderProtocol


def extract_features(
    traffic: Traffic,
    features: List[str],
    init_features: List[str] = [],
) -> np.ndarray:
    """Extract features from Traffic data according to the feature list.

    Parameters
    ----------
    traffic: Traffic
    features: List[str]
        Labels of the columns to extract from the underlying dataframe of
        Traffic object.
    init_features: List[str]
        Labels of the features to extract from the first row of each Flight
        underlying dataframe.
    Returns
    -------
    np.ndarray
        Feature vector `(N, HxL)` with `N` number of flights, `H` the number
        of features and `L` the sequence length.
    """
    X = np.stack(list(f.data[features].values.ravel() for f in traffic))

    if len(init_features) > 0:
        init_ = np.stack(
            list(f.data[init_features].iloc[0].values.ravel() for f in traffic)
        )
        X = np.concatenate((init_, X), axis=1)

    return X


def get_dataloaders(
    dataset: Dataset,
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
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        test_loader = val_loader

    return train_loader, val_loader, test_loader


# fmt: off
def init_dataframe(
    data: np.ndarray, features: List[str], init_features: List[str] = [],
) -> pd.DataFrame:
    """ TODO:
    """
    # handle dense features (features)
    dense: np.ndarray = data[:, len(init_features):]
    nb_samples = data.shape[0]
    dense = dense.reshape(nb_samples, -1, len(features))
    nb_obs = dense.shape[1]

    # handle sparce features (init_features)
    if len(init_features) > 0:
        sparce = data[:, :len(init_features)]
        sparce = sparce[:, np.newaxis]
        sparce = np.insert(
            sparce, [1] * (nb_obs - 1), [np.nan] * len(init_features), axis=1
        )
        dense = np.concatenate((dense, sparce), axis=2)
        features = features + init_features

    # generate dataframe
    df = pd.DataFrame(
        {feature: dense[:, :, i].ravel() for i, feature in enumerate(features)}
    )

    return df


# fmt: on
def traffic_from_data(
    data: np.ndarray,
    features: List[str],
    init_features: List[str] = [],
    builder: Optional[BuilderProtocol] = None,
) -> Traffic:

    df = init_dataframe(data, features, init_features)

    if builder is not None:
        df = builder(df)

    return Traffic(df)


def plot_traffic(traffic: Traffic) -> Figure:
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            1, figsize=(5, 5), subplot_kw=dict(projection=EuroPP())
        )
        traffic[1].plot(ax, c="orange", label="reconstructed")
        traffic[0].plot(ax, c="purple", label="original")
        ax.legend()

    return fig


"""
    Function below from https://github.com/JulesBelveze/time-series-autoencoder
"""


def init_hidden(
    x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True
):
    """
    Initialize hidden.
    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(
            torch.zeros(num_dir, x.size(0), hidden_size)
        ).to(x.device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(x.device)


def cli_main(
    cls: LightningModule, dataset_cls: Dataset, data_mode: str, seed: int = 42
) -> None:
    """TODO: define `AE` to type `cls`."""
    pl.seed_everything(seed, workers=True)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=Path,
        default=Path("./data/denoised_v3.pkl").absolute(),
    )
    parser.add_argument(
        "--features",
        dest="features",
        nargs="+",
        default=["latitude", "longitude", "altitude", "timedelta"],
    )
    parser.add_argument(
        "--init_features",
        dest="init_features",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--train_ratio", dest="train_ratio", type=float, default=0.8
    )
    parser.add_argument(
        "--val_ratio", dest="val_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1000
    )
    parser.add_argument(
        "--test_batch_size",
        dest="test_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--early_stop", dest="early_stop", action="store_true")
    parser.add_argument(
        "--no_early_stop", dest="early_stop", action="store_false"
    )
    parser.set_defaults(early_stop=False)
    parser.add_argument(
        "--show_latent", dest="show_latent", action="store_true"
    )
    parser.add_argument(
        "--no_show_latent", dest="show_latent", action="store_false"
    )
    parser.set_defaults(show_latent=False)
    parser = Trainer.add_argparse_args(parser)
    parser = cls.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = dataset_cls(
        args.data_path,
        features=args.features,
        init_features=args.init_features,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        mode=data_mode,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        args.train_ratio,
        args.val_ratio,
        args.batch_size,
        args.test_batch_size,
    )

    # ------------
    # logger
    # ------------
    tb_logger = TensorBoardLogger(
        "lightning_logs/",
        name=args.network_name,
        default_hp_metric=False,
        log_graph=True,
    )

    # ------------
    # model
    # ------------
    model = cls(
        input_dim=dataset.input_dim,
        seq_len=dataset.seq_len,
        scaler=dataset.scaler,
        config=args,
    )

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(monitor="hp/valid_loss")
    if args.early_stop:
        early_stopping = EarlyStopping("hp/valid_loss")
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
        )
    else:
        trainer = Trainer.from_argparse_args(
            args, callbacks=[checkpoint_callback], logger=tb_logger
        )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)
