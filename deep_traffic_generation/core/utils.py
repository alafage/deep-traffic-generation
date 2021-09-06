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
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats._distn_infrastructure import rv_continuous
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from traffic.core import Traffic
from traffic.core.projection import EuroPP

from deep_traffic_generation.core.datasets import Infos, TrafficDataset

from .protocols import BuilderProtocol


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
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
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

    if val_size > 0:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=test_batch_size
            if test_batch_size is not None
            else len(val_dataset),
            shuffle=True,
            num_workers=num_workers,
        )
    else:
        val_loader = None

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
        test_loader = None

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
    cls: LightningModule,
    dataset_cls: TrafficDataset,
    data_shape: str,
    seed: int = 42,
) -> None:
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
        "--info_features",
        dest="info_features",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--info_index",
        dest="info_index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--label",
        dest="label",
        type=str,
        default=None,
    )
    parser.add_argument("--navpts", dest="navpts", action="store_true")
    parser.add_argument("--no_navpts", dest="navpts", action="store_false")
    parser.set_defaults(navpts=False)
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
    parser.add_argument(
        "--early_stop", dest="early_stop", type=int, default=None
    )
    parser.add_argument(
        "--show_latent", dest="show_latent", action="store_true"
    )
    parser.add_argument(
        "--no_show_latent", dest="show_latent", action="store_false"
    )
    parser.set_defaults(show_latent=False)
    parser = Trainer.add_argparse_args(parser)
    parser, _ = cls.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = dataset_cls.from_file(
        args.data_path,
        features=args.features,
        shape=data_shape,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        info_params=Infos(features=args.info_features, index=args.info_index),
    )
    print(dataset.input_dim)
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
        dataset_params=dataset.parameters,
        config=args,
    )

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(monitor="hp/valid_loss")
    # checkpoint_callback = ModelCheckpoint()
    if args.early_stop is not None:
        early_stopping = EarlyStopping(
            "hp/valid_loss", patience=args.early_stop
        )
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
        )
    else:
        trainer = Trainer.from_argparse_args(
            args, callbacks=[checkpoint_callback], logger=tb_logger
        )

    if val_loader is not None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    if test_loader is not None:
        trainer.test(test_dataloaders=test_loader)


def build_weights(size: int, builder: rv_continuous, **kwargs) -> np.ndarray:
    """Build weight array according to a density law."""
    w = np.array(
        [builder.pdf(i / (size + 1), **kwargs) for i in range(1, size + 1)]
    )
    return w


def plot_clusters(traffic: Traffic, cluster_label: str = "cluster") -> Figure:
    assert (
        cluster_label in traffic.data.columns
    ), f"Underlying dataframe should have a {cluster_label} column"
    clusters = sorted(list(traffic.data[cluster_label].value_counts().keys()))
    n_clusters = len(clusters)
    # -- dealing with the grid
    if n_clusters > 3:
        nb_cols = 3
        nb_lines = n_clusters // nb_cols + ((n_clusters % nb_cols) > 0)

        with plt.style.context("traffic"):
            fig, axs = plt.subplots(
                nb_lines,
                nb_cols,
                figsize=(10, 15),
                subplot_kw=dict(projection=EuroPP()),
            )

            for n, cluster in enumerate(clusters):
                ax = axs[n // nb_cols][n % nb_cols]
                ax.set_title(f"cluster {cluster}")
                t_cluster = traffic.query(f"{cluster_label} == {cluster}")
                t_cluster.plot(ax, alpha=0.5)
                t_cluster.centroid(nb_samples=None, projection=EuroPP()).plot(
                    ax, color="red", alpha=1
                )
    else:
        with plt.style.context("traffic"):
            fig, axs = plt.subplots(
                n_clusters,
                figsize=(10, 15),
                subplot_kw=dict(projection=EuroPP()),
            )

            for n, cluster in enumerate(clusters):
                ax = axs[n]
                ax.set_title(f"cluster {cluster}")
                t_cluster = traffic.query(f"{cluster_label} == {cluster}")
                t_cluster.plot(ax, alpha=0.5)
                t_cluster.centroid(nb_samples=None, projection=EuroPP()).plot(
                    ax, color="red", alpha=1
                )
    return fig


def unpad_sequence(padded: torch.Tensor, lengths: torch.Tensor) -> List:
    return [padded[i][: lengths[i]] for i in range(len(padded))]
