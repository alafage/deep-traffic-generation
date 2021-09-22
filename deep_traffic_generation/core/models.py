from argparse import Namespace
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data.dataset import TensorDataset

from .utils import get_dataloaders


class TCVAE:
    def __init__(encoding_dim: int, h_dims: List[int]) -> None:
        pass

    def data_shape(self) -> str:
        raise NotImplementedError()

    def set_seed(self, seed: int) -> None:
        pl.seed_everything(int, workers=True)

    def fit(self, X: torch.Tensor):
        """[summary]

        Args:
            X (torch.Tensor): [description]
        """
        # ------------
        # data
        # ------------
        dataset = TensorDataset(X)
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset,
            self.train_ratio,
            self.val_ratio,
            self.batch_size,
            self.test_batch_size,
        )

        # ------------
        # logger
        # ------------
        tb_logger = (
            TensorBoardLogger(
                "lightning_logs/",
                name=self.network_name(),
                default_hp_metric=False,
                log_graph=True,
            )
            if self.use_tensorboard
            else False
        )

        # ------------
        # training
        # ------------
        checkpoint_callback = ModelCheckpoint(monitor="hp/valid_loss")
        if self.early_stop is not None:
            early_stopping = EarlyStopping(
                "hp/valid_loss", patience=self.early_stop
            )
            trainer = Trainer.from_argparse_args(
                Namespace(**self.trainer_params),
                callbacks=[checkpoint_callback, early_stopping],
                logger=tb_logger if self.use_tensorboard else False,
            )
        else:
            trainer = Trainer.from_argparse_args(
                Namespace(**self.trainer_params),
                callbacks=[checkpoint_callback],
                logger=tb_logger,
            )

        if val_loader is not None:
            trainer.fit(self, train_loader, val_loader)
        else:
            trainer.fit(self, train_loader)

        # ------------
        # testing
        # ------------
        if test_loader is not None:
            trainer.test(test_dataloaders=test_loader)
