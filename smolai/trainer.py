from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Type

import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from smolai.callbacks import Callback, CallbackManager
from smolai.metrics import Metric
from smolai.utils import to_device


def expand_callback_factories(callbacks_and_callback_factories):
    """Turn callback factories into callbacks."""
    cbs = []
    for cb in callbacks_and_callback_factories:
        if isinstance(cb, type):  # if the callback is a class
            cbs.extend(cb.as_factory())
        else:  # if the callback is an object instance
            cbs.append(cb)
    return cbs


@dataclass
class Trainer:
    """Train a model

    Args:
        model (Module): pytorch model.
        criterion (Callable): loss function.
        opt_func (Callable): optimizer function.
        callbacks (Sequence[Callback], optional): Callbacks and callback factories."""

    model: Module
    criterion: Callable
    opt_func: Callable[..., Optimizer]
    callbacks: Optional[Sequence[Callback]] = None

    def __post_init__(self):
        if self.callbacks is None:
            self.callbacks = []
        self.callbacks = expand_callback_factories(self.callbacks)
        self.model = to_device(self.model)
        self.callback_manager = CallbackManager(self.callbacks, context=self)

    @logger.catch
    def fit(
        self,
        train_dl: DataLoader,
        test_dl: Optional[DataLoader] = None,
        n_epochs: int = 1,
        lr: float = 1e-3,
    ):
        """Fit a model.

        Args:
            train_dl (DataLoader): training data loader.
            test_dl (DataLoader, optional): evaluation data loader. If None, skip evaluation.
            n_epochs (int, optional): Number of epochs. Defaults to 1.
            lr (float, optional): Learning rate. Defaults to 1e-3."""

        self.n_epochs = n_epochs
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.lr = lr
        self.opt = self.opt_func(self.model.parameters(), lr=self.lr)

        with self.callback_manager.fit(), plt.style.context("bmh"):
            for self.epoch in range(self.n_epochs):
                with self.callback_manager.epoch():
                    with self.callback_manager.train():
                        self.one_epoch(self.train_dl, training=True)
                    if test_dl:
                        with self.callback_manager.test(), torch.no_grad():
                            self.one_epoch(self.test_dl, training=False)

        return self

    def one_epoch(self, dl: DataLoader, training: bool) -> List[Metric]:
        self.model.training = training
        for self.batch_idx, self.batch in enumerate(dl):
            with self.callback_manager.batch():
                self.X, self.y = self.batch
                self.X, self.y = map(to_device, [self.X, self.y])
                self.y_pred = self.model(self.X)
                self.loss = self.criterion(self.y_pred, self.y)
                if training:
                    self.loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()
