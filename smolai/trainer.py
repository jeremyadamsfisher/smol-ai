from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Type

import torch
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from smolai.callbacks import Callback, CallbackManager
from smolai.metrics import Metric
from smolai.utils import to_device


class CancelBatch(Exception):
    pass


class CancelEpoch(Exception):
    pass


class CancelTrain(Exception):
    pass


class CancelTest(Exception):
    pass


@dataclass
class Trainer:
    """Train a model

    Args:
        model (Module): pytorch model.
        criterion (Callable): loss function.
        opt_func (Callable): optimizer function.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        metric_factories (Sequence[Type[Metric]]], optional): Metric factories. Defaults to None.
        callbacks (Sequence[Callback], optional): Callbacks. Defaults to None."""

    model: Module
    criterion: Callable
    opt_func: Callable[..., Optimizer]
    lr: float = 1e-3
    metric_factories: Optional[Sequence[Type[Metric]]] = None
    callbacks: Optional[Sequence[Callback]] = None

    def __post_init__(self):
        if self.metric_factories is None:
            self.metric_factories = []
        if self.callbacks is None:
            self.callbacks = []
        self.model = to_device(self.model)
        self.opt = self.opt_func(self.model.parameters(), lr=self.lr)
        self.batch = None
        self.epochwise_metrics = []
        self.callback_manager = CallbackManager(self.callbacks, context=self)

    @logger.catch
    def fit(
        self,
        train_dl: DataLoader,
        test_dl: Optional[DataLoader] = None,
        n_epochs: int = 1,
    ):
        """Fit a model.

        Args:
            train_dl (DataLoader): training data loader.
            test_dl (DataLoader, optional): evaluation data loader. If None, skip evaluation.
            n_epochs (int, optional): Number of epochs. Defaults to 1."""

        self.n_epochs = n_epochs
        self.train_dl = train_dl
        self.test_dl = test_dl

        with self.callback_manager.fit():
            for self.epoch in range(self.n_epochs):
                with self.callback_manager.epoch():
                    metrics_trn, metrics_tst = None, None
                    with self.callback_manager.train():
                        metrics_trn = self.one_epoch(self.train_dl, training=True)
                    if test_dl:
                        with self.callback_manager.test(), torch.no_grad():
                            metrics_tst = self.one_epoch(self.test_dl, training=False)
                    self.epochwise_metrics.append((metrics_trn, metrics_tst))

        return self

    def one_epoch(self, dl: DataLoader, training: bool) -> List[Metric]:
        self.model.training = training
        self.batch_metrics = [metric() for metric in self.metric_factories]
        for self.batch_idx, self.batch in enumerate(dl):
            with self.callback_manager.batch():
                X, y = self.batch
                X, y = map(to_device, [X, y])
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                if training:
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()
                with torch.no_grad():
                    for metric in self.batch_metrics:
                        metric.add_batch(y=y, y_pred=y_pred, loss=loss)
        return self.batch_metrics
