from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Type

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from smolai.callbacks import Callback, CallbackManager
from smolai.metrics import Metric
from smolai.utils import DotDict, to_device


def fit(
    model: Module,
    criterion: Callable,
    train_dl: DataLoader,
    test_dl: DataLoader,
    opt_func: Callable[..., Optimizer],
    n_epochs: int = 1,
    lr: float = 1e-3,
    metric_factories: Optional[Sequence[Type[Metric]]] = None,
    callbacks: Optional[Sequence[Callback]] = None,
):
    """Fit a model.

    Args:
        learner (Learner): model and training state.
        train_dl (DataLoader): training data loader.
        test_dl (DataLoader): evaluation data loader.
        opt_func (Callable): optimizer function.
        n_epochs (int, optional): Number of epochs. Defaults to 1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        metric_factories (Sequence[Type[Metric]]], optional): Metric factories. Defaults to None.
        callbacks (Sequence[Callback], optional): Callbacks. Defaults to None."""

    if metric_factories is None:
        metric_factories = []
    if callbacks is None:
        callbacks = []
    model = to_device(model)
    opt = opt_func(model.parameters(), lr=lr)

    context = DotDict(
        model=model,
        criterion=criterion,
        train_dl=train_dl,
        test_dl=test_dl,
        opt=opt,
        n_epochs=n_epochs,
        lr=lr,
        epoch=0,
        epochwise_metrics=[],
    )

    callback = CallbackManager(context, callbacks)

    for context.epoch in range(n_epochs):
        with callback.epoch():
            with callback.train():
                metrics_trn = one_epoch(
                    context, metric_factories, train_dl, training=True
                )
            with callback.test(), torch.no_grad():
                metrics_tst = one_epoch(
                    context, metric_factories, test_dl, training=False
                )
            context.epochwise_metrics.append((metrics_trn, metrics_tst))

    return context


def one_epoch(
    context,
    metric_factories: Sequence[Type[Metric]],
    dl: DataLoader,
    training: bool,
) -> List[Metric]:
    """One epoch of training or evaluation.

    Args:
        model (Module): model.
        criterion (Callable): loss function.
        metric_factories (Sequence[Type[Metric]]): Metric factories.
        dl (DataLoader): data loader.
        training (bool): training or evaluation.
        opt (Optional[Optimizer], optional): optimizer. Defaults to None.
    Returns:
        Tuple[Learner, Stats]: Trained learner and corresponding training statistics."""

    context.model.training = training
    metrics = [metric() for metric in metric_factories]
    for batch in dl:
        X, y = batch
        X, y = map(to_device, [X, y])
        y_pred = context.model(X)
        loss = context.criterion(y_pred, y)
        if training:
            loss.backward()
            context.opt.step()
            context.opt.zero_grad()
        with torch.no_grad():
            for metric in metrics:
                metric.add_y_batch(y, y_pred)
    return metrics
