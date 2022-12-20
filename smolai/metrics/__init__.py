from abc import ABCMeta, abstractmethod

import torch


class Metric(metaclass=ABCMeta):
    """Abstract base class for metrics"""

    def __init_subclass__(cls, metric_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if metric_name is not None:
            cls.metric_name = metric_name
        else:
            cls.metric_name = cls.__name__.lower()

    @abstractmethod
    def add_batch(self, y: torch.tensor, y_pred: torch.tensor, loss: torch.tensor):
        """Add a batch of predictions to the metric."""

    @abstractmethod
    def summarize(self) -> torch.tensor:
        """Summarize the epoch."""

    def __repr__(self) -> str:
        summary = self.summarize()
        if isinstance(summary, float):
            summary = f"{summary:.2f}"
        return f"{self.__class__.metric_name}: {summary}"


from .accuracy import Accuracy
from .loss import Loss

__all__ = ["Accuracy", "Loss"]
