from abc import ABCMeta, abstractmethod
from typing import List, TypeVar

import torch

BatchMetricType = TypeVar("BatchMetricType")
EpochMetricType = TypeVar("EpochMetricType")


class Metric(metaclass=ABCMeta):
    """Abstract base class for metrics, keeping track of batchwise and epochwise
    state."""

    metric_name: str = "metric"

    def __init_subclass__(cls, metric_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if metric_name is not None:
            cls.metric_name = cls.__name__.lower()

    def get_metric_name(self):
        return self.__cls__.metric_name

    def __init__(self):
        self.batchwise_metrics: List[BatchMetricType] = []

    def add_y_batch(self, y: torch.tensor, y_pred: torch.tensor):
        """Add a batch of predictions to the metric."""
        metric = self.compute_batchwise(y, y_pred)
        metric.n = y.shape[0]
        self.batchwise_metrics.append(metric)

    @abstractmethod
    def compute_batchwise(
        self, y: torch.tensor, y_pred: torch.tensor
    ) -> BatchMetricType:
        """Compute batchwise metric. Does not alter state."""
        ...

    @abstractmethod
    def compute_epoch(
        self, batchwise_metrics: List[BatchMetricType]
    ) -> EpochMetricType:
        """Compute epochwise metric. Does not alter state."""
        ...

    def __repr__(self) -> str:
        return f"{self.get_metric_name()}: {self.compute_epoch(self.batchwise_metrics)}"


from .accuracy import Accuracy

__all__ = ["Accuracy"]
