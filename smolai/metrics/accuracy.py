from typing import List, TypeVar

import torch
from pydantic import BaseModel

from smolai.metrics import Metric

BatchMetricType = TypeVar("BatchMetricType")
EpochMetricType = TypeVar("EpochMetricType")


class AccuracyBatch(BaseModel):
    n: int
    n_correct: int


class Accuracy(Metric):
    def compute_batchwise(self, y: torch.tensor, y_pred: torch.tensor) -> AccuracyBatch:
        n = y.shape[0]
        n_correct = (y_pred.argmax(dim=1) == y).float().mean()
        return AccuracyBatch(n=n, n_correct=n_correct)

    def compute_epoch(self, batchwise_metrics: List[AccuracyBatch]) -> float:
        n_correct = sum(s.n_correct for s in batchwise_metrics)
        return n_correct / sum(s.n for s in batchwise_metrics)
