import torch
from loguru import logger

from smolai.callbacks import Callback


def run_only_for_relevant_split(f):
    def run_only_for_relevant_split_inner(self, context, *args, **kwargs):
        if context.model.training == self.training:
            return f(self, context, *args, **kwargs)
        else:
            # need to return generator-like to work with normal callback lifecycle
            return [None]

    return run_only_for_relevant_split_inner


class Metric(Callback):
    """Abstract base class for metrics"""

    def __init_subclass__(cls, metric_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if metric_name is not None:
            cls.metric_name = metric_name
        else:
            cls.metric_name = cls.__name__.lower()

    def __init__(self, training: bool) -> None:
        self.training = training

    @classmethod
    def as_factory(cls):
        trn = cls(training=True)
        tst = cls(training=False)
        return [trn, tst]

    def summarize(self) -> torch.tensor:
        """Summarize the epoch."""
        raise NotImplementedError

    def __repr__(self) -> str:
        summary = self.summarize()
        if isinstance(summary, float):
            summary = f"{summary:.2f}"
        return f"{self.__class__.metric_name}: {summary}"


from .accuracy import Accuracy
from .loss import Loss

__all__ = ["Accuracy", "Loss"]
