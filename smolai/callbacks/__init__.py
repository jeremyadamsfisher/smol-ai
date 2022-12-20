from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, List

from loguru import logger

if TYPE_CHECKING:
    from smolai.trainer import Trainer


class CancellationException(Exception):
    def __init__(self, skip_after_event_functionality=False):
        super().__init__()
        self.skip_after_event_functionality = skip_after_event_functionality


class CancelBatch(CancellationException):
    pass


class CancelEpoch(CancellationException):
    pass


class CancelTrain(CancellationException):
    pass


class CancelTest(CancellationException):
    pass


class CancelFit(CancellationException):
    pass


def after(f):
    """Run method after lifecycle method."""

    def inner_after(*args, **kwargs):
        yield
        f(*args, **kwargs)

    return inner_after


def before(f):
    """Run method before lifecycle method."""

    def inner_before(*args, **kwargs):
        f(*args, **kwargs)
        yield

    return inner_before


class Callback:
    """Callback base class.

    Callbacks are called during the training loop. They can be used to implement
    custom training logic, such as logging, early stopping, or model checkpointing.

    Args:
        context (Trainer): Trainer instance managing the callback
    """

    def __init_subclass__(cls, priority=0, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.priority = priority

    def setup(self, context):
        pass

    def batch(self, context):
        raise NotImplementedError

    def epoch(self, context):
        raise NotImplementedError

    def train(self, context):
        raise NotImplementedError

    def test(self, context):
        raise NotImplementedError

    def fit(self, context):
        raise NotImplementedError

    NONLIFECYCLE_METHODS = {"setup"}


@dataclass
class CallbackManager:
    """Callback container."""

    cbs: List[Callback]
    context: "Trainer"

    def __post_init__(self):
        for cb in self.cbs:
            cb.setup(self.context)

    def setup_for_lifecycle(self, lifecycle, lifecycle_kwargs):
        """Instatiate all relevant lifecycle hooks"""
        lifecycle_cbs = []
        for cb in sorted(self.cbs, key=lambda cb_: cb_.__class__.priority):
            lifecycle_cb_f = getattr(cb, lifecycle)
            try:
                lifecycle_cb_gen = lifecycle_cb_f(self.context, **lifecycle_kwargs)
            except NotImplementedError:
                continue
            else:
                lifecycle_cb_iter = iter(lifecycle_cb_gen)
                lifecycle_cbs.append(lifecycle_cb_iter)
        return lifecycle_cbs

    @contextmanager
    def run_lifecycle(self, lifecycle, **kwargs):
        """Run callbacks in a lifecycle.

        Args:
            lifecycle (str): Lifecycle to run.
            **kwargs: Keyword arguments to pass to lifecycle method.

        Raises:
            Cancel{Batch|Epoch|Train|Test|Fit}: SKip lifecycle."""
        cbs = self.setup_for_lifecycle(lifecycle, kwargs)
        try:
            for cb in cbs:
                next(cb)
            yield
        except (CancelBatch, CancelEpoch, CancelTrain, CancelTest, CancelFit) as e:
            logger.debug("Caught {} in {} hook.", e.__class__.__name__, lifecycle)
            if e.skip_after_event_functionality:
                logger.warning("Skipping the teardown stage of {}.", lifecycle)
                return
        for cb in cbs:
            try:
                next(cb)
            except StopIteration:
                pass

    def __getattribute__(self, attribute):
        if (
            attribute in Callback.__dict__
            and attribute not in Callback.NONLIFECYCLE_METHODS
        ):
            return partial(self.run_lifecycle, attribute)
        return super().__getattribute__(attribute)
