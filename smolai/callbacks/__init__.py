from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Type

from loguru import logger

from smolai.utils import advance

if TYPE_CHECKING:
    from smolai.trainer import Trainer


class CallbackSetupError(Exception):
    pass


class CancelBatch(Exception):
    pass


class CancelEpoch(Exception):
    pass


class CancelTrain(Exception):
    pass


class CancelTest(Exception):
    pass


class CancelFit(Exception):
    pass


lifecycle2cancel_exception = {
    "batch": CancelBatch,
    "epoch": CancelEpoch,
    "train": CancelTrain,
    "test": CancelTest,
    "fit": CancelFit,
}


def no_context(f):
    """Run lifecycle hook without accessing training state."""

    def inner_no_context(self, _context, *args, **kwargs):
        return f(self, *args, **kwargs)

    return inner_no_context


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

    def __init_subclass__(cls, priority=1, requires_grad=False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.priority = priority
        cls.requires_grad = requires_grad

    @classmethod
    def as_factory(cls):
        """If a callback is supplied as a class, this method is called and the
        resulting list is added to the full callback list."""
        return [cls()]

    def setup(self, context):
        pass

    def require_other_callback(self, context, callback_type: Type[Callback]):
        """Check if given callback types is present in the context.

        Returns:
            List[Callback]: List of callbacks of given type. If all callbacks
                are metrics, they are sorted by training then testing."""
        from smolai.metrics import Metric

        cbs = [cb for cb in context.callbacks if isinstance(cb, callback_type)]
        if not cbs:
            raise RuntimeError(
                f"Callback {callback_type.__name__} is required for {self}"
            )
        for cb in cbs:
            if cb.__class__.priority > self.__class__.priority:
                raise RuntimeError(
                    f"Callback {cb.__class__.__name__} has lower priority than {self.__class__.__name__} "
                    f"which requested it. Please increase the priority of {cb.__class__.__name__}."
                )
        if all(isinstance(cb, Metric) for cb in cbs):
            return sorted(cbs, key=lambda cb: cb.training, reverse=True)
        return cbs

    LIFECYCLE_METHODS = {"batch", "epoch", "train", "test", "fit"}

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


@dataclass
class CallbackManager:
    """Callback container."""

    cbs: List[Callback]
    context: "Trainer"

    def __post_init__(self):
        for cb in self.cbs:
            cb.setup(self.context)

    def setup_for_lifecycle(self, lifecycle, lifecycle_kwargs) -> Callable:
        """Instatiate all relevant lifecycle hooks and create function
        to call `next` upon them."""
        lifecycle_cbs = []
        for cb in sorted(self.cbs, key=lambda cb_: cb_.__class__.priority):
            lifecycle_cb_f = getattr(cb, lifecycle)
            try:
                lifecycle_cb_gen = lifecycle_cb_f(self.context, **lifecycle_kwargs)
            except NotImplementedError:
                continue
            else:
                try:
                    lifecycle_cb_iter = iter(lifecycle_cb_gen)
                except TypeError:
                    raise CallbackSetupError(
                        f"{cb.__class__.__name__}.{lifecycle}() must be a generator."
                    )
                lifecycle_cbs.append((cb.requires_grad, lifecycle_cb_iter))

        return lambda: [
            advance(cb_gen, requires_grad) for requires_grad, cb_gen in lifecycle_cbs
        ]

    @contextmanager
    def run_lifecycle(self, lifecycle, **kwargs):
        """Run callbacks in a lifecycle.

        Args:
            lifecycle (str): Lifecycle to run.
            **kwargs: Keyword arguments to pass to lifecycle method.

        Raises:
            Cancel{Batch|Epoch|Train|Test|Fit}: Immediately terminate lifecycle event."""
        advance_cbs = self.setup_for_lifecycle(lifecycle, kwargs)
        try:
            if lifecycle != "batch":
                logger.debug("Starting {}.", lifecycle)
            advance_cbs()
            yield
            if lifecycle != "batch":
                logger.debug("Finishing {}.", lifecycle)
            advance_cbs()
        except lifecycle2cancel_exception[lifecycle] as e:
            logger.debug("Caught {}.", e.__class__.__name__)

    def __getattribute__(self, attribute):
        if attribute in Callback.LIFECYCLE_METHODS:
            return partial(self.run_lifecycle, attribute)
        return super().__getattribute__(attribute)
