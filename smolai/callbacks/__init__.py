from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

from smolai.utils import DotDict


class Callback:
    def batch(self, context):
        raise NotImplementedError

    def epoch(self, context):
        raise NotImplementedError

    def train(self, context):
        raise NotImplementedError

    def test(self, context):
        raise NotImplementedError


@dataclass
class CallbackManager:
    """Callback container."""

    context: DotDict
    cbs: List[Callback]

    @contextmanager
    def run_lifecycle(self, lifecycle):
        cbs = []
        for cb in self.cbs:
            try:
                lifecycle_cb = getattr(cb, lifecycle)(self.context)
            except NotImplementedError:
                continue
            else:
                i_lifecycle_cb = iter(lifecycle_cb)
                next(i_lifecycle_cb)
                cbs.append(i_lifecycle_cb)
        yield
        for cb in cbs:
            try:
                next(cb)
            except StopIteration:
                pass

    def __getattribute__(self, attribute):
        if attribute in Callback.__dict__:
            return lambda: self.run_lifecycle(attribute)
        return super().__getattribute__(attribute)
