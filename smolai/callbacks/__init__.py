from contextlib import contextmanager


class Callback:
    @contextmanager
    def batch(self, context):
        yield

    @contextmanager
    def epoch(self, context):
        yield

    @contextmanager
    def train(self, context):
        yield

    @contextmanager
    def test(self, context):
        yield
