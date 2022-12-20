import numpy as np

from smolai import metrics
from smolai.callbacks import after, no_context
from smolai.metrics import Metric


class Loss(Metric):
    """Loss metric, used with ReportAverageLossWithPlot to
    show loss in real-time"""

    @no_context
    def setup(self):
        self.losses = []

    # @after
    @metrics.run_only_for_relevant_split
    def batch(self, context):
        yield
        n = context.y.shape[0]
        self.losses.append((n, context.loss.item()))

    def summarize(self):
        try:
            ns, losses = zip(*self.losses)
        except ValueError:
            return np.nan
        return sum(losses) / len(ns)
