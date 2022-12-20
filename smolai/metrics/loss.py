import numpy as np

from smolai import metrics
from smolai.callbacks import after, no_context
from smolai.metrics import Metric


class Loss(Metric):
    """Loss metric, used with RealTimeLoss to show loss in real-time.
    Assumes that the loss is a mean reduction, which means that it
    needs to be multiplied by the batch size before being summed and
    divided by the total number of samples."""

    @no_context
    def setup(self):
        self.ns = []
        self.losses = []

    @after
    @metrics.run_only_for_relevant_split
    def batch(self, context):
        self.ns.append(context.y.shape[0])
        self.losses.append(context.loss.item())

    def summarize(self):
        loss_total = 0
        n_total = 0
        for n, loss in zip(self.ns, self.losses):
            loss_total += n * loss
            n_total += n
        try:
            return loss_total / n_total
        except ZeroDivisionError:
            return np.nan
