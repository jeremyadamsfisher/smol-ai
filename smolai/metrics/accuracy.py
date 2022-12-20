import numpy as np

from smolai import metrics
from smolai.callbacks import after, no_context
from smolai.metrics import Metric


class Accuracy(Metric):
    @no_context
    def setup(self):
        self.n = 0
        self.n_correct = 0

    @after
    @metrics.run_only_for_relevant_split
    def batch(self, context):
        n = context.y.shape[0]
        nc = (context.y_pred.argmax(dim=1) == context.y).float().sum().item()
        self.n += n
        self.n_correct += nc

    def summarize(self) -> float:
        try:
            return self.n_correct / self.n
        except ZeroDivisionError:
            return np.nan
