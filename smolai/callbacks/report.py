from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from loguru import logger

from smolai.callbacks import Callback, after, no_context
from smolai.metrics import Loss, Metric


class ReportEpochs(Callback, priority=2):
    """Report the start and end of each epoch."""

    def epoch(self, context):
        curr, total = context.epoch + 1, context.n_epochs
        logger.info("epoch {}/{}...", curr, total)
        yield
        logger.info("...epoch {}/{} done", curr, total)


class ReportMetricsWithLogger(Callback):
    """Log the latest epochwise metrics."""

    @after
    def epoch(self, context):
        metric_callbacks = [cb for cb in context.callbacks if isinstance(cb, Metric)]
        metrics_trn = [cb for cb in metric_callbacks if cb.training]
        metrics_tst = [cb for cb in metric_callbacks if not cb.training]
        logger.info("metrics (training): {}", pformat(metrics_trn))
        logger.info("metrics (test):     {}", pformat(metrics_tst))


class RealTimeLoss(Callback, priority=2):
    """Plot the loss in real time."""

    def setup(self, context):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(7, 3.5))
        self.loss_trn, self.loss_tst = self.require_other_callback(context, Loss)
        self.display = None

    @after
    def batch(self, context):
        # only plot every 5 batches, but not the first
        if 0 == context.batch_idx or context.batch_idx % 5 != 0:
            return

        for ax, split, metric in zip(
            self.axes, ["test", "train"], [self.loss_tst, self.loss_trn]
        ):
            ax.clear()
            ax.set(xlabel="Steps", ylabel="Average loss", title=split.title())
            # assuming a mean-reduction of the loss
            total_losses_per_batch = np.array(metric.ns) * np.array(metric.losses)
            cum_ns = np.cumsum(metric.ns)
            cum_losses = np.cumsum(total_losses_per_batch)
            avg_losses = cum_losses / cum_ns
            ax.plot(cum_ns, avg_losses)

        if self.display is None:
            self.fig.tight_layout()
            self.display = display(self.fig, display_id=True)
        else:
            self.display.update(self.fig)

    @after
    @no_context
    def epoch(self):
        plt.close(self.fig)
