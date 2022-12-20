from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from loguru import logger

from smolai.callbacks import Callback, after
from smolai.metrics import Loss


class ReportMetricsWithLogger(Callback):
    def epoch(self, context):
        """Log the latest epochwise metrics."""
        logger.info("epoch {}/{}...", context.epoch + 1, context.n_epochs)
        yield
        try:
            train, test = context.epochwise_metrics[-1]
        except (IndexError, ValueError):  # no metrics
            logger.info("...done")
        else:
            if len(train) == 1 and len(test) == 1:
                (train,) = train
                (test,) = test
            logger.info("...done: {}", pformat({"train": train, "test": test}))


class ReportAverageLossWithPlot(Callback):
    def setup(self, context):
        # Ensure that loss is being tracked
        if not any(m is Loss for m in context.metric_factories):
            context.metric_factories.append(Loss)

    @after
    def batch(self, context, *_, batch_metrics, batch_idx, training):

        # only plot every 10 batches, but not the first
        if 0 == batch_idx or batch_idx % 100 != 0:
            return

        losses = {"test": [], "train": []}

        # grab the epoch metrics, if they exist
        if context.epochwise_metrics:
            for metrics_train, metrics_test in context.epochwise_metrics:
                for split, ms in [("train", metrics_train), ("test", metrics_test)]:
                    for m in ms:
                        if isinstance(m, Loss):
                            losses[split].extend(m.losses)
                            break

        # add the current batch's metrics
        split = "train" if training else "test"
        for m in batch_metrics:
            if isinstance(m, Loss):
                losses[split].extend(m.losses)

        with plt.style.context("ggplot"):
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            for ax, split in zip(axes, ["test", "train"]):
                ax.set(xlabel="Steps", ylabel="Average loss", title=split.title())
                try:
                    ns, ls = zip(*losses[split])
                except ValueError:
                    continue  # no metrics to plot
                cum_ns = np.cumsum(ns)
                cum_losses = np.cumsum(ls)
                avg_losses = cum_losses / cum_ns
                ax.plot(cum_ns, avg_losses)
            plt.tight_layout()
            clear_output()
            display(fig)
            plt.close()
