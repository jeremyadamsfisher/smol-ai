from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from loguru import logger

from smolai.callbacks import Callback, after, no_context
from smolai.metrics import Loss, Metric


class ReportEpochs(Callback):
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
