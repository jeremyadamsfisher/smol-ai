import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from smolai.callbacks import (Callback, CancelFit, CancelTest, after, before,
                              no_context)
from smolai.metrics import Loss


@dataclass
class LrFinder(Callback):

    lr_mult: float = 1.3

    def __post_init__(self):
        self.res = []
        self.min = math.inf

    def setup(self, context):
        # Ensure that loss is being tracked
        if not any(m is Loss for m in context.metric_factories):
            context.metric_factories.append(Loss)

    @after
    def batch(self, context):
        (group,) = context.opt.param_groups
        (loss_metric,) = [m for m in context.batch_metrics if isinstance(m, Loss)]
        _, loss = loss_metric.losses[-1]
        self.res.append((group["lr"], loss))
        self.min = min((loss, self.min))
        if loss > self.min * 3:
            raise CancelFit
        group["lr"] *= self.lr_mult

    def suggest(self):
        lrs, losses = zip(*self.res)
        lr_suggested_idx = np.gradient(losses).argmin()
        lr_suggested = lrs[lr_suggested_idx]
        loss_grad_min = losses[lr_suggested_idx]
        return (lr_suggested, loss_grad_min)

    def plot(self):
        lrs, losses = zip(*self.res)
        lr_suggested, loss_grad_min = self.suggest()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(title="Learning rate finder", xlabel="Learning rate", ylabel="Loss")
        ax.set_xscale("log")
        ax.plot(lrs, losses)
        ax.scatter([lr_suggested], [loss_grad_min], c="red")
        display(fig)
        plt.close()

    @before
    @no_context
    def test(self):
        # We are only interested in the effect of the learning rate
        # on the loss, so the performance of the model itself on
        # a hold-out set is irrelevant
        raise CancelTest
