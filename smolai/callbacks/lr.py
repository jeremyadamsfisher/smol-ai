import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from smolai.callbacks import (Callback, CancelFit, CancelTest, after, before,
                              no_context)


@dataclass
class LrFinderResult:
    lr: float
    loss_grad_min: float


@dataclass
class LrFinder(Callback):

    lr_mult: float = 1.3

    @no_context
    def setup(self):
        self.res = []
        self.min = math.inf

    @after
    def batch(self, context):
        (group,) = context.opt.param_groups
        loss = context.loss.item()
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
        return LrFinderResult(lr_suggested, loss_grad_min)

    def plot(self):
        lrs, losses = zip(*self.res)
        suggestion = self.suggest()
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
        ax.set(title="Learning rate finder", xlabel="Learning rate", ylabel="Loss")
        ax.set_xscale("log")
        ax.plot(lrs, losses)
        ax.scatter([suggestion.lr], [suggestion.loss_grad_min], c="red")
        display(fig)
        plt.close()

    @before
    @no_context
    def test(self):
        # We are only interested in the effect of the learning rate
        # on the loss, so the performance of the model itself on
        # a hold-out set is irrelevant
        raise CancelTest
