from typing import List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from smolai.callbacks import Callback, after, before, no_context


class ActivationStats(Callback):
    """Callback to compute the mean and std of the activations in a model."""

    @no_context
    def setup(self):
        self.activations = {"mean": [], "std": []}
        self.fig, self.axes = plt.subplots(1, 2, figsize=(7, 3.5))
        self.fig.tight_layout()
        self.display = None

        def capture_activation(_):
            def hook(model, _, output):
                if model.training is False:
                    return
                activation = output.detach()
                self.activations["mean"][-1].append(activation.mean().item())
                self.activations["std"][-1].append(activation.std().item())

            return hook

        self.capture_activation = capture_activation

    def fit(self, context):
        hooks = []
        for name, layer in context.model.named_modules():
            hook = layer.register_forward_hook(self.capture_activation(name))
            hooks.append(hook)
        yield
        for hook in hooks:
            del hook

    def batch(self, context):
        for summary_stat in ["mean", "std"]:
            self.activations[summary_stat].append([])
        yield
        if context.batch_idx % 10 != 0:
            return
        for stat, ax in zip(["mean", "std"], self.axes):
            ax.clear()
            ax.set(xlabel="Batches", ylabel=stat.title())
            activation_statistics_by_batch = self.activations[stat]

            max_ = max(max(x) for x in activation_statistics_by_batch)

            n_bins = 10
            canvas = np.zeros((n_bins, context.batch_idx + 1))
            for i, activation_statistics in enumerate(activation_statistics_by_batch):
                frequency, _ = np.histogram(
                    activation_statistics, bins=n_bins, range=(0, max_)
                )
                canvas[:, i] = frequency
            ax.imshow(canvas, aspect="auto")
            ax.set_yticklabels([f"{t:.2f}" for t in np.linspace(0, max_, n_bins)])
            if self.display is None:
                self.display = display(self.fig, display_id=True)
            else:
                self.display.update(self.fig)
