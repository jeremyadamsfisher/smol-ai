import torch
import torch.nn as nn

from smolai.callbacks import Callback, before


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)


class InitWeights(Callback):
    """Use Kaiming initialization for all layers in the model and scale
    the input apropriately.

    See: https://colab.research.google.com/drive/1J1E5a_WtZ2tJt-9MRASxWR_lHqIbDb1G?usp=sharing"""

    @before
    def batch(self, context):
        X, y = context.batch
        X = (X - X.mean()) / X.std()
        context.batch = (X, y)

    @before
    def fit(self, context):
        context.model.apply(init_weights)
