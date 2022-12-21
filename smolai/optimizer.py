from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class SGD:
    """Stochastic Gradient Descent

    Args:
        params (Iterable): Iterable of parameters to optimize
        lr (float): Learning rate
        weight_decay (float): Weight decay (L2 penalty)"""

    params: Iterable
    lr: float
    weight_decay: float

    def __post_init__(self):
        self.params = list(self.params)
        self.i = 0

    def step(self):
        for p in self.params:
            self.opt_step(p)
            self.reg_step(p)
        self.i += 1

    def opt_step(self, p):
        p -= p.grad * self.lr

    def reg_step(self, p):
        """Regularization step.

        In L2 regularization, the sum of the squared weights are added to
        the loss function, so we need to determine the derivate.

        L = loss + weight_decay * sum(weight**2)
        dL/dw = dloss/dw + 2 * weight_decay * weight

        Therefore, after one optimization step:

        weight  = weight - lr * 2 * weight_decay * weight
                = weight (1 - lr * 2 * weight_decay)
        weight *= 1 - lr * 2 * weight_decay

        Let K = 2 * weight_decay

        weight *= 1 - lr * K
        """
        if self.weight_decay:
            p *= 1 - self.lr * self.weight_decay

    def zero_grad(self):
        for p in self.params:
            p.grad.data.zero_()
