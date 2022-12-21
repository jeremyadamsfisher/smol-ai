import torch

from smolai.callbacks import Callback, after, before


class Scheduler(Callback):
    """Callback to update a scheduler after each epoch."""

    def __init__(self, scheduler_func, **kwargs):
        self.scheduler_func = scheduler_func
        self.scheduler_func_kwargs = kwargs

    @before
    def fit(self, context):
        self.scheduler = self.scheduler_func(context.opt, **self.scheduler_func_kwargs)

    @after
    def batch(self, context):
        if context.model.training:
            self.scheduler.step()


class CosineAnnealingScheduler(Scheduler):
    """Cosine annealing scheduler."""

    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.CosineAnnealingLR, *args, **kwargs)

    def fit(self, context):
        self.scheduler_func_kwargs["T_max"] = len(context.train_dl) * context.n_epochs
        return super().fit(context)


class OneCycleScheduler(Scheduler):
    """Cosine annealing scheduler."""

    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.OneCycleLR, *args, **kwargs)

    def fit(self, context):
        self.scheduler_func_kwargs["total_steps"] = (
            len(context.train_dl) * context.n_epochs
        )
        self.scheduler_func_kwargs["max_lr"] = context.lr
        return super().fit(context)
