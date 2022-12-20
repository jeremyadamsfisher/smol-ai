import torch
from loguru import logger
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from smolai.callbacks.lr import LrFinder
from smolai.callbacks.report import ReportEpochs, ReportMetricsWithLogger
from smolai.metrics import Accuracy, Loss
from smolai.trainer import Trainer


class MnistModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = Sequential(
            Linear(28 * 28, 30),
            ReLU(),
            Linear(30, 10),
        )

    def forward(self, xb):
        return self.net(xb.view(xb.size(0), -1))


def main():
    model = MnistModel()
    train_ds = MNIST(root="data", train=True, download=True, transform=ToTensor())
    train_dl = lambda: DataLoader(train_ds, batch_size=256, shuffle=True)
    test_ds = MNIST(root="data", train=False, download=True, transform=ToTensor())
    test_dl = lambda: DataLoader(test_ds, batch_size=256, shuffle=False)

    trainer_args = dict(criterion=CrossEntropyLoss(), opt_func=torch.optim.AdamW)

    lrf = LrFinder()
    lrf_trainer = Trainer(model=model, callbacks=[lrf, Loss], **trainer_args)
    lrf_trainer.fit(train_dl=train_dl())
    lr = lrf.suggest().lr

    logger.info("Using lr={} for training.", lr)

    Trainer(
        model=model,
        callbacks=[
            ReportMetricsWithLogger(),
            ReportEpochs(),
            Accuracy,
            Loss,
        ],
        **trainer_args
    ).fit(
        lr=lr,
        train_dl=train_dl(),
        test_dl=test_dl(),
        n_epochs=2,
    )


if __name__ == "__main__":
    main()
