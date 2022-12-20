import torch
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from smolai import fit
from smolai.callbacks.report import ReportMetricsWithLogger
from smolai.metrics import Accuracy, Loss


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
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_ds = MNIST(root="data", train=False, download=True, transform=ToTensor())
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    cbs = [ReportMetricsWithLogger()]

    fit(
        model=model,
        criterion=CrossEntropyLoss(),
        train_dl=train_dl,
        test_dl=test_dl,
        opt_func=torch.optim.AdamW,
        metric_factories=[Accuracy, Loss],
        callbacks=cbs,
        n_epochs=2,
    )


if __name__ == "__main__":
    main()