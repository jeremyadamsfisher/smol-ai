from smolai.metrics import Metric


class Accuracy(Metric, metric_name="accuracy_score"):
    def __init__(self):
        self.n = 0
        self.n_correct = 0

    def add_batch(self, y, y_pred, **_):
        self.n += y.shape[0]
        self.n_correct += (y_pred.argmax(dim=1) == y).float().sum().item()

    def summarize(self) -> float:
        return self.n_correct / self.n
