from smolai.metrics import Metric


class Loss(Metric):
    """Loss metric, used with ReportAverageLossWithPlot to
    show loss in real-time"""

    def __init__(self):
        self.losses = []

    def add_batch(self, y, loss, **_):
        n = y.shape[0]
        self.losses.append((n, loss.item()))

    def summarize(self):
        ns, losses = zip(*self.losses)
        return sum(losses) / len(ns)
