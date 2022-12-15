from pprint import pformat

from loguru import logger

from smolai.callbacks import Callback


class Report(Callback):
    def epoch(self, context):
        """Log the latest epochwise metrics."""
        logger.info("epoch {}/{}...", context.epoch + 1, context.n_epochs)
        yield
        try:
            trn, tst = context.epochwise_metrics[-1]
        except (IndexError, ValueError):  # no metrics
            logger.info("...done")
        else:
            logger.info("...done: {}", pformat({"trn": trn, "tst": tst}))
