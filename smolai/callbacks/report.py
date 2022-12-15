from pprint import pformat

from loguru import logger

from smolai.callbacks import Callback


class Report(Callback):
    def epoch(self, context):
        """Log the latest epochwise metrics."""
        logger.info(f"{self.about}...")
        yield
        try:
            trn, tst = context.epochwise_metrics[-1]
            logger.info("...done {}: {}", self.about, pformat({"trn": trn, "tst": tst}))
        except (IndexError, ValueError):
            logger.info("...done {}", self.about)
