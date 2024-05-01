import sys

import logging
import logging.config

from typing import Optional

import coloredlogs

from elemelek.settings import LOGGING_FORMAT, LOGGING_LEVEL, LOGGING_PLAIN


def add_handler(logger, handler, level, fmt):
    handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(
    name: str = "",
    level: str = LOGGING_LEVEL,
    file_handler_path: str = None,
    fmt: str = LOGGING_FORMAT,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        return logger

    if file_handler_path:
        add_handler(logger, logging.FileHandler(file_handler_path), level, fmt)
    else:
        if LOGGING_PLAIN:
            add_handler(logger, logging.StreamHandler(sys.stdout), level, fmt)
        else:
            coloredlogs.install(level=level, logger=logger, fmt=fmt)

    logger.propagate = False

    return logger


class SelfLogging:
    _logger: Optional[logging.Logger] = None

    def _create_logger(self):
        return get_logger(self.__class__.__name__)

    @property
    def logger(self):
        if not self._logger:
            self._logger = self._create_logger()
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def debug(self, msg, **kwargs):
        self.logger.debug(msg, **kwargs)

    def info(self, msg, **kwargs):
        self.logger.info(msg, **kwargs)

    def error(self, msg, **kwargs):
        self.logger.error(msg, **kwargs)

    def warn(self, msg, **kwargs):
        self.logger.warning(msg, **kwargs)

    def exception(self, msg, **kwargs):
        self.logger.exception(msg, **kwargs)
