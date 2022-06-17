import os
import sys
import logging


def build_logger(log_path=None, stdout=True):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handlers = []
    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        handlers.append(ch)

    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        handlers.append(fh)
    logging.basicConfig(
        level=logging.DEBUG, handlers=handlers)
    return logging.getLogger()


class Logger:
    def __init__(self, log_path=None, stdout=True):
        self.logger = build_logger(log_path, stdout)

    def log(self, msg):
        self.logger.log(logging.INFO, msg)