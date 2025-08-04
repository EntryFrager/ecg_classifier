import logging
import os

logger = None


def setup_logger():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(os.getcwd(), "main.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")

    fmt = "%(message)s"
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_fmt = "%(message)s"
    console_formatter = logging.Formatter(console_fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False


def log_output(msg: str, level: int = logging.INFO):
    logger.log(level, msg)
