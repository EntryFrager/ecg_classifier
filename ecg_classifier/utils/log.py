import logging
import os

logger = None


def setup_logger():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), "main.log")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")

    fmt = "%(asctime)s %(levelname)-5s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False


def log_output(msg: str, level: int = logging.INFO):
    logger.log(level, msg)
