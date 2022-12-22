from __future__ import annotations
import os
import sys
import math
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMATTER = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
MAX_LOG_FILE_SIZE = 2 * int(math.pow(10, 6))  # 2 MB
BACKUP_COUNT = 10
LOGS_DIR = "logfiles"


# Stores logs in LOG_DIR
# Log files are named "logfile_name"
def get_logger(logger_name: str):
    curr_date_time = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    Path(LOGS_DIR).mkdir(exist_ok=True)
    logfile = Path(f"{LOGS_DIR}/{logger_name}-{curr_date_time}.log")

    return _make_logger(logger_name=logger_name, logfile=logfile)


def _make_logger(logger_name: str, logfile: os.PathLike[str]):
    logger = logging.getLogger(logger_name)
    logger.addHandler(_get_console_handler())
    logger.addHandler(_get_file_handler(logfile))
    logger.propagate = False
    return logger


def _get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def _get_file_handler(logfile: os.PathLike[str]):
    file_handler = RotatingFileHandler(
        filename=logfile, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(FORMATTER)
    return file_handler
