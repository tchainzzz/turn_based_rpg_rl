import datetime
import logging
import os
import time

def get_file_logger(fname, level=logging.DEBUG, overwrite=True):
    if os.path.isfile(fname):
        os.unlink(fname)
    file_logger = logging.FileHandler(fname)
    file_logger.setLevel(level)
    file_logger.setFormatter(fmt)
    logger.addHandler(file_logger)

logger = logging.getLogger('ROOT')
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d - %(message)s\n")

stdout_logger = logging.StreamHandler()
stdout_logger.setLevel(logging.INFO)
stdout_logger.setFormatter(fmt)
logger.addHandler(stdout_logger)

human_readable_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
get_file_logger(f"logs/log_{human_readable_time}.log")
get_file_logger(f"logs/latest.log")

