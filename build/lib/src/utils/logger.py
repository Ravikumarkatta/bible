# src/utils/logger.py
import logging
import os

def setup_logger(name, log_file):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger