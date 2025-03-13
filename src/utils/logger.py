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

def get_logger(name):
    # Use setup_logger to create a logger if not already set up
    log_file = os.path.join("logs", f"{name}.log")
    return setup_logger(name, log_file)