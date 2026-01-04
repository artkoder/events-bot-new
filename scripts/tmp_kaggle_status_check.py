
import os
import json
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_status_check")

def check():
    username = os.getenv("KAGGLE_USERNAME")
    slug = "eventsbot-e2e-tests"
    kernel_ref = f"{username}/{slug}"
    
    try:
        api = KaggleApi()
        api.authenticate()
        status = api.kernels_status(kernel_ref)
        logger.info(f"Kernel {kernel_ref} status: {status}")
    except Exception as e:
        logger.error(f"Status check failed: {e}")

if __name__ == "__main__":
    check()
