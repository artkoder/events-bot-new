
import os
import json
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_status_diag")

def diag():
    username = os.getenv("KAGGLE_USERNAME")
    kernel_ref = f"{username}/eventsbot-e2e-tests"
    logger.info(f"Checking status for: {kernel_ref}")
    
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Authentication successful!")
        
        status = api.kernels_status(kernel_ref)
        logger.info(f"Status result: {status}")
        
    except Exception as e:
        logger.error(f"Kaggle API error: {e}")

if __name__ == "__main__":
    diag()
