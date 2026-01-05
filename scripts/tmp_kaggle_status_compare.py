
import os
import json
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_status_compare")

def diag():
    username = os.getenv("KAGGLE_USERNAME")
    kernels = ["eventsbot-e2e-tests", "preview-3d", "events-bot-e2e-test-run"]
    
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Authentication successful!")
        
        for slug in kernels:
            ref = f"{username}/{slug}"
            try:
                status = api.kernels_status(ref)
                logger.info(f"Ref: {ref} -> Status: {status}")
            except Exception as e:
                logger.error(f"Ref: {ref} -> Error: {e}")
                
    except Exception as e:
        logger.error(f"Kaggle API error: {e}")

if __name__ == "__main__":
    diag()
