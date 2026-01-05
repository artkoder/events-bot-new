
import os
import json
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_diag")

def diag():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    logger.info(f"Checking Kaggle for user: {username} (key length: {len(key) if key else 0})")
    
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Authentication successful!")
        
        logger.info("Listing kernels...")
        kernels = api.kernels_list(user=username, page_size=20)
        for k in kernels:
            logger.info(f" - {getattr(k, 'ref', 'no ref')}: status={getattr(k, 'status', 'N/A')} lastRunTime={getattr(k, 'lastRunTime', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Kaggle API error: {e}")

if __name__ == "__main__":
    diag()
