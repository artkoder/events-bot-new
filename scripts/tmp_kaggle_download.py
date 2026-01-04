
import os
import json
import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_download")

def download():
    username = os.getenv("KAGGLE_USERNAME")
    slug = "eventsbot-e2e-tests"
    kernel_ref = f"{username}/{slug}"
    output_dir = Path("e2e_results_new")
    output_dir.mkdir(exist_ok=True)
    
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading output for {kernel_ref}...")
        api.kernels_output(kernel_ref, path=str(output_dir), force=True)
        logger.info("Download complete!")
    except Exception as e:
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    download()
