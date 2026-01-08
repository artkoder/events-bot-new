
import os
import json
import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_download")

def download():
    project_root = Path(__file__).resolve().parents[1]
    username = os.getenv("KAGGLE_USERNAME")
    slug = "eventsbot-e2e-tests"
    kernel_ref = f"{username}/{slug}"
    default_output_dir = project_root / "artifacts" / "e2e" / "e2e_results_new"
    output_dir = Path(os.getenv("OUTPUT_DIR", str(default_output_dir))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
