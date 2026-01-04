
import os
import json
import logging
from pathlib import Path
import tempfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kaggle_push_debug")

def push_debug():
    username = os.getenv("KAGGLE_USERNAME")
    # Using a slightly different name to avoid potential conflicts
    slug = "events-bot-e2e-test-run"
    kernel_id = f"{username}/{slug}"
    
    kernel_dir = Path("kaggle/E2ETests")
    
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Authenticated as {username}")
        
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Copy all files from kernel_dir
            for item in kernel_dir.iterdir():
                if item.name == "kernel-metadata.json":
                    continue
                dest = tmp_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            # Create fresh metadata
            meta = {
                "id": kernel_id,
                "title": "Events Bot E2E Test Run",
                "code_file": "e2e_tests.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "slug": slug
            }
            (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta))
            
            logger.info(f"Pushing to {kernel_id}...")
            api.kernels_push(str(tmp_path))
            logger.info("Push successful!")
            
    except Exception as e:
        logger.error(f"Push failed: {e}")

if __name__ == "__main__":
    push_debug()
