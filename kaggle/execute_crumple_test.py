#!/usr/bin/env python3
"""
Execute CrumpleVideo test on Kaggle.

This script:
1. Creates a temporary dataset with test afisha images
2. Pushes the CrumpleVideo kernel to Kaggle
3. Waits for execution to complete
4. Downloads the resulting video

Usage:
    python kaggle/execute_crumple_test.py

Requirements:
    - KAGGLE_USERNAME and KAGGLE_KEY environment variables
    - Test afisha images in video_announce/test_afisha/
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from video_announce.kaggle_client import KaggleClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("crumple_test")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
KERNEL_DIR = PROJECT_ROOT / "kaggle" / "CrumpleVideo"
TEST_AFISHA_DIR = PROJECT_ROOT / "video_announce" / "test_afisha"
ASSETS_DIR = PROJECT_ROOT / "video_announce" / "assets"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "crumple_test"


def create_test_payload(afisha_dir: Path, output_path: Path) -> None:
    """Create a payload.json for testing using images from afisha_dir."""
    images = list(afisha_dir.glob("*.jpg")) + list(afisha_dir.glob("*.png"))
    
    scenes = []
    for i, img in enumerate(images[:5]):  # Test run: 5 posters
        scenes.append({
            "about": f"Тестовое событие {i + 1}",
            "date": "25 декабря",
            "location": "Калининград",
            "images": [img.name]  # Relative to dataset root
        })
    
    test_date_start = date.today() + timedelta(days=1)
    test_date_end = test_date_start + timedelta(days=2)
    payload = {
        "intro": {
            "count": len(scenes),
            "text": "СОБЫТИЯ КОТОРЫЕ СТОИТ ПОСЕТИТЬ",
            "date": "25-27 декабря",
            # Include multiple cities to exercise safe spacing with weekday-range titles.
            "cities": ["Калининград", "Черняховск"],
            "date_start": test_date_start.isoformat(),
            "date_end": test_date_end.isoformat(),
        },
        "selection_params": {
            "mode": "test",
            "is_test": True,
        },
        "scenes": scenes
    }
    
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info(f"Created payload with {len(scenes)} scenes: {output_path}")


async def run_crumple_test():
    """
    Run CrumpleVideo test on Kaggle.
    """
    client = KaggleClient()
    
    # Check kernel exists
    if not KERNEL_DIR.exists():
        logger.error(f"Kernel directory not found: {KERNEL_DIR}")
        return False
    
    # Check test afisha exists
    if not TEST_AFISHA_DIR.exists():
        logger.error(f"Test afisha directory not found: {TEST_AFISHA_DIR}")
        return False
    
    # Create temporary dataset folder
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = Path(tmp) / "crumple-test-dataset"
        dataset_path.mkdir()
        
        # Copy test afisha images
        for img in TEST_AFISHA_DIR.iterdir():
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy2(img, dataset_path / img.name)
                logger.info(f"Copied: {img.name}")
        
        # Copy fonts
        for font in ASSETS_DIR.glob("*.ttf"):
            shutil.copy2(font, dataset_path / font.name)
        for font in ASSETS_DIR.glob("*.otf"):
            shutil.copy2(font, dataset_path / font.name)
        logger.info("Copied fonts")
        
        # Copy audio
        audio_file = ASSETS_DIR / "The_xx_-_Intro.mp3"
        if audio_file.exists():
            shutil.copy2(audio_file, dataset_path / audio_file.name)
            logger.info("Copied audio file")
        
        # Copy Blender XPBD scripts (critical for rendering!)
        for script in ["blender_xpbd_paper.py", "run.py", "make_showreel_audio.py"]:
            script_path = KERNEL_DIR / script
            if script_path.exists():
                shutil.copy2(script_path, dataset_path / script)
                logger.info(f"Copied script: {script}")
        
        # Create payload.json
        create_test_payload(TEST_AFISHA_DIR, dataset_path / "payload.json")
        
        # Create dataset-metadata.json
        username = os.getenv("KAGGLE_USERNAME", "zigomaro")
        dataset_slug = f"{username}/crumple-video-test"
        dataset_meta = {
            "id": dataset_slug,
            "title": "Crumple Video Test",
            "licenses": [{"name": "CC0-1.0"}]
        }
        (dataset_path / "dataset-metadata.json").write_text(
            json.dumps(dataset_meta, indent=2)
        )
        
        # Create/update dataset
        logger.info(f"Creating dataset: {dataset_slug}")
        try:
            client.create_dataset(dataset_path)
        except Exception as e:
            # If dataset exists, try to update it
            logger.warning(f"Dataset create failed (may already exist): {e}")
            try:
                api = client._get_api()
                api.dataset_create_version(
                    str(dataset_path),
                    version_notes="Test run",
                    quiet=True
                )
            except Exception as e2:
                logger.error(f"Failed to update dataset: {e2}")
                return False
        
        # Wait for dataset to be ready
        logger.info("Waiting 30s for dataset to propagate...")
        await asyncio.sleep(30)
        
        # Push kernel with our test dataset
        # Note: We push directly to avoid deploy_kernel_update's hardcoded slug
        logger.info(f"Pushing CrumpleVideo kernel with dataset: {dataset_slug}")
        try:
            # Use push_kernel directly with the correct dataset
            assets_dataset = os.getenv("CRUMPLE_ASSETS_DATASET", "zigomaro/video-announce-assets")
            client.push_kernel(
                kernel_path=KERNEL_DIR,
                dataset_sources=[dataset_slug, assets_dataset],
            )
            # Get kernel ref from metadata
            meta = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
            kernel_ref = meta.get("id", "zigomaro/crumple-video")
        except Exception as e:
            logger.error(f"Failed to deploy kernel: {e}")
            return False
        
        logger.info(f"Kernel deployed: {kernel_ref}")
        
        # Poll for completion
        start_time = time.time()
        max_wait = 9000  # 150 minutes (Blender rendering is slow)
        has_started = False
        
        while time.time() - start_time < max_wait:
            try:
                status_info = client.get_kernel_status(kernel_ref)
                status = status_info.get("status", "UNKNOWN").upper()
            except Exception as e:
                logger.warning(f"Status check failed: {e}")
                await asyncio.sleep(30)
                continue
            
            logger.info(f"Status: {status}")
            
            if status in ["QUEUED", "RUNNING"]:
                has_started = True
            
            if status == "COMPLETE":
                if has_started or (time.time() - start_time > 60):
                    logger.info("✅ Kernel completed successfully!")
                    break
                else:
                    logger.info("Waiting for new run to start...")
            
            elif status in ["ERROR", "FAILED", "CANCELLED"]:
                if has_started or (time.time() - start_time > 60):
                    logger.error(f"❌ Kernel failed: {status_info.get('failureMessage')}")
                    return False
            
            await asyncio.sleep(30)
        
        # Download output
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading outputs...")
        try:
            files = client.download_kernel_output(kernel_ref, path=OUTPUT_DIR, force=True)
            logger.info(f"Downloaded files: {files}")
        except Exception as e:
            logger.error(f"Failed to download output: {e}")
            return False
        
        # Check for video file
        video_file = OUTPUT_DIR / "crumple_video_final.mp4"
        if video_file.exists():
            video_size = video_file.stat().st_size
            logger.info(f"Video saved: {video_file}")
            logger.info(f"Video size: {video_size / 1024 / 1024:.2f} MB")
            
            # Validate video size (should be > 500KB for real content)
            if video_size < 500000:
                logger.error(f"❌ VALIDATION FAILED: Video too small ({video_size/1024:.1f} KB), likely empty!")
                return False
            
            logger.info("🎉 SUCCESS! Video size validation passed!")
            return True
        else:
            # Check for any mp4
            mp4_files = list(OUTPUT_DIR.glob("*.mp4"))
            if mp4_files:
                video_size = mp4_files[0].stat().st_size
                if video_size < 500000:
                    logger.error(f"❌ Video too small ({video_size/1024:.1f} KB)")
                    return False
                logger.info(f"🎉 SUCCESS! Found video: {mp4_files[0]} ({video_size/1024/1024:.2f} MB)")
                return True
            else:
                logger.warning("No video file found in output")
                logger.info(f"Output contents: {list(OUTPUT_DIR.iterdir())}")
                return False


if __name__ == "__main__":
    success = asyncio.run(run_crumple_test())
    exit(0 if success else 1)
