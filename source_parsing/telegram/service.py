import json
import logging
import os
import shutil
import tempfile
import base64
from pathlib import Path

from db import Database
from video_announce.kaggle_client import KaggleClient
from source_parsing.telegram.deduplication import get_month_context_urls
from .split_secrets import encrypt_secret

logger = logging.getLogger(__name__)

# Dataset containing the encrypted session string (and other config)
CONFIG_DATASET_CIPHER = "telegram-monitor-cipher"
# Dataset containing the Fernet key to decrypt
CONFIG_DATASET_KEY = "telegram-monitor-key"

KERNEL_REF = "artkoder/telegram-monitor-bot" # update with actual ref
KERNEL_PATH = Path("kaggle/TelegramMonitor")

async def run_telegram_monitor(db: Database, tg_session: str, channels: list[str]):
    """
    Orchestrate the Telegram Monitor job with Secure Session Splitting.
    
    1. Prepare Context (Telegraph URLs).
    2. Encrypt Session string using Fernet.
    3. Update 'telegram-monitor-cipher' (contains encrypted session)
       and 'telegram-monitor-key' (contains key) private datasets.
    4. Push/Run the 'TelegramMonitor' kernel (attached to both).
    """
    logger.info("Starting Telegram Monitor job...")
    
    # 1. Prepare Context
    telegraph_urls = await get_month_context_urls(db)
    logger.info(f"Context: {len(telegraph_urls)} Telegraph URLs found.")
    
    # 2. Encrypt Secrets
    # We encrypt the session string. Config can be in cleartext in the cipher dataset (it's private anyway),
    # but strictly speaking the 'key' dataset should ONLY have the key.
    
    encrypted_session, fernet_key = encrypt_secret(tg_session)
    
    # We store the encrypted session as a base64 string to avoid binary issues in JSON.
    encrypted_session_b64 = base64.b64encode(encrypted_session).decode('utf-8')
    # fernet_key is binary, we write it to .key file directly, but for logging/checking we handle it carefully.
    
    # Dataset 1: Cipher (Config + Encrypted Data)
    config_data = {
        "TG_SESSION_ENCRYPTED": encrypted_session_b64,
        "CHANNELS": channels,
        "TELEGRAPH_URLS": telegraph_urls,
        "TG_API_ID": os.environ.get("TG_API_ID"), 
        "TG_API_HASH": os.environ.get("TG_API_HASH"),
        "GEMMA_API_KEY": os.environ.get("GEMMA_API_KEY") 
    }
    
    client = KaggleClient()
    api = client._get_api()
    user = os.environ.get("KAGGLE_USERNAME", "artkoder")
    
    # Helper to update dataset files
    def update_dataset_files(slug_suffix, title, file_writer_func):
        slug = f"{user}/{slug_suffix}"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Allow caller to write files
            file_writer_func(tmp_path)
            
            metadata = {
                "title": title,
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(tmp_path / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f)
                
            try:
                api.dataset_list_files(slug)
                logger.info(f"Updating dataset {slug}...")
                api.dataset_create_version(
                    str(tmp_path), version_notes="Automated update", quiet=True, dir_mode="zip"
                )
            except Exception:
                logger.info(f"Creating dataset {slug}...")
                api.dataset_create_new(
                    str(tmp_path), public=False, quiet=True, dir_mode="zip"
                )
        return slug

    # 3. Update Datasets
    
    def write_cipher(path):
        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, ensure_ascii=False)
            
    def write_key(path):
        with open(path / "fernet.key", "wb") as f:
            f.write(fernet_key)

    slug_cipher = update_dataset_files(CONFIG_DATASET_CIPHER, "Telegram Monitor Cipher", write_cipher)
    slug_key = update_dataset_files(CONFIG_DATASET_KEY, "Telegram Monitor Key", write_key)
    
    logger.info("Datasets updated. Preparing Kernel...")

    # 4. Push Kernel
    # Patch metadata to include BOTH datasets
    meta_path = KERNEL_PATH / "kernel-metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        sources = meta.get("dataset_sources", [])
        
        # Ensure both are present
        for s in [slug_cipher, slug_key]:
            if s not in sources:
                sources.append(s)
            
        meta["dataset_sources"] = sources
        meta_path.write_text(json.dumps(meta, indent=2))
            
    # Push kernel
    logger.info(f"Pushing kernel {KERNEL_REF} from {KERNEL_PATH}...")
    client.push_kernel(kernel_path=KERNEL_PATH)
    
    logger.info("Job submitted. Monitoring should start shortly on Kaggle.")
