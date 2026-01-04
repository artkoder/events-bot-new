"""Secrets management for Universal Festival Parser.

Decrypts Google API key from two private Kaggle datasets:
- Dataset A: Contains encrypted key (google_api_key.enc)
- Dataset B: Contains Fernet key (fernet.key)

This separation ensures that no single dataset contains everything
needed to recover the API key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_api_key_from_datasets(
    cipher_dataset_path: str = "/kaggle/input/gemma-cipher",
    key_dataset_path: str = "/kaggle/input/gemma-key",
) -> Optional[str]:
    """Decrypt Google API key from two private Kaggle datasets.
    
    Args:
        cipher_dataset_path: Path to dataset with encrypted key
        key_dataset_path: Path to dataset with Fernet key
        
    Returns:
        Decrypted API key or None if not available
        
    Security notes:
    - Key is decrypted in memory only
    - Never written to disk
    - Datasets should be private and owned by trusted account
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.error("cryptography package not installed")
        return None
    
    cipher_path = Path(cipher_dataset_path) / "google_api_key.enc"
    key_path = Path(key_dataset_path) / "fernet.key"
    
    if not cipher_path.exists():
        logger.warning("Cipher file not found: %s", cipher_path)
        return None
    
    if not key_path.exists():
        logger.warning("Fernet key file not found: %s", key_path)
        return None
    
    try:
        # Read Fernet key
        fernet_key = key_path.read_bytes().strip()
        fernet = Fernet(fernet_key)
        
        # Read and decrypt API key
        encrypted_key = cipher_path.read_bytes()
        decrypted_key = fernet.decrypt(encrypted_key).decode("utf-8").strip()
        
        # Validate key format (should start with AIza for Google API keys)
        if not decrypted_key.startswith("AIza"):
            logger.warning("Decrypted key has unexpected format")
        
        logger.info("Successfully decrypted API key from datasets")
        return decrypted_key
        
    except Exception as e:
        logger.error("Failed to decrypt API key: %s", e)
        return None


def get_api_key() -> Optional[str]:
    """Get Google API key from available sources.
    
    Tries in order:
    1. GOOGLE_API_KEY environment variable
    2. Kaggle secrets (via kaggle_secrets)
    3. Private Kaggle datasets (Fernet encrypted)
    
    Returns:
        API key or None if not found
    """
    # 1. Try environment variable
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        logger.info("Using API key from GOOGLE_API_KEY environment variable")
        return env_key
    
    # 2. Try Kaggle secrets (may fail in API context)
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        kaggle_key = secrets.get_secret("GOOGLE_API_KEY")
        if kaggle_key:
            logger.info("Using API key from Kaggle secrets")
            return kaggle_key
    except Exception as e:
        logger.debug("Kaggle secrets not available: %s", e)
    
    # 3. Try private datasets with Fernet encryption
    dataset_key = get_api_key_from_datasets()
    if dataset_key:
        return dataset_key
    
    logger.error("No API key found from any source")
    return None


def create_encrypted_key_files(
    api_key: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Create encrypted key files for Kaggle datasets.
    
    This is a helper function for setting up the datasets.
    Run this locally to generate the files to upload.
    
    Args:
        api_key: The Google API key to encrypt
        output_dir: Directory to save files
        
    Returns:
        Tuple of (cipher_file_path, key_file_path)
    """
    from cryptography.fernet import Fernet
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate new Fernet key
    fernet_key = Fernet.generate_key()
    fernet = Fernet(fernet_key)
    
    # Encrypt API key
    encrypted_key = fernet.encrypt(api_key.encode("utf-8"))
    
    # Save files
    cipher_path = output_dir / "google_api_key.enc"
    key_path = output_dir / "fernet.key"
    
    cipher_path.write_bytes(encrypted_key)
    key_path.write_bytes(fernet_key)
    
    print(f"Created {cipher_path} (upload to gemma-cipher dataset)")
    print(f"Created {key_path} (upload to gemma-key dataset)")
    print("IMPORTANT: Keep these datasets PRIVATE!")
    
    return cipher_path, key_path
