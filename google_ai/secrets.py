"""Secrets provider with fallback chain.

Fallback order:
1. Environment variables (os.getenv)
2. Kaggle Secrets (kaggle_secrets.UserSecretsClient)
3. Encrypted datasets (Fernet + 2 private Kaggle datasets)

Based on existing implementation in kaggle/UniversalFestivalParser/src/secrets.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google_ai.exceptions import SecretsError

logger = logging.getLogger(__name__)


class SecretsProvider:
    """Unified secrets provider with fallback chain.
    
    Supports:
    - Single secrets: get_secret("GOOGLE_API_KEY")
    - Secret pools: get_secret_pool("GOOGLE_API_KEY") -> ["key1", "key2", ...]
    
    For pools, looks for GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, etc.
    """
    
    def __init__(
        self,
        cipher_dataset_path: str = "/kaggle/input/eve-secrets-cipher",
        key_dataset_path: str = "/kaggle/input/eve-secrets-key",
        # Legacy paths for backward compatibility
        legacy_cipher_path: str = "/kaggle/input/gemma-cipher",
        legacy_key_path: str = "/kaggle/input/gemma-key",
    ):
        self.cipher_dataset_path = cipher_dataset_path
        self.key_dataset_path = key_dataset_path
        self.legacy_cipher_path = legacy_cipher_path
        self.legacy_key_path = legacy_key_path
        self._cache: dict[str, str] = {}
        self._bundle_loaded = False
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret by name.
        
        Fallback order:
        1. Environment variable
        2. Kaggle Secrets
        3. Encrypted bundle from datasets
        
        Returns None if not found.
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        # 1. Try environment variable
        value = os.getenv(name)
        if value:
            logger.debug("Secret %s found in environment", name)
            self._cache[name] = value
            return value
        
        # 2. Try Kaggle Secrets
        value = self._get_from_kaggle_secrets(name)
        if value:
            logger.debug("Secret %s found in Kaggle Secrets", name)
            self._cache[name] = value
            return value
        
        # 3. Try encrypted datasets
        value = self._get_from_encrypted_bundle(name)
        if value:
            logger.debug("Secret %s found in encrypted bundle", name)
            self._cache[name] = value
            return value
        
        logger.warning("Secret %s not found in any source", name)
        return None
    
    def get_secret_pool(self, prefix: str) -> list[str]:
        """Get all secrets matching a prefix pattern.
        
        For prefix "GOOGLE_API_KEY", returns values of:
        - GOOGLE_API_KEY
        - GOOGLE_API_KEY_2
        - GOOGLE_API_KEY_3
        - etc.
        
        Returns list of values (not keys).
        """
        values = []
        
        # First key (no suffix)
        value = self.get_secret(prefix)
        if value:
            values.append(value)
        
        # Numbered keys
        for i in range(2, 100):  # Support up to 99 keys
            value = self.get_secret(f"{prefix}_{i}")
            if value:
                values.append(value)
            else:
                break  # Stop on first missing
        
        return values
    
    def _get_from_kaggle_secrets(self, name: str) -> Optional[str]:
        """Try to get secret from Kaggle Secrets API."""
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            value = secrets.get_secret(name)
            if value:
                return value
        except ImportError:
            logger.debug("kaggle_secrets not available")
        except Exception as e:
            logger.debug("Kaggle Secrets error for %s: %s", name, e)
        return None
    
    def _get_from_encrypted_bundle(self, name: str) -> Optional[str]:
        """Try to get secret from encrypted dataset bundle."""
        if not self._bundle_loaded:
            self._load_bundle()
        return self._cache.get(name)
    
    def _load_bundle(self) -> None:
        """Load and decrypt secrets bundle from datasets."""
        self._bundle_loaded = True
        
        # Try new bundle format first
        bundle = self._try_load_bundle(
            self.cipher_dataset_path,
            self.key_dataset_path,
            bundle_file="secrets.enc",
            key_file="fernet.keys",
        )
        
        if bundle:
            self._cache.update(bundle)
            return
        
        # Try legacy format (single key file)
        legacy_key = self._try_load_legacy_key()
        if legacy_key:
            self._cache["GOOGLE_API_KEY"] = legacy_key
    
    def _try_load_bundle(
        self,
        cipher_path: str,
        key_path: str,
        bundle_file: str,
        key_file: str,
    ) -> Optional[dict[str, str]]:
        """Try to load and decrypt a secrets bundle."""
        cipher_file = Path(cipher_path) / bundle_file
        keys_file = Path(key_path) / key_file
        
        if not cipher_file.exists() or not keys_file.exists():
            return None
        
        try:
            from cryptography.fernet import Fernet, MultiFernet
            
            # Load key ring (multiple Fernet keys for rotation)
            key_lines = keys_file.read_text().strip().split("\n")
            fernets = [Fernet(k.strip().encode()) for k in key_lines if k.strip()]
            
            if not fernets:
                logger.warning("No valid Fernet keys in %s", keys_file)
                return None
            
            multi_fernet = MultiFernet(fernets)
            
            # Decrypt bundle
            encrypted = cipher_file.read_bytes()
            decrypted = multi_fernet.decrypt(encrypted)
            bundle = json.loads(decrypted.decode("utf-8"))
            
            logger.info("Loaded %d secrets from encrypted bundle", len(bundle))
            return bundle
            
        except ImportError:
            logger.warning("cryptography package not installed")
        except Exception as e:
            logger.error("Failed to decrypt bundle: %s", e)
        
        return None
    
    def _try_load_legacy_key(self) -> Optional[str]:
        """Try to load single key from legacy dataset format."""
        cipher_file = Path(self.legacy_cipher_path) / "google_api_key.enc"
        key_file = Path(self.legacy_key_path) / "fernet.key"
        
        if not cipher_file.exists() or not key_file.exists():
            return None
        
        try:
            from cryptography.fernet import Fernet
            
            fernet_key = key_file.read_bytes().strip()
            fernet = Fernet(fernet_key)
            
            encrypted = cipher_file.read_bytes()
            decrypted = fernet.decrypt(encrypted).decode("utf-8").strip()
            
            # Validate Google API key format
            if not decrypted.startswith("AIza"):
                logger.warning("Decrypted key has unexpected format")
            
            logger.info("Loaded API key from legacy encrypted datasets")
            return decrypted
            
        except ImportError:
            logger.warning("cryptography package not installed")
        except Exception as e:
            logger.error("Failed to decrypt legacy key: %s", e)
        
        return None


# Module-level singleton
_provider: Optional[SecretsProvider] = None


def get_provider() -> SecretsProvider:
    """Get or create the global secrets provider."""
    global _provider
    if _provider is None:
        _provider = SecretsProvider()
    return _provider


def get_secret(name: str) -> Optional[str]:
    """Get a secret by name (module-level convenience function)."""
    return get_provider().get_secret(name)


def get_secret_pool(prefix: str) -> list[str]:
    """Get all secrets matching a prefix (module-level convenience function)."""
    return get_provider().get_secret_pool(prefix)


def create_encrypted_bundle(
    secrets: dict[str, str],
    output_dir: str | Path,
    cipher_filename: str = "secrets.enc",
    key_filename: str = "fernet.keys",
) -> tuple[Path, Path]:
    """Create encrypted bundle files for Kaggle datasets.
    
    This is a helper function for setting up the datasets.
    Run this locally to generate the files to upload.
    
    Args:
        secrets: Dict of secret_name -> secret_value
        output_dir: Directory to save files
        cipher_filename: Name of cipher file
        key_filename: Name of key file
        
    Returns:
        Tuple of (cipher_file_path, key_file_path)
    """
    from cryptography.fernet import Fernet
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate new Fernet key
    fernet_key = Fernet.generate_key()
    fernet = Fernet(fernet_key)
    
    # Encrypt bundle as JSON
    bundle_json = json.dumps(secrets, ensure_ascii=False)
    encrypted = fernet.encrypt(bundle_json.encode("utf-8"))
    
    # Save files
    cipher_path = output_dir / cipher_filename
    key_path = output_dir / key_filename
    
    cipher_path.write_bytes(encrypted)
    key_path.write_text(fernet_key.decode("utf-8"))
    
    print(f"Created {cipher_path} (upload to eve-secrets-cipher dataset)")
    print(f"Created {key_path} (upload to eve-secrets-key dataset)")
    print("IMPORTANT: Keep these datasets PRIVATE!")
    
    return cipher_path, key_path
