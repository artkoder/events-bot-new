
import base64
import os
import random
from typing import Tuple, Optional
from pathlib import Path

# Try to import cryptography, but don't fail immediately if not present (server might not have it installed yet)
try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

def _load_static_key() -> Optional[bytes]:
    key_raw = (os.getenv("TG_MONITORING_FERNET_KEY") or "").strip()
    if key_raw:
        return key_raw.encode("utf-8")
    key_path = (os.getenv("TG_MONITORING_FERNET_KEY_PATH") or "").strip()
    if key_path:
        path = Path(key_path)
        if path.exists():
            return path.read_bytes().strip()
    return None


def encrypt_secret(secret: str) -> Tuple[bytes, bytes]:
    """
    Encrypt a secret using Fernet (symmetric encryption).
    
    Returns:
        (encrypted_data, key)
    """
    if Fernet is None:
        raise ImportError("cryptography package is required for secure secret splitting")
        
    if not secret:
        return b"", b""
        
    key = _load_static_key() or Fernet.generate_key()
    f = Fernet(key)
    encrypted_data = f.encrypt(secret.encode("utf-8"))
    
    return encrypted_data, key

def decrypt_secret(encrypted_data: bytes, key: bytes) -> str:
    """
    Decrypt a secret using Fernet key.
    """
    if Fernet is None:
        raise ImportError("cryptography package is required for secure secret splitting")
        
    if not encrypted_data or not key:
        return ""
        
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode("utf-8")
