
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
        
    key = Fernet.generate_key()
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
