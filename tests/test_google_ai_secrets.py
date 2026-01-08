"""Tests for google_ai.secrets module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from google_ai.secrets import (
    SecretsProvider,
    get_secret,
    get_secret_pool,
    create_encrypted_bundle,
)


class TestSecretsProvider:
    """Tests for SecretsProvider class."""
    
    def test_get_secret_from_env(self):
        """Test getting secret from environment variable."""
        with patch.dict(os.environ, {"TEST_SECRET": "test_value"}):
            provider = SecretsProvider()
            assert provider.get_secret("TEST_SECRET") == "test_value"
    
    def test_get_secret_caching(self):
        """Test that secrets are cached."""
        with patch.dict(os.environ, {"CACHED_SECRET": "cached_value"}):
            provider = SecretsProvider()
            
            # First call
            result1 = provider.get_secret("CACHED_SECRET")
            
            # Clear env
            os.environ.pop("CACHED_SECRET", None)
            
            # Second call should return cached value
            result2 = provider.get_secret("CACHED_SECRET")
            
            assert result1 == result2 == "cached_value"
    
    def test_get_secret_not_found(self):
        """Test returning None when secret not found."""
        provider = SecretsProvider()
        result = provider.get_secret("NONEXISTENT_SECRET_12345")
        assert result is None
    
    def test_get_secret_pool_single(self):
        """Test getting pool with single key."""
        with patch.dict(os.environ, {"POOL_KEY": "value1"}):
            provider = SecretsProvider()
            result = provider.get_secret_pool("POOL_KEY")
            assert result == ["value1"]
    
    def test_get_secret_pool_multiple(self):
        """Test getting pool with multiple keys."""
        with patch.dict(os.environ, {
            "MULTI_KEY": "value1",
            "MULTI_KEY_2": "value2",
            "MULTI_KEY_3": "value3",
        }):
            provider = SecretsProvider()
            result = provider.get_secret_pool("MULTI_KEY")
            assert result == ["value1", "value2", "value3"]
    
    def test_get_secret_pool_gap_stops(self):
        """Test that pool stops at first missing key."""
        with patch.dict(os.environ, {
            "GAP_KEY": "value1",
            "GAP_KEY_2": "value2",
            # GAP_KEY_3 missing
            "GAP_KEY_4": "value4",
        }):
            provider = SecretsProvider()
            result = provider.get_secret_pool("GAP_KEY")
            # Should stop at GAP_KEY_3
            assert result == ["value1", "value2"]
    
    def test_get_secret_pool_empty(self):
        """Test getting empty pool."""
        provider = SecretsProvider()
        result = provider.get_secret_pool("EMPTY_POOL_12345")
        assert result == []
    
    @patch("google_ai.secrets.SecretsProvider._get_from_kaggle_secrets")
    def test_fallback_to_kaggle_secrets(self, mock_kaggle):
        """Test fallback to Kaggle Secrets when env not available."""
        mock_kaggle.return_value = "kaggle_value"
        
        provider = SecretsProvider()
        result = provider.get_secret("KAGGLE_SECRET")
        
        assert result == "kaggle_value"
        mock_kaggle.assert_called_once_with("KAGGLE_SECRET")


class TestEncryptedBundle:
    """Tests for encrypted bundle functionality."""
    
    def test_create_and_load_bundle(self):
        """Test creating and loading encrypted bundle."""
        # Skip if cryptography not installed
        pytest.importorskip("cryptography")
        
        secrets = {
            "SECRET_A": "value_a",
            "SECRET_B": "value_b",
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create bundle
            cipher_path, key_path = create_encrypted_bundle(
                secrets=secrets,
                output_dir=tmpdir,
            )
            
            assert cipher_path.exists()
            assert key_path.exists()
            
            # Create subdirs to simulate Kaggle datasets
            cipher_dataset = Path(tmpdir) / "cipher_dataset"
            key_dataset = Path(tmpdir) / "key_dataset"
            cipher_dataset.mkdir()
            key_dataset.mkdir()
            
            # Copy files
            (cipher_dataset / "secrets.enc").write_bytes(cipher_path.read_bytes())
            (key_dataset / "fernet.keys").write_text(key_path.read_text())
            
            # Load bundle
            provider = SecretsProvider(
                cipher_dataset_path=str(cipher_dataset),
                key_dataset_path=str(key_dataset),
            )
            
            assert provider.get_secret("SECRET_A") == "value_a"
            assert provider.get_secret("SECRET_B") == "value_b"


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_secret(self):
        """Test module-level get_secret function."""
        with patch.dict(os.environ, {"MODULE_SECRET": "module_value"}):
            # Reset provider singleton
            import google_ai.secrets
            google_ai.secrets._provider = None
            
            result = get_secret("MODULE_SECRET")
            assert result == "module_value"
    
    def test_get_secret_pool(self):
        """Test module-level get_secret_pool function."""
        with patch.dict(os.environ, {
            "MODULE_POOL": "val1",
            "MODULE_POOL_2": "val2",
        }):
            # Reset provider singleton
            import google_ai.secrets
            google_ai.secrets._provider = None
            
            result = get_secret_pool("MODULE_POOL")
            assert result == ["val1", "val2"]
