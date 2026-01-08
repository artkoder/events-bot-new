"""Tests for google_ai.client module."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google_ai.client import GoogleAIClient, ReserveResult, UsageInfo, RequestContext
from google_ai.exceptions import RateLimitError, ProviderError


class TestGoogleAIClient:
    """Tests for GoogleAIClient class."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Create mock Supabase client."""
        client = MagicMock()
        client.rpc.return_value.execute.return_value.data = [{
            "ok": True,
            "api_key_id": "test-key-id",
            "env_var_name": "GOOGLE_API_KEY",
            "key_alias": "test-key",
            "minute_bucket": "2024-01-01T00:00:00Z",
            "day_bucket": "2024-01-01",
        }]
        return client
    
    @pytest.fixture
    def client(self, mock_supabase):
        """Create GoogleAIClient with mocks."""
        return GoogleAIClient(
            supabase_client=mock_supabase,
            consumer="test",
            dry_run=True,  # Skip actual API calls
        )
    
    @pytest.mark.asyncio
    async def test_generate_content_success(self, client):
        """Test successful content generation."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
            response, usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="Hello, world!",
            )
            
            assert "[DRY RUN]" in response
            assert usage.total_tokens > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_no_retry(self, mock_supabase):
        """Test that RateLimitError is raised immediately without retry."""
        # Configure mock to return blocked
        mock_supabase.rpc.return_value.execute.return_value.data = [{
            "ok": False,
            "blocked_reason": "rpm",
            "retry_after_ms": 30000,
        }]
        
        client = GoogleAIClient(
            supabase_client=mock_supabase,
            consumer="test",
        )
        
        with pytest.raises(RateLimitError) as exc_info:
            await client.generate_content_async(
                model="gemma-3-27b",
                prompt="Test",
            )
        
        assert exc_info.value.blocked_reason == "rpm"
        assert exc_info.value.retry_after_ms == 30000
        
        # Should only call reserve once (no retries)
        assert mock_supabase.rpc.call_count == 1
    
    @pytest.mark.asyncio
    async def test_provider_error_retries(self, mock_supabase):
        """Test that ProviderError triggers retries."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
            client = GoogleAIClient(
                supabase_client=mock_supabase,
                consumer="test",
                dry_run=False,  # Need real calls to test retries
            )
            
            # Mock genai to raise error
            mock_genai = MagicMock()
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(
                side_effect=Exception("Connection timeout")
            )
            mock_genai.GenerativeModel.return_value = mock_model
            client._genai = mock_genai
            
            with pytest.raises(ProviderError) as exc_info:
                await client.generate_content_async(
                    model="gemma-3-27b",
                    prompt="Test",
                )
            
            # Should retry up to MAX_RETRIES times
            assert mock_model.generate_content_async.call_count == GoogleAIClient.MAX_RETRIES
    
    def test_calculate_reserved_tpm(self, client):
        """Test TPM reservation calculation."""
        reserved = client._calculate_reserved_tpm(8192)
        assert reserved == 8192 + client.DEFAULT_TPM_RESERVE_EXTRA
    
    def test_classify_error_retryable(self, client):
        """Test error classification for retryable errors."""
        error = Exception("Connection timeout")
        result = client._classify_error(error)
        
        assert result.retryable is True
        assert "timeout" in result.error_message.lower()
    
    def test_classify_error_non_retryable(self, client):
        """Test error classification for non-retryable errors."""
        error = ValueError("Invalid argument")
        result = client._classify_error(error)
        
        assert result.retryable is False


class TestReserveResult:
    """Tests for ReserveResult dataclass."""
    
    def test_successful_reserve(self):
        """Test creating successful reserve result."""
        result = ReserveResult(
            ok=True,
            api_key_id="key-123",
            env_var_name="GOOGLE_API_KEY",
        )
        
        assert result.ok is True
        assert result.api_key_id == "key-123"
        assert result.blocked_reason is None
    
    def test_blocked_reserve(self):
        """Test creating blocked reserve result."""
        result = ReserveResult(
            ok=False,
            blocked_reason="rpd",
            retry_after_ms=None,
        )
        
        assert result.ok is False
        assert result.blocked_reason == "rpd"


class TestUsageInfo:
    """Tests for UsageInfo dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        usage = UsageInfo()
        
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
    
    def test_with_values(self):
        """Test with actual values."""
        usage = UsageInfo(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150


class TestExceptions:
    """Tests for custom exceptions."""
    
    def test_rate_limit_error_str(self):
        """Test RateLimitError string representation."""
        error = RateLimitError(
            blocked_reason="tpm",
            retry_after_ms=45000,
        )
        
        assert "tpm" in str(error)
        assert "45000" in str(error)
    
    def test_provider_error_str(self):
        """Test ProviderError string representation."""
        error = ProviderError(
            error_type="ApiError",
            error_code="QUOTA_EXCEEDED",
            error_message="Daily quota exceeded",
        )
        
        assert "ApiError" in str(error)
        assert "QUOTA_EXCEEDED" in str(error)
        assert "Daily quota" in str(error)


class TestNoSupabase:
    """Tests for client without Supabase (local dev mode)."""
    
    @pytest.mark.asyncio
    async def test_no_supabase_skips_reserve(self):
        """Test that missing Supabase skips rate limiting."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
            client = GoogleAIClient(
                supabase_client=None,
                consumer="test",
                dry_run=True,
            )
            
            # Should not raise, just log warning
            response, usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="Test",
            )
            
            assert "[DRY RUN]" in response
