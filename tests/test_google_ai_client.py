"""Tests for google_ai.client module."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google_ai.client import (
    GoogleAIClient,
    RequestContext,
    ReserveResult,
    UsageInfo,
    _DEFAULT_ENV_CANDIDATE_CACHE,
)
from google_ai.exceptions import RateLimitError, ProviderError, ReservationError


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
    async def test_reserve_rpc_missing_uses_direct_fallback_when_enabled(self, mock_supabase):
        """Missing reserve RPC should not disable LLM completely when fallback is enabled."""
        mock_supabase.rpc.return_value.execute.side_effect = Exception(
            "PGRST202: Route POST:/rpc/google_ai_reserve not found"
        )

        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test-api-key", "GOOGLE_AI_ALLOW_RESERVE_FALLBACK": "1"},
        ):
            client = GoogleAIClient(
                supabase_client=mock_supabase,
                consumer="test",
                dry_run=True,
            )
            response, _usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="Test reserve fallback",
            )

        assert "[DRY RUN]" in response
        assert mock_supabase.rpc.call_count >= 1

    @pytest.mark.asyncio
    async def test_reserve_fallback_respects_custom_default_env_var(self, mock_supabase):
        mock_supabase.rpc.return_value.execute.side_effect = Exception(
            "PGRST202: Route POST:/rpc/google_ai_reserve not found"
        )

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY2": "test-api-key-2",
                "GOOGLE_AI_ALLOW_RESERVE_FALLBACK": "1",
            },
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=mock_supabase,
                consumer="guide-test",
                default_env_var_name="GOOGLE_API_KEY2",
                dry_run=True,
            )
            response, _usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="Test reserve fallback with custom env",
            )

        assert "[DRY RUN]" in response

    @pytest.mark.asyncio
    async def test_reserve_rpc_missing_raises_when_fallback_disabled(self, mock_supabase):
        """Fallback can be explicitly disabled for strict environments."""
        mock_supabase.rpc.return_value.execute.side_effect = Exception(
            "PGRST202: Could not find function public.google_ai_reserve in schema cache"
        )

        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test-api-key", "GOOGLE_AI_ALLOW_RESERVE_FALLBACK": "0"},
        ):
            client = GoogleAIClient(
                supabase_client=mock_supabase,
                consumer="test",
                dry_run=True,
            )
            with pytest.raises(ReservationError):
                await client.generate_content_async(
                    model="gemma-3-27b",
                    prompt="Test strict reserve",
                )

    @pytest.mark.asyncio
    async def test_reserve_rpc_missing_cached_mode_rechecks_and_recovers(self):
        """Cached fallback should periodically recheck reserve RPC and recover."""
        supabase = MagicMock()
        reserve_calls = 0

        def _rpc_side_effect(fn_name, _payload):
            nonlocal reserve_calls
            resp = MagicMock()
            if fn_name == "google_ai_reserve":
                reserve_calls += 1
                if reserve_calls == 1:
                    resp.execute.side_effect = Exception(
                        "PGRST202: Route POST:/rpc/google_ai_reserve not found"
                    )
                else:
                    resp.execute.return_value = MagicMock(
                        data=[
                            {
                                "ok": True,
                                "api_key_id": "test-key-id",
                                "env_var_name": "GOOGLE_API_KEY",
                                "key_alias": "recovered-key",
                                "minute_bucket": "2024-01-01T00:00:00Z",
                                "day_bucket": "2024-01-01",
                            }
                        ]
                    )
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-api-key",
                "GOOGLE_AI_ALLOW_RESERVE_FALLBACK": "1",
                "GOOGLE_AI_RESERVE_RPC_RECHECK_SECONDS": "60",
            },
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=supabase,
                consumer="test",
                dry_run=True,
            )
            ctx = RequestContext(
                request_uid="req-reserve-recheck",
                consumer="test",
                account_name="local",
                model="gemma-3-27b",
                reserved_tpm=1000,
            )

            with patch("google_ai.client._monotonic", side_effect=[10.0, 30.0, 75.0]):
                first = await client._reserve(ctx, attempt_no=1, candidate_key_ids=None)
                second = await client._reserve(ctx, attempt_no=2, candidate_key_ids=None)
                third = await client._reserve(ctx, attempt_no=3, candidate_key_ids=None)

        assert first.ok is True
        assert first.blocked_reason == "reserve_rpc_missing"
        assert second.ok is True
        assert second.key_alias in {"reserve-fallback-no-rpc-cached", "local-fallback-no-rpc-cached"}
        assert third.ok is True
        assert third.key_alias == "recovered-key"
        assert reserve_calls == 2
        assert client._reserve_rpc_missing is False

    @pytest.mark.asyncio
    async def test_local_reserve_uses_custom_default_env_var(self):
        client = GoogleAIClient(
            supabase_client=None,
            consumer="guide-test",
            default_env_var_name="GOOGLE_API_KEY2",
            dry_run=True,
        )
        ctx = RequestContext(
            request_uid="req-local-custom-env",
            consumer="guide-test",
            account_name="guide",
            model="gemma-3-27b",
            reserved_tpm=1000,
        )
        reserve = await client._reserve(ctx, attempt_no=1, candidate_key_ids=None)
        assert reserve.env_var_name == "GOOGLE_API_KEY2"

    @pytest.mark.asyncio
    async def test_reserve_scopes_to_default_env_metadata_when_candidates_not_provided(self):
        _DEFAULT_ENV_CANDIDATE_CACHE.clear()
        supabase = MagicMock()
        keys_table = MagicMock()
        keys_table.select.return_value = keys_table
        keys_table.eq.return_value = keys_table
        keys_table.in_.return_value = keys_table
        keys_table.order.return_value = keys_table
        keys_table.execute.return_value = MagicMock(
            data=[
                {"id": "generic-id", "env_var_name": "GOOGLE_API_KEY", "priority": 10},
                {"id": "guide-id", "env_var_name": "GOOGLE_API_KEY2", "priority": 5},
            ]
        )
        supabase.table.return_value = keys_table
        supabase.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "ok": True,
                    "api_key_id": "generic-id",
                    "env_var_name": "GOOGLE_API_KEY",
                    "key_alias": "generic",
                    "minute_bucket": "2024-01-01T00:00:00Z",
                    "day_bucket": "2024-01-01",
                }
            ]
        )

        client = GoogleAIClient(
            supabase_client=supabase,
            consumer="smart-update-test",
            dry_run=True,
        )
        ctx = RequestContext(
            request_uid="req-scope-default",
            consumer="smart_update",
            account_name="prod",
            model="gemma-3-27b",
            reserved_tpm=1000,
        )
        reserve = await client._reserve(ctx, attempt_no=1, candidate_key_ids=None)

        assert reserve.api_key_id == "generic-id"
        payload = supabase.rpc.call_args.args[1]
        assert payload["p_candidate_key_ids"] == ["generic-id"]

    @pytest.mark.asyncio
    async def test_reserve_keeps_explicit_candidate_ids(self):
        _DEFAULT_ENV_CANDIDATE_CACHE.clear()
        supabase = MagicMock()
        supabase.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "ok": True,
                    "api_key_id": "explicit-id",
                    "env_var_name": "GOOGLE_API_KEY2",
                    "key_alias": "guide-explicit",
                    "minute_bucket": "2024-01-01T00:00:00Z",
                    "day_bucket": "2024-01-01",
                }
            ]
        )

        client = GoogleAIClient(
            supabase_client=supabase,
            consumer="guide-test",
            default_env_var_name="GOOGLE_API_KEY",
            dry_run=True,
        )
        ctx = RequestContext(
            request_uid="req-explicit-candidates",
            consumer="guide-test",
            account_name="guide",
            model="gemma-3-27b",
            reserved_tpm=1000,
        )
        reserve = await client._reserve(ctx, attempt_no=1, candidate_key_ids=["explicit-id"])

        assert reserve.api_key_id == "explicit-id"
        payload = supabase.rpc.call_args.args[1]
        assert payload["p_candidate_key_ids"] == ["explicit-id"]

    @pytest.mark.asyncio
    async def test_reserve_rpc_transient_error_retries_before_fallback(self):
        supabase = MagicMock()
        reserve_calls = 0

        def _rpc_side_effect(fn_name, _payload):
            nonlocal reserve_calls
            resp = MagicMock()
            if fn_name == "google_ai_reserve":
                reserve_calls += 1
                if reserve_calls == 1:
                    resp.execute.side_effect = Exception(
                        "Server disconnected without sending a response."
                    )
                else:
                    resp.execute.return_value = MagicMock(
                        data=[
                            {
                                "ok": True,
                                "api_key_id": "test-key-id",
                                "env_var_name": "GOOGLE_API_KEY",
                                "key_alias": "retry-ok",
                                "minute_bucket": "2024-01-01T00:00:00Z",
                                "day_bucket": "2024-01-01",
                            }
                        ]
                    )
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-api-key",
                "GOOGLE_AI_ALLOW_RESERVE_FALLBACK": "1",
                "GOOGLE_AI_RESERVE_RPC_RETRY_ATTEMPTS": "2",
                "GOOGLE_AI_RESERVE_RPC_RETRY_BASE_DELAY_MS": "50",
            },
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=supabase,
                consumer="test",
                dry_run=True,
            )
            text, _usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="retry transient reserve rpc",
            )
            assert "[DRY RUN]" in text

        assert reserve_calls == 2
        assert client._reserve_rpc_missing is False
    
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

    @pytest.mark.asyncio
    async def test_provider_429_is_not_retried(self, mock_supabase):
        """Provider 429 should bubble up so outer workflows can handle waiting."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
            client = GoogleAIClient(
                supabase_client=mock_supabase,
                consumer="test",
                dry_run=False,
            )

            mock_genai = MagicMock()
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(
                side_effect=Exception("429 Resource exhausted. Please retry in 58s.")
            )
            mock_genai.GenerativeModel.return_value = mock_model
            client._genai = mock_genai

            with pytest.raises(ProviderError) as exc_info:
                await client.generate_content_async(
                    model="gemma-3-27b",
                    prompt="Test",
                )

            assert exc_info.value.status_code == 429
            assert mock_model.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_finalize_fallback_uses_legacy_after_new_rpc_missing(self):
        """If new finalize RPC is missing, client should keep using legacy finalize."""
        supabase = MagicMock()

        def _rpc_side_effect(fn_name, _payload):
            resp = MagicMock()
            if fn_name == "google_ai_finalize":
                resp.execute.side_effect = Exception(
                    "PGRST202: Could not find function public.google_ai_finalize in schema cache"
                )
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect
        client = GoogleAIClient(supabase_client=supabase, consumer="test", dry_run=True)

        ctx = RequestContext(
            request_uid="req-1",
            consumer="test",
            account_name=None,
            model="gemma-3-27b",
            reserved_tpm=1000,
        )
        ctx.api_key_id = "key-1"
        usage = UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15)

        await client._finalize(ctx, attempt_no=1, usage=usage, duration_ms=10)
        await client._finalize(ctx, attempt_no=2, usage=usage, duration_ms=12)

        called_rpcs = [call.args[0] for call in supabase.rpc.call_args_list]
        assert called_rpcs == [
            "google_ai_finalize",
            "finalize_google_ai_usage",
            "finalize_google_ai_usage",
        ]

    @pytest.mark.asyncio
    async def test_mark_sent_retries_transient_rpc_error(self):
        supabase = MagicMock()
        calls = {"n": 0}

        def _rpc_side_effect(fn_name, _payload):
            resp = MagicMock()
            if fn_name == "google_ai_mark_sent":
                calls["n"] += 1
                if calls["n"] == 1:
                    resp.execute.side_effect = Exception("Server disconnected without sending a response.")
                else:
                    resp.execute.return_value = MagicMock(data=None)
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect
        client = GoogleAIClient(supabase_client=supabase, consumer="test", dry_run=True)
        ctx = RequestContext(
            request_uid="req-mark-sent",
            consumer="test",
            account_name="local",
            model="gemma-3-27b",
            reserved_tpm=1000,
        )

        async def _noop_sleep(_sec: float) -> None:
            return None

        with patch.dict(os.environ, {"GOOGLE_AI_RESERVE_RPC_RETRY_ATTEMPTS": "2"}, clear=False):
            with patch("google_ai.client.asyncio.sleep", new=_noop_sleep):
                await client._mark_sent(ctx, 1)

        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_finalize_retries_transient_rpc_error(self):
        supabase = MagicMock()
        calls = {"n": 0}

        def _rpc_side_effect(fn_name, _payload):
            resp = MagicMock()
            if fn_name == "google_ai_finalize":
                calls["n"] += 1
                if calls["n"] == 1:
                    resp.execute.side_effect = Exception("SSL handshake timeout")
                else:
                    resp.execute.return_value = MagicMock(data=None)
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect
        client = GoogleAIClient(supabase_client=supabase, consumer="test", dry_run=True)
        ctx = RequestContext(
            request_uid="req-finalize",
            consumer="test",
            account_name="local",
            model="gemma-3-27b",
            reserved_tpm=1000,
        )
        ctx.api_key_id = "key-1"
        usage = UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15)

        async def _noop_sleep(_sec: float) -> None:
            return None

        with patch.dict(os.environ, {"GOOGLE_AI_RESERVE_RPC_RETRY_ATTEMPTS": "2"}, clear=False):
            with patch("google_ai.client.asyncio.sleep", new=_noop_sleep):
                await client._finalize(ctx, attempt_no=1, usage=usage, duration_ms=10)

        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_mark_sent_missing_rpc_is_cached_case_insensitive(self):
        """Missing mark_sent RPC should be detected and skipped on next calls."""
        supabase = MagicMock()
        supabase.rpc.return_value.execute.side_effect = Exception(
            "pgrst202: route post:/rpc/google_ai_mark_sent not found"
        )
        client = GoogleAIClient(supabase_client=supabase, consumer="test", dry_run=True)
        ctx = RequestContext(
            request_uid="req-1",
            consumer="test",
            account_name=None,
            model="gemma-3-27b",
            reserved_tpm=1000,
        )

        await client._mark_sent(ctx, 1)
        assert client._mark_sent_rpc_missing is True

        supabase.rpc.reset_mock()
        await client._mark_sent(ctx, 2)
        supabase.rpc.assert_not_called()
    
    def test_calculate_reserved_tpm(self, client):
        """Test TPM reservation calculation."""
        prompt = "Hello, world!"
        reserved = client._calculate_reserved_tpm(prompt=prompt, max_output_tokens=8192)
        assert reserved > 8192  # includes prompt estimate + extra

    def test_estimate_prompt_tokens_is_more_conservative_for_cyrillic_ocr(self, client):
        prompt = ("АФИША\n" + ("Концерт 19:00 Калининград\n" * 1200)).strip()
        estimate = client._estimate_prompt_tokens(prompt)
        legacy = int((len(prompt.encode("utf-8")) / 4.0) * 1.15) + 50
        assert estimate > legacy
    
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

    def test_classify_error_retry_after_ms(self, client):
        error = Exception("429 You exceeded your current quota. Please retry in 51.1s.")
        result = client._classify_error(error)
        assert result.status_code == 429
        assert result.retry_after_ms is not None
        assert 50_000 <= result.retry_after_ms <= 60_000


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


class TestLoggingModelNames:
    """Structured log should include concrete invoked model names."""

    def test_log_event_contains_requested_and_provider_model_names(self, caplog):
        client = GoogleAIClient(supabase_client=None, consumer="test", dry_run=True)
        ctx = RequestContext(
            request_uid="req-123",
            consumer="test",
            account_name="local",
            model="gemma-3-27b",
            requested_model="gemma-3-27b",
            provider_model="gemma-3-27b-it",
            provider_model_name="models/gemma-3-27b-it",
            reserved_tpm=8192,
        )

        with caplog.at_level("INFO", logger="google_ai.client"):
            client._log_event("google_ai.test", ctx, attempt_no=1)

        payload = json.loads(caplog.records[-1].message)
        assert payload["model"] == "gemma-3-27b"
        assert payload["requested_model"] == "gemma-3-27b"
        assert payload["provider_model"] == "gemma-3-27b-it"
        assert payload["provider_model_name"] == "models/gemma-3-27b-it"
        assert payload["invoked_model"] == "models/gemma-3-27b-it"


class TestModelChainPolicy:
    """Text quality policy for model chains."""

    def test_chain_prioritizes_27b_and_skips_models_below_12b(self):
        with patch.dict(
            os.environ,
            {"GOOGLE_AI_FALLBACK_MODELS": "gemma-3-4b,gemma-3-12b,gemma-3-1b"},
            clear=False,
        ):
            client = GoogleAIClient(supabase_client=None, consumer="test", dry_run=True)
            chain = client._build_model_chain("gemma-3-12b")

        assert chain[0] == "gemma-3-27b"
        assert "gemma-3-12b" in chain
        assert not any("gemma-3-4b" in m for m in chain)
        assert not any("gemma-3-1b" in m for m in chain)

    def test_low_requested_model_is_replaced_by_primary_27b(self):
        with patch.dict(
            os.environ,
            {"GOOGLE_AI_FALLBACK_MODELS": ""},
            clear=False,
        ):
            client = GoogleAIClient(supabase_client=None, consumer="test", dry_run=True)
            chain = client._build_model_chain("gemma-3-4b")

        assert chain == ["gemma-3-27b"]


class TestIncidentNotifications:
    @pytest.mark.asyncio
    async def test_provider_error_sends_incident(self):
        calls: list[tuple[str, dict]] = []

        async def _notifier(kind: str, payload: dict) -> None:
            calls.append((kind, payload))

        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test-api-key", "GOOGLE_AI_MAX_RETRIES": "1"},
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=None,
                consumer="test",
                dry_run=False,
                incident_notifier=_notifier,
            )
            mock_genai = MagicMock()
            failing_model = MagicMock()
            failing_model.generate_content_async = AsyncMock(
                side_effect=Exception("503 service unavailable")
            )
            mock_genai.GenerativeModel.return_value = failing_model
            client._genai = mock_genai

            with pytest.raises(ProviderError):
                await client.generate_content_async(
                    model="gemma-3-27b",
                    prompt="hello",
                )

        assert calls
        kind, payload = calls[-1]
        assert kind == "provider_error"
        assert payload.get("requested_model") == "gemma-3-27b"
        assert "models/gemma-3-27b-it" in str(payload.get("invoked_model"))

    @pytest.mark.asyncio
    async def test_reserve_rpc_missing_sends_incident(self):
        calls: list[tuple[str, dict]] = []

        async def _notifier(kind: str, payload: dict) -> None:
            calls.append((kind, payload))

        supabase = MagicMock()

        def _rpc_side_effect(fn_name, _payload):
            resp = MagicMock()
            if fn_name == "google_ai_reserve":
                resp.execute.side_effect = Exception(
                    "PGRST202: Route POST:/rpc/google_ai_reserve not found"
                )
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-api-key",
                "GOOGLE_AI_ALLOW_RESERVE_FALLBACK": "1",
                "GOOGLE_AI_MAX_RETRIES": "1",
                "GOOGLE_AI_RESERVE_DIRECT_RETRY": "0",
            },
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=supabase,
                consumer="test",
                dry_run=True,
                incident_notifier=_notifier,
            )
            text, _usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="probe",
            )
            assert "[DRY RUN]" in text

        assert any(kind == "reserve_rpc_missing" for kind, _payload in calls)
        payload = next(payload for kind, payload in calls if kind == "reserve_rpc_missing")
        assert str(payload.get("severity")).lower() == "warning"

    @pytest.mark.asyncio
    async def test_fallback_model_chain_recovers_after_primary_failure(self):
        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-api-key",
                "GOOGLE_AI_MAX_RETRIES": "1",
                "GOOGLE_AI_FALLBACK_MODELS": "gemma-3-12b",
            },
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=None,
                consumer="test",
                dry_run=False,
            )

            used_models: list[str] = []
            mock_genai = MagicMock()

            def _model_factory(model_name: str):
                used_models.append(model_name)
                model_obj = MagicMock()
                if "gemma-3-27b-it" in model_name:
                    model_obj.generate_content_async = AsyncMock(
                        side_effect=Exception("503 service unavailable")
                    )
                else:
                    model_obj.generate_content_async = AsyncMock(
                        return_value=SimpleNamespace(text="ok", usage_metadata=None)
                    )
                return model_obj

            mock_genai.GenerativeModel.side_effect = _model_factory
            client._genai = mock_genai

            text, usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="hello",
            )

        assert text == "ok"
        assert usage.total_tokens == 0
        assert any("gemma-3-27b-it" in m for m in used_models)
        assert any("gemma-3-12b-it" in m for m in used_models)

    @pytest.mark.asyncio
    async def test_fallback_chain_uses_unique_attempt_numbers(self):
        supabase = MagicMock()

        def _rpc_side_effect(fn_name, payload):
            resp = MagicMock()
            if fn_name == "google_ai_reserve":
                resp.execute.return_value = MagicMock(
                    data=[
                        {
                            "ok": True,
                            "api_key_id": "key-1",
                            "env_var_name": "GOOGLE_API_KEY",
                            "key_alias": "k1",
                        }
                    ]
                )
            else:
                resp.execute.return_value = MagicMock(data=None)
            return resp

        supabase.rpc.side_effect = _rpc_side_effect

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-api-key",
                "GOOGLE_AI_MAX_RETRIES": "1",
                "GOOGLE_AI_FALLBACK_MODELS": "gemma-3-12b",
            },
            clear=False,
        ):
            client = GoogleAIClient(
                supabase_client=supabase,
                consumer="test",
                dry_run=False,
            )

            mock_genai = MagicMock()
            used_models: list[str] = []

            def _model_factory(model_name: str):
                used_models.append(model_name)
                model_obj = MagicMock()
                if "gemma-3-27b-it" in model_name:
                    model_obj.generate_content_async = AsyncMock(
                        side_effect=Exception("503 service unavailable")
                    )
                else:
                    model_obj.generate_content_async = AsyncMock(
                        return_value=SimpleNamespace(text="ok", usage_metadata=None)
                    )
                return model_obj

            mock_genai.GenerativeModel.side_effect = _model_factory
            client._genai = mock_genai

            text, _usage = await client.generate_content_async(
                model="gemma-3-27b",
                prompt="hello",
            )

        assert text == "ok"
        reserve_calls = [
            kwargs["p_attempt_no"]
            for name, kwargs in (
                (call.args[0], call.args[1]) for call in supabase.rpc.call_args_list
            )
            if name == "google_ai_reserve"
        ]
        assert reserve_calls == [1, 2]
