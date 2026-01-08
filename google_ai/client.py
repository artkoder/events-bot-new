"""Google AI client with Supabase-based rate limiting.

Features:
- Wrapper over google.generativeai
- Atomic reserve/finalize through Supabase RPC
- NO_WAIT policy: raises RateLimitError immediately on limit exceeded
- Retries only on provider errors (max 3)
- Structured logging (JSON lines)
- Idempotency via request_uid
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING

from google_ai.exceptions import RateLimitError, ProviderError, ReservationError

if TYPE_CHECKING:
    from supabase import Client as SupabaseClient

logger = logging.getLogger(__name__)


@dataclass
class ReserveResult:
    """Result of a successful rate limit reservation."""
    ok: bool
    api_key_id: Optional[str] = None
    env_var_name: Optional[str] = None
    key_alias: Optional[str] = None
    minute_bucket: Optional[str] = None
    day_bucket: Optional[str] = None
    limits: Optional[dict] = None
    used_after: Optional[dict] = None
    blocked_reason: Optional[str] = None
    retry_after_ms: Optional[int] = None


@dataclass
class UsageInfo:
    """Token usage information from provider response."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RequestContext:
    """Context for a single request (may have multiple attempts)."""
    request_uid: str
    consumer: str
    account_name: Optional[str]
    model: str
    reserved_tpm: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GoogleAIClient:
    """Google AI client with rate limiting and retry logic.
    
    Usage:
        client = GoogleAIClient(
            supabase_client=get_supabase_client(),
            secrets_provider=get_provider(),
        )
        
        response = await client.generate_content_async(
            model="gemma-3-27b",
            prompt="Hello, world!",
        )
    """
    
    # Default values
    DEFAULT_MAX_OUTPUT_TOKENS = 8192
    DEFAULT_TPM_RESERVE_EXTRA = 1000
    MAX_RETRIES = 3
    RETRY_DELAYS_MS = [250, 500, 1000]  # Backoff delays
    
    def __init__(
        self,
        supabase_client: Optional["SupabaseClient"] = None,
        secrets_provider: Optional[Any] = None,
        consumer: str = "bot",
        account_name: Optional[str] = None,
        dry_run: bool = False,
    ):
        """Initialize the client.
        
        Args:
            supabase_client: Supabase client for rate limiting RPC calls
            secrets_provider: Provider for API keys (if None, uses env directly)
            consumer: Consumer identifier (bot/kaggle/script)
            account_name: Account name for logging (from GOOGLE_API_LOCALNAME)
            dry_run: If True, skip actual API calls (for testing)
        """
        self.supabase = supabase_client
        self.secrets_provider = secrets_provider
        self.consumer = consumer
        self.account_name = account_name or os.getenv("GOOGLE_API_LOCALNAME")
        self.dry_run = dry_run
        
        # Lazy import google.generativeai
        self._genai = None
    
    @property
    def genai(self):
        """Lazy-load google.generativeai module."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._genai
    
    async def generate_content_async(
        self,
        model: str,
        prompt: str,
        generation_config: Optional[dict] = None,
        safety_settings: Optional[list] = None,
        max_output_tokens: Optional[int] = None,
        candidate_key_ids: Optional[list[str]] = None,
    ) -> tuple[str, UsageInfo]:
        """Generate content with rate limiting and retries.
        
        Args:
            model: Model name (e.g., "gemma-3-27b")
            prompt: Input prompt
            generation_config: Optional generation config
            safety_settings: Optional safety settings
            max_output_tokens: Max output tokens (for TPM reservation)
            candidate_key_ids: Optional list of API key IDs to try
            
        Returns:
            Tuple of (response_text, usage_info)
            
        Raises:
            RateLimitError: If rate limits exceeded (NO_WAIT)
            ProviderError: If provider error after max retries
        """
        # Prepare request context
        request_uid = str(uuid.uuid4())
        reserved_tpm = self._calculate_reserved_tpm(
            max_output_tokens or self.DEFAULT_MAX_OUTPUT_TOKENS
        )
        
        ctx = RequestContext(
            request_uid=request_uid,
            consumer=self.consumer,
            account_name=self.account_name,
            model=model,
            reserved_tpm=reserved_tpm,
        )
        
        # Retry loop (only for provider errors)
        last_error: Optional[Exception] = None
        
        for attempt_no in range(1, self.MAX_RETRIES + 1):
            try:
                return await self._attempt_generate(
                    ctx=ctx,
                    attempt_no=attempt_no,
                    prompt=prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    max_output_tokens=max_output_tokens,
                    candidate_key_ids=candidate_key_ids,
                )
            except RateLimitError:
                # NO_WAIT: don't retry on rate limits
                raise
            except ProviderError as e:
                last_error = e
                if not e.retryable or attempt_no >= self.MAX_RETRIES:
                    raise
                
                # Retry with backoff + jitter
                delay_ms = self.RETRY_DELAYS_MS[min(attempt_no - 1, len(self.RETRY_DELAYS_MS) - 1)]
                jitter_ms = random.randint(0, 100)
                await asyncio.sleep((delay_ms + jitter_ms) / 1000)
                
                self._log_event("google_ai.retry", ctx, attempt_no=attempt_no, error=str(e))
        
        # Should not reach here, but just in case
        raise last_error or ProviderError(error_type="unknown", error_message="Max retries exceeded")
    
    async def _attempt_generate(
        self,
        ctx: RequestContext,
        attempt_no: int,
        prompt: str,
        generation_config: Optional[dict],
        safety_settings: Optional[list],
        max_output_tokens: Optional[int],
        candidate_key_ids: Optional[list[str]],
    ) -> tuple[str, UsageInfo]:
        """Single attempt to generate content."""
        
        # 1. Reserve rate limit slot
        reserve_result = await self._reserve(ctx, attempt_no, candidate_key_ids)
        
        if not reserve_result.ok:
            raise RateLimitError(
                blocked_reason=reserve_result.blocked_reason or "unknown",
                retry_after_ms=reserve_result.retry_after_ms,
                model=ctx.model,
                api_key_id=reserve_result.api_key_id,
                minute_bucket=reserve_result.minute_bucket,
                day_bucket=reserve_result.day_bucket,
            )
        
        self._log_event("google_ai.reserve_ok", ctx, attempt_no=attempt_no, reserve=reserve_result)
        
        # 2. Get API key
        api_key = self._get_api_key(reserve_result.env_var_name)
        if not api_key:
            raise ReservationError(f"API key not found: {reserve_result.env_var_name}")
        
        # 3. Mark as sent (before actual call)
        await self._mark_sent(ctx.request_uid, attempt_no)
        
        # 4. Call provider
        start_time = time.monotonic()
        try:
            if self.dry_run:
                # Dry run mode for testing
                response_text = f"[DRY RUN] Response for: {prompt[:50]}..."
                usage = UsageInfo(input_tokens=100, output_tokens=50, total_tokens=150)
            else:
                response_text, usage = await self._call_provider(
                    api_key=api_key,
                    model=ctx.model,
                    prompt=prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    max_output_tokens=max_output_tokens,
                )
            
            duration_ms = int((time.monotonic() - start_time) * 1000)
            
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            
            # Classify error
            provider_error = self._classify_error(e)
            
            # Finalize with error
            await self._finalize(
                ctx=ctx,
                attempt_no=attempt_no,
                usage=None,
                duration_ms=duration_ms,
                error=provider_error,
            )
            
            self._log_event(
                "google_ai.call_error",
                ctx,
                attempt_no=attempt_no,
                duration_ms=duration_ms,
                error=provider_error,
            )
            
            raise provider_error
        
        # 5. Finalize (update usage, reconcile TPM)
        await self._finalize(
            ctx=ctx,
            attempt_no=attempt_no,
            usage=usage,
            duration_ms=duration_ms,
        )
        
        self._log_event(
            "google_ai.call_ok",
            ctx,
            attempt_no=attempt_no,
            duration_ms=duration_ms,
            usage=usage,
        )
        
        return response_text, usage
    
    async def _reserve(
        self,
        ctx: RequestContext,
        attempt_no: int,
        candidate_key_ids: Optional[list[str]],
    ) -> ReserveResult:
        """Reserve rate limit slot via Supabase RPC."""
        if not self.supabase:
            # No Supabase = no rate limiting (for local dev)
            logger.warning("No Supabase client, skipping rate limit reservation")
            return ReserveResult(
                ok=True,
                env_var_name="GOOGLE_API_KEY",
            )
        
        try:
            result = self.supabase.rpc(
                "google_ai_reserve",
                {
                    "p_request_uid": ctx.request_uid,
                    "p_attempt_no": attempt_no,
                    "p_consumer": ctx.consumer,
                    "p_account_name": ctx.account_name,
                    "p_model": ctx.model,
                    "p_reserved_tpm": ctx.reserved_tpm,
                    "p_candidate_key_ids": candidate_key_ids,
                }
            ).execute()
            
            data = result.data
            if isinstance(data, list) and data:
                data = data[0]
            
            return ReserveResult(
                ok=data.get("ok", False),
                api_key_id=data.get("api_key_id"),
                env_var_name=data.get("env_var_name"),
                key_alias=data.get("key_alias"),
                minute_bucket=data.get("minute_bucket"),
                day_bucket=data.get("day_bucket"),
                limits=data.get("limits"),
                used_after=data.get("used_after"),
                blocked_reason=data.get("blocked_reason"),
                retry_after_ms=data.get("retry_after_ms"),
            )
            
        except Exception as e:
            logger.error("Failed to call google_ai_reserve: %s", e)
            raise ReservationError(f"Reserve RPC failed: {e}")
    
    async def _mark_sent(self, request_uid: str, attempt_no: int) -> None:
        """Mark request as sent (before calling provider)."""
        if not self.supabase:
            return
        
        try:
            self.supabase.rpc(
                "google_ai_mark_sent",
                {
                    "p_request_uid": request_uid,
                    "p_attempt_no": attempt_no,
                }
            ).execute()
        except Exception as e:
            logger.warning("Failed to mark_sent: %s", e)
    
    async def _finalize(
        self,
        ctx: RequestContext,
        attempt_no: int,
        usage: Optional[UsageInfo],
        duration_ms: int,
        error: Optional[ProviderError] = None,
    ) -> None:
        """Finalize request (record usage, reconcile TPM)."""
        if not self.supabase:
            return
        
        try:
            self.supabase.rpc(
                "google_ai_finalize",
                {
                    "p_request_uid": ctx.request_uid,
                    "p_attempt_no": attempt_no,
                    "p_usage_input_tokens": usage.input_tokens if usage else None,
                    "p_usage_output_tokens": usage.output_tokens if usage else None,
                    "p_usage_total_tokens": usage.total_tokens if usage else None,
                    "p_duration_ms": duration_ms,
                    "p_provider_status": "succeeded" if not error else "failed",
                    "p_error_type": error.error_type if error else None,
                    "p_error_code": error.error_code if error else None,
                    "p_error_message": error.error_message if error else None,
                }
            ).execute()
        except Exception as e:
            logger.warning("Failed to finalize: %s", e)
    
    async def _call_provider(
        self,
        api_key: str,
        model: str,
        prompt: str,
        generation_config: Optional[dict],
        safety_settings: Optional[list],
        max_output_tokens: Optional[int],
    ) -> tuple[str, UsageInfo]:
        """Call Google AI provider."""
        # Configure API key
        self.genai.configure(api_key=api_key)
        
        # Build generation config
        config = generation_config or {}
        if max_output_tokens and "max_output_tokens" not in config:
            config["max_output_tokens"] = max_output_tokens
        
        # Create model (use full model path for gemma)
        model_name = model if model.startswith("models/") else f"models/{model}-it"
        gen_model = self.genai.GenerativeModel(model_name)
        
        # Generate content
        response = await gen_model.generate_content_async(
            prompt,
            generation_config=config,
            safety_settings=safety_settings,
        )
        
        # Extract usage
        usage = UsageInfo()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage.input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage.output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            usage.total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
        
        # Extract text
        response_text = response.text if hasattr(response, "text") else ""
        
        return response_text, usage
    
    def _get_api_key(self, env_var_name: Optional[str]) -> Optional[str]:
        """Get API key from environment or secrets provider."""
        name = env_var_name or "GOOGLE_API_KEY"
        
        if self.secrets_provider:
            return self.secrets_provider.get_secret(name)
        
        return os.getenv(name)
    
    def _calculate_reserved_tpm(self, max_output_tokens: int) -> int:
        """Calculate tokens to reserve for TPM check."""
        # Conservative: max_output + extra buffer
        return max_output_tokens + self.DEFAULT_TPM_RESERVE_EXTRA
    
    def _classify_error(self, error: Exception) -> ProviderError:
        """Classify exception into ProviderError."""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Check for retryable errors
        retryable = any(x in error_str.lower() for x in [
            "timeout",
            "connection",
            "temporary",
            "rate limit",
            "503",
            "502",
            "504",
        ])
        
        return ProviderError(
            error_type=error_type,
            error_message=error_str[:500],  # Limit message length
            retryable=retryable,
        )
    
    def _log_event(
        self,
        event: str,
        ctx: RequestContext,
        attempt_no: int = 1,
        **kwargs,
    ) -> None:
        """Log structured event (JSON lines format)."""
        log_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "request_uid": ctx.request_uid,
            "attempt_no": attempt_no,
            "consumer": ctx.consumer,
            "account_name": ctx.account_name,
            "model": ctx.model,
            "reserved_tpm": ctx.reserved_tpm,
        }
        
        # Add optional fields
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "__dict__"):
                    log_data[key] = value.__dict__
                else:
                    log_data[key] = value
        
        logger.info(json.dumps(log_data, ensure_ascii=False, default=str))
