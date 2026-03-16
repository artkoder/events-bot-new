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
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional, TYPE_CHECKING
from time import monotonic as _monotonic

from google_ai.exceptions import RateLimitError, ProviderError, ReservationError

if TYPE_CHECKING:
    from supabase import Client as SupabaseClient

logger = logging.getLogger(__name__)
IncidentNotifier = Callable[[str, dict[str, Any]], Any]

_DEFAULT_ENV_CANDIDATE_CACHE: dict[tuple[str, tuple[str, ...]], tuple[str, ...] | None] = {}


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
    requested_model: Optional[str] = None
    provider_model: Optional[str] = None
    provider_model_name: Optional[str] = None
    api_key_id: Optional[str] = None
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
    # Heuristic budget for prompt token estimation. We intentionally overestimate
    # to avoid passing Supabase reserve checks and then hitting provider 429
    # on input-token-per-minute quotas.
    _BYTES_PER_TOKEN_ESTIMATE = 4.0
    MAX_RETRIES = 3
    RETRY_DELAYS_MS = [250, 500, 1000]  # Backoff delays
    RESERVE_FALLBACK_ENV = "GOOGLE_AI_ALLOW_RESERVE_FALLBACK"
    LOCAL_FALLBACK_ENV = "GOOGLE_AI_LOCAL_LIMITER_FALLBACK"
    LOCAL_FALLBACK_ON_ERROR_ENV = "GOOGLE_AI_LOCAL_LIMITER_ON_RESERVE_ERROR"
    LOCAL_RPM_ENV = "GOOGLE_AI_LOCAL_RPM"
    LOCAL_TPM_ENV = "GOOGLE_AI_LOCAL_TPM"
    LOCAL_RPD_ENV = "GOOGLE_AI_LOCAL_RPD"
    RESERVE_RPC_RECHECK_ENV = "GOOGLE_AI_RESERVE_RPC_RECHECK_SECONDS"
    RESERVE_RPC_RETRY_ATTEMPTS_ENV = "GOOGLE_AI_RESERVE_RPC_RETRY_ATTEMPTS"
    RESERVE_RPC_RETRY_BASE_DELAY_MS_ENV = "GOOGLE_AI_RESERVE_RPC_RETRY_BASE_DELAY_MS"
    INCIDENT_NOTIFICATIONS_ENV = "GOOGLE_AI_INCIDENT_NOTIFICATIONS"
    INCIDENT_COOLDOWN_ENV = "GOOGLE_AI_INCIDENT_COOLDOWN_SECONDS"
    TEXT_PRIMARY_MODEL = "gemma-3-27b"
    TEXT_MIN_GEMMA_B = 12
    # When Supabase client is configured with a non-public schema, PostgREST can
    # return 404 for RPC that exists in public. In that case we can retry via
    # direct REST call with explicit schema headers.
    RESERVE_DIRECT_RETRY_ENV = "GOOGLE_AI_RESERVE_DIRECT_RETRY"
    RESERVE_DIRECT_SCHEMA_ENV = "GOOGLE_AI_RESERVE_DIRECT_SCHEMA"
    RESERVE_SCOPE_TO_DEFAULT_ENV_ENV = "GOOGLE_AI_RESERVE_SCOPE_TO_DEFAULT_ENV"

    # Process-local limiter (used when Supabase reserve RPC is missing/flaky).
    _local_limiter_lock = asyncio.Lock()
    _local_limiter_minute_bucket: int | None = None
    _local_limiter_used_rpm: int = 0
    _local_limiter_used_tpm: int = 0
    _local_limiter_day_bucket: str | None = None
    _local_limiter_used_rpd: int = 0

    @staticmethod
    def _normalize_rate_limit_model(model: str) -> str:
        """Normalize model id used in Supabase RPC quota tables.

        Provider may use interactive Gemma variants (`-it`), while quota tables
        in some projects store base model ids (`gemma-3-27b`).
        """
        raw = (model or "").strip()
        if raw.startswith("models/"):
            raw = raw.split("/", 1)[1].strip()
        if raw.startswith("gemma-") and raw.endswith("-it"):
            return raw[:-3]
        return raw

    @classmethod
    def _gemma_b_size(cls, model: str) -> Optional[int]:
        normalized = cls._normalize_rate_limit_model(model).strip().lower()
        match = re.match(r"^gemma-\d+(?:\.\d+)?-(\d+)b$", normalized)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @classmethod
    def _is_gemma_model(cls, model: str) -> bool:
        normalized = cls._normalize_rate_limit_model(model).strip().lower()
        return normalized.startswith("gemma-")

    @classmethod
    def _is_disallowed_text_model(cls, model: str) -> bool:
        size = cls._gemma_b_size(model)
        return size is not None and size < cls.TEXT_MIN_GEMMA_B

    @staticmethod
    def _resolve_provider_model(model: str) -> tuple[str, str]:
        """Resolve model id passed to provider and fully-qualified model_name.

        Returns:
            Tuple of (provider_model, provider_model_name):
            - provider_model: short provider id (e.g. "gemma-3-27b-it")
            - provider_model_name: fully-qualified API id (e.g. "models/gemma-3-27b-it")
        """
        raw = (model or "").strip()
        if raw.startswith("models/"):
            provider_model = raw.split("/", 1)[1].strip() or raw
            return provider_model, raw

        provider_model = raw
        if provider_model.startswith("gemma-") and not provider_model.endswith("-it"):
            provider_model = f"{provider_model}-it"
        return provider_model, f"models/{provider_model}"
    
    def __init__(
        self,
        supabase_client: Optional["SupabaseClient"] = None,
        secrets_provider: Optional[Any] = None,
        consumer: str = "bot",
        account_name: Optional[str] = None,
        default_env_var_name: Optional[str] = None,
        dry_run: bool = False,
        incident_notifier: Optional[IncidentNotifier] = None,
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
        self.default_env_var_name = (default_env_var_name or "GOOGLE_API_KEY").strip() or "GOOGLE_API_KEY"
        self.dry_run = dry_run
        self.allow_reserve_fallback = (
            os.getenv(self.RESERVE_FALLBACK_ENV, "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.allow_local_limiter_fallback = (
            os.getenv(self.LOCAL_FALLBACK_ENV, "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.allow_local_limiter_on_reserve_error = (
            os.getenv(self.LOCAL_FALLBACK_ON_ERROR_ENV, "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.incident_notifier = incident_notifier
        self.incident_notifications_enabled = (
            os.getenv(self.INCIDENT_NOTIFICATIONS_ENV, "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.incident_cooldown_seconds = self._read_int_env(
            self.INCIDENT_COOLDOWN_ENV,
            900,
        )
        self.reserve_rpc_recheck_seconds = self._read_int_env(
            self.RESERVE_RPC_RECHECK_ENV,
            60,
        )
        self.max_retries = self._read_int_env("GOOGLE_AI_MAX_RETRIES", self.MAX_RETRIES)
        self.retry_delays_ms = self._read_retry_delays()
        self.fallback_models = self._read_fallback_models()
        self._incident_last_sent: dict[str, float] = {}
        self.scope_reserve_to_default_env = (
            os.getenv(self.RESERVE_SCOPE_TO_DEFAULT_ENV_ENV, "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # Cache missing Supabase RPCs to avoid noisy per-request fallbacks when
        # the Supabase project hasn't been migrated yet (PGRST202).
        self._reserve_rpc_missing = False
        self._reserve_rpc_missing_since = 0.0
        self._mark_sent_rpc_missing = False
        self._finalize_rpc_missing = False
        self._legacy_finalize_rpc_missing = False
        self._missing_rpc_logged: set[str] = set()
        
        # Lazy import google.generativeai
        self._genai = None

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            return max(1, value)
        except Exception:
            return default

    def _read_retry_delays(self) -> list[int]:
        raw = (os.getenv("GOOGLE_AI_RETRY_DELAYS_MS") or "").strip()
        if not raw:
            return list(self.RETRY_DELAYS_MS)
        out: list[int] = []
        for part in raw.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.append(max(50, int(p)))
            except Exception:
                continue
        return out or list(self.RETRY_DELAYS_MS)

    @staticmethod
    def _default_env_aliases(name: str | None) -> list[str]:
        raw = (name or "").strip()
        if not raw:
            return []
        names = [raw]
        if raw == "GOOGLE_API_KEY2":
            names.append("GOOGLE_API_KEY_2")
        elif raw == "GOOGLE_API_KEY_2":
            names.append("GOOGLE_API_KEY2")
        return names

    def _resolve_default_env_candidate_key_ids(
        self,
        *,
        consumer: str,
    ) -> list[str] | None:
        if self.supabase is None or not self.scope_reserve_to_default_env:
            return None
        env_names = tuple(self._default_env_aliases(self.default_env_var_name))
        if not env_names:
            return None
        cache_key = (consumer, env_names)
        if cache_key in _DEFAULT_ENV_CANDIDATE_CACHE:
            cached = _DEFAULT_ENV_CANDIDATE_CACHE[cache_key]
            return list(cached) if cached else None
        try:
            result = (
                self.supabase.table("google_ai_api_keys")
                .select("id, env_var_name, priority")
                .eq("is_active", True)
                .in_("env_var_name", list(env_names))
                .order("priority")
                .order("id")
                .execute()
            )
            rows = list(result.data or [])
        except Exception as exc:
            logger.warning(
                "google_ai.default_env_candidates_failed consumer=%s env=%s err=%s",
                consumer,
                ",".join(env_names),
                exc,
            )
            _DEFAULT_ENV_CANDIDATE_CACHE[cache_key] = None
            return None
        ids = tuple(
            str(row.get("id"))
            for row in rows
            if row.get("id") and str(row.get("env_var_name") or "") in env_names
        )
        if not ids:
            logger.warning(
                "google_ai.default_env_candidates_missing consumer=%s env=%s",
                consumer,
                ",".join(env_names),
            )
            _DEFAULT_ENV_CANDIDATE_CACHE[cache_key] = None
            return None
        _DEFAULT_ENV_CANDIDATE_CACHE[cache_key] = ids
        return list(ids)

    async def _call_supabase_rpc_with_retries(
        self,
        fn_name: str,
        payload: dict[str, Any],
        *,
        log_label: str,
    ) -> Any:
        """Call a Supabase RPC with short retries on transient transport errors."""
        retry_attempts = self._read_int_env(self.RESERVE_RPC_RETRY_ATTEMPTS_ENV, 2)
        retry_attempts = max(1, min(retry_attempts, 6))
        retry_base_delay_ms = self._read_int_env(
            self.RESERVE_RPC_RETRY_BASE_DELAY_MS_ENV,
            350,
        )
        retry_base_delay_ms = max(50, min(retry_base_delay_ms, 5000))
        last_exc: Exception | None = None
        for rpc_attempt in range(1, retry_attempts + 1):
            try:
                return self.supabase.rpc(fn_name, payload).execute()
            except Exception as exc:
                last_exc = exc
                transient = self._is_transient_reserve_rpc_error(exc)
                if not transient or rpc_attempt >= retry_attempts:
                    raise
                delay_ms = int(retry_base_delay_ms * (2 ** (rpc_attempt - 1)))
                delay_ms += random.randint(0, max(30, retry_base_delay_ms // 2))
                delay_ms = min(delay_ms, 7000)
                logger.warning(
                    "google_ai.%s_rpc_transient_retry fn=%s attempt=%s/%s delay_ms=%s err=%s",
                    log_label,
                    fn_name,
                    rpc_attempt,
                    retry_attempts,
                    delay_ms,
                    exc,
                )
                await asyncio.sleep(delay_ms / 1000.0)
        raise last_exc or RuntimeError(f"RPC failed: {fn_name}")

    @staticmethod
    def _is_transient_reserve_rpc_error(exc: Exception) -> bool:
        cls_name = exc.__class__.__name__.lower()
        msg = str(exc or "").lower()
        if "timeout" in cls_name or "ssl" in cls_name or "connection" in cls_name:
            return True
        markers = (
            "timed out",
            "timeout",
            "handshake",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "eof",
            "unexpected eof",
            "temporarily unavailable",
            "tls",
            "ssl",
        )
        return any(token in msg for token in markers)

    def _read_fallback_models(self) -> list[str]:
        raw = (os.getenv("GOOGLE_AI_FALLBACK_MODELS") or "").strip()
        if not raw:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for part in raw.split(","):
            model = part.strip()
            if not model:
                continue
            key = model.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(model)
        return out

    def _build_model_chain(self, requested_model: str) -> list[str]:
        requested = (requested_model or "").strip()
        chain: list[str] = []
        seen: set[str] = set()
        for model in [requested, *self.fallback_models]:
            m = (model or "").strip()
            if not m:
                continue
            if self._is_disallowed_text_model(m):
                logger.warning(
                    "google_ai.model_chain_skip model=%s reason=below_%sb_text_policy",
                    m,
                    self.TEXT_MIN_GEMMA_B,
                )
                continue
            key = self._normalize_rate_limit_model(m).lower()
            if key in seen:
                continue
            seen.add(key)
            chain.append(m)
        has_gemma = self._is_gemma_model(requested) or any(
            self._is_gemma_model(m) for m in self.fallback_models
        )
        if has_gemma:
            primary = self.TEXT_PRIMARY_MODEL
            primary_key = self._normalize_rate_limit_model(primary).lower()
            head = next(
                (m for m in chain if self._normalize_rate_limit_model(m).lower() == primary_key),
                primary,
            )
            tail = [
                m
                for m in chain
                if self._normalize_rate_limit_model(m).lower() != primary_key
            ]
            chain = [head, *tail]
        return chain or [self.TEXT_PRIMARY_MODEL if has_gemma else requested_model]

    async def _notify_incident(
        self,
        kind: str,
        *,
        ctx: RequestContext | None = None,
        severity: str = "critical",
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.incident_notifier or not self.incident_notifications_enabled:
            return

        model = (ctx.requested_model if ctx else None) or (ctx.model if ctx else None) or ""
        base = f"{kind}:{self.consumer}:{model}"
        dedupe_key = base
        if details:
            code = details.get("error_code") or details.get("blocked_reason") or details.get("error_type")
            if code:
                dedupe_key = f"{base}:{code}"

        now = _monotonic()
        last = self._incident_last_sent.get(dedupe_key)
        if last is not None and (now - last) < float(self.incident_cooldown_seconds):
            return
        self._incident_last_sent[dedupe_key] = now

        payload: dict[str, Any] = {
            "kind": kind,
            "severity": severity,
            "consumer": self.consumer,
            "account_name": self.account_name,
            "message": message,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if ctx:
            payload.update(
                {
                    "request_uid": ctx.request_uid,
                    "model": ctx.model,
                    "requested_model": ctx.requested_model or ctx.model,
                    "provider_model": ctx.provider_model,
                    "provider_model_name": ctx.provider_model_name,
                    "invoked_model": ctx.provider_model_name or ctx.requested_model or ctx.model,
                }
            )
        if details:
            payload.update(details)
        try:
            maybe = self.incident_notifier(kind, payload)
            if asyncio.iscoroutine(maybe):
                await maybe
        except Exception as exc:
            logger.warning("google_ai incident notifier failed: %s", exc)
    
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
        request_uid = str(uuid.uuid4())
        reserved_tpm = self._calculate_reserved_tpm(
            prompt=prompt,
            max_output_tokens=max_output_tokens or self.DEFAULT_MAX_OUTPUT_TOKENS,
        )
        requested_model = (model or "").strip()
        model_chain = self._build_model_chain(requested_model)
        attempt_cursor = 0

        last_error: Optional[Exception] = None
        for model_index, model_name in enumerate(model_chain):
            limit_model = self._normalize_rate_limit_model(model_name)
            provider_model, provider_model_name = self._resolve_provider_model(
                model_name or limit_model
            )
            ctx = RequestContext(
                request_uid=request_uid,
                consumer=self.consumer,
                account_name=self.account_name,
                model=limit_model,
                requested_model=model_name or limit_model,
                provider_model=provider_model,
                provider_model_name=provider_model_name,
                reserved_tpm=reserved_tpm,
            )

            for local_attempt_no in range(1, self.max_retries + 1):
                attempt_cursor += 1
                attempt_no = attempt_cursor
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
                except RateLimitError as e:
                    if (e.blocked_reason or "").strip().lower() in {"no_keys", "model_not_found"}:
                        await self._notify_incident(
                            "rate_limit_blocked",
                            ctx=ctx,
                            severity="critical",
                            message=str(e),
                            details={
                                "blocked_reason": e.blocked_reason,
                                "retry_after_ms": e.retry_after_ms,
                            },
                        )
                    raise
                except ReservationError as e:
                    last_error = e
                    await self._notify_incident(
                        "reservation_error",
                        ctx=ctx,
                        severity="critical",
                        message=str(e),
                    )
                    raise
                except ProviderError as e:
                    last_error = e
                    # Keep provider-side 429 fail-fast here. Higher-level flows
                    # already decide whether to wait, defer, or retry, and an
                    # extra retry loop in the client multiplies end-to-end delay.
                    if int(getattr(e, "status_code", 0) or 0) == 429:
                        raise
                    can_retry = bool(e.retryable) and local_attempt_no < self.max_retries
                    if can_retry:
                        delay_ms = self.retry_delays_ms[
                            min(local_attempt_no - 1, len(self.retry_delays_ms) - 1)
                        ]
                        if e.retry_after_ms:
                            delay_ms = max(int(delay_ms), int(e.retry_after_ms))
                        jitter_ms = random.randint(0, 100)
                        await asyncio.sleep((delay_ms + jitter_ms) / 1000)
                        self._log_event("google_ai.retry", ctx, attempt_no=attempt_no, error=str(e))
                        continue

                    has_fallback = model_index < (len(model_chain) - 1)
                    await self._notify_incident(
                        "provider_error_fallback" if has_fallback else "provider_error",
                        ctx=ctx,
                        severity="warning" if has_fallback else "critical",
                        message=str(e),
                        details={
                            "error_type": e.error_type,
                            "error_code": e.error_code,
                            "status_code": e.status_code,
                            "retryable": int(bool(e.retryable)),
                            "attempt_no": attempt_no,
                            "max_retries": self.max_retries,
                            "next_model": model_chain[model_index + 1] if has_fallback else None,
                        },
                    )
                    if has_fallback:
                        self._log_event(
                            "google_ai.model_fallback",
                            ctx,
                            attempt_no=attempt_no,
                            next_model=model_chain[model_index + 1],
                            error=e,
                        )
                        break
                    raise

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

        ctx.api_key_id = reserve_result.api_key_id
        self._log_event("google_ai.reserve_ok", ctx, attempt_no=attempt_no, reserve=reserve_result)
        
        # 2. Get API key
        api_key = self._get_api_key(reserve_result.env_var_name)
        if not api_key:
            await self._notify_incident(
                "missing_api_key",
                ctx=ctx,
                severity="critical",
                message=f"API key not found: {reserve_result.env_var_name}",
                details={"env_var_name": reserve_result.env_var_name},
            )
            raise ReservationError(f"API key not found: {reserve_result.env_var_name}")
        
        # 3. Mark as sent (before actual call)
        await self._mark_sent(ctx, attempt_no)
        
        # 4. Call provider
        start_time = _monotonic()
        try:
            if self.dry_run:
                # Dry run mode for testing
                response_text = f"[DRY RUN] Response for: {prompt[:50]}..."
                usage = UsageInfo(input_tokens=100, output_tokens=50, total_tokens=150)
            else:
                response_text, usage = await self._call_provider(
                    api_key=api_key,
                    model=ctx.requested_model or ctx.model,
                    prompt=prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    max_output_tokens=max_output_tokens,
                )
            
            duration_ms = int((_monotonic() - start_time) * 1000)
            
        except Exception as e:
            duration_ms = int((_monotonic() - start_time) * 1000)
            
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
                env_var_name=self.default_env_var_name,
            )
        was_cached_missing = self._reserve_rpc_missing
        if was_cached_missing:
            now = _monotonic()
            age = (
                now - self._reserve_rpc_missing_since
                if self._reserve_rpc_missing_since > 0
                else 0.0
            )
            if age < self.reserve_rpc_recheck_seconds:
                if self.allow_local_limiter_fallback:
                    return await self._local_reserve(
                        ctx,
                        attempt_no=attempt_no,
                        key_alias="local-fallback-no-rpc-cached",
                        blocked_reason="reserve_rpc_missing",
                    )
                return ReserveResult(
                    ok=True,
                    env_var_name=self.default_env_var_name,
                    key_alias="reserve-fallback-no-rpc-cached",
                    blocked_reason="reserve_rpc_missing",
                )
            # Cooldown elapsed: retry RPC once instead of staying in cached fallback forever.
            self._reserve_rpc_missing = False
            logger.warning(
                "google_ai.reserve_rpc_recheck consumer=%s model=%s age_s=%.1f",
                ctx.consumer,
                ctx.model,
                age,
            )

        if self._reserve_rpc_missing:
            if self.allow_local_limiter_fallback:
                return await self._local_reserve(
                    ctx,
                    attempt_no=attempt_no,
                    key_alias="local-fallback-no-rpc-cached",
                    blocked_reason="reserve_rpc_missing",
                )
            return ReserveResult(
                ok=True,
                env_var_name=self.default_env_var_name,
                key_alias="reserve-fallback-no-rpc-cached",
                blocked_reason="reserve_rpc_missing",
            )
        
        scoped_candidate_key_ids = candidate_key_ids
        if scoped_candidate_key_ids is None:
            scoped_candidate_key_ids = self._resolve_default_env_candidate_key_ids(
                consumer=ctx.consumer,
            )

        payload = {
            "p_request_uid": ctx.request_uid,
            "p_attempt_no": attempt_no,
            "p_consumer": ctx.consumer,
            "p_account_name": ctx.account_name,
            "p_model": ctx.model,
            "p_reserved_tpm": ctx.reserved_tpm,
            "p_candidate_key_ids": scoped_candidate_key_ids,
        }

        try:
            retry_attempts = self._read_int_env(self.RESERVE_RPC_RETRY_ATTEMPTS_ENV, 2)
            retry_attempts = max(1, min(retry_attempts, 6))
            retry_base_delay_ms = self._read_int_env(self.RESERVE_RPC_RETRY_BASE_DELAY_MS_ENV, 350)
            retry_base_delay_ms = max(50, min(retry_base_delay_ms, 5000))
            rpc_error: Exception | None = None
            result = None
            for rpc_attempt in range(1, retry_attempts + 1):
                try:
                    result = self.supabase.rpc("google_ai_reserve", payload).execute()
                    rpc_error = None
                    break
                except Exception as exc:
                    rpc_error = exc
                    transient = self._is_transient_reserve_rpc_error(exc)
                    if not transient or rpc_attempt >= retry_attempts:
                        raise
                    delay_ms = int(retry_base_delay_ms * (2 ** (rpc_attempt - 1)))
                    delay_ms += random.randint(0, max(30, retry_base_delay_ms // 2))
                    delay_ms = min(delay_ms, 7000)
                    logger.warning(
                        "google_ai.reserve_rpc_transient_retry consumer=%s model=%s attempt=%s/%s delay_ms=%s err=%s",
                        ctx.consumer,
                        ctx.model,
                        rpc_attempt,
                        retry_attempts,
                        delay_ms,
                        str(exc)[:260],
                    )
                    await asyncio.sleep(delay_ms / 1000.0)
            if result is None:
                if rpc_error:
                    raise rpc_error
                raise RuntimeError("google_ai_reserve returned no result")
            
            data = result.data
            if isinstance(data, list) and data:
                data = data[0]

            if was_cached_missing:
                logger.info(
                    "google_ai.reserve_rpc_recovered consumer=%s model=%s",
                    ctx.consumer,
                    ctx.model,
                )
                self._reserve_rpc_missing_since = 0.0
                self._missing_rpc_logged.discard("google_ai_reserve")
            
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
            if (
                self.allow_reserve_fallback
                and self._is_missing_reserve_rpc_error(e)
                and (not self.dry_run)
                and (os.getenv(self.RESERVE_DIRECT_RETRY_ENV, "1").strip().lower() in {"1", "true", "yes", "on"})
            ):
                # Retry via direct REST call with explicit schema headers.
                direct = await self._reserve_via_direct_rest(ctx, attempt_no=attempt_no, payload=payload)
                if direct is not None:
                    if was_cached_missing:
                        self._reserve_rpc_missing_since = 0.0
                        self._missing_rpc_logged.discard("google_ai_reserve")
                    return direct

            if self.allow_reserve_fallback and self._is_missing_reserve_rpc_error(e):
                msg = str(e)
                self._reserve_rpc_missing = True
                self._reserve_rpc_missing_since = _monotonic()
                if "google_ai_reserve" not in self._missing_rpc_logged:
                    self._missing_rpc_logged.add("google_ai_reserve")
                    logger.error(
                        "Supabase RPC google_ai_reserve is missing in this Supabase project "
                        "(PGRST202). Rate limiting via Supabase is disabled; using direct env key "
                        "%s instead. Set %s=0 to fail hard. error=%s",
                        self.default_env_var_name,
                        self.RESERVE_FALLBACK_ENV,
                        msg,
                    )
                self._log_event(
                    "google_ai.reserve_fallback_no_rpc",
                    ctx,
                    attempt_no=attempt_no,
                    error=msg[:500],
                )
                await self._notify_incident(
                    "reserve_rpc_missing",
                    ctx=ctx,
                    severity="warning",
                    message=(
                        "Supabase RPC google_ai_reserve missing; "
                        "switched to process-local limiter fallback (direct API key)."
                    ),
                    details={"error": msg[:500]},
                )
                if self.allow_local_limiter_fallback:
                    return await self._local_reserve(
                        ctx,
                        attempt_no=attempt_no,
                        key_alias="local-fallback",
                        blocked_reason="reserve_rpc_missing",
                    )
                return ReserveResult(
                    ok=True,
                    env_var_name=self.default_env_var_name,
                    key_alias="reserve-fallback",
                    blocked_reason="reserve_rpc_missing",
                )
            if (
                self.allow_reserve_fallback
                and self.allow_local_limiter_on_reserve_error
                and self.allow_local_limiter_fallback
            ):
                msg = str(e)
                logger.warning("Reserve RPC failed; using local limiter fallback. error=%s", msg)
                await self._notify_incident(
                    "reserve_rpc_error_fallback",
                    ctx=ctx,
                    severity="warning",
                    message=f"Reserve RPC failed; using local limiter fallback: {e}",
                    details={"error": msg[:500]},
                )
                return await self._local_reserve(
                    ctx,
                    attempt_no=attempt_no,
                    key_alias="local-fallback-reserve-error",
                    blocked_reason="reserve_rpc_error",
                )
            logger.error("Failed to call google_ai_reserve: %s", e)
            await self._notify_incident(
                "reserve_rpc_error",
                ctx=ctx,
                severity="critical",
                message=f"Reserve RPC failed: {e}",
                details={"error": str(e)[:500]},
            )
            raise ReservationError(f"Reserve RPC failed: {e}")

    async def _reserve_via_direct_rest(
        self,
        ctx: RequestContext,
        *,
        attempt_no: int,
        payload: dict[str, Any],
    ) -> ReserveResult | None:
        """Call google_ai_reserve via REST endpoint, forcing schema headers.

        This is a fallback for environments where Supabase client is configured with
        a different schema (e.g. 'private') and PostgREST returns 404 for an RPC
        that exists in 'public'.
        """
        base_url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
        key = (os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()
        if not base_url or not key:
            return None
        schema = (os.getenv(self.RESERVE_DIRECT_SCHEMA_ENV) or "public").strip() or "public"
        endpoint = f"{base_url}/rest/v1/rpc/google_ai_reserve"
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }

        def _do() -> tuple[int, str]:
            import requests

            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            return int(resp.status_code), resp.text or ""

        try:
            status, body = await asyncio.to_thread(_do)
        except Exception as exc:
            self._log_event(
                "google_ai.reserve_direct_error",
                ctx,
                attempt_no=attempt_no,
                error=str(exc)[:300],
            )
            return None

        if status == 404:
            # Still missing in this schema/project.
            return None
        if status >= 400:
            self._log_event(
                "google_ai.reserve_direct_http_error",
                ctx,
                attempt_no=attempt_no,
                status=status,
                body_head=(body or "").replace("\n", " ")[:240],
            )
            return None

        try:
            data = json.loads(body) if body else {}
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                return None
        except Exception:
            return None

        self._log_event(
            "google_ai.reserve_direct_ok",
            ctx,
            attempt_no=attempt_no,
            status=status,
            schema=schema,
        )
        return ReserveResult(
            ok=bool(data.get("ok", False)),
            api_key_id=data.get("api_key_id"),
            env_var_name=data.get("env_var_name"),
            key_alias=data.get("key_alias") or f"direct:{schema}",
            minute_bucket=data.get("minute_bucket"),
            day_bucket=data.get("day_bucket"),
            limits=data.get("limits"),
            used_after=data.get("used_after"),
            blocked_reason=data.get("blocked_reason"),
            retry_after_ms=data.get("retry_after_ms"),
        )

    def _read_local_limit(self, name: str, default: int) -> int:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(float(raw)))
        except Exception:
            return default

    async def _local_reserve(
        self,
        ctx: RequestContext,
        *,
        attempt_no: int,
        key_alias: str,
        blocked_reason: str,
    ) -> ReserveResult:
        """Process-local limiter used when Supabase reserve RPC is missing/flaky."""
        rpm_limit = self._read_local_limit(self.LOCAL_RPM_ENV, 20)
        tpm_limit = self._read_local_limit(self.LOCAL_TPM_ENV, 12000)
        rpd_limit = self._read_local_limit(self.LOCAL_RPD_ENV, 5000)

        now = time.time()
        minute_bucket = int(now // 60) * 60
        day_bucket = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
        required_tpm = max(1, int(ctx.reserved_tpm))

        async with self._local_limiter_lock:
            if self._local_limiter_minute_bucket != minute_bucket:
                self._local_limiter_minute_bucket = minute_bucket
                self._local_limiter_used_rpm = 0
                self._local_limiter_used_tpm = 0
            if self._local_limiter_day_bucket != day_bucket:
                self._local_limiter_day_bucket = day_bucket
                self._local_limiter_used_rpd = 0

            limits = {"rpd": rpd_limit, "rpm": rpm_limit, "tpm": tpm_limit}
            used_before = {
                "rpd": self._local_limiter_used_rpd,
                "rpm": self._local_limiter_used_rpm,
                "tpm": self._local_limiter_used_tpm,
            }

            if self._local_limiter_used_rpd + 1 > rpd_limit:
                midnight = (
                    datetime.fromtimestamp(now, timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(days=1)
                )
                retry_after_ms = max(1000, int((midnight.timestamp() - now) * 1000))
                return ReserveResult(
                    ok=False,
                    env_var_name=self.default_env_var_name,
                    key_alias=key_alias,
                    minute_bucket=datetime.fromtimestamp(minute_bucket, timezone.utc).isoformat(),
                    day_bucket=day_bucket,
                    limits=limits,
                    used_after=used_before,
                    blocked_reason="rpd",
                    retry_after_ms=retry_after_ms,
                )
            if self._local_limiter_used_rpm + 1 > rpm_limit:
                retry_after_ms = max(250, int((minute_bucket + 60 - now) * 1000))
                return ReserveResult(
                    ok=False,
                    env_var_name=self.default_env_var_name,
                    key_alias=key_alias,
                    minute_bucket=datetime.fromtimestamp(minute_bucket, timezone.utc).isoformat(),
                    day_bucket=day_bucket,
                    limits=limits,
                    used_after=used_before,
                    blocked_reason="rpm",
                    retry_after_ms=retry_after_ms,
                )
            if self._local_limiter_used_tpm + required_tpm > tpm_limit:
                retry_after_ms = max(250, int((minute_bucket + 60 - now) * 1000))
                return ReserveResult(
                    ok=False,
                    env_var_name=self.default_env_var_name,
                    key_alias=key_alias,
                    minute_bucket=datetime.fromtimestamp(minute_bucket, timezone.utc).isoformat(),
                    day_bucket=day_bucket,
                    limits=limits,
                    used_after=used_before,
                    blocked_reason="tpm",
                    retry_after_ms=retry_after_ms,
                )

            self._local_limiter_used_rpd += 1
            self._local_limiter_used_rpm += 1
            self._local_limiter_used_tpm += required_tpm
            used_after = {
                "rpd": self._local_limiter_used_rpd,
                "rpm": self._local_limiter_used_rpm,
                "tpm": self._local_limiter_used_tpm,
            }

            reserve = ReserveResult(
                ok=True,
                env_var_name=self.default_env_var_name,
                key_alias=key_alias,
                minute_bucket=datetime.fromtimestamp(minute_bucket, timezone.utc).isoformat(),
                day_bucket=day_bucket,
                limits=limits,
                used_after=used_after,
                blocked_reason=blocked_reason,
                retry_after_ms=None,
            )

        self._log_event("google_ai.reserve_local_fallback_ok", ctx, attempt_no=attempt_no, reserve=reserve)
        return reserve

    @staticmethod
    def _is_missing_reserve_rpc_error(error: Exception) -> bool:
        return GoogleAIClient._is_missing_rpc_error(error, "google_ai_reserve")

    @staticmethod
    def _is_missing_rpc_error(error: Exception, rpc_name: str) -> bool:
        message = str(error).lower()
        rpc_name_l = rpc_name.lower()
        if rpc_name_l not in message:
            return False
        markers = (
            "pgrst202",
            "route post:/rpc/",
            "not found",
            "schema cache",
        )
        return any(marker in message for marker in markers)
    
    async def _mark_sent(self, ctx: RequestContext, attempt_no: int) -> None:
        """Mark request as sent (before calling provider)."""
        if not self.supabase:
            return
        if self._mark_sent_rpc_missing:
            return
        request_uid = ctx.request_uid

        try:
            await self._call_supabase_rpc_with_retries(
                "google_ai_mark_sent",
                {
                    "p_request_uid": request_uid,
                    "p_attempt_no": attempt_no,
                },
                log_label="mark_sent",
            )
        except Exception as e:
            message = str(e)
            if self._is_missing_rpc_error(e, "google_ai_mark_sent"):
                self._mark_sent_rpc_missing = True
                if "google_ai_mark_sent" not in self._missing_rpc_logged:
                    self._missing_rpc_logged.add("google_ai_mark_sent")
                    logger.warning(
                        "Supabase RPC google_ai_mark_sent is missing (PGRST202). "
                        "Will skip it for the rest of the process. error=%s",
                        message,
                    )
                return
            logger.warning("Failed to mark_sent: %s", e)
            await self._notify_incident(
                "mark_sent_rpc_error",
                ctx=ctx,
                severity="warning",
                message=message,
                details={"attempt_no": attempt_no, "rpc": "google_ai_mark_sent"},
            )
    
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

        legacy_payload = {
            "p_request_uid": ctx.request_uid,
            "p_api_key_id": ctx.api_key_id,
            "p_model": ctx.model,
            "p_actual_input_tokens": usage.input_tokens if usage else None,
            "p_actual_output_tokens": usage.output_tokens if usage else None,
            "p_status": "success" if not error else "failed",
        }

        if self._finalize_rpc_missing:
            await self._finalize_legacy(legacy_payload)
            return

        payload = {
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

        try:
            await self._call_supabase_rpc_with_retries(
                "google_ai_finalize",
                payload,
                log_label="finalize",
            )
            return
        except Exception as e:
            if not self._is_missing_rpc_error(e, "google_ai_finalize"):
                logger.warning("Failed to finalize: %s", e)
                await self._notify_incident(
                    "finalize_rpc_error",
                    ctx=ctx,
                    severity="warning",
                    message=str(e),
                    details={"attempt_no": attempt_no, "rpc": "google_ai_finalize"},
                )
                return
            logger.info("google_ai_finalize missing, falling back to finalize_google_ai_usage")
            # Don't try google_ai_finalize again in this process.
            self._finalize_rpc_missing = True

        await self._finalize_legacy(legacy_payload)

    async def _finalize_legacy(self, legacy_payload: dict[str, Any]) -> None:
        """Legacy finalize fallback for Supabase projects without google_ai_finalize."""
        if self._legacy_finalize_rpc_missing:
            return

        try:
            self.supabase.rpc("finalize_google_ai_usage", legacy_payload).execute()
        except Exception as legacy_error:
            if self._is_missing_rpc_error(legacy_error, "finalize_google_ai_usage"):
                self._legacy_finalize_rpc_missing = True
                if "finalize_google_ai_usage" not in self._missing_rpc_logged:
                    self._missing_rpc_logged.add("finalize_google_ai_usage")
                    logger.warning(
                        "Supabase RPC finalize_google_ai_usage is missing (PGRST202). "
                        "Finalize is disabled for the rest of the process. error=%s",
                        legacy_error,
                    )
                return
            logger.warning("Failed to finalize_google_ai_usage: %s", legacy_error)
    
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
        config = dict(generation_config or {})
        if max_output_tokens and "max_output_tokens" not in config:
            config["max_output_tokens"] = max_output_tokens
        
        # Create model:
        # - google.generativeai expects model_name like "models/gemma-3-27b-it"
        # - For Gemma, "-it" is the tested interactive-tuned variant in this project.
        # - For Gemini, use the model name as-is (no "-it" suffix).
        _provider_model, model_name = self._resolve_provider_model(model)

        # Gemma models frequently reject JSON-mode knobs like `response_mime_type`.
        # Callers still request JSON in the prompt and then parse it from text.
        if self._is_gemma_model(model_name) or self._is_gemma_model(model):
            stripped = []
            for key in ("response_mime_type", "response_schema", "response_schema_name"):
                if key in config:
                    stripped.append(key)
                    config.pop(key, None)
            if stripped:
                logger.info(
                    "google_ai: stripped_generation_config model=%s provider_model=%s stripped=%s",
                    model,
                    model_name,
                    ",".join(stripped),
                )
        gen_model = self.genai.GenerativeModel(model_name)
        
        # Generate content
        response = await gen_model.generate_content_async(
            prompt,
            generation_config=config,
            safety_settings=safety_settings,
        )
        
        def _get_usage(resp: Any) -> UsageInfo:
            usage = UsageInfo()
            meta = getattr(resp, "usage_metadata", None)
            if not meta:
                return usage
            try:
                if isinstance(meta, dict):
                    usage.input_tokens = int(meta.get("prompt_token_count") or 0)
                    usage.output_tokens = int(meta.get("candidates_token_count") or 0)
                    usage.total_tokens = int(meta.get("total_token_count") or 0)
                else:
                    usage.input_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
                    usage.output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
                    usage.total_tokens = int(getattr(meta, "total_token_count", 0) or 0)
            except Exception:
                # Best-effort only; token accounting must not break requests.
                pass
            return usage

        def _extract_text(resp: Any) -> str:
            # Old `google.generativeai`: response.text
            try:
                text = getattr(resp, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception:
                pass

            # Newer responses often store content in candidates[].content.parts[].text
            parts: list[str] = []
            cands = getattr(resp, "candidates", None)
            if cands:
                for cand in list(cands):
                    content = getattr(cand, "content", None)
                    if content is None and isinstance(cand, dict):
                        content = cand.get("content")
                    if content is None:
                        continue
                    cand_parts = getattr(content, "parts", None)
                    if cand_parts is None and isinstance(content, dict):
                        cand_parts = content.get("parts")
                    if cand_parts:
                        for part in list(cand_parts):
                            t = getattr(part, "text", None)
                            if t is None and isinstance(part, dict):
                                t = part.get("text")
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
                    else:
                        t = getattr(content, "text", None)
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
            if parts:
                return "\n".join(parts).strip()

            # Last resort: stringify.
            try:
                return str(resp).strip()
            except Exception:
                return ""

        usage = _get_usage(response)
        response_text = _extract_text(response)
        if not response_text:
            raise ProviderError(
                error_type="empty_response",
                error_message=(
                    "Provider returned empty text "
                    f"(requested_model={model}, provider_model_name={model_name})"
                ),
                retryable=True,
            )
        return response_text, usage
    
    def _get_api_key(self, env_var_name: Optional[str]) -> Optional[str]:
        """Get API key from environment or secrets provider."""
        name = env_var_name or self.default_env_var_name or "GOOGLE_API_KEY"
        
        if self.secrets_provider:
            return self.secrets_provider.get_secret(name)
        
        return os.getenv(name)
    
    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Best-effort token estimate for prompts.

        We can't depend on provider-side countTokens here (it would also require
        an API call). We use conservative byte/char heuristics because long
        Cyrillic/OCR prompts can tokenize much denser than a simple bytes/4
        estimate and otherwise slip past reserve() only to hit provider 429.
        """
        if not prompt:
            return 1
        try:
            size = len(prompt.encode("utf-8", errors="ignore"))
        except Exception:
            size = len(prompt)
        chars = len(prompt)
        non_ascii = sum(1 for ch in prompt if ord(ch) > 127)
        non_ascii_ratio = (non_ascii / chars) if chars > 0 else 0.0

        bytes_est = size / float(self._BYTES_PER_TOKEN_ESTIMATE)
        if non_ascii_ratio >= 0.30:
            chars_est = chars * 0.72
            bytes_est = size / 2.6
        else:
            chars_est = chars * 0.30

        est = int(max(bytes_est, chars_est))
        # Add overhead for JSON, escaping, and tokenization variance.
        est = int(est * 1.15) + 50
        return max(1, est)

    def _calculate_reserved_tpm(self, *, prompt: str, max_output_tokens: int) -> int:
        """Calculate tokens to reserve for TPM check.

        Supabase reservation must cover BOTH prompt (input) and output tokens.
        Under-reserving here can lead to provider 429 (ResourceExhausted) even
        when Supabase reserve() returned ok=true.
        """
        input_est = self._estimate_prompt_tokens(prompt)
        output_budget = max(1, int(max_output_tokens))
        return input_est + output_budget + int(self.DEFAULT_TPM_RESERVE_EXTRA)
    
    def _classify_error(self, error: Exception) -> ProviderError:
        """Classify exception into ProviderError."""
        if isinstance(error, ProviderError):
            return error
        error_str = str(error)
        error_lower = error_str.lower()
        error_type = type(error).__name__

        retry_after_ms: Optional[int] = None
        # Gemini/Gemma errors often include "Please retry in <seconds>s."
        m_retry = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", error_lower)
        if m_retry:
            try:
                retry_after_ms = int(float(m_retry.group(1)) * 1000)
            except Exception:
                retry_after_ms = None
        
        # Check for retryable errors
        retryable = any(x in error_lower for x in [
            "timeout",
            "connection",
            "temporary",
            "rate limit",
            "503",
            "502",
            "504",
            "resource_exhausted",
            "unavailable",
            "deadline exceeded",
            "internal",
            "try again",
            "econnreset",
            "connection reset",
            "socket",
        ])
        status_code: Optional[int] = None
        for code in ("429", "500", "502", "503", "504"):
            if code in error_lower:
                try:
                    status_code = int(code)
                except Exception:
                    status_code = None
                break
        # Some exceptions don't include "resource_exhausted" in the string, but
        # the type name is still informative.
        if not retryable and error_type.lower() in {"resourceexhausted", "unavailable"}:
            retryable = True
        if not retryable and status_code == 429:
            retryable = True
        
        return ProviderError(
            error_type=error_type,
            error_message=error_str[:500],  # Limit message length
            retryable=retryable,
            status_code=status_code,
            retry_after_ms=retry_after_ms,
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
            "requested_model": ctx.requested_model or ctx.model,
            "provider_model": ctx.provider_model,
            "provider_model_name": ctx.provider_model_name,
            "invoked_model": ctx.provider_model_name or ctx.requested_model or ctx.model,
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
