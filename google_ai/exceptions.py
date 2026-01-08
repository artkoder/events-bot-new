"""Custom exceptions for Google AI SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RateLimitError(Exception):
    """Raised when rate limits are exceeded.
    
    NO_WAIT policy: this error is raised immediately without waiting.
    """
    blocked_reason: str  # 'rpm' | 'tpm' | 'rpd'
    retry_after_ms: Optional[int] = None
    model: Optional[str] = None
    api_key_id: Optional[str] = None
    minute_bucket: Optional[str] = None
    day_bucket: Optional[str] = None
    
    def __str__(self) -> str:
        msg = f"Rate limit exceeded: {self.blocked_reason}"
        if self.retry_after_ms:
            msg += f" (retry after {self.retry_after_ms}ms)"
        return msg


@dataclass
class ProviderError(Exception):
    """Raised when Google AI provider returns an error.
    
    Retryable errors will be retried up to 3 times.
    """
    error_type: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    status_code: Optional[int] = None
    
    def __str__(self) -> str:
        msg = f"Provider error: {self.error_type}"
        if self.error_code:
            msg += f" ({self.error_code})"
        if self.error_message:
            msg += f": {self.error_message}"
        return msg


class SecretsError(Exception):
    """Raised when secrets cannot be retrieved or decrypted."""
    pass


class ReservationError(Exception):
    """Raised when rate limit reservation fails unexpectedly."""
    pass
