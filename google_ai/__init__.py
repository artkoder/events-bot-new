"""Google AI SDK with rate limiting and secrets management.

This module provides:
- SecretsProvider: Unified secret retrieval (env -> Kaggle Secrets -> encrypted datasets)
- GoogleAIClient: Wrapper over google.generativeai with Supabase-based rate limiting
- RateLimitError, ProviderError: Custom exceptions
"""

from google_ai.exceptions import RateLimitError, ProviderError
from google_ai.secrets import SecretsProvider, get_secret, get_secret_pool
from google_ai.client import GoogleAIClient

__all__ = [
    "SecretsProvider",
    "get_secret",
    "get_secret_pool",
    "GoogleAIClient",
    "RateLimitError",
    "ProviderError",
]
