from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CANDIDATE_KEY_ID_CACHE: dict[tuple[str, ...], tuple[str, ...] | None] = {}


class GuideSecretsProviderAdapter:
    def __init__(self, base: Any, *, primary_key_env: str, fallback_key_env: str):
        self.base = base
        self.primary_key_env = primary_key_env
        self.fallback_key_env = fallback_key_env

    def get_secret(self, name: str) -> str | None:
        if name in {"GOOGLE_API_KEY", self.primary_key_env}:
            return self.base.get_secret(self.primary_key_env) or self.base.get_secret(self.fallback_key_env)
        return self.base.get_secret(name)


def guide_account_name(*, primary_account_env: str, fallback_account_env: str) -> str | None:
    return (
        os.getenv(primary_account_env)
        or os.getenv(fallback_account_env)
        or ""
    ).strip() or None


def _key_env_aliases(name: str | None) -> list[str]:
    raw = (name or "").strip()
    if not raw:
        return []
    names = [raw]
    if raw == "GOOGLE_API_KEY2":
        names.append("GOOGLE_API_KEY_2")
    elif raw == "GOOGLE_API_KEY_2":
        names.append("GOOGLE_API_KEY2")
    return names


def resolve_candidate_key_ids(
    *,
    supabase: Any,
    primary_key_env: str,
    fallback_key_env: str,
    consumer: str,
) -> list[str] | None:
    if supabase is None:
        return None
    primary_envs = _key_env_aliases(primary_key_env)
    fallback_envs = [name for name in _key_env_aliases(fallback_key_env) if name not in primary_envs]
    env_names = [*primary_envs, *fallback_envs]
    if not env_names:
        return None
    cache_key = (consumer, *env_names)
    if cache_key in _CANDIDATE_KEY_ID_CACHE:
        cached = _CANDIDATE_KEY_ID_CACHE[cache_key]
        return list(cached) if cached else None

    try:
        result = (
            supabase.table("google_ai_api_keys")
            .select("id, env_var_name, priority")
            .eq("is_active", True)
            .in_("env_var_name", env_names)
            .order("priority")
            .order("id")
            .execute()
        )
        rows = list(result.data or [])
    except Exception as exc:
        logger.warning("guide_llm_support: failed to resolve candidate keys for %s: %s", consumer, exc)
        _CANDIDATE_KEY_ID_CACHE[cache_key] = None
        return None

    primary_ids = [
        str(row.get("id"))
        for row in rows
        if row.get("id") and str(row.get("env_var_name") or "") in primary_envs
    ]
    fallback_ids = [
        str(row.get("id"))
        for row in rows
        if row.get("id") and str(row.get("env_var_name") or "") in fallback_envs
    ]

    if primary_envs and not primary_ids:
        logger.warning(
            "guide_llm_support: missing primary key metadata for %s envs=%s; falling back=%s",
            consumer,
            ",".join(primary_envs),
            bool(fallback_ids),
        )

    resolved = tuple(primary_ids or fallback_ids)
    _CANDIDATE_KEY_ID_CACHE[cache_key] = resolved or None
    return list(resolved) if resolved else None
