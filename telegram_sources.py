from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from pathlib import Path
from functools import lru_cache

import yaml


TRUST_LEVELS = ("low", "medium", "high")
_TRUST_PRIORITY = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True, slots=True)
class TelegramSourceSpec:
    username: str
    trust_level: str
    default_location: str | None = None
    festival_series: str | None = None
    filters: dict[str, Any] | None = None
    notes: str | None = None


_USERNAME_RE = re.compile(r"^[a-z0-9_]{4,64}$")


def normalize_tg_username(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = raw.lstrip("@").strip()
    raw = re.sub(r"^https?://", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^tg://resolve\\?domain=", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(?:www\\.)?(?:t\\.me|telegram\\.me)/", "", raw, flags=re.IGNORECASE)
    raw = raw.split("?", 1)[0].split("#", 1)[0]
    raw = raw.split("/", 1)[0]
    raw = raw.strip().lstrip("@").strip().lower()
    if not raw:
        return ""
    if not _USERNAME_RE.match(raw):
        return ""
    return raw


def trust_priority(value: str | None) -> int:
    key = (value or "").strip().lower()
    return _TRUST_PRIORITY.get(key, 0)


def _load_canonical_sources() -> list[TelegramSourceSpec]:
    root = Path(__file__).resolve().parent
    path = root / "docs" / "features" / "telegram-monitoring" / "sources.yml"
    if not path.exists():
        raise FileNotFoundError(f"Canonical telegram sources file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or int(payload.get("version") or 0) != 1:
        raise ValueError("Invalid sources.yml: expected version: 1")
    raw_sources = payload.get("sources") or []
    if not isinstance(raw_sources, list):
        raise ValueError("Invalid sources.yml: sources must be a list")
    out: list[TelegramSourceSpec] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        username = normalize_tg_username(item.get("username"))
        trust = str(item.get("trust_level") or "").strip().lower()
        if not username or trust not in TRUST_LEVELS:
            continue
        default_location = str(item.get("default_location") or "").strip() or None
        festival_series = str(item.get("festival_series") or "").strip() or None
        filters = item.get("filters")
        if filters is not None and not isinstance(filters, dict):
            filters = None
        notes = str(item.get("notes") or "").strip() or None
        out.append(
            TelegramSourceSpec(
                username=username,
                trust_level=trust,
                default_location=default_location,
                festival_series=festival_series,
                filters=dict(filters) if isinstance(filters, dict) else None,
                notes=notes,
            )
        )
    # Keep deterministic order for pagination and reproducibility.
    out.sort(key=lambda s: s.username)
    return out


@lru_cache(maxsize=1)
def canonical_tg_sources() -> tuple[TelegramSourceSpec, ...]:
    return tuple(_load_canonical_sources())
