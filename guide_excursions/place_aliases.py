from __future__ import annotations

import os
from functools import lru_cache

from location_reference import normalize_venue_key

from .parser import collapse_ws


@lru_cache(maxsize=1)
def _read_alias_lines() -> tuple[str, ...]:
    path = os.path.join("docs", "reference", "guide-place-aliases.md")
    if not os.path.exists(path):
        return ()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except Exception:
        return ()
    return tuple(lines)


@lru_cache(maxsize=1)
def _read_alias_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _read_alias_lines():
        if "=>" not in line:
            continue
        alias_raw, public_raw = line.split("=>", 1)
        alias_key = normalize_venue_key(alias_raw)
        public_value = collapse_ws(public_raw)
        if alias_key and public_value:
            out[alias_key] = public_value
    return out


def normalize_public_place(value: object | None) -> str | None:
    text = collapse_ws("" if value is None else str(value))
    if not text:
        return None
    alias = _read_alias_map().get(normalize_venue_key(text))
    return alias or text
