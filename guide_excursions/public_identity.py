from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

from telegram_sources import normalize_tg_username

from .parser import collapse_ws
from .telethon_client import create_telethon_runtime_client

logger = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"(?<!\\w)@([A-Za-z0-9_]{4,64})")
GUIDE_CONTEXT_RE = re.compile(
    r"(?:\bмы\s+с\b|\bвместе\s+с\b|\bгид(?:ы|ом|а)?\b|\bэкскурсовод(?:ы|ом|а)?\b|"
    r"\bпровед[её]\w*\b|\bвед[её]\w*\b|\bавтор(?:ы|ом)?\s+маршрута\b)",
    re.I,
)
BOOKING_CONTEXT_RE = re.compile(
    r"(?:\bзапись\b|\bзапис[ьа]\w*\b|\bбронь\b|\bбронир\w*\b|\bпишите\b|\bпиши\b|"
    r"\bл/с\b|\bлс\b|\bличк\w*\b|\bв\s+личк\w*\b|\bзвоните\b|\bтелефон\b|\bwhatsapp\b|\bватсап\b)",
    re.I,
)


def extract_public_usernames(text: object | None, *, limit: int = 6) -> list[str]:
    raw = collapse_ws("" if text is None else str(text))
    if not raw:
        return []
    out: list[str] = []
    for match in USERNAME_RE.findall(raw):
        username = normalize_tg_username(match)
        if not username or username in out:
            continue
        out.append(username)
        if len(out) >= limit:
            break
    return out


def extract_public_guide_usernames(text: object | None, *, limit: int = 6) -> list[str]:
    raw = collapse_ws("" if text is None else str(text))
    if not raw:
        return []
    out: list[str] = []
    segments = re.split(r"(?<=[\n.!?;])\s+", raw)
    for segment in segments:
        if not segment:
            continue
        if BOOKING_CONTEXT_RE.search(segment):
            continue
        if not GUIDE_CONTEXT_RE.search(segment):
            continue
        for username in extract_public_usernames(segment, limit=limit):
            if username in out:
                continue
            out.append(username)
            if len(out) >= limit:
                return out
    return out


def _display_name_from_entity(entity: Any) -> str | None:
    first = collapse_ws(getattr(entity, "first_name", None))
    last = collapse_ws(getattr(entity, "last_name", None))
    full = collapse_ws(" ".join(part for part in (first, last) if part))
    if len(full.split()) >= 2:
        return full
    title = collapse_ws(getattr(entity, "title", None))
    if len(title.split()) >= 2:
        return title
    return None


def _surname_key(value: str | None) -> str:
    parts = [part for part in collapse_ws(value).split() if part]
    if len(parts) < 2:
        return ""
    return parts[-1].casefold().replace("ё", "е")


def _text_key(value: str | None) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", collapse_ws(value).casefold().replace("ё", "е"), flags=re.I)


def _merge_resolved_names(guide_names: Sequence[str], resolved_names: Sequence[str]) -> list[str]:
    current = [collapse_ws(item) for item in guide_names if collapse_ws(item)]
    current = list(dict.fromkeys(current))
    if not resolved_names:
        return current
    out = list(current)
    for resolved in resolved_names:
        surname = _surname_key(resolved)
        if surname:
            replaced = False
            for idx, existing in enumerate(out):
                if _surname_key(existing) == surname:
                    out[idx] = resolved
                    replaced = True
                    break
            if replaced:
                continue
        if resolved not in out:
            out.append(resolved)
    return out[:4]


def _collapse_profile_aliases(row: Mapping[str, Any], guide_names: Sequence[str]) -> list[str]:
    current = [collapse_ws(item) for item in guide_names if collapse_ws(item)]
    current = list(dict.fromkeys(current))
    canonical = collapse_ws(row.get("guide_profile_display_name"))
    marketing = collapse_ws(row.get("guide_profile_marketing_name"))
    source_username = normalize_tg_username(row.get("source_username"))
    alias_keys = {_text_key(marketing)}
    if source_username:
        alias_keys.add(_text_key(source_username))
        alias_keys.add(_text_key(f"@{source_username}"))
    if not canonical:
        return current
    out: list[str] = []
    inserted = False
    canonical_key = _text_key(canonical)
    for item in current:
        key = _text_key(item)
        if key == canonical_key or key in alias_keys:
            if not inserted:
                out.append(canonical)
                inserted = True
            continue
        out.append(item)
    if not out and canonical:
        out.append(canonical)
    return list(dict.fromkeys(out))[:4]


async def resolve_public_guide_names(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = [dict(row) for row in rows]
    usernames: list[str] = []
    by_occurrence: dict[int, list[str]] = {}
    for row in prepared:
        occurrence_id = int(row.get("id") or row.get("occurrence_id") or 0)
        if occurrence_id <= 0:
            continue
        raw_usernames = extract_public_guide_usernames(row.get("dedup_source_text"), limit=6)
        source_username = normalize_tg_username(row.get("source_username"))
        filtered = [item for item in raw_usernames if item and item != source_username]
        if filtered:
            by_occurrence[occurrence_id] = filtered
            for username in filtered:
                if username not in usernames:
                    usernames.append(username)
    resolved: dict[str, dict[str, str]] = {}
    if usernames:
        client = None
        try:
            client = await create_telethon_runtime_client()
            for username in usernames:
                try:
                    entity = await client.get_entity(username)
                except Exception:
                    logger.info("guide_public_identity: failed to resolve @%s", username, exc_info=True)
                    continue
                display_name = _display_name_from_entity(entity)
                if not display_name:
                    continue
                resolved[username] = {
                    "display_name": display_name,
                    "username": username,
                }
        except Exception:
            logger.warning("guide_public_identity: telethon resolver unavailable", exc_info=True)
        finally:
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    logger.warning("guide_public_identity: failed to disconnect telethon client", exc_info=True)

    out: list[dict[str, Any]] = []
    for row in prepared:
        occurrence_id = int(row.get("id") or row.get("occurrence_id") or 0)
        guide_names = row.get("guide_names") or []
        if not isinstance(guide_names, list):
            guide_names = []
        guide_names = _collapse_profile_aliases(row, guide_names)
        row["guide_names"] = list(guide_names)
        resolved_names = [
            resolved[item]["display_name"]
            for item in by_occurrence.get(occurrence_id, [])
            if item in resolved
        ]
        if resolved_names:
            row["guide_names"] = _merge_resolved_names(guide_names, resolved_names)
            row["resolved_guide_profiles"] = [
                resolved[item]
                for item in by_occurrence.get(occurrence_id, [])
                if item in resolved
            ]
        out.append(row)
    return out
