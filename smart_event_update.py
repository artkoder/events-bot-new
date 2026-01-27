from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Iterable, Sequence

from sqlalchemy import and_, or_, select

from db import Database
from models import Event, EventPoster, EventSource

logger = logging.getLogger(__name__)

_HALL_HINT_RE = re.compile(
    r"\b(зал|аудитория|лекторий|сцена|фойе|этаж|корпус)\b\s+([^\s,.;:]+)(?:\s+([^\s,.;:]+))?(?:\s+([^\s,.;:]+))?",
    re.IGNORECASE,
)

SMART_UPDATE_LLM = os.getenv("SMART_UPDATE_LLM", "gemma").strip().lower()
SMART_UPDATE_MODEL = os.getenv(
    "SMART_UPDATE_MODEL",
    os.getenv("TG_MONITORING_TEXT_MODEL", "gemma-3-27b-it"),
).strip()


@dataclass(slots=True)
class PosterCandidate:
    catbox_url: str | None = None
    sha256: str | None = None
    phash: str | None = None
    ocr_text: str | None = None
    ocr_title: str | None = None


@dataclass(slots=True)
class EventCandidate:
    source_type: str
    source_url: str | None
    source_text: str
    title: str | None = None
    date: str | None = None
    time: str | None = None
    end_date: str | None = None
    festival: str | None = None
    location_name: str | None = None
    location_address: str | None = None
    city: str | None = None
    ticket_link: str | None = None
    ticket_price_min: int | None = None
    ticket_price_max: int | None = None
    ticket_status: str | None = None
    event_type: str | None = None
    emoji: str | None = None
    is_free: bool | None = None
    pushkin_card: bool | None = None
    search_digest: str | None = None
    raw_excerpt: str | None = None
    posters: list[PosterCandidate] = field(default_factory=list)
    source_chat_username: str | None = None
    source_chat_id: int | None = None
    source_message_id: int | None = None
    creator_id: int | None = None
    trust_level: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass(slots=True)
class SmartUpdateResult:
    status: str
    event_id: int | None = None
    created: bool = False
    merged: bool = False
    added_posters: int = 0
    added_sources: bool = False
    added_facts: list[str] = field(default_factory=list)
    skipped_conflicts: list[str] = field(default_factory=list)
    reason: str | None = None


MATCH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventMatch",
        "schema": {
            "type": "object",
            "properties": {
                "match_event_id": {"type": ["integer", "null"]},
                "confidence": {"type": "number"},
                "reason_short": {"type": "string"},
            },
            "required": ["match_event_id", "confidence", "reason_short"],
            "additionalProperties": False,
        },
    },
}

MERGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventMerge",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "ticket_link": {"type": ["string", "null"]},
                "ticket_price_min": {"type": ["integer", "null"]},
                "ticket_price_max": {"type": ["integer", "null"]},
                "ticket_status": {"type": ["string", "null"]},
                "added_facts": {"type": "array", "items": {"type": "string"}},
                "skipped_conflicts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["description", "added_facts", "skipped_conflicts"],
            "additionalProperties": False,
        },
    },
}

MATCH_SCHEMA = MATCH_RESPONSE_FORMAT["json_schema"]["schema"]
MERGE_SCHEMA = MERGE_RESPONSE_FORMAT["json_schema"]["schema"]


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_location(value: str | None) -> str:
    if not value:
        return ""
    return _norm_space(value)


def _location_matches(a: str | None, b: str | None) -> bool:
    na = _normalize_location(a)
    nb = _normalize_location(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


@lru_cache(maxsize=1)
def _get_gemma_client():
    try:
        from google_ai import GoogleAIClient, SecretsProvider
        from main import get_supabase_client
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("smart_update: gemma client unavailable: %s", exc)
        return None
    supabase = get_supabase_client()
    return GoogleAIClient(
        supabase_client=supabase,
        secrets_provider=SecretsProvider(),
        consumer="smart_update",
    )


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


async def _ask_gemma_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    max_tokens: int,
    label: str,
) -> dict[str, Any] | None:
    client = _get_gemma_client()
    if client is None:
        return None
    schema_text = json.dumps(schema, ensure_ascii=False)
    full_prompt = (
        f"{prompt}\n\n"
        "Верни только JSON без markdown и комментариев.\n"
        f"JSON schema:\n{schema_text}"
    )
    try:
        raw, _usage = await client.generate_content_async(
            model=SMART_UPDATE_MODEL,
            prompt=full_prompt,
            generation_config={"temperature": 0},
            max_output_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - provider failures
        logger.warning("smart_update: gemma %s failed: %s", label, exc)
        return None
    data = _extract_json(raw or "")
    if data is not None:
        return data
    fix_prompt = (
        "Исправь JSON под схему. Верни только JSON без markdown.\n"
        f"Schema:\n{schema_text}\n\n"
        f"Input:\n{raw}"
    )
    try:
        raw_fix, _usage = await client.generate_content_async(
            model=SMART_UPDATE_MODEL,
            prompt=fix_prompt,
            generation_config={"temperature": 0},
            max_output_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - provider failures
        logger.warning("smart_update: gemma %s json_fix failed: %s", label, exc)
        return None
    return _extract_json(raw_fix or "")


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = value.split("..", 1)[0].strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _event_date_range(ev: Event) -> tuple[date | None, date | None]:
    start = _parse_iso_date(ev.date or "")
    end = _parse_iso_date(ev.end_date) if ev.end_date else None
    if not end and ev.date and ".." in ev.date:
        end = _parse_iso_date(ev.date.split("..", 1)[1])
    if start and not end:
        end = start
    return start, end


def _candidate_date_range(candidate: EventCandidate) -> tuple[date | None, date | None]:
    start = _parse_iso_date(candidate.date)
    end = _parse_iso_date(candidate.end_date) if candidate.end_date else None
    if start and not end:
        end = start
    return start, end


def _ranges_overlap(a_start: date | None, a_end: date | None, b_start: date | None, b_end: date | None) -> bool:
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return not (a_end < b_start or b_end < a_start)


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    value = url.strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        value = value.rstrip("/")
    return value


def _is_http_url(url: str | None) -> bool:
    if not url:
        return False
    value = url.strip().lower()
    return value.startswith("http://") or value.startswith("https://")


def _normalize_event_type_value(
    title: str | None, description: str | None, event_type: str | None
) -> str | None:
    if not event_type:
        return None
    try:
        from main import normalize_event_type
    except Exception:  # pragma: no cover - defensive
        return event_type
    return normalize_event_type(title or "", description or "", event_type)


def _clean_search_digest(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from digest_helper import clean_search_digest
    except Exception:  # pragma: no cover - defensive
        return value.strip()
    return clean_search_digest(value) or None


def _trust_priority(level: str | None) -> int:
    if not level:
        return 2
    key = level.strip().lower()
    if key == "high":
        return 3
    if key == "medium":
        return 2
    if key == "low":
        return 1
    return 2


def _extract_hall_hint(text: str | None) -> str | None:
    if not text:
        return None
    match = _HALL_HINT_RE.search(text)
    if not match:
        return None
    parts = [p for p in match.groups() if p]
    if not parts:
        return None
    return _norm_space(" ".join(parts))


@lru_cache(maxsize=1)
def _load_location_flags() -> dict[str, dict[str, Any]]:
    path = os.path.join("docs", "reference", "location-flags.md")
    flags: dict[str, dict[str, Any]] = {}
    if not os.path.exists(path):
        return flags
    current: str | None = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m_loc = re.match(r"-\s*location_name:\s*\"?(.+?)\"?$", line)
                if m_loc:
                    current = m_loc.group(1).strip()
                    flags[current] = {"allow_parallel_events": False}
                    continue
                if current:
                    m_flag = re.match(r"allow_parallel_events:\s*(true|false)", line, re.I)
                    if m_flag:
                        flags[current]["allow_parallel_events"] = m_flag.group(1).lower() == "true"
    except Exception as exc:
        logger.warning("smart_update: failed to read location flags: %s", exc)
    return flags


def _allow_parallel_events(location_name: str | None) -> bool:
    if not location_name:
        return False
    flags = _load_location_flags()
    for name, data in flags.items():
        if _normalize_location(name) == _normalize_location(location_name):
            return bool(data.get("allow_parallel_events"))
    return False


def _clip(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    raw = text.strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3].rstrip() + "..."


def _clip_title(text: str | None, limit: int = 80) -> str:
    if not text:
        return ""
    raw = text.strip()
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "…"


async def _fetch_event_posters_map(
    db: Database, event_ids: Sequence[int]
) -> dict[int, list[EventPoster]]:
    if not event_ids:
        return {}
    async with db.get_session() as session:
        result = await session.execute(
            select(EventPoster).where(EventPoster.event_id.in_(event_ids))
        )
        posters = list(result.scalars().all())
    grouped: dict[int, list[EventPoster]] = {}
    for poster in posters:
        grouped.setdefault(poster.event_id, []).append(poster)
    return grouped


def _poster_hashes(posters: Iterable[PosterCandidate]) -> set[str]:
    hashes: set[str] = set()
    for poster in posters:
        if poster.sha256:
            hashes.add(poster.sha256)
    return hashes


async def _llm_match_event(
    candidate: EventCandidate,
    events: Sequence[Event],
    *,
    posters_map: dict[int, list[EventPoster]] | None = None,
) -> tuple[int | None, float, str]:
    if not events:
        return None, 0.0, "shortlist_empty"
    if SMART_UPDATE_LLM != "gemma":
        try:
            from main import ask_4o
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("smart_update: ask_4o unavailable: %s", exc)
            return None, 0.0, "llm_unavailable"

    posters_map = posters_map or {}
    candidates_payload: list[dict[str, Any]] = []
    for ev in events:
        posters = posters_map.get(ev.id or 0, [])
        poster_texts = [p.ocr_text for p in posters if p.ocr_text][:2]
        candidates_payload.append(
            {
                "id": ev.id,
                "title": ev.title,
                "date": ev.date,
                "time": ev.time,
                "end_date": ev.end_date,
                "location_name": ev.location_name,
                "location_address": ev.location_address,
                "city": ev.city,
                "ticket_link": ev.ticket_link,
                "description": _clip(ev.description, 600),
                "source_text": _clip(ev.source_text, 600),
                "poster_texts": poster_texts,
            }
        )

    payload = {
        "candidate": {
            "title": candidate.title,
            "date": candidate.date,
            "time": candidate.time,
            "end_date": candidate.end_date,
            "location_name": candidate.location_name,
            "location_address": candidate.location_address,
            "city": candidate.city,
            "ticket_link": candidate.ticket_link,
            "text": _clip(candidate.source_text, 1200),
            "raw_excerpt": _clip(candidate.raw_excerpt, 800),
            "poster_texts": [
                _clip(p.ocr_text, 400) for p in candidate.posters if p.ocr_text
            ][:3],
        },
        "events": candidates_payload[:10],
    }
    prompt = (
        "Ты сопоставляешь анонс события с уже существующими событиями. "
        "Найди наиболее вероятное совпадение или верни null. "
        "Учитывай дату, время, площадку, участников, ссылки, афиши и OCR. "
        "Ответь строго JSON."
        "\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    if SMART_UPDATE_LLM == "gemma":
        data = await _ask_gemma_json(
            prompt,
            MATCH_SCHEMA,
            max_tokens=400,
            label="match",
        )
        if data is None:
            return None, 0.0, "llm_bad_json"
    else:
        try:
            raw = await ask_4o(
                prompt,
                response_format=MATCH_RESPONSE_FORMAT,
                max_tokens=400,
            )
        except Exception as exc:  # pragma: no cover - network / llm failures
            logger.warning("smart_update: match llm failed: %s", exc)
            return None, 0.0, "llm_error"
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            logger.warning("smart_update: match invalid json: %s", raw)
            return None, 0.0, "llm_bad_json"
    match_id = data.get("match_event_id")
    confidence = data.get("confidence")
    reason = data.get("reason_short") or ""
    try:
        conf_val = float(confidence)
    except Exception:
        conf_val = 0.0
    if match_id is None:
        return None, conf_val, reason
    try:
        match_id = int(match_id)
    except Exception:
        return None, conf_val, reason
    return match_id, conf_val, reason


async def _llm_merge_event(
    candidate: EventCandidate,
    event: Event,
    *,
    conflicting_anchor_fields: dict[str, Any] | None = None,
    poster_texts: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    if SMART_UPDATE_LLM != "gemma":
        try:
            from main import ask_4o
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("smart_update: ask_4o unavailable for merge: %s", exc)
            return None

    payload = {
        "event_before": {
            "title": event.title,
            "description": _clip(event.description, 2000),
            "ticket_link": event.ticket_link,
            "ticket_price_min": event.ticket_price_min,
            "ticket_price_max": event.ticket_price_max,
            "ticket_status": getattr(event, "ticket_status", None),
        },
        "candidate": {
            "title": candidate.title,
            "raw_excerpt": _clip(candidate.raw_excerpt, 1200),
            "text": _clip(candidate.source_text, 2000),
            "ticket_link": candidate.ticket_link,
            "ticket_price_min": candidate.ticket_price_min,
            "ticket_price_max": candidate.ticket_price_max,
            "ticket_status": candidate.ticket_status,
            "poster_texts": [
                _clip(p.ocr_text, 400) for p in candidate.posters if p.ocr_text
            ][:3],
        },
        "constraints": {
            "anchor_fields_do_not_change": [
                "date",
                "time",
                "end_date",
                "location_name",
                "location_address",
            ],
            "conflicting_do_not_use": conflicting_anchor_fields or {},
        },
    }
    if poster_texts:
        payload["candidate"]["existing_poster_texts"] = list(poster_texts)[:3]

    prompt = (
        "Ты объединяешь информацию о событии. "
        "Никогда не меняй якорные поля (дата/время/площадка/адрес). "
        "Если кандидат содержит противоречия в якорных полях, игнорируй их. "
        "Добавляй только непротиворечивые факты. "
        "Верни JSON с полями title (если нужно улучшить), description (обязательно), "
        "ticket_link, ticket_price_min/max, ticket_status, added_facts, skipped_conflicts."
        "\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    if SMART_UPDATE_LLM == "gemma":
        data = await _ask_gemma_json(
            prompt,
            MERGE_SCHEMA,
            max_tokens=700,
            label="merge",
        )
        if data is None:
            logger.warning("smart_update: merge invalid json (gemma)")
            return None
        return data
    try:
        raw = await ask_4o(
            prompt,
            response_format=MERGE_RESPONSE_FORMAT,
            max_tokens=700,
        )
    except Exception as exc:  # pragma: no cover - network / llm failures
        logger.warning("smart_update: merge llm failed: %s", exc)
        return None
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("smart_update: merge invalid json: %s", raw)
        return None


def _apply_ticket_fields(
    event: Event,
    *,
    ticket_link: str | None,
    ticket_price_min: int | None,
    ticket_price_max: int | None,
    ticket_status: str | None,
    candidate_trust: str | None,
) -> list[str]:
    added: list[str] = []
    cand_priority = _trust_priority(candidate_trust)
    existing_priority = _trust_priority(getattr(event, "ticket_trust_level", None))

    def _can_override(existing: Any) -> bool:
        if existing in (None, ""):
            return True
        return cand_priority > existing_priority

    if ticket_link and _can_override(event.ticket_link):
        event.ticket_link = ticket_link
        event.ticket_trust_level = candidate_trust
        added.append("ticket_link")
    if ticket_price_min is not None and _can_override(event.ticket_price_min):
        event.ticket_price_min = ticket_price_min
        event.ticket_trust_level = candidate_trust
        added.append("ticket_price_min")
    if ticket_price_max is not None and _can_override(event.ticket_price_max):
        event.ticket_price_max = ticket_price_max
        event.ticket_trust_level = candidate_trust
        added.append("ticket_price_max")
    if ticket_status and _can_override(getattr(event, "ticket_status", None)):
        setattr(event, "ticket_status", ticket_status)
        event.ticket_trust_level = candidate_trust
        added.append("ticket_status")
    return added


def _candidate_has_new_text(candidate: EventCandidate, event: Event) -> bool:
    candidate_text = (candidate.raw_excerpt or candidate.source_text or "").strip()
    if not candidate_text:
        return False
    if not event.description:
        return True
    if len(candidate_text) < 40:
        return False
    return candidate_text not in event.description


async def smart_event_update(
    db: Database,
    candidate: EventCandidate,
    *,
    check_source_url: bool = True,
    schedule_tasks: bool = True,
    schedule_kwargs: dict[str, Any] | None = None,
) -> SmartUpdateResult:
    logger.info(
        "smart_update.start source_type=%s source_url=%s title=%s date=%s time=%s location=%s city=%s posters=%d trust=%s",
        candidate.source_type,
        candidate.source_url,
        _clip_title(candidate.title),
        candidate.date,
        candidate.time,
        _clip_title(candidate.location_name, 60),
        candidate.city,
        len(candidate.posters),
        candidate.trust_level,
    )
    if not candidate.date:
        logger.warning(
            "smart_update.invalid reason=missing_date source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.title),
        )
        return SmartUpdateResult(status="invalid", reason="missing_date")
    if not candidate.title:
        logger.warning(
            "smart_update.invalid reason=missing_title source_type=%s source_url=%s",
            candidate.source_type,
            candidate.source_url,
        )
        return SmartUpdateResult(status="invalid", reason="missing_title")
    if not candidate.location_name:
        logger.warning(
            "smart_update.invalid reason=missing_location source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.title),
        )
        return SmartUpdateResult(status="invalid", reason="missing_location")

    if check_source_url and candidate.source_url:
        async with db.get_session() as session:
            exists = (
                await session.execute(
                    select(EventSource.id).where(EventSource.source_url == candidate.source_url)
                )
            ).scalar_one_or_none()
            if exists:
                logger.info(
                    "smart_update.skip reason=source_url_exists source_type=%s source_url=%s title=%s",
                    candidate.source_type,
                    candidate.source_url,
                    _clip_title(candidate.title),
                )
                return SmartUpdateResult(status="skipped_same_source_url", reason="source_url_exists")

    cand_start, cand_end = _candidate_date_range(candidate)
    if not cand_start or not cand_end:
        return SmartUpdateResult(status="invalid", reason="invalid_date")

    async with db.get_session() as session:
        stmt = select(Event).where(
            and_(
                Event.date >= cand_start.isoformat(),
                Event.date <= cand_end.isoformat(),
                or_(Event.end_date.is_(None), Event.end_date >= cand_start.isoformat()),
            )
        )
        if candidate.city:
            stmt = stmt.where(Event.city == candidate.city)
        res = await session.execute(stmt)
        shortlist = list(res.scalars().all())

    if candidate.location_name:
        shortlist = [
            ev for ev in shortlist if _location_matches(ev.location_name, candidate.location_name)
        ]

    allow_parallel = _allow_parallel_events(candidate.location_name)
    candidate_hall = _extract_hall_hint(candidate.source_text)
    if allow_parallel and candidate_hall:
        filtered: list[Event] = []
        for ev in shortlist:
            hall = _extract_hall_hint((ev.source_text or "") + "\n" + (ev.description or ""))
            if hall and hall != candidate_hall:
                continue
            filtered.append(ev)
        shortlist = filtered

    if not shortlist:
        match_event = None
        match_reason = "shortlist_empty"
        posters_map: dict[int, list[EventPoster]] = {}
    else:
        event_ids = [ev.id for ev in shortlist if ev.id]
        posters_map = await _fetch_event_posters_map(db, event_ids)

        candidate_hashes = _poster_hashes(candidate.posters)
        ticket_norm = _normalize_url(candidate.ticket_link)

        strong_matches: dict[int, int] = {}
        if ticket_norm:
            for ev in shortlist:
                if _normalize_url(ev.ticket_link) == ticket_norm and ev.id:
                    strong_matches[ev.id] = strong_matches.get(ev.id, 0) + 3
        if candidate_hashes:
            for ev in shortlist:
                hashes = {p.poster_hash for p in posters_map.get(ev.id or 0, [])}
                overlap = len(candidate_hashes & hashes)
                if overlap and ev.id:
                    strong_matches[ev.id] = strong_matches.get(ev.id, 0) + overlap

        logger.info(
            "smart_update.shortlist count=%d allow_parallel=%s source_type=%s source_url=%s",
            len(shortlist),
            bool(allow_parallel),
            candidate.source_type,
            candidate.source_url,
        )
        match_event = None
        match_reason = ""
        if strong_matches:
            best = max(strong_matches.items(), key=lambda item: item[1])
            match_event = next((ev for ev in shortlist if ev.id == best[0]), None)
            match_reason = "strong_match"
            logger.info(
                "smart_update.match type=strong event_id=%s score=%s",
                getattr(match_event, "id", None),
                best[1],
            )

        if match_event is None:
            match_id, confidence, reason = await _llm_match_event(
                candidate, shortlist[:10], posters_map=posters_map
            )
            match_reason = reason
            if match_id:
                match_event = next((ev for ev in shortlist if ev.id == match_id), None)
                if match_event is None:
                    confidence = 0.0
                threshold = 0.85 if allow_parallel and len(shortlist) > 1 else 0.6
                if confidence < threshold:
                    match_event = None
                    match_reason = f"llm_conf_{confidence:.2f}<={threshold:.2f}"
            else:
                match_event = None
            logger.info(
                "smart_update.match type=llm match_id=%s confidence=%.2f reason=%s",
                match_id,
                float(confidence or 0.0),
                match_reason,
            )

    if match_event is None:
        normalized_event_type = _normalize_event_type_value(
            candidate.title, candidate.raw_excerpt or candidate.source_text, candidate.event_type
        )
        normalized_digest = _clean_search_digest(candidate.search_digest)
        is_free_value: bool
        if candidate.is_free is True:
            is_free_value = True
        elif candidate.is_free is False:
            is_free_value = False
        else:
            is_free_value = bool(
                candidate.ticket_price_min == 0
                and (candidate.ticket_price_max in (0, None))
            )
        new_event = Event(
            title=candidate.title or "",
            description=(candidate.raw_excerpt or candidate.source_text or candidate.title or "").strip(),
            festival=candidate.festival,
            date=candidate.date or "",
            time=candidate.time or "",
            location_name=candidate.location_name or "",
            location_address=candidate.location_address,
            city=candidate.city or "Калининград",
            ticket_price_min=candidate.ticket_price_min,
            ticket_price_max=candidate.ticket_price_max,
            ticket_link=candidate.ticket_link,
            ticket_status=candidate.ticket_status,
            ticket_trust_level=candidate.trust_level,
            event_type=normalized_event_type or candidate.event_type,
            emoji=candidate.emoji,
            end_date=candidate.end_date,
            is_free=is_free_value,
            pushkin_card=bool(candidate.pushkin_card),
            source_text=candidate.source_text or "",
            source_texts=[candidate.source_text] if candidate.source_text else [],
            source_post_url=candidate.source_url if _is_http_url(candidate.source_url) else None,
            source_chat_id=candidate.source_chat_id,
            source_message_id=candidate.source_message_id,
            creator_id=candidate.creator_id,
            search_digest=normalized_digest,
            photo_urls=[p.catbox_url for p in candidate.posters if p.catbox_url],
            photo_count=len([p for p in candidate.posters if p.catbox_url]),
        )
        if candidate.source_url and _is_http_url(candidate.source_url):
            try:
                from main import is_vk_wall_url
            except Exception:  # pragma: no cover - defensive
                is_vk_wall_url = None
            if is_vk_wall_url and is_vk_wall_url(candidate.source_url):
                new_event.source_vk_post_url = candidate.source_url

        async with db.get_session() as session:
            session.add(new_event)
            await session.commit()
            await session.refresh(new_event)

            added_posters = await _apply_posters(session, new_event.id, candidate.posters)
            added_sources, _same_source = await _ensure_event_source(
                session, new_event.id, candidate
            )
            if candidate.source_text:
                await _sync_source_texts(session, new_event)
            await session.commit()

        await _classify_topics(db, new_event.id)

        if schedule_tasks:
            try:
                from main import schedule_event_update_tasks
                async with db.get_session() as session:
                    refreshed = await session.get(Event, new_event.id)
                if refreshed:
                    await schedule_event_update_tasks(db, refreshed, **(schedule_kwargs or {}))
            except Exception:
                logger.warning("smart_update: schedule/update failed for event %s", new_event.id, exc_info=True)

        logger.info(
            "smart_update.created event_id=%s added_posters=%d added_sources=%s reason=%s",
            new_event.id,
            added_posters,
            int(bool(added_sources)),
            match_reason if "match_reason" in locals() else None,
        )
        return SmartUpdateResult(
            status="created",
            event_id=new_event.id,
            created=True,
            merged=False,
            added_posters=added_posters,
            added_sources=added_sources,
            reason=match_reason if "match_reason" in locals() else None,
        )

    # Merge path
    existing = match_event
    existing_start, existing_end = _event_date_range(existing)
    conflicting = {}
    if existing_start and cand_start and existing_start != cand_start:
        conflicting["date"] = candidate.date
    if existing.time and candidate.time and existing.time != candidate.time:
        conflicting["time"] = candidate.time
    if existing.location_name and candidate.location_name and not _location_matches(existing.location_name, candidate.location_name):
        conflicting["location_name"] = candidate.location_name
    if existing.location_address and candidate.location_address and existing.location_address != candidate.location_address:
        conflicting["location_address"] = candidate.location_address
    if existing_end and cand_end and existing_end != cand_end:
        conflicting["end_date"] = candidate.end_date

    new_hashes = _poster_hashes(candidate.posters)
    existing_hashes = {p.poster_hash for p in posters_map.get(existing.id or 0, [])}
    has_new_posters = bool(new_hashes - existing_hashes)
    has_new_text = _candidate_has_new_text(candidate, existing)

    should_merge = has_new_posters or has_new_text or any(
        [
            candidate.ticket_link and candidate.ticket_link != existing.ticket_link,
            candidate.ticket_price_min is not None and candidate.ticket_price_min != existing.ticket_price_min,
            candidate.ticket_price_max is not None and candidate.ticket_price_max != existing.ticket_price_max,
            candidate.ticket_status and candidate.ticket_status != getattr(existing, "ticket_status", None),
        ]
    )

    added_facts: list[str] = []
    skipped_conflicts: list[str] = []
    updated_fields = False
    updated_keys: list[str] = []

    async with db.get_session() as session:
        event_db = await session.get(Event, existing.id)
        if not event_db:
            return SmartUpdateResult(status="error", reason="event_missing")

        if should_merge:
            posters_texts = [p.ocr_text for p in posters_map.get(existing.id or 0, []) if p.ocr_text]
            merge_data = await _llm_merge_event(
                candidate,
                event_db,
                conflicting_anchor_fields=conflicting,
                poster_texts=posters_texts,
            )
            if merge_data:
                title = merge_data.get("title")
                description = merge_data.get("description")
                if isinstance(title, str) and title.strip():
                    event_db.title = title.strip()
                    updated_fields = True
                    updated_keys.append("title")
                if isinstance(description, str) and description.strip():
                    event_db.description = description.strip()
                    updated_fields = True
                    updated_keys.append("description")
                ticket_updates = _apply_ticket_fields(
                    event_db,
                    ticket_link=merge_data.get("ticket_link"),
                    ticket_price_min=merge_data.get("ticket_price_min"),
                    ticket_price_max=merge_data.get("ticket_price_max"),
                    ticket_status=merge_data.get("ticket_status"),
                    candidate_trust=candidate.trust_level,
                )
                if ticket_updates:
                    updated_fields = True
                    updated_keys.extend(ticket_updates)
                added_facts = list(merge_data.get("added_facts") or [])
                skipped_conflicts = list(merge_data.get("skipped_conflicts") or [])
        else:
            ticket_updates = _apply_ticket_fields(
                event_db,
                ticket_link=candidate.ticket_link,
                ticket_price_min=candidate.ticket_price_min,
                ticket_price_max=candidate.ticket_price_max,
                ticket_status=candidate.ticket_status,
                candidate_trust=candidate.trust_level,
            )
            if ticket_updates:
                updated_fields = True
                updated_keys.extend(ticket_updates)

        if not event_db.location_address and candidate.location_address:
            event_db.location_address = candidate.location_address
            updated_fields = True
            updated_keys.append("location_address")
        if not event_db.city and candidate.city:
            event_db.city = candidate.city
            updated_fields = True
            updated_keys.append("city")
        if not event_db.end_date and candidate.end_date:
            event_db.end_date = candidate.end_date
            updated_fields = True
            updated_keys.append("end_date")
        if not event_db.festival and candidate.festival:
            event_db.festival = candidate.festival
            updated_fields = True
            updated_keys.append("festival")
        if candidate.event_type and not event_db.event_type:
            normalized = _normalize_event_type_value(
                event_db.title, event_db.description, candidate.event_type
            )
            event_db.event_type = normalized or candidate.event_type
            updated_fields = True
            updated_keys.append("event_type")
        if candidate.emoji and not event_db.emoji:
            event_db.emoji = candidate.emoji
            updated_fields = True
            updated_keys.append("emoji")
        if candidate.search_digest and not event_db.search_digest:
            event_db.search_digest = _clean_search_digest(candidate.search_digest)
            updated_fields = True
            updated_keys.append("search_digest")
        if candidate.pushkin_card is True and not event_db.pushkin_card:
            event_db.pushkin_card = True
            updated_fields = True
            updated_keys.append("pushkin_card")
        if not event_db.is_free:
            if candidate.is_free is True:
                event_db.is_free = True
                updated_fields = True
                updated_keys.append("is_free")
            elif (
                event_db.ticket_price_min == 0
                and (event_db.ticket_price_max in (0, None))
            ):
                event_db.is_free = True
                updated_fields = True
                updated_keys.append("is_free")
        if not event_db.source_post_url and candidate.source_url and _is_http_url(candidate.source_url):
            event_db.source_post_url = candidate.source_url
            updated_fields = True
            updated_keys.append("source_post_url")
        if candidate.source_url and _is_http_url(candidate.source_url):
            try:
                from main import is_vk_wall_url
            except Exception:  # pragma: no cover - defensive
                is_vk_wall_url = None
            if is_vk_wall_url and is_vk_wall_url(candidate.source_url):
                if not event_db.source_vk_post_url:
                    event_db.source_vk_post_url = candidate.source_url
                    updated_fields = True
                    updated_keys.append("source_vk_post_url")
        if not event_db.creator_id and candidate.creator_id:
            event_db.creator_id = candidate.creator_id
            updated_fields = True
            updated_keys.append("creator_id")

        added_posters = await _apply_posters(session, event_db.id, candidate.posters)
        if added_posters:
            updated_fields = True
            updated_keys.append("posters")

        added_sources, same_source = await _ensure_event_source(
            session, event_db.id, candidate
        )
        if candidate.source_text:
            if same_source:
                event_db.source_text = candidate.source_text
                updated_fields = True
                updated_keys.append("source_text")
            if await _sync_source_texts(session, event_db):
                updated_fields = True
                updated_keys.append("source_texts")

        if updated_fields:
            session.add(event_db)
        await session.commit()

    if updated_fields or added_posters:
        await _classify_topics(db, existing.id)
        if schedule_tasks:
            try:
                from main import schedule_event_update_tasks
                async with db.get_session() as session:
                    refreshed = await session.get(Event, existing.id)
                if refreshed:
                    await schedule_event_update_tasks(db, refreshed, **(schedule_kwargs or {}))
            except Exception:
                logger.warning("smart_update: schedule/update failed for event %s", existing.id, exc_info=True)

    status = "merged" if updated_fields or added_posters else "skipped_nochange"
    logger.info(
        "smart_update.merge event_id=%s status=%s updated=%s added_posters=%d added_sources=%s updated_keys=%s added_facts=%d skipped_conflicts=%d reason=%s",
        existing.id,
        status,
        int(bool(updated_fields)),
        added_posters,
        int(bool(added_sources)),
        ",".join(updated_keys[:12]) if updated_keys else "",
        len(added_facts),
        len(skipped_conflicts),
        match_reason if "match_reason" in locals() else None,
    )
    return SmartUpdateResult(
        status=status,
        event_id=existing.id,
        created=False,
        merged=updated_fields,
        added_posters=added_posters,
        added_sources=added_sources,
        added_facts=added_facts,
        skipped_conflicts=skipped_conflicts,
        reason=match_reason if "match_reason" in locals() else None,
    )


async def _apply_posters(
    session,
    event_id: int | None,
    posters: Sequence[PosterCandidate],
) -> int:
    if not event_id or not posters:
        return 0
    existing = (
        await session.execute(select(EventPoster).where(EventPoster.event_id == event_id))
    ).scalars()
    existing_map = {row.poster_hash: row for row in existing}
    added = 0
    now = datetime.now(timezone.utc)
    extra_urls: list[str] = []

    for poster in posters:
        digest = poster.sha256
        if not digest:
            if poster.catbox_url:
                extra_urls.append(poster.catbox_url)
            continue
        row = existing_map.get(digest)
        if row:
            if poster.catbox_url:
                row.catbox_url = poster.catbox_url
            if poster.phash:
                row.phash = poster.phash
            if poster.ocr_text is not None:
                row.ocr_text = poster.ocr_text
            if poster.ocr_title is not None:
                row.ocr_title = poster.ocr_title
            row.updated_at = now
        else:
            session.add(
                EventPoster(
                    event_id=event_id,
                    catbox_url=poster.catbox_url,
                    poster_hash=digest,
                    phash=poster.phash,
                    ocr_text=poster.ocr_text,
                    ocr_title=poster.ocr_title,
                    updated_at=now,
                )
            )
            added += 1

    # Update event.photo_urls if possible
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if event:
        current = list(event.photo_urls or [])
        for poster in posters:
            if poster.catbox_url and poster.catbox_url not in current:
                current.append(poster.catbox_url)
        for url in extra_urls:
            if url not in current:
                current.append(url)
        # Prefer posters with OCR text/title (proxy for "quality")
        preferred_urls: list[str] = []
        scored: list[tuple[int, str]] = []
        for poster in posters:
            if not poster.catbox_url:
                continue
            score = 0
            if poster.ocr_title:
                score += len(poster.ocr_title)
            if poster.ocr_text:
                score += len(poster.ocr_text)
            if score > 0:
                scored.append((score, poster.catbox_url))
        if scored:
            for _score, url in sorted(scored, key=lambda item: item[0], reverse=True):
                if url not in preferred_urls:
                    preferred_urls.append(url)
            reordered = preferred_urls + [url for url in current if url not in preferred_urls]
            current = reordered
        event.photo_urls = current
        event.photo_count = len(current)
        session.add(event)

    return added


async def _ensure_event_source(
    session,
    event_id: int | None,
    candidate: EventCandidate,
) -> tuple[bool, bool]:
    if not event_id or not candidate.source_url:
        return False, False
    existing = (
        await session.execute(
            select(EventSource).where(
                EventSource.event_id == event_id,
                EventSource.source_url == candidate.source_url,
            )
        )
    ).scalar_one_or_none()
    if existing:
        updated = False
        if candidate.source_text and candidate.source_text != existing.source_text:
            existing.source_text = candidate.source_text
            existing.imported_at = datetime.now(timezone.utc)
            updated = True
            logger.info(
                "smart_update.source_text_update event_id=%s source_url=%s",
                event_id,
                candidate.source_url,
            )
        if candidate.trust_level and not existing.trust_level:
            existing.trust_level = candidate.trust_level
            updated = True
        if updated:
            session.add(existing)
        return False, True
    session.add(
        EventSource(
            event_id=event_id,
            source_type=candidate.source_type,
            source_url=candidate.source_url,
            source_chat_username=candidate.source_chat_username,
            source_chat_id=candidate.source_chat_id,
            source_message_id=candidate.source_message_id,
            source_text=candidate.source_text or None,
            imported_at=datetime.now(timezone.utc),
            trust_level=candidate.trust_level,
        )
    )
    return True, False


async def _sync_source_texts(session, event: Event) -> bool:
    if not event:
        return False
    rows = (
        await session.execute(
            select(EventSource.source_text, EventSource.imported_at)
            .where(EventSource.event_id == event.id)
            .order_by(EventSource.imported_at)
        )
    ).all()
    texts: list[str] = []
    for text, _ts in rows:
        if not text:
            continue
        if text not in texts:
            texts.append(text)
    if texts != list(event.source_texts or []):
        event.source_texts = texts
        logger.info(
            "smart_update.source_texts_sync event_id=%s count=%d",
            event.id,
            len(texts),
        )
        return True
    return False


async def _classify_topics(db: Database, event_id: int | None) -> None:
    if not event_id:
        return
    try:
        from main import assign_event_topics
    except Exception:
        return
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        if not event or event.topics_manual:
            return
        try:
            await assign_event_topics(event)
        except Exception:
            logger.warning("smart_update: topic classification failed event_id=%s", event_id, exc_info=True)
            return
        session.add(event)
        await session.commit()
