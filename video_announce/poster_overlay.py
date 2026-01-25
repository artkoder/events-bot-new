from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from db import Database
from google_ai import GoogleAIClient, ProviderError, RateLimitError
from google_ai.secrets import get_provider
from main import get_supabase_client
from models import EventPoster

from .selection import _has_meaningful_ocr_text

logger = logging.getLogger(__name__)

# Default per-user request is short; keep output small and strictly JSON.
_GEMMA_MODEL_RAW = os.getenv("VIDEO_ANNOUNCE_POSTER_CHECK_MODEL", "gemma-3-27b")


def _normalize_gemma_model_id(value: str) -> str:
    """Keep app-level model id compatible with Supabase limits seed (e.g. 'gemma-3-27b').

    UniversalFestivalParser uses provider model_name 'models/gemma-3-27b-it' while
    treating the canonical id as 'gemma-3-27b' for config/limits.
    """

    v = (value or "").strip()
    if v.startswith("models/"):
        v = v[len("models/") :]
    # Drop provider "-it" suffix if someone passes it in env.
    if v.startswith("gemma-") and v.endswith("-it"):
        v = v[: -len("-it")]
    return v or "gemma-3-27b"


_GEMMA_MODEL = _normalize_gemma_model_id(_GEMMA_MODEL_RAW)

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "\U00002300-\U000023FF"
    "]+"
)


@dataclass(frozen=True)
class PosterCheck:
    has_title: bool
    has_date: bool
    has_time: bool
    has_location: bool
    missing: tuple[str, ...]
    rationale: str | None = None
    model: str | None = None


def _collapse_ws(text: str | None) -> str:
    raw = _EMOJI_RE.sub("", str(text or ""))
    return " ".join(raw.replace("\n", " ").split()).strip()


def _shorten(text: str | None, limit: int) -> str:
    raw = _collapse_ws(text)
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "…"


def _heuristic_check(*, ocr_text: str, title: str, date: str, location: str) -> PosterCheck:
    """Cheap fallback when Gemma is rate-limited/unavailable."""

    ocr_norm = ocr_text.casefold()
    title_norm = title.casefold()
    location_norm = location.casefold()

    # Title: at least one non-trivial token from event title appears in OCR.
    title_tokens = [
        t
        for t in re.split(r"[^\w]+", title_norm, flags=re.UNICODE)
        if len(t) >= 4
    ]
    has_title = any(t in ocr_norm for t in title_tokens[:6]) if title_tokens else False

    # Date: common patterns + ru month names.
    has_date = bool(
        re.search(r"\b\d{1,2}[./-]\d{1,2}\b", ocr_text)
        or re.search(
            r"\b(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\b",
            ocr_norm,
        )
    )

    # Time: 19:00 / 7:30 etc.
    has_time = bool(re.search(r"\b\d{1,2}[:.]\d{2}\b", ocr_text))

    # Location: check if city/venue tokens appear in OCR.
    loc_tokens = [
        t
        for t in re.split(r"[^\w]+", location_norm, flags=re.UNICODE)
        if len(t) >= 4
    ]
    has_location = any(t in ocr_norm for t in loc_tokens[:6]) if loc_tokens else False

    missing: list[str] = []
    if not has_title:
        missing.append("title")
    if not has_date:
        missing.append("date")
    if not has_time:
        missing.append("time")
    if not has_location:
        missing.append("location")
    return PosterCheck(
        has_title=has_title,
        has_date=has_date,
        has_time=has_time,
        has_location=has_location,
        missing=tuple(missing),
        rationale="heuristic_fallback",
        model=None,
    )


def _build_gemma_prompt(*, ocr_text: str, title: str, date: str, location: str) -> str:
    # Keep the instruction concise and hard to misinterpret for JSON parsing.
    return (
        "Ты проверяешь, есть ли на афише (в OCR-тексте) данные для решения идти на событие.\n"
        "Верни СТРОГО JSON (без markdown/комментариев) по схеме:\n"
        "{\n"
        '  "has_title": true|false,\n'
        '  "has_date": true|false,\n'
        '  "has_time": true|false,\n'
        '  "has_location": true|false,\n'
        '  "missing": ["title"|"date"|"time"|"location", ...],\n'
        '  "rationale": "коротко, до 200 символов"\n'
        "}\n\n"
        "Правила:\n"
        "- has_title=true если в OCR есть понятное название события (не обязательно точное совпадение с title).\n"
        "- has_date=true если есть дата (например 24 января / 24.01 / 24-25.01).\n"
        "- has_time=true если есть время (например 19:00 / 7:30).\n"
        "- has_location=true если есть место проведения (площадка/адрес/город).\n"
        "- missing должен соответствовать has_* (включай только отсутствующие).\n\n"
        f"Ожидаемые данные события:\n- title: {title}\n- date: {date}\n- location: {location}\n\n"
        f"OCR:\n{ocr_text}\n"
    )


def _parse_gemma_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    # Try to extract the first JSON object if the model adds extra text.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        text = m.group(0)
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_check(obj: dict[str, Any], *, model: str) -> PosterCheck | None:
    def _b(key: str) -> bool | None:
        v = obj.get(key)
        return v if isinstance(v, bool) else None

    has_title = _b("has_title")
    has_date = _b("has_date")
    has_time = _b("has_time")
    has_location = _b("has_location")
    if None in (has_title, has_date, has_time, has_location):
        return None

    missing_raw = obj.get("missing", [])
    missing: list[str] = []
    if isinstance(missing_raw, list):
        for item in missing_raw:
            if isinstance(item, str):
                v = item.strip().casefold()
                if v in {"title", "date", "time", "location"} and v not in missing:
                    missing.append(v)

    # If model forgot to fill missing, derive it.
    derived: list[str] = []
    if not has_title:
        derived.append("title")
    if not has_date:
        derived.append("date")
    if not has_time:
        derived.append("time")
    if not has_location:
        derived.append("location")
    if set(missing) != set(derived):
        missing = derived

    rationale = obj.get("rationale")
    rationale_s = str(rationale).strip() if isinstance(rationale, str) else None
    if rationale_s:
        rationale_s = rationale_s[:200]
    return PosterCheck(
        has_title=bool(has_title),
        has_date=bool(has_date),
        has_time=bool(has_time),
        has_location=bool(has_location),
        missing=tuple(missing),
        rationale=rationale_s,
        model=model,
    )


async def _gemma_check(*, ocr_text: str, title: str, date: str, location: str) -> PosterCheck | None:
    prompt = _build_gemma_prompt(
        ocr_text=_shorten(ocr_text, 6000),
        title=_shorten(title, 200),
        date=_shorten(date, 120),
        location=_shorten(location, 200),
    )
    client = GoogleAIClient(
        supabase_client=get_supabase_client(),
        secrets_provider=get_provider(),
        consumer="bot",
    )
    raw, _usage = await client.generate_content_async(
        model=_GEMMA_MODEL,
        prompt=prompt,
        generation_config={"temperature": 0.0},
        max_output_tokens=512,
    )
    obj = _parse_gemma_json(raw)
    if obj is None:
        return None
    return _coerce_check(obj, model=_GEMMA_MODEL)


async def _load_latest_ocr_texts(db: Database, event_ids: list[int]) -> dict[int, str]:
    if not event_ids:
        return {}
    async with db.get_session() as session:
        result = await session.execute(
            select(EventPoster)
            .where(EventPoster.event_id.in_(event_ids))
            .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
        )
        posters = result.scalars().all()
    seen: set[int] = set()
    out: dict[int, str] = {}
    for poster in posters:
        if poster.event_id in seen:
            continue
        text = (poster.ocr_text or "").strip()
        if _has_meaningful_ocr_text(text):
            title = (poster.ocr_title or "").strip()
            combined = text
            if _has_meaningful_ocr_text(title) and title.casefold() not in text.casefold():
                combined = f"{title}\n{text}"
            out[poster.event_id] = combined
            seen.add(poster.event_id)
    return out


def _build_overlay_text(*, missing: tuple[str, ...], title: str, date: str, location: str) -> str | None:
    # Always produce a compact 1-3 line block.
    lines: list[str] = []
    if "title" in missing:
        t = _shorten(title, 64)
        if t:
            lines.append(t)
    if "date" in missing or "time" in missing:
        d = _shorten(date, 48)
        if d:
            lines.append(d)
    if "location" in missing:
        loc = _shorten(location, 64)
        if loc:
            lines.append(loc)
    if not lines:
        return None
    return "\n".join(lines)


async def enrich_payload_with_poster_overlays(db: Database, payload_json: str) -> str:
    """Attach per-scene `poster_overlay` and drop scenes with empty OCR text.

    The kernel then renders an extra badge onto the poster without covering existing text
    (best-effort).
    """

    try:
        payload_obj = json.loads(payload_json)
    except Exception:
        return payload_json
    if not isinstance(payload_obj, dict):
        return payload_json
    scenes = payload_obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return payload_json
    selection_params = payload_obj.get("selection_params")
    allow_empty_ocr = bool(
        selection_params.get("allow_empty_ocr") if isinstance(selection_params, dict) else False
    )

    event_ids: list[int] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        ev_id = scene.get("event_id")
        if isinstance(ev_id, int):
            event_ids.append(ev_id)
    if not event_ids:
        return payload_json

    ocr_map = await _load_latest_ocr_texts(db, event_ids)

    # Evaluate sequentially with a small concurrency window; rate limits are handled by GoogleAIClient.
    sem = asyncio.Semaphore(3)

    async def _process_scene(scene: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        ev_id = scene.get("event_id")
        if not isinstance(ev_id, int):
            return True, scene
        ocr_text = ocr_map.get(ev_id, "")
        title = _collapse_ws(scene.get("title"))
        date = _collapse_ws(scene.get("date"))
        location = _collapse_ws(scene.get("location"))

        if not _has_meaningful_ocr_text(ocr_text):
            if allow_empty_ocr:
                missing_all = ("title", "date", "time", "location")
                overlay_text = _build_overlay_text(
                    missing=missing_all,
                    title=title,
                    date=date,
                    location=location,
                )
                if overlay_text:
                    scene["poster_overlay"] = {
                        "v": 1,
                        "missing": list(missing_all),
                        "text": overlay_text,
                        "model": "empty_ocr_fallback",
                    }
                return True, scene
            return False, scene
        # Default to "missing everything" if required info isn't even available from event record.
        if not title or not date or not location:
            missing_all = tuple(
                k
                for k, v in (("title", title), ("date", date), ("time", date), ("location", location))
                if not v
            )
            overlay_text = _build_overlay_text(
                missing=missing_all,
                title=title,
                date=date,
                location=location,
            )
            if overlay_text:
                scene["poster_overlay"] = {
                    "v": 1,
                    "missing": list(missing_all),
                    "text": overlay_text,
                    "model": "fallback_missing_event_fields",
                }
            return True, scene

        check: PosterCheck | None = None
        async with sem:
            try:
                check = await _gemma_check(
                    ocr_text=ocr_text, title=title, date=date, location=location
                )
            except RateLimitError as e:
                logger.warning(
                    "video_announce: gemma poster check rate-limited event_id=%s reason=%s",
                    ev_id,
                    getattr(e, "blocked_reason", None),
                )
            except ProviderError as e:
                logger.warning(
                    "video_announce: gemma poster check provider error event_id=%s type=%s",
                    ev_id,
                    getattr(e, "error_type", None),
                )
            except Exception:
                logger.exception("video_announce: gemma poster check failed event_id=%s", ev_id)

        if check is None:
            check = _heuristic_check(
                ocr_text=ocr_text, title=title, date=date, location=location
            )

        if check.missing:
            overlay_text = _build_overlay_text(
                missing=check.missing, title=title, date=date, location=location
            )
            if overlay_text:
                scene["poster_overlay"] = {
                    "v": 1,
                    "missing": list(check.missing),
                    "text": overlay_text,
                    "model": check.model or "heuristic",
                    "rationale": check.rationale,
                }

        return True, scene

    tasks = [_process_scene(scene) for scene in scenes if isinstance(scene, dict)]
    results = await asyncio.gather(*tasks)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for keep, scene in results:
        if keep:
            kept.append(scene)
        else:
            dropped += 1

    if dropped:
        logger.info("video_announce: dropped scenes with empty ocr_text dropped=%d", dropped)
        payload_obj["scenes"] = kept
        intro = payload_obj.get("intro")
        if isinstance(intro, dict):
            intro["count"] = len(kept)

    return json.dumps(payload_obj, ensure_ascii=False, indent=2)
