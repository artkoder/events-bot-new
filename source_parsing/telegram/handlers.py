import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from db import Database
from event_utils import strip_city_from_address
from models import EventSource, TelegramSource, TelegramScannedMessage
from smart_event_update import EventCandidate, PosterCandidate, smart_event_update

logger = logging.getLogger(__name__)

_LONG_EVENT_TYPES = {"выставка", "ярмарка", "exhibition", "fair"}
_EVENT_TYPE_ALIASES = {
    "exhibition": "выставка",
    "fair": "ярмарка",
}


@dataclass(slots=True)
class TelegramMonitorReport:
    run_id: str | None = None
    generated_at: str | None = None
    sources_total: int = 0
    messages_scanned: int = 0
    messages_skipped: int = 0
    messages_with_events: int = 0
    events_extracted: int = 0
    events_created: int = 0
    events_merged: int = 0
    events_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    created_events: list["TelegramMonitorEventInfo"] = field(default_factory=list)
    merged_events: list["TelegramMonitorEventInfo"] = field(default_factory=list)


@dataclass(slots=True)
class TelegramMonitorEventInfo:
    event_id: int
    title: str
    date: str | None
    time: str | None
    source_link: str | None
    telegraph_url: str | None
    ics_url: str | None
    log_cmd: str | None
    fact_stats: dict[str, int] | None
    photo_count: int | None
    added_posters: int | None
    metrics: dict[str, Any] | None
    source_excerpt: str | None


def _event_telegraph_url(event) -> str | None:
    url = getattr(event, "telegraph_url", None)
    if url:
        return url
    path = getattr(event, "telegraph_path", None)
    if path:
        return f"https://telegra.ph/{path.lstrip('/')}"
    return None


def _build_excerpt(text: str | None, *, max_len: int = 160) -> str | None:
    if not text:
        return None
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _parse_iso_date(value: str | None) -> date | None:
    raw = str(value or "").split("..", 1)[0].strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _candidate_is_long_event(candidate: EventCandidate) -> bool:
    event_type = str(candidate.event_type or "").strip().casefold()
    return bool(event_type and event_type in _LONG_EVENT_TYPES)


def _normalize_event_type(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return _EVENT_TYPE_ALIASES.get(raw.casefold(), raw)


def _should_skip_past_event_candidate(candidate: EventCandidate, *, today: date | None = None) -> bool:
    """Past single-date events are skipped; long-running events are valid until end_date."""
    now = today or date.today()
    start_date = _parse_iso_date(candidate.date)
    if start_date is None or start_date >= now:
        return False
    if _candidate_is_long_event(candidate):
        end_date = _parse_iso_date(candidate.end_date)
        # Ongoing long events are valid even if start date is in the past.
        if end_date is None or end_date >= now:
            return False
    return True


async def _load_latest_source_fact_stats(
    db: Database,
    *,
    event_id: int,
    source_url: str | None,
) -> dict[str, int] | None:
    """Return per-status fact counts for the most recent log batch for (event_id, source_url)."""
    if not source_url:
        return None
    from sqlalchemy import func
    from models import EventSourceFact

    async with db.get_session() as session:
        source = (
            await session.execute(
                select(EventSource).where(
                    EventSource.event_id == int(event_id),
                    EventSource.source_url == str(source_url),
                )
            )
        ).scalar_one_or_none()
        if not source:
            return None
        ts = await session.scalar(
            select(func.max(EventSourceFact.created_at)).where(
                EventSourceFact.event_id == int(event_id),
                EventSourceFact.source_id == int(source.id),
            )
        )
        if not ts:
            return None
        rows = (
            await session.execute(
                select(EventSourceFact.status, func.count())
                .where(
                    EventSourceFact.event_id == int(event_id),
                    EventSourceFact.source_id == int(source.id),
                    EventSourceFact.created_at == ts,
                )
                .group_by(EventSourceFact.status)
            )
        ).all()
    out: dict[str, int] = {}
    for status, cnt in rows:
        key = (str(status or "added")).strip().lower() or "added"
        out[key] = int(cnt or 0)
    return out or None


async def _build_event_info(
    db: Database,
    *,
    event_id: int | None,
    source_link: str | None,
    source_text: str | None,
    metrics: dict[str, Any] | None = None,
    added_posters: int | None = None,
) -> TelegramMonitorEventInfo | None:
    if not event_id:
        return None
    from models import Event

    async with db.get_session() as session:
        event = await session.get(Event, event_id)
    if not event:
        return None
    fact_stats = await _load_latest_source_fact_stats(
        db,
        event_id=int(event_id),
        source_url=source_link,
    )
    photo_count = None
    try:
        raw = getattr(event, "photo_count", None)
        if raw is None:
            urls = getattr(event, "photo_urls", None)
            if isinstance(urls, list):
                raw = len([u for u in urls if (str(u or "").strip())])
        if raw is not None:
            photo_count = int(raw or 0)
    except Exception:
        photo_count = None
    return TelegramMonitorEventInfo(
        event_id=event_id,
        title=getattr(event, "title", "") or "",
        date=getattr(event, "date", None),
        time=getattr(event, "time", None),
        source_link=source_link,
        telegraph_url=_event_telegraph_url(event),
        ics_url=getattr(event, "ics_url", None),
        log_cmd=f"/log {event_id}",
        fact_stats=fact_stats,
        photo_count=photo_count,
        added_posters=added_posters,
        metrics=metrics,
        source_excerpt=_build_excerpt(source_text),
    )


async def refresh_telegram_monitor_event_info(
    db: Database,
    info: TelegramMonitorEventInfo,
) -> TelegramMonitorEventInfo:
    """Refresh Telegraph/ICS URLs + latest per-source fact stats for an existing report item."""
    from models import Event

    if not info or not getattr(info, "event_id", None):
        return info
    async with db.get_session() as session:
        event = await session.get(Event, int(info.event_id))
    if event:
        info.title = getattr(event, "title", "") or info.title
        info.date = getattr(event, "date", None)
        info.time = getattr(event, "time", None)
        info.telegraph_url = _event_telegraph_url(event)
        info.ics_url = getattr(event, "ics_url", None)
        try:
            raw = getattr(event, "photo_count", None)
            if raw is None:
                urls = getattr(event, "photo_urls", None)
                if isinstance(urls, list):
                    raw = len([u for u in urls if (str(u or "").strip())])
            if raw is not None:
                info.photo_count = int(raw or 0)
        except Exception:
            pass
    info.log_cmd = f"/log {int(info.event_id)}"
    try:
        info.fact_stats = await _load_latest_source_fact_stats(
            db,
            event_id=int(info.event_id),
            source_url=info.source_link,
        )
    except Exception:
        info.fact_stats = info.fact_stats
    return info


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _clean_url(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if not re.match(r"https?://", raw):
        return None
    if re.match(r"^https?://t\.me/addlist/", raw):
        return None
    return raw


def _parse_tg_source_url(value: str | None) -> tuple[str | None, int | None]:
    raw = _clean_url(value)
    if not raw:
        return None, None
    m = re.search(r"t\.me/([^/]+)/([0-9]+)", raw, flags=re.IGNORECASE)
    if not m:
        return None, None
    username = str(m.group(1) or "").strip() or None
    message_id = _to_int(m.group(2))
    return username, message_id


async def _attach_linked_sources(
    db: Database,
    *,
    event_id: int | None,
    linked_urls: list[str] | None,
    trust_level: str | None,
) -> int:
    if not event_id:
        return 0

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in linked_urls or []:
        url = _clean_url(raw)
        if not url:
            continue
        key = url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        normalized.append(url)
    if not normalized:
        return 0

    added = 0
    async with db.get_session() as session:
        for url in normalized:
            exists = (
                await session.execute(
                    select(EventSource.id).where(
                        EventSource.event_id == int(event_id),
                        EventSource.source_url == url,
                    )
                )
            ).scalar_one_or_none()
            if exists:
                continue
            username, message_id = _parse_tg_source_url(url)
            session.add(
                EventSource(
                    event_id=int(event_id),
                    source_type="telegram",
                    source_url=url,
                    source_chat_username=username,
                    source_message_id=message_id,
                    trust_level=trust_level,
                )
            )
            added += 1
        if added:
            await session.commit()
    return added


def _norm_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _location_matches(a: str | None, b: str | None) -> bool:
    na = _norm_space(a)
    nb = _norm_space(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


_SCHED_LINE_RE = re.compile(r"(^|\s)(\d{1,2})[./](\d{1,2})\s*\|", re.IGNORECASE)
_SCHED_LINE_START_RE = re.compile(r"^\s*\d{1,2}[./]\d{1,2}\s*\|", re.IGNORECASE)
_TIME_TOKEN_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _date_tokens_from_iso(iso_date: str | None) -> list[str]:
    if not iso_date:
        return []
    raw = str(iso_date).split("..", 1)[0].strip()
    if not raw:
        return []
    try:
        y, m, d = raw.split("-", 2)
        mm = int(m)
        dd = int(d)
    except Exception:
        return []
    return [
        f"{dd:02d}.{mm:02d}",
        f"{dd}.{mm}",
        f"{dd:02d}/{mm:02d}",
        f"{dd}/{mm}",
    ]


def _extract_time_tokens(text: str | None) -> list[str]:
    raw = str(text or "")
    out: list[str] = []
    seen: set[str] = set()
    for hh, mm in _TIME_TOKEN_RE.findall(raw):
        token = f"{int(hh):02d}:{mm}"
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _infer_time_from_event_text(text: str | None, *, event_date: str | None) -> str | None:
    """Best-effort time fallback for per-event schedule snippets.

    Telegram monitor may extract date/title from OCR but occasionally leave `time`
    empty even when poster text contains a single explicit `HH:MM` for that date.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    date_tokens = _date_tokens_from_iso(event_date)
    if date_tokens and lines:
        dated_times: list[str] = []
        for ln in lines:
            low = ln.lower()
            if any(tok in low for tok in date_tokens):
                dated_times.extend(_extract_time_tokens(ln))
        uniq = sorted(set(dated_times))
        if len(uniq) == 1:
            return uniq[0]

    uniq_all = sorted(set(_extract_time_tokens(raw)))
    if len(uniq_all) == 1:
        return uniq_all[0]
    return None


def _filter_schedule_source_text(text: str, *, event_date: str | None, event_title: str | None) -> str:
    """Reduce multi-event schedule posts to the segment relevant to this event.

    Telegram repertoire posts often contain many lines like `DD.MM | Title`. Passing
    the whole post into Smart Update can leak other event titles into the description.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # If this doesn't look like a schedule (multiple `DD.MM |` anchors), keep as-is.
    if len(_SCHED_LINE_RE.findall(raw)) < 2:
        return raw

    # Normalize and ensure schedule anchors appear on separate lines.
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+(?=\d{1,2}[./]\d{1,2}\s*\|)", "\n", normalized)
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return raw

    tokens = _date_tokens_from_iso(event_date)
    idx: int | None = None
    if tokens:
        for i, ln in enumerate(lines):
            if any(tok in ln for tok in tokens):
                idx = i
                break
    if idx is not None:
        out = [lines[idx]]
        for j in range(idx + 1, len(lines)):
            if _SCHED_LINE_START_RE.search(lines[j]):
                break
            out.append(lines[j])
        filtered = "\n".join(out).strip()
        return filtered or raw

    # Fallback: try to keep only lines that mention the title.
    title_norm = _norm_space(event_title)
    words = [w for w in title_norm.split() if len(w) >= 4]
    if words:
        matched = [ln for ln in lines if any(w in _norm_space(ln) for w in words)]
        if matched:
            return "\n".join(matched[:3]).strip()

    return "\n".join(lines[:3]).strip() or raw


def _norm_match(s: str | None) -> str:
    raw = (s or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("ё", "е")
    raw = re.sub(r"[^\w\s:./-]+", " ", raw, flags=re.U)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _title_tokens(title: str | None) -> list[str]:
    t = _norm_match(title)
    if not t:
        return []
    stop = {"и", "в", "на", "по", "из", "от", "до", "для", "или", "о", "об", "про", "со", "к"}
    out: list[str] = []
    for w in t.split():
        if len(w) < 4:
            continue
        if w in stop:
            continue
        out.append(w)
    return out


def _looks_like_generic_schedule_poster(ocr_text: str) -> bool:
    t = _norm_match(ocr_text)
    if not t:
        return False
    if "неделя" in t and "театре" in t:
        return True
    if len(re.findall(r"\b\d{1,2}[./]\d{1,2}\b", t)) >= 3:
        return True
    return False


def _poster_match_score(
    poster: PosterCandidate,
    *,
    event_title: str | None,
    event_date: str | None,
    event_time: str | None,
) -> int:
    ocr = _norm_match(poster.ocr_text) or _norm_match(poster.ocr_title)
    if not ocr:
        return 0
    if _looks_like_generic_schedule_poster(ocr):
        return 0

    score = 0
    for tok in _date_tokens_from_iso(event_date):
        if tok and tok in ocr:
            score += 4
            break
    tm = _norm_match(event_time)
    if tm and re.search(rf"\b{re.escape(tm)}\b", ocr):
        score += 2

    tokens = _title_tokens(event_title)
    if tokens:
        hit = sum(1 for w in tokens if w in ocr)
        if hit:
            # Title hits should dominate for schedule posts; dates/times are often shared.
            score += 6
            if any(len(w) >= 7 and w in ocr for w in tokens):
                score += 2
            if hit >= max(2, int(len(tokens) * 0.6)):
                score += 2
    return score


def _filter_posters_for_event(
    posters: list[PosterCandidate],
    *,
    event_title: str | None,
    event_date: str | None,
    event_time: str | None,
) -> list[PosterCandidate]:
    if not posters:
        return []
    tokens = _title_tokens(event_title)
    has_many = len(posters) >= 2
    scored: list[tuple[int, int, PosterCandidate]] = []
    for i, p in enumerate(posters):
        scored.append(
            (
                _poster_match_score(
                    p, event_title=event_title, event_date=event_date, event_time=event_time
                ),
                i,
                p,
            )
        )
    scored.sort(key=lambda x: (-x[0], x[1]))
    kept: list[PosterCandidate] = []
    for s, _i, p in scored:
        if s < 6:
            continue
        if has_many and tokens:
            # In multi-poster schedule posts, require at least one solid title hit
            # (long token or multiple hits). Date/time alone is too weak.
            ocr = _norm_match(p.ocr_text) or _norm_match(p.ocr_title) or ""
            hits = [w for w in tokens if w in ocr]
            long_hit = any(len(w) >= 7 for w in hits)
            if not (long_hit or len(hits) >= 2):
                continue
        kept.append(p)
    return kept[:3]


async def _get_or_create_source(db: Database, username: str) -> TelegramSource:
    async with db.get_session() as session:
        result = await session.execute(
            select(TelegramSource).where(TelegramSource.username == username)
        )
        source = result.scalar_one_or_none()
        if source:
            return source
        source = TelegramSource(username=username, enabled=True)
        session.add(source)
        await session.commit()
        await session.refresh(source)
        return source


async def _is_message_scanned(
    db: Database, source_id: int, message_id: int
) -> TelegramScannedMessage | None:
    async with db.get_session() as session:
        return await session.get(TelegramScannedMessage, (source_id, message_id))


async def _mark_message_scanned(
    db: Database,
    *,
    source_id: int,
    message_id: int,
    message_date: datetime | None,
    status: str,
    events_extracted: int,
    events_imported: int,
    error: str | None,
) -> None:
    for attempt in range(1, 8):
        try:
            async with db.get_session() as session:
                row = await session.get(TelegramScannedMessage, (source_id, message_id))
                if row:
                    row.message_date = message_date or row.message_date
                    row.processed_at = datetime.now(timezone.utc)
                    row.status = status
                    row.events_extracted = events_extracted
                    row.events_imported = events_imported
                    row.error = error
                else:
                    row = TelegramScannedMessage(
                        source_id=source_id,
                        message_id=message_id,
                        message_date=message_date,
                        processed_at=datetime.now(timezone.utc),
                        status=status,
                        events_extracted=events_extracted,
                        events_imported=events_imported,
                        error=error,
                    )
                    session.add(row)
                await session.commit()
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt >= 7:
                raise
            await asyncio.sleep(0.15 * attempt)


async def _update_source_scan_meta(
    db: Database, source_id: int, message_id: int | None
) -> None:
    if message_id is None:
        return
    for attempt in range(1, 8):
        try:
            async with db.get_session() as session:
                source = await session.get(TelegramSource, source_id)
                if not source:
                    return
                if (
                    source.last_scanned_message_id is None
                    or message_id > source.last_scanned_message_id
                ):
                    source.last_scanned_message_id = message_id
                source.last_scan_at = datetime.now(timezone.utc)
                session.add(source)
                await session.commit()
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt >= 7:
                raise
            await asyncio.sleep(0.15 * attempt)


def _build_candidate(
    source: TelegramSource,
    message: dict[str, Any],
    event_data: dict[str, Any],
) -> EventCandidate:
    username = str(message.get("source_username") or "").strip()
    message_id = _to_int(message.get("message_id"))
    source_link = message.get("source_link")
    if not source_link and username and message_id:
        source_link = f"https://t.me/{username}/{message_id}"
    title = event_data.get("title")
    date_raw = event_data.get("date")
    time_raw = event_data.get("time") or ""
    end_date = event_data.get("end_date")
    extracted_location = event_data.get("location_name")
    location_name = extracted_location or source.default_location
    location_address = event_data.get("location_address")
    if extracted_location and source.default_location and not _location_matches(
        extracted_location, source.default_location
    ):
        logger.warning(
            "telegram: location mismatch for @%s msg=%s extracted=%s default=%s",
            username,
            message_id,
            extracted_location,
            source.default_location,
        )
        location_name = source.default_location
        location_address = None
    if not location_name and location_address:
        location_name, location_address = location_address, None
    city = event_data.get("city") or "Калининград"
    if location_address:
        location_address = strip_city_from_address(location_address, city)
    ticket_link = _clean_url(event_data.get("ticket_link")) or _clean_url(source.default_ticket_link)
    ticket_price_min = _to_int(event_data.get("ticket_price_min"))
    ticket_price_max = _to_int(event_data.get("ticket_price_max"))
    ticket_status = event_data.get("ticket_status")
    raw_excerpt = event_data.get("raw_excerpt")
    event_type = _normalize_event_type(event_data.get("event_type"))
    emoji = event_data.get("emoji")
    is_free = event_data.get("is_free")
    pushkin_card = event_data.get("pushkin_card")
    search_digest = event_data.get("search_digest") or event_data.get("search_description")

    # Prefer per-event posters/source_text if the monitor provided them (assignment),
    # but keep the *message-level* poster scope for cleanup of previously-attached
    # "foreign" posters from the same album/schedule post.
    message_posters_payload = message.get("posters") or []
    assigned_posters_payload = event_data.get("posters") or []
    posters_payload = assigned_posters_payload or message_posters_payload or []
    event_source_text = event_data.get("source_text") or event_data.get("description") or ""
    event_source_text_raw = str(event_source_text or "")
    message_text = message.get("text") or ""
    event_source_text_s = str(event_source_text or "").strip()
    message_text_s = str(message_text or "").strip()
    # Schedule monitor may provide per-event source_text as a short header
    # like "12.02 | Фигаро". Prefer the richer message text and then filter it
    # down to the event segment to keep factual lines for this event.
    if message_text_s and (
        not event_source_text_s
        or len(event_source_text_s) < 80
        or (
            len(_SCHED_LINE_RE.findall(message_text_s)) >= 2
            and len(message_text_s) > len(event_source_text_s) + 60
        )
    ):
        event_source_text = message_text_s
    else:
        event_source_text = event_source_text_s
    event_source_text = _filter_schedule_source_text(
        str(event_source_text),
        event_date=str(date_raw).strip() if date_raw else None,
        event_title=str(title).strip() if title else None,
    )
    if not str(time_raw or "").strip():
        time_probe_text = str(event_source_text or "")
        raw_probe = event_source_text_raw.strip()
        if raw_probe and raw_probe not in time_probe_text:
            time_probe_text = f"{time_probe_text}\n{raw_probe}".strip()
        inferred_time = _infer_time_from_event_text(
            time_probe_text,
            event_date=str(date_raw).strip() if date_raw else None,
        )
        if inferred_time:
            time_raw = inferred_time
            logger.info(
                "telegram: inferred missing event time from source text source=%s message_id=%s title=%r time=%s",
                username,
                message_id,
                title,
                inferred_time,
            )

    posters: list[PosterCandidate] = []
    seen_hashes: set[str] = set()
    scope_hashes: set[str] = set()

    def _payload_to_posters(payload: list[dict[str, Any]]) -> list[PosterCandidate]:
        out: list[PosterCandidate] = []
        local_seen: set[str] = set()
        for item in payload:
            sha = (item or {}).get("sha256")
            if sha and isinstance(sha, str):
                sha = sha.strip()
            if sha and sha in local_seen:
                continue
            if sha:
                local_seen.add(sha)
            out.append(
                PosterCandidate(
                    catbox_url=(item or {}).get("catbox_url"),
                    supabase_url=(item or {}).get("supabase_url"),
                    supabase_path=(item or {}).get("supabase_path"),
                    sha256=sha,
                    phash=(item or {}).get("phash"),
                    ocr_text=(item or {}).get("ocr_text"),
                    ocr_title=(item or {}).get("ocr_title"),
                )
            )
        return out

    # Scope hashes: use the message-level posters when available (album posts), otherwise
    # fall back to whatever posters_payload we have.
    scope_payload = message_posters_payload or posters_payload
    for item in scope_payload:
        sha = (item or {}).get("sha256")
        if sha and isinstance(sha, str):
            sha = sha.strip()
            if sha:
                scope_hashes.add(sha)
    for item in posters_payload:
        sha = item.get("sha256")
        if sha and sha in seen_hashes:
            continue
        if sha:
            seen_hashes.add(sha)
        posters.append(
            PosterCandidate(
                catbox_url=item.get("catbox_url"),
                supabase_url=item.get("supabase_url"),
                supabase_path=item.get("supabase_path"),
                sha256=sha,
                phash=item.get("phash"),
                ocr_text=item.get("ocr_text"),
                ocr_title=item.get("ocr_title"),
            )
        )

    posters = _filter_posters_for_event(
        posters,
        event_title=str(title).strip() if title else None,
        event_date=str(date_raw).strip() if date_raw else None,
        event_time=str(time_raw).strip() if time_raw else None,
    )

    if not posters and assigned_posters_payload:
        # Telegram monitor may already map posters to concrete event cards.
        # Keep this event-level assignment as a fallback when strict OCR matching
        # is inconclusive (e.g., missing event time in message).
        assigned = _payload_to_posters(assigned_posters_payload)
        relaxed: list[PosterCandidate] = []
        for p in assigned:
            score = _poster_match_score(
                p,
                event_title=str(title).strip() if title else None,
                event_date=str(date_raw).strip() if date_raw else None,
                event_time=str(time_raw).strip() if time_raw else None,
            )
            if score >= 4:
                relaxed.append(p)
        if not relaxed:
            for p in assigned:
                ocr = _norm_match(p.ocr_text) or _norm_match(p.ocr_title)
                if ocr and _looks_like_generic_schedule_poster(ocr):
                    continue
                relaxed.append(p)
        posters = relaxed[:3]
        if posters:
            logger.info(
                "telegram: posters fallback kept assigned posters for %s/%s title=%r count=%s",
                username,
                message_id,
                title,
                len(posters),
            )

    return EventCandidate(
        source_type="telegram",
        source_url=source_link or None,
        source_text=event_source_text,
        title=str(title).strip() if title else None,
        date=str(date_raw).strip() if date_raw else None,
        time=str(time_raw).strip() if time_raw else "",
        end_date=str(end_date).strip() if end_date else None,
        festival=event_data.get("festival"),
        location_name=str(location_name).strip() if location_name else None,
        location_address=str(location_address).strip() if location_address else None,
        city=str(city).strip() if city else None,
        ticket_link=ticket_link,
        ticket_price_min=ticket_price_min,
        ticket_price_max=ticket_price_max,
        ticket_status=str(ticket_status).strip() if ticket_status else None,
        event_type=str(event_type).strip() if event_type else None,
        emoji=str(emoji).strip() if emoji else None,
        is_free=is_free if isinstance(is_free, bool) else None,
        pushkin_card=pushkin_card if isinstance(pushkin_card, bool) else None,
        search_digest=str(search_digest).strip() if search_digest else None,
        raw_excerpt=str(raw_excerpt).strip() if raw_excerpt else None,
        posters=posters,
        poster_scope_hashes=sorted(scope_hashes or seen_hashes),
        source_chat_username=username or None,
        source_chat_id=_to_int(message.get("source_chat_id")),
        source_message_id=message_id,
        trust_level=source.trust_level,
        metrics=message.get("metrics"),
    )


async def process_telegram_results(
    results_path: str | Path,
    db: Database,
) -> TelegramMonitorReport:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"telegram_results.json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    report = TelegramMonitorReport(
        run_id=data.get("run_id"),
        generated_at=data.get("generated_at"),
    )

    stats = data.get("stats") or {}
    report.sources_total = int(stats.get("sources_total") or 0)
    report.messages_scanned = int(stats.get("messages_scanned") or 0)
    report.messages_with_events = int(stats.get("messages_with_events") or 0)
    report.events_extracted = int(stats.get("events_extracted") or 0)
    logger.info(
        "tg_monitor.results run_id=%s generated_at=%s sources_total=%d messages_scanned=%d messages_with_events=%d events_extracted=%d",
        report.run_id,
        report.generated_at,
        report.sources_total,
        report.messages_scanned,
        report.messages_with_events,
        report.events_extracted,
    )

    for message in data.get("messages") or []:
        username = str(message.get("source_username") or "").strip()
        if not username:
            continue
        message_id = _to_int(message.get("message_id"))
        if message_id is None:
            report.errors.append(f"missing message_id for source {username}")
            continue
        source_link = message.get("source_link")
        if not source_link and username and message_id:
            source_link = f"https://t.me/{username}/{message_id}"
        source_text = message.get("text") or ""
        source = await _get_or_create_source(db, username)
        if not source.enabled:
            report.messages_skipped += 1
            logger.info(
                "tg_monitor.message skip reason=source_disabled run_id=%s source=%s message_id=%s",
                report.run_id,
                username,
                message_id,
            )
            await _mark_message_scanned(
                db,
                source_id=source.id,
                message_id=message_id,
                message_date=_parse_datetime(message.get("message_date")),
                status="skipped",
                events_extracted=0,
                events_imported=0,
                error="source_disabled",
            )
            continue

        existing = await _is_message_scanned(db, source.id, message_id)
        if existing and existing.status == "done":
            report.messages_skipped += 1
            logger.info(
                "tg_monitor.message skip reason=already_scanned run_id=%s source=%s message_id=%s",
                report.run_id,
                username,
                message_id,
            )
            continue

        events = message.get("events") or []

        # If the monitor didn't pre-assign posters per event, try to do it here.
        # Multi-event posts (repertoire/schedule) often contain several posters;
        # attaching all posters to every extracted event produces "foreign" posters
        # on Telegraph pages.
        try:
            posters_payload = message.get("posters") or []
            has_event_level_posters = any(
                isinstance(ev, dict) and bool(ev.get("posters")) for ev in events
            )
            if posters_payload and len(events) >= 2 and not has_event_level_posters:
                poster_candidates: list[PosterCandidate] = []
                for item in posters_payload:
                    if not isinstance(item, dict):
                        continue
                    poster_candidates.append(
                        PosterCandidate(
                            catbox_url=item.get("catbox_url"),
                            supabase_url=item.get("supabase_url"),
                            supabase_path=item.get("supabase_path"),
                            sha256=item.get("sha256"),
                            phash=item.get("phash"),
                            ocr_text=item.get("ocr_text"),
                            ocr_title=item.get("ocr_title"),
                        )
                    )

                assignments: list[list[dict[str, Any]]] = [[] for _ in events]
                for idx, poster in enumerate(poster_candidates):
                    src_item = posters_payload[idx]
                    best_i: int | None = None
                    best_score = 0
                    for i, ev in enumerate(events):
                        if not isinstance(ev, dict):
                            continue
                        score = _poster_match_score(
                            poster,
                            event_title=str(ev.get("title") or "") or None,
                            event_date=str(ev.get("date") or "") or None,
                            event_time=str(ev.get("time") or "") or None,
                        )
                        if score > best_score:
                            best_score = score
                            best_i = i
                    # Use the same minimal threshold as _filter_posters_for_event.
                    if best_i is not None and best_score >= 6:
                        assignments[best_i].append(src_item)

                # OCR can be empty for text-heavy posters. In schedule-like posts where
                # monitor extracted N events and there are exactly N posters, prefer
                # deterministic positional mapping over dropping all posters.
                assigned_total = sum(len(items) for items in assignments)
                schedule_like = len(_SCHED_LINE_RE.findall(str(message.get("text") or ""))) >= 2
                if (
                    assigned_total == 0
                    and schedule_like
                    and len(events) >= 2
                    and len(posters_payload) == len(events)
                ):
                    for i, ev in enumerate(events):
                        if isinstance(ev, dict):
                            ev["posters"] = [posters_payload[i]]
                    logger.info(
                        "telegram: posters fallback positional mapping source=%s message_id=%s count=%s",
                        username,
                        message_id,
                        len(events),
                    )

                for i, ev in enumerate(events):
                    if not isinstance(ev, dict):
                        continue
                    if assignments[i]:
                        ev["posters"] = assignments[i]
        except Exception:
            logger.warning("telegram: poster assignment failed", exc_info=True)
        events_extracted = len(events)
        events_imported = 0
        logger.info(
            "tg_monitor.message start run_id=%s source=%s message_id=%s events=%d",
            report.run_id,
            username,
            message_id,
            events_extracted,
        )

        for event_data in events:
            try:
                candidate = _build_candidate(source, message, event_data)
                # Telegram monitoring extracts multiple events from schedule posts.
                # Skip only truly past events: long-running events (exhibitions/fairs)
                # remain valid while end_date is current/future.
                if _should_skip_past_event_candidate(candidate):
                    report.events_skipped += 1
                    logger.info(
                        "tg_monitor.event skip reason=past_event source=%s message_id=%s title=%s date=%s end_date=%s event_type=%s",
                        username,
                        message_id,
                        (candidate.title or "")[:80],
                        candidate.date,
                        candidate.end_date,
                        candidate.event_type,
                    )
                    continue
                # Telegram monitoring should not enqueue VK publishing jobs: they are irrelevant
                # for the monitoring workflow and slow down local/E2E environments.
                result = await smart_event_update(
                    db,
                    candidate,
                    check_source_url=False,
                    schedule_kwargs={"skip_vk_sync": True},
                )
                linked_added = 0
                if result.event_id:
                    linked_added = await _attach_linked_sources(
                        db,
                        event_id=result.event_id,
                        linked_urls=list(event_data.get("linked_source_urls") or []),
                        trust_level=source.trust_level,
                    )
                if result.status == "created":
                    report.events_created += 1
                    events_imported += 1
                    info = await _build_event_info(
                        db,
                        event_id=result.event_id,
                        source_link=source_link,
                        source_text=source_text,
                        metrics=message.get("metrics") if isinstance(message.get("metrics"), dict) else None,
                        added_posters=int(getattr(result, "added_posters", 0) or 0),
                    )
                    if info:
                        report.created_events.append(info)
                elif result.status == "merged":
                    report.events_merged += 1
                    events_imported += 1
                    info = await _build_event_info(
                        db,
                        event_id=result.event_id,
                        source_link=source_link,
                        source_text=source_text,
                        metrics=message.get("metrics") if isinstance(message.get("metrics"), dict) else None,
                        added_posters=int(getattr(result, "added_posters", 0) or 0),
                    )
                    if info:
                        report.merged_events.append(info)
                elif result.status.startswith("skipped"):
                    report.events_skipped += 1
                logger.info(
                    "tg_monitor.event result=%s event_id=%s source=%s message_id=%s title=%s linked_added=%s",
                    result.status,
                    result.event_id,
                    username,
                    message_id,
                    (candidate.title or "")[:80],
                    linked_added,
                )
            except Exception as exc:
                report.errors.append(f"{username}/{message_id}: {exc}")
                logger.exception("telegram_results: smart update failed")
        logger.info(
            "tg_monitor.message done run_id=%s source=%s message_id=%s imported=%d",
            report.run_id,
            username,
            message_id,
            events_imported,
        )

        await _mark_message_scanned(
            db,
            source_id=source.id,
            message_id=message_id,
            message_date=_parse_datetime(message.get("message_date")),
            status="done",
            events_extracted=events_extracted,
            events_imported=events_imported,
            error=None,
        )
        await _update_source_scan_meta(db, source.id, message_id)

    return report
