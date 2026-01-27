import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from db import Database
from event_utils import strip_city_from_address
from models import TelegramSource, TelegramScannedMessage
from smart_event_update import EventCandidate, PosterCandidate, smart_event_update

logger = logging.getLogger(__name__)


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


async def _update_source_scan_meta(
    db: Database, source_id: int, message_id: int | None
) -> None:
    if message_id is None:
        return
    async with db.get_session() as session:
        source = await session.get(TelegramSource, source_id)
        if not source:
            return
        if source.last_scanned_message_id is None or message_id > source.last_scanned_message_id:
            source.last_scanned_message_id = message_id
        source.last_scan_at = datetime.now(timezone.utc)
        session.add(source)
        await session.commit()


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
    location_name = event_data.get("location_name") or source.default_location
    location_address = event_data.get("location_address")
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
    event_type = event_data.get("event_type")
    emoji = event_data.get("emoji")
    is_free = event_data.get("is_free")
    pushkin_card = event_data.get("pushkin_card")
    search_digest = event_data.get("search_digest")

    posters: list[PosterCandidate] = []
    seen_hashes: set[str] = set()
    for item in message.get("posters") or []:
        sha = item.get("sha256")
        if sha and sha in seen_hashes:
            continue
        if sha:
            seen_hashes.add(sha)
        posters.append(
            PosterCandidate(
                catbox_url=item.get("catbox_url"),
                sha256=sha,
                phash=item.get("phash"),
                ocr_text=item.get("ocr_text"),
                ocr_title=item.get("ocr_title"),
            )
        )

    return EventCandidate(
        source_type="telegram",
        source_url=source_link or None,
        source_text=message.get("text") or "",
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
                result = await smart_event_update(db, candidate, check_source_url=False)
                if result.status == "created":
                    report.events_created += 1
                    events_imported += 1
                elif result.status == "merged":
                    report.events_merged += 1
                    events_imported += 1
                elif result.status.startswith("skipped"):
                    report.events_skipped += 1
                logger.info(
                    "tg_monitor.event result=%s event_id=%s source=%s message_id=%s title=%s",
                    result.status,
                    result.event_id,
                    username,
                    message_id,
                    (candidate.title or "")[:80],
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
