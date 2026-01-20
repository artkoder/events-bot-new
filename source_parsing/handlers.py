"""Handlers for source parsing operations."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from aiogram import Bot

from db import Database
from source_parsing.kaggle_runner import run_kaggle_kernel
from source_parsing.parser import (
    TheatreEvent,
    parse_theatre_json,
    normalize_location_name,
    find_existing_event,
    should_update_event,
    find_linked_events,
    limit_photos_for_source,
)
from poster_media import PosterMedia, process_media

logger = logging.getLogger(__name__)

PARSE_EVENT_TIMEOUT_SECONDS = int(os.getenv("SOURCE_PARSING_EVENT_TIMEOUT_SECONDS", "180"))
SOURCE_PARSING_OCR_TIMEOUT_SECONDS = int(os.getenv("SOURCE_PARSING_OCR_TIMEOUT_SECONDS", "60"))
SOURCE_PARSING_DIAG_TITLE = os.getenv("SOURCE_PARSING_DIAG_TITLE", "джотто").strip().lower()
SOURCE_PARSING_DISABLE_OCR_SOURCES = {
    s.strip().lower()
    for s in os.getenv("SOURCE_PARSING_DISABLE_OCR_SOURCES", "tretyakov").split(",")
    if s.strip()
}

# Delay between adding events to avoid overloading the system
EVENT_ADD_DELAY_SECONDS = 5  # Delay for Telegraph creation

# TEMPORARY: Limit events for debugging (set to None to disable)
DEBUG_MAX_EVENTS = None


@dataclass
class SourceParsingStats:
    """Statistics for a source parsing run."""
    source: str
    total_received: int = 0
    new_added: int = 0
    ticket_updated: int = 0
    already_exists: int = 0
    failed: int = 0
    skipped: int = 0  # Events that already existed and were updated (not new)
    added_event_ids: list[int] = field(default_factory=list)
    updated_event_ids: list[int] = field(default_factory=list)  # For displaying Telegraph links


@dataclass
class SourceParsingResult:
    """Complete result of a source parsing run."""
    stats_by_source: dict[str, SourceParsingStats] = field(default_factory=dict)
    total_events: int = 0
    kernel_duration: float = 0.0
    processing_duration: float = 0.0
    log_file_path: str = ""
    json_file_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    chat_id: int | None = None  # For progress messages
    added_events: list["AddedEventInfo"] = field(default_factory=list)
    updated_events: list["UpdatedEventInfo"] = field(default_factory=list)


@dataclass
class AddedEventInfo:
    """Newly added event with Telegraph link for reporting."""
    event_id: int
    title: str
    telegraph_url: str
    date: str | None
    time: str | None
    source: str | None


@dataclass
class UpdatedEventInfo:
    """Updated event (ticket status change) with Telegraph link for reporting."""
    event_id: int
    title: str
    telegraph_url: str
    date: str | None
    time: str | None
    source: str | None
    update_type: str  # 'ticket_status', 'full_update'


def _event_telegraph_url(event) -> str | None:
    url = getattr(event, "telegraph_url", None)
    if url:
        return url
    path = getattr(event, "telegraph_path", None)
    if path:
        return f"https://telegra.ph/{path.lstrip('/')}"
    return None


async def _ensure_telegraph_url(db: Database, event_id: int) -> str | None:
    import sys

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if not main_mod or not hasattr(main_mod, "update_telegraph_event_page"):
        logger.warning("source_parsing: main module missing for telegraph build")
        return None
    try:
        return await main_mod.update_telegraph_event_page(event_id, db, None)
    except Exception as e:
        logger.warning(
            "source_parsing: telegraph build failed event_id=%d error=%s",
            event_id,
            e,
        )
        return None


async def build_added_event_info(
    db: Database,
    event_id: int,
    source: str | None,
) -> AddedEventInfo | None:
    from models import Event

    async with db.get_session() as session:
        event = await session.get(Event, event_id)

    if not event:
        return None

    url = _event_telegraph_url(event)
    if not url:
        await _ensure_telegraph_url(db, event_id)
        async with db.get_session() as session:
            event = await session.get(Event, event_id)
        if not event:
            return None
        url = _event_telegraph_url(event)

    if not url:
        logger.warning(
            "source_parsing: telegraph url missing after build event_id=%d",
            event_id,
        )

    return AddedEventInfo(
        event_id=event_id,
        title=event.title or "",
        telegraph_url=url or "",
        date=getattr(event, "date", None),
        time=getattr(event, "time", None),
        source=source,
    )


async def build_updated_event_info(
    db: Database,
    event_id: int,
    source: str | None,
    update_type: str,
) -> UpdatedEventInfo | None:
    """Build UpdatedEventInfo for a modified event."""
    from models import Event

    async with db.get_session() as session:
        event = await session.get(Event, event_id)

    if not event:
        return None

    url = _event_telegraph_url(event)
    # Don't force rebuild for updates - URL should already exist
    if not url:
        logger.debug(
            "source_parsing: no telegraph url for updated event_id=%d",
            event_id,
        )

    return UpdatedEventInfo(
        event_id=event_id,
        title=event.title or "",
        telegraph_url=url or "",
        date=getattr(event, "date", None),
        time=getattr(event, "time", None),
        source=source,
        update_type=update_type,
    )


async def download_images(urls: list[str]) -> list[tuple[bytes, str]]:
    """Download images from URLs."""
    import main as main_mod
    session = main_mod.get_http_session()
    result = []
    seen = set()
    
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            # Use short timeout to avoid blocking
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    # Try to get filename from URL or content-disposition
                    name = url.split("/")[-1] or "image.jpg"
                    result.append((data, name))
        except Exception as e:
            logger.warning("source_parsing: download failed url=%s error=%s", url, e)
            
    return result


async def update_event_ticket_status(
    db: Database,
    event_id: int,
    ticket_status: str,
    ticket_link: str | None = None,
) -> bool:
    """Update ticket status for an existing event.
    
    Args:
        db: Database instance
        event_id: Event ID to update
        ticket_status: New ticket status
        ticket_link: Optional ticket link to update
    
    Returns:
        True if updated successfully
    """
    from models import Event
    
    try:
        async with db.get_session() as session:
            event = await session.get(Event, event_id)
            if not event:
                return False
            
            old_status = event.ticket_status
            event.ticket_status = ticket_status
            
            if ticket_link and not event.ticket_link:
                event.ticket_link = ticket_link
            
            await session.commit()
            
            logger.info(
                "source_parsing: updated ticket_status event_id=%d old=%s new=%s",
                event_id,
                old_status,
                ticket_status,
            )
            
            # Rebuild Telegraph page with updated data
            await _ensure_telegraph_url(db, event_id)
            
            return True
    except Exception as e:
        logger.error(
            "source_parsing: update failed event_id=%d error=%s",
            event_id,
            e,
        )
        return False


async def update_event_full(
    db: Database,
    event_id: int,
    theatre_event: TheatreEvent,
) -> bool:
    """Fully update an event with new data (for placeholder 00:00 events).
    
    Args:
        db: Database instance
        event_id: Event ID to update
        theatre_event: New event data
    
    Returns:
        True if updated successfully
    """
    from models import Event
    
    try:
        async with db.get_session() as session:
            event = await session.get(Event, event_id)
            if not event:
                return False
            
            # Update time
            if theatre_event.parsed_time and theatre_event.parsed_time != "00:00":
                event.time = theatre_event.parsed_time
            
            # Update ticket info
            event.ticket_status = theatre_event.ticket_status
            if theatre_event.url:
                event.ticket_link = theatre_event.url
            
            # Update pushkin card
            event.pushkin_card = theatre_event.pushkin_card
            
            # Don't overwrite description with raw parser text
            # Description should only come from LLM (short_description)
            # Keeping existing description if present
            
            # Update photos
            photos = limit_photos_for_source(
                theatre_event.photos,
                theatre_event.source_type,
            )
            if photos and not event.photo_urls:
                event.photo_urls = photos
                event.photo_count = len(photos)
            
            await session.commit()
            
            logger.info(
                "source_parsing: full update event_id=%d time=%s",
                event_id,
                event.time,
            )
            return True
    except Exception as e:
        logger.error(
            "source_parsing: full update failed event_id=%d error=%s",
            event_id,
            e,
        )
        return False


async def update_linked_events(
    db: Database,
    event_id: int,
    location_name: str,
    title: str,
) -> None:
    """Update linked_event_ids for an event and its related events.
    
    Args:
        db: Database instance
        event_id: Event ID to update
        location_name: Location name for finding linked events
        title: Title for fuzzy matching
    """
    from models import Event
    
    try:
        linked_ids = await find_linked_events(db, location_name, title, event_id)
        
        if not linked_ids:
            return
        
        async with db.get_session() as session:
            # Update this event with linked IDs
            event = await session.get(Event, event_id)
            if event:
                event.linked_event_ids = linked_ids
            
            # Update linked events to include this one
            for linked_id in linked_ids:
                linked_event = await session.get(Event, linked_id)
                if linked_event:
                    existing_links = set(linked_event.linked_event_ids or [])
                    existing_links.add(event_id)
                    existing_links.discard(linked_id)  # Don't link to self
                    linked_event.linked_event_ids = list(existing_links)
            
            await session.commit()
            
        logger.info(
            "source_parsing: linked events event_id=%d linked_count=%d",
            event_id,
            len(linked_ids),
        )
    except Exception as e:
        logger.warning(
            "source_parsing: linking failed event_id=%d error=%s",
            event_id,
            e,
        )


async def schedule_existing_event_update(db: Database, event_id: int) -> None:
    """Enqueue deferred page rebuilds for an updated existing event."""
    import sys
    from models import Event

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if not main_mod or not hasattr(main_mod, "schedule_event_update_tasks"):
        logger.warning(
            "source_parsing: schedule_event_update_tasks unavailable event_id=%d",
            event_id,
        )
        return

    async with db.get_session() as session:
        event = await session.get(Event, event_id)
    if not event:
        return

    await main_mod.schedule_event_update_tasks(
        db,
        event,
        drain_nav=False,
        skip_vk_sync=True,
    )


async def add_new_event_via_queue(
    db: Database,
    bot: Bot | None,
    theatre_event: TheatreEvent,
    progress_current: int,
    progress_total: int,
    poster_media: Sequence[PosterMedia] | None = None,
) -> tuple[int | None, bool]:
    """Add a new event through the existing LLM queue system.
    
    Uses build_event_drafts_from_vk for consistent event creation.
    
    Args:
        db: Database instance
        bot: Telegram bot for notifications
        theatre_event: Event data from parser
        progress_current: Current progress number
        progress_total: Total events to process
    
    Returns:
        New event ID or None if failed
    """
    from vk_intake import build_event_drafts_from_vk
    import main as main_mod
    
    try:
        # Build description with all available info
        description_parts = []
        if theatre_event.description:
            description_parts.append(theatre_event.description)
        if theatre_event.age_restriction:
            description_parts.append(f"Возраст: {theatre_event.age_restriction}")
        if theatre_event.scene:
            description_parts.append(f"Сцена: {theatre_event.scene}")
        
        full_description = "\n\n".join(description_parts) if description_parts else theatre_event.title
        
        # Build source text for LLM - include title explicitly for normalization
        source_text = f"Название: {theatre_event.title}\n\n{full_description}"
        
        location_name = normalize_location_name(theatre_event.location, theatre_event.scene)
        
        # Limit photos for source
        photos = limit_photos_for_source(
            theatre_event.photos,
            theatre_event.source_type,
        )
        
        # Prefer Catbox URLs from poster_media if available
        if poster_media:
            catbox_photos = [
                p.catbox_url for p in poster_media 
                if p.catbox_url
            ]
            if catbox_photos:
                # Use catbox photos, but still respect source limits if needed
                # (though usually we want all processed photos)
                photos = catbox_photos
        
        # Log progress
        logger.info(
            "source_parsing: adding event %d/%d title=%s location=%s",
            progress_current,
            progress_total,
            theatre_event.title[:50],
            location_name,
        )
        
        # Use existing event creation logic
        diag_enabled = bool(SOURCE_PARSING_DIAG_TITLE) and SOURCE_PARSING_DIAG_TITLE in theatre_event.title.lower()
        llm_start = time.monotonic()
        try:
            drafts, _ = await asyncio.wait_for(
                build_event_drafts_from_vk(
                    text=source_text,
                    source_name=f"theatre:{theatre_event.source_type}",
                    location_hint=location_name,
                    default_ticket_link=theatre_event.url,
                    poster_media=poster_media,
                ),
                timeout=PARSE_EVENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "source_parsing: LLM timeout after %ss title=%s",
                PARSE_EVENT_TIMEOUT_SECONDS,
                theatre_event.title[:50],
            )
            return None, False
        if diag_enabled:
            logger.info(
                "source_parsing: diag LLM done title=%s drafts=%d duration=%.2fs",
                theatre_event.title[:120],
                len(drafts or []),
                time.monotonic() - llm_start,
            )
        
        if not drafts:
            logger.warning(
                "source_parsing: no drafts returned title=%s",
                theatre_event.title,
            )
            return None, False
        
        draft = drafts[0]
        
        # Override with parsed values
        if theatre_event.parsed_date:
            draft.date = theatre_event.parsed_date
        if theatre_event.parsed_time:
            draft.time = theatre_event.parsed_time
            
        # Override prices if available
        if theatre_event.ticket_price_min is not None:
            draft.ticket_price_min = theatre_event.ticket_price_min
        if theatre_event.ticket_price_max is not None:
            draft.ticket_price_max = theatre_event.ticket_price_max
        
        draft.venue = location_name
        draft.ticket_link = theatre_event.url
        draft.pushkin_card = theatre_event.pushkin_card
        
        # Use lightweight bulk insert - no Telegraph rebuild, no VK posting
        try:
            import sys
            from datetime import datetime, timezone
            from models import Event
            
            main_mod = sys.modules.get("main") or sys.modules.get("__main__")
            if main_mod is None:
                logger.error("source_parsing: main module not found")
                return None
            
            upsert_event = main_mod.upsert_event
            upsert_event_posters = main_mod.upsert_event_posters
            assign_event_topics = main_mod.assign_event_topics
            
            # Build final description - should come from LLM (short_description)
            # If LLM didn't return it, use title as fallback and log warning
            final_description = draft.description
            if not final_description:
                logger.warning(
                    "source_parsing: LLM returned empty short_description title=%s",
                    theatre_event.title[:50],
                )
                final_description = draft.title
            
            # Build Event object
            event = Event(
                title=draft.title,
                description=final_description,
                festival=(draft.festival or None),
                date=draft.date or datetime.now(timezone.utc).date().isoformat(),
                time=draft.time or "00:00",
                location_name=draft.venue or "",
                location_address=draft.location_address or None,
                city=draft.city or None,
                ticket_price_min=draft.ticket_price_min,
                ticket_price_max=draft.ticket_price_max,
                ticket_link=theatre_event.url,  # Always use parser URL, not LLM links
                event_type=draft.event_type or None,
                emoji=draft.emoji or None,
                end_date=draft.end_date or None,
                is_free=bool(draft.is_free),
                pushkin_card=bool(draft.pushkin_card),
                source_text=draft.source_text or draft.title,
                photo_urls=photos,
                photo_count=len(photos),
                source_post_url=theatre_event.url,
                search_digest=draft.search_digest,
            )
            
            # Assign topics (LLM classification)
            await assign_event_topics(event)
            
            # Save to database
            async with db.get_session() as session:
                saved, was_added = await upsert_event(session, event)
                await upsert_event_posters(session, saved.id, poster_media)
            
            event_id = saved.id
            
            # Reload saved event for schedule_event_update_tasks
            async with db.get_session() as session:
                saved = await session.get(Event, event_id)
            
            # Create Telegraph pages and other artifacts (NO VK posting, NO nav drain - updated in batch later)
            schedule_event_update_tasks = main_mod.schedule_event_update_tasks
            await schedule_event_update_tasks(db, saved, drain_nav=False, skip_vk_sync=True)
            
            logger.info(
                "source_parsing: event created event_id=%d title=%s (Telegraph only, no nav update)",
                event_id,
                theatre_event.title[:50],
            )
            
            # Update ticket status
            await update_event_ticket_status(
                db,
                event_id,
                theatre_event.ticket_status,
                theatre_event.url,
            )
            
            # Update linked events
            await update_linked_events(
                db,
                event_id,
                location_name,
                theatre_event.title,
            )
            
            return event_id, was_added
                
        except Exception as persist_err:
            logger.error(
                "source_parsing: persist failed title=%s error=%s",
                theatre_event.title,
                persist_err,
                exc_info=True,
            )
            return None, False
        
    except Exception as e:
        logger.error(
            "source_parsing: add failed title=%s error=%s",
            theatre_event.title,
            e,
            exc_info=True,
        )
        return None, False


def escape_md(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    chars = "_*[]()~`>#+-=|{}.!"
    for c in chars:
        text = text.replace(c, f"\\{c}")
    return text


def _format_kaggle_status(status: dict | None) -> str:
    if not status:
        return "неизвестен"
    state = status.get("status")
    failure_msg = status.get("failureMessage") or status.get("failure_message")
    if not state:
        return "неизвестен"
    result = str(state)
    if failure_msg:
        result += f" ({failure_msg})"
    return result


def _format_kaggle_phase(phase: str) -> str:
    labels = {
        "prepare": "подготовка",
        "pushed": "запуск в Kaggle",
        "poll": "выполнение",
        "complete": "завершено",
        "failed": "ошибка",
        "timeout": "таймаут",
        "not_found": "kernel не найден",
        "metadata_missing": "нет kernel-metadata.json",
        "metadata_error": "ошибка метаданных",
        "push_failed": "ошибка отправки",
    }
    return labels.get(phase, phase)


def _format_kaggle_status_message(
    phase: str,
    kernel_ref: str,
    status: dict | None,
) -> str:
    lines = [
        "🛰️ Kaggle: ParseTheatres",
        f"Kernel: {kernel_ref or '—'}",
        f"Этап: {_format_kaggle_phase(phase)}",
    ]
    if status is not None:
        lines.append(f"Статус Kaggle: {_format_kaggle_status(status)}")
    return "\n".join(lines)


def format_parsing_report(result: SourceParsingResult) -> str:
    """Format parsing result as a human-readable report.
    
    Args:
        result: Parsing result
    
    Returns:
        Formatted summary string
    """
    lines = [
        f"🏁 **Парсинг источников завершен**",
        f"Продолжительность: {result.processing_duration:.1f} сек",
        f"Всего событий: {result.total_events}",
        "",
        "**По источникам:**"
    ]
    
    total_added = 0
    total_updated = 0
    total_failed = 0
    total_skipped = 0
    
    for source, stats in result.stats_by_source.items():
        total_added += stats.new_added
        total_updated += stats.ticket_updated
        total_failed += stats.failed
        total_skipped += stats.skipped
        
        # Use descriptive labels if available
        source_label = {
            "dramteatr": "Драмтеатр",
            "muzteatr": "Музтеатр",
            "sobor": "Собор",
            "tretyakov": "🎨 Третьяковка",
        }.get(source, source)
        
        lines.append(f"• **{escape_md(source_label)}**:")
        lines.append(f"  ✅ Добавлено: {stats.new_added}")
        if stats.ticket_updated:
            lines.append(f"  🔄 Обновлено: {stats.ticket_updated}")
        if stats.failed:
            lines.append(f"  ❌ Ошибок: {stats.failed}")
        if stats.skipped:
            lines.append(f"  ⏭️ Пропущено: {stats.skipped}")
    
    lines.append("")
    lines.append(f"**Итого:**")
    lines.append(f"✅ Всего добавлено: {total_added}")
    if total_updated:
        lines.append(f"🔄 Всего обновлено: {total_updated}")
    if total_failed:
        lines.append(f"❌ Всего ошибок: {total_failed}")
    
    if result.errors:
        lines.append("")
        lines.append("**Ошибки выполнения:**")
        # Show first 3 errors to avoid overflow
        for err in result.errors[:3]:
            # Escape error text as it may contain underscores/paths
            lines.append(f"⚠️ {escape_md(str(err))}")
        if len(result.errors) > 3:
            lines.append(f"... и еще {len(result.errors) - 3}")

    # Add JSON file paths if available
    if result.json_file_paths:
        lines.append("")
        lines.append("**Сохраненные файлы:**")
        for path in result.json_file_paths:
            lines.append(f"📄 {escape_md(Path(path).name)}")
            
    return "\n".join(lines)


async def run_source_parsing(
    db: Database,
    bot: Bot | None = None,
    chat_id: int | None = None,
) -> SourceParsingResult:
    """Run full source parsing pipeline.
    
    Args:
        db: Database instance
        bot: Optional bot instance for progress updates
        chat_id: Optional chat ID for progress updates
    
    Returns:
        Result statistics
    """
    start_time = time.time()
    result = SourceParsingResult(chat_id=chat_id)
    kaggle_status_message_id: int | None = None
    kaggle_kernel_ref = ""
    run_id = uuid.uuid4().hex[:8]
    log_handler: logging.Handler | None = None

    class _SourceParsingFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.name.startswith("source_parsing")

    log_dir = Path("/data/parse_debug")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"source_parsing_{run_id}.log"
    try:
        log_handler = logging.FileHandler(log_path, encoding="utf-8")
        log_handler.setLevel(logging.INFO)
        log_handler.addFilter(_SourceParsingFilter())
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(log_handler)
        result.log_file_path = str(log_path)
    except Exception as e:
        logger.warning("source_parsing: failed to init log file: %s", e)

    async def _update_kaggle_status(
        phase: str,
        kernel_ref: str,
        status: dict | None,
    ) -> None:
        nonlocal kaggle_status_message_id, kaggle_kernel_ref
        kaggle_kernel_ref = kernel_ref or kaggle_kernel_ref
        if not bot or not chat_id:
            return
        text = _format_kaggle_status_message(phase, kernel_ref, status)
        try:
            if kaggle_status_message_id is None:
                sent = await bot.send_message(chat_id, text)
                kaggle_status_message_id = sent.message_id
            else:
                await bot.edit_message_text(
                    text=text,
                    chat_id=chat_id,
                    message_id=kaggle_status_message_id,
                )
        except Exception:
            logger.warning(
                "source_parsing: failed to update kaggle status message",
                exc_info=True,
            )
    if DEBUG_MAX_EVENTS:
        logger.info(
            "source_parsing: DEBUG limit active max_new_events=%d",
            DEBUG_MAX_EVENTS,
        )
    
    try:
        # 1. Run Kaggle kernel
        try:
            status, output_files, duration = await run_kaggle_kernel(
                status_callback=_update_kaggle_status,
            )
            result.kernel_duration = duration
            result.json_file_paths = [
                path for path in output_files
                if Path(path).suffix.lower() == ".json" and Path(path).exists()
            ]
            
            if status != "complete":
                result.errors.append(f"Kaggle kernel failed: {status}")
                return result
                
            logger.info(
                "source_parsing: kaggle complete duration=%.1fs files=%d",
                duration,
                len(output_files),
            )
            
        except Exception as e:
            logger.error("source_parsing: kaggle error: %s", e, exc_info=True)
            result.errors.append(f"Kaggle error: {str(e)}")
            return result

        # 2. Parse all files first
        events_by_source = {}

        def _normalize_source_name(raw_name: str) -> str:
            normalized = raw_name.strip().lower()
            normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
            if "dramteatr" in normalized or "драм" in normalized:
                return "dramteatr"
            if "muzteatr" in normalized or "муз" in normalized:
                return "muzteatr"
            if "sobor" in normalized or "собор" in normalized:
                return "sobor"
            if "tretyakov" in normalized or "трет" in normalized:
                return "tretyakov"
            return normalized
        
        for file_path_str in output_files:
            try:
                file_path = Path(file_path_str)
                if file_path.suffix.lower() != ".json":
                    continue
                # Determine source from filename (e.g. sobor.json -> sobor)
                source_name = _normalize_source_name(file_path.stem)
                
                # Parse JSON
                raw_content = file_path.read_text(encoding="utf-8")
                events = parse_theatre_json(raw_content, source_name)
                
                if not events:
                    logger.warning("source_parsing: no events found in %s", file_path)
                    continue
                    
                events_by_source[source_name] = events
                
            except Exception as e:
                logger.error("source_parsing: failed to parse %s: %s", file_path_str, e, exc_info=True)
                result.errors.append(f"File {file_path_str}: {str(e)}")
        
        # 3. Process events
        total_count = sum(len(ev) for ev in events_by_source.values())
        result.total_events = total_count
        
        current_index = 0
        progress_message_id = None
        
        for source, events in events_by_source.items():
            try:
                stats, progress_message_id = await process_source_events(
                    db,
                    bot,
                    events,
                    source,
                    current_index,
                    total_count,
                    chat_id=chat_id,
                    progress_message_id=progress_message_id,
                    added_events=result.added_events if chat_id else None,
                    updated_events=result.updated_events if chat_id else None,
                )
                result.stats_by_source[source] = stats
                current_index += len(events)
            except Exception as e:
                logger.error("source_parsing: failed to process events from %s: %s", source, e, exc_info=True)
                # Escape error text as it may contain underscores/paths
                result.errors.append(f"Source {escape_md(source)}: {escape_md(str(e))}")
                
        months = sorted({
            event.parsed_date[:7]
            for events in events_by_source.values()
            for event in events
            if event.parsed_date
        })
        if months:
            logger.info("source_parsing: ensuring month_pages tasks for months=%s", months)
            try:
                import sys
                from sqlalchemy import select
                from models import Event, JobTask
            except Exception as e:
                logger.error(
                    "source_parsing: failed to prepare month_pages enqueue: %s",
                    e,
                )
            else:
                main_mod = sys.modules.get("main") or sys.modules.get("__main__")
                if main_mod is None:
                    logger.error("source_parsing: main module not found for month_pages enqueue")
                else:
                    enqueue_job = main_mod.enqueue_job
                    mark_pages_dirty = main_mod.mark_pages_dirty
                    month_event_ids: dict[str, int] = {}
                    async with db.get_session() as session:
                        for month in months:
                            res = await session.execute(
                                select(Event.id)
                                .where(Event.date.like(f"{month}%"))
                                .order_by(Event.id.desc())
                                .limit(1)
                            )
                            event_id = res.scalar_one_or_none()
                            if event_id:
                                month_event_ids[month] = event_id

                    for month in months:
                        event_id = month_event_ids.get(month)
                        if not event_id:
                            continue
                        await enqueue_job(
                            db,
                            event_id,
                            JobTask.month_pages,
                            coalesce_key=f"month_pages:{month}",
                        )
                        await mark_pages_dirty(db, month)

        # Final progress update
        if bot and chat_id and progress_message_id:
            try:
                total_new = sum(s.new_added for s in result.stats_by_source.values())
                total_fail = sum(s.failed for s in result.stats_by_source.values())
                
                final_text = (
                    f"✅ Обработка завершена\n"
                    f"Добавлено: {total_new}\n"
                    f"Ошибок: {total_fail}"
                )
                await bot.edit_message_text(
                    text=final_text,
                    chat_id=chat_id,
                    message_id=progress_message_id,
                )
            except Exception as e:
                logger.warning("source_parsing: failed to update final progress: %s", e)
        
        result.processing_duration = time.time() - start_time
        
        logger.info(
            "source_parsing: complete total=%d kernel=%.1fs processing=%.1fs",
            total_count,
            result.kernel_duration,
            result.processing_duration,
        )
        
        return result
    finally:
        if log_handler:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()


async def process_source_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    source: str,
    start_index: int,
    total_count: int,
    chat_id: int | None = None,
    progress_message_id: int | None = None,
    added_events: list[AddedEventInfo] | None = None,
    updated_events: list[UpdatedEventInfo] | None = None,
) -> tuple[SourceParsingStats, int | None]:
    """Process events from a single source.
    
    Args:
        db: Database instance
        bot: Telegram bot
        events: List of events to process
        source: Source identifier
        start_index: Starting index for progress
        total_count: Total events across all sources
        chat_id: Chat ID for progress messages
        progress_message_id: Message ID to edit for progress updates
        added_events: List to populate with newly added events
        updated_events: List to populate with updated events
    
    Returns:
        Tuple of (statistics, updated progress_message_id)
    """
    stats = SourceParsingStats(source=source, total_received=len(events))
    
    # Source label for messages
    source_label = {
        "dramteatr": "🎭 Драмтеатр",
        "muzteatr": "🎵 Музтеатр",
        "sobor": "⛪ Собор",
        "tretyakov": "🎨 Третьяковка",
    }.get(source, source)
    
    for i, event in enumerate(events):
        current_progress = start_index + i + 1
        event_start = time.monotonic()
        result_tag = "unknown"
        event_id: int | None = None
        llm_used = False
        diag_enabled = bool(SOURCE_PARSING_DIAG_TITLE) and SOURCE_PARSING_DIAG_TITLE in (event.title or "").lower()

        # Update progress message for every event (new or existing).
        if bot and chat_id:
            try:
                progress_text = f"📝 Обработка {current_progress}/{total_count}: {event.title[:40]}"
                if progress_message_id:
                    await bot.edit_message_text(
                        text=progress_text,
                        chat_id=chat_id,
                        message_id=progress_message_id,
                    )
                else:
                    msg = await bot.send_message(chat_id, progress_text)
                    progress_message_id = msg.message_id
            except Exception as e:
                logger.warning("source_parsing: failed to update progress: %s", e)
        
        if not event.parsed_date:
            logger.warning(
                "source_parsing: skipping event without date title=%s",
                event.title,
            )
            stats.failed += 1
            result_tag = "missing_date"
            logger.info(
                "source_parsing: event_result source=%s title=%s result=%s duration=%.2fs",
                source,
                event.title[:80],
                result_tag,
                time.monotonic() - event_start,
            )
            continue

        if diag_enabled:
            desc_len = len(event.description or "")
            photos_count = len(event.photos or [])
            logger.info(
                "source_parsing: diag title=%s date_raw=%s parsed=%s %s url=%s photos=%d desc_len=%d",
                event.title[:120],
                event.date_raw,
                event.parsed_date,
                event.parsed_time or "",
                event.url,
                photos_count,
                desc_len,
            )
        
        try:
            location_name = normalize_location_name(event.location, event.scene)
            if diag_enabled:
                logger.info(
                    "source_parsing: diag normalized_location=%s",
                    location_name,
                )
            
            # Check for existing event
            existing_id, needs_full_update = await find_existing_event(
                db,
                location_name,
                event.parsed_date,
                event.parsed_time or "00:00",
                event.title,
            )
            
            if existing_id:
                event_id = existing_id
                if needs_full_update:
                    # Update the placeholder event fully
                    success = await update_event_full(db, existing_id, event)
                    if success:
                        await schedule_existing_event_update(db, existing_id)
                        stats.ticket_updated += 1
                        result_tag = "existing_full_update"
                        # Track updated event for reporting
                        if updated_events is not None:
                            info = await build_updated_event_info(db, existing_id, source, "full_update")
                            if info:
                                updated_events.append(info)
                    else:
                        stats.failed += 1
                        result_tag = "existing_full_update_failed"
                else:
                    # Just update ticket status
                    success = await update_event_ticket_status(
                        db,
                        existing_id,
                        event.ticket_status,
                        event.url,
                    )
                    if success:
                        await schedule_existing_event_update(db, existing_id)
                        stats.ticket_updated += 1
                        result_tag = "existing_ticket_update"
                        # Track updated event for reporting
                        if updated_events is not None:
                            info = await build_updated_event_info(db, existing_id, source, "ticket_status")
                            if info:
                                updated_events.append(info)
                    else:
                        stats.already_exists += 1
                        result_tag = "existing_ticket_update_failed"
                
                # Always update linked events
                await update_linked_events(db, existing_id, location_name, event.title)
            else:
                if diag_enabled:
                    logger.info(
                        "source_parsing: diag no existing match title=%s",
                        event.title[:120],
                    )
                # Prepare images for OCR if any
                poster_media_list = []
                
                # Filter photos first
                target_photos = limit_photos_for_source(
                    event.photos,
                    event.source_type,
                )

                if event.source_type in SOURCE_PARSING_DISABLE_OCR_SOURCES:
                    if diag_enabled:
                        logger.info(
                            "source_parsing: diag ocr skipped source=%s",
                            event.source_type,
                        )
                    target_photos = []
                
                if target_photos:
                    try:
                        if diag_enabled:
                            logger.info(
                                "source_parsing: diag ocr start title=%s photos=%d",
                                event.title[:120],
                                len(target_photos),
                            )
                        raw_images = await asyncio.wait_for(
                            download_images(target_photos),
                            timeout=SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                        )
                        
                        if raw_images:
                            poster_media_list, _ = await asyncio.wait_for(
                                process_media(
                                    raw_images,
                                    need_catbox=True,
                                    need_ocr=True,
                                ),
                                timeout=SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                            )
                        if diag_enabled:
                            logger.info(
                                "source_parsing: diag ocr done title=%s raw=%d posters=%d",
                                event.title[:120],
                                len(raw_images),
                                len(poster_media_list),
                            )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "source_parsing: ocr timeout title=%s after %ss",
                            event.title,
                            SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                        )
                    except Exception as e:
                        logger.warning("source_parsing: ocr failed event=%s error=%s", event.title, e)

                # Add new event
                llm_used = True
                new_id, was_added = await add_new_event_via_queue(
                    db,
                    bot,
                    event,
                    current_progress,
                    total_count,
                    poster_media=poster_media_list,
                )
                
                if new_id:
                    event_id = new_id
                    if was_added:
                        stats.new_added += 1
                        if added_events is not None:
                            info = await build_added_event_info(db, new_id, source)
                            if info:
                                added_events.append(info)
                        result_tag = "new_added"
                    else:
                        stats.skipped += 1  # Event existed, was updated but not new
                        result_tag = "new_updated"
                    # Delay between additions
                    await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)
                    
                    # DEBUG: Stop after max events
                    if DEBUG_MAX_EVENTS and stats.new_added >= DEBUG_MAX_EVENTS:
                        logger.info("source_parsing: DEBUG limit reached (%d events)", DEBUG_MAX_EVENTS)
                        break
                else:
                    stats.failed += 1
                    result_tag = "new_failed"
        except Exception:
            stats.failed += 1
            result_tag = "exception"
            logger.exception(
                "source_parsing: event_exception source=%s title=%s",
                source,
                event.title[:80],
            )

        logger.info(
            "source_parsing: event_result source=%s title=%s date=%s time=%s result=%s event_id=%s llm=%s duration=%.2fs",
            source,
            event.title[:80],
            event.parsed_date,
            event.parsed_time,
            result_tag,
            event_id,
            int(llm_used),
            time.monotonic() - event_start,
        )
    
    return stats, progress_message_id
async def run_diagnostic_parse(
    bot: Bot,
    chat_id: int,
    source: str,
) -> None:
    """Run diagnostic parse for a specific source and send result JSON.
    
    Args:
        bot: Bot instance
        chat_id: Chat ID to send results to
        source: Source identifier (dramteatr, muzteatr, sobor, tretyakov, or all)
    """
    from aiogram.types import FSInputFile
    
    await bot.send_message(
        chat_id,
        f"🔍 Запуск диагностического парсинга: {source}...\nПожалуйста, подождите (около 2-5 минут)."
    )
    
    start_time = time.time()
    kaggle_status_message_id: int | None = None
    kaggle_kernel_ref = ""

    async def _update_kaggle_status(
        phase: str,
        kernel_ref: str,
        status: dict | None,
    ) -> None:
        nonlocal kaggle_status_message_id, kaggle_kernel_ref
        kaggle_kernel_ref = kernel_ref or kaggle_kernel_ref
        
        text = _format_kaggle_status_message(phase, kernel_ref, status)
        try:
            if kaggle_status_message_id is None:
                sent = await bot.send_message(chat_id, text)
                kaggle_status_message_id = sent.message_id
            else:
                await bot.edit_message_text(
                    text=text,
                    chat_id=chat_id,
                    message_id=kaggle_status_message_id,
                )
        except Exception:
            pass

    # Run kernel with config
    status, output_files, duration = await run_kaggle_kernel(
        status_callback=_update_kaggle_status,
        run_config={"target_source": source}
    )
    
    if status != "complete":
        await bot.send_message(chat_id, f"❌ Ошибка запуска: статус {status}")
        return

    # Find the specific JSON file
    target_filename = f"{source}.json"
    target_path = None
    
    # If source is 'tretyakov', look for 'tretyakov.json' etc.
    # Note: notebook saves lowercase filenames
    
    found_files = []
    
    for fpath in output_files:
        path = Path(fpath)
        if path.name == target_filename:
            target_path = fpath
        if path.suffix == ".json":
            found_files.append(path)
            
    if target_path and Path(target_path).exists():
        await bot.send_document(
            chat_id,
            FSInputFile(target_path),
            caption=f"✅ Диагностика {source} завершена за {duration:.1f}с.\nРезультат в файле."
        )
    elif found_files:
        # If exact match not found but other JSONs exist (maybe naming mismatch), send them
        await bot.send_message(chat_id, f"⚠️ Файл {target_filename} не найден, отправляю найденные JSON:")
        for fpath in found_files:
             await bot.send_document(
                chat_id,
                FSInputFile(fpath),
                caption=f"📄 {fpath.name}"
            )
    else:
        await bot.send_message(
            chat_id, 
            f"⚠️ Файл {target_filename} не найден в результатах.\nСтатус: {status}\nФайлы: {len(output_files)}"
        )
