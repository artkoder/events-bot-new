"""Handlers for source parsing operations."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from aiogram import Bot

from db import Database
from source_parsing.parser import (
    TheatreEvent,
    normalize_location_name,
    find_existing_event,
    should_update_event,
    find_linked_events,
    limit_photos_for_source,
)

logger = logging.getLogger(__name__)

# Delay between adding events to avoid overloading the system
EVENT_ADD_DELAY_SECONDS = 20


@dataclass
class SourceParsingStats:
    """Statistics for a source parsing run."""
    source: str
    total_received: int = 0
    new_added: int = 0
    ticket_updated: int = 0
    already_exists: int = 0
    failed: int = 0


@dataclass
class SourceParsingResult:
    """Complete result of a source parsing run."""
    stats_by_source: dict[str, SourceParsingStats] = field(default_factory=dict)
    total_events: int = 0
    kernel_duration: float = 0.0
    processing_duration: float = 0.0
    log_file_path: str = ""
    errors: list[str] = field(default_factory=list)


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
            
            # Update description if provided
            if theatre_event.description and not event.description:
                event.description = theatre_event.description
            
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


async def add_new_event_via_queue(
    db: Database,
    bot: Bot | None,
    theatre_event: TheatreEvent,
    progress_current: int,
    progress_total: int,
) -> int | None:
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
    from vk_intake import build_event_drafts_from_vk, EventDraft
    from runtime import get_running_main
    
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
        
        # Build source text for LLM
        source_text = f"""
Название: {theatre_event.title}
Дата: {theatre_event.date_raw}
Место: {theatre_event.location}
{full_description}

Билеты: {theatre_event.url}
Пушкинская карта: {'Да' if theatre_event.pushkin_card else 'Нет'}
"""
        
        location_name = normalize_location_name(theatre_event.location)
        
        # Limit photos for Музтеатр
        photos = limit_photos_for_source(
            theatre_event.photos,
            theatre_event.source_type,
        )
        
        # Log progress
        logger.info(
            "source_parsing: adding event %d/%d title=%s",
            progress_current,
            progress_total,
            theatre_event.title[:50],
        )
        
        # Use existing event creation logic
        drafts, _ = await build_event_drafts_from_vk(
            text=source_text,
            source_name=f"theatre:{theatre_event.source_type}",
            location_hint=location_name,
            default_ticket_link=theatre_event.url,
        )
        
        if not drafts:
            logger.warning(
                "source_parsing: no drafts returned title=%s",
                theatre_event.title,
            )
            return None
        
        draft = drafts[0]
        
        # Override with parsed values
        if theatre_event.parsed_date:
            draft.date = theatre_event.parsed_date
        if theatre_event.parsed_time:
            draft.time = theatre_event.parsed_time
        
        draft.venue = location_name
        draft.ticket_link = theatre_event.url
        draft.pushkin_card = theatre_event.pushkin_card
        
        # Create event in database
        main = get_running_main()
        if main and hasattr(main, "persist_event_draft"):
            result = await main.persist_event_draft(
                db,
                draft,
                photos=photos,
                silent=True,  # Don't send notifications for bulk imports
            )
            if result:
                event_id = result.event_id
                
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
                
                return event_id
        
        return None
        
    except Exception as e:
        logger.error(
            "source_parsing: add failed title=%s error=%s",
            theatre_event.title,
            e,
            exc_info=True,
        )
        return None


async def process_source_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    source: str,
    start_index: int,
    total_count: int,
) -> SourceParsingStats:
    """Process events from a single source.
    
    Args:
        db: Database instance
        bot: Telegram bot
        events: List of events to process
        source: Source identifier
        start_index: Starting index for progress
        total_count: Total events across all sources
    
    Returns:
        Statistics for this source
    """
    stats = SourceParsingStats(source=source, total_received=len(events))
    
    for i, event in enumerate(events):
        current_progress = start_index + i + 1
        
        if not event.parsed_date:
            logger.warning(
                "source_parsing: skipping event without date title=%s",
                event.title,
            )
            stats.failed += 1
            continue
        
        location_name = normalize_location_name(event.location)
        
        # Check for existing event
        existing_id, needs_full_update = await find_existing_event(
            db,
            location_name,
            event.parsed_date,
            event.parsed_time or "00:00",
            event.title,
        )
        
        if existing_id:
            if needs_full_update:
                # Update the placeholder event fully
                success = await update_event_full(db, existing_id, event)
                if success:
                    stats.ticket_updated += 1
                else:
                    stats.failed += 1
            else:
                # Just update ticket status
                success = await update_event_ticket_status(
                    db,
                    existing_id,
                    event.ticket_status,
                    event.url,
                )
                if success:
                    stats.ticket_updated += 1
                else:
                    stats.already_exists += 1
            
            # Always update linked events
            await update_linked_events(db, existing_id, location_name, event.title)
        else:
            # Add new event
            new_id = await add_new_event_via_queue(
                db,
                bot,
                event,
                current_progress,
                total_count,
            )
            
            if new_id:
                stats.new_added += 1
                # Delay between additions
                await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)
            else:
                stats.failed += 1
    
    return stats


async def run_source_parsing(
    db: Database,
    bot: Bot | None = None,
    test_data: dict[str, list[TheatreEvent]] | None = None,
) -> SourceParsingResult:
    """Main entry point for source parsing.
    
    Args:
        db: Database instance
        bot: Telegram bot for notifications
        test_data: Optional test data to use instead of Kaggle
    
    Returns:
        Complete parsing result with statistics
    """
    start_time = time.time()
    result = SourceParsingResult()
    
    # Get events from Kaggle or test data
    if test_data:
        events_by_source = test_data
        result.log_file_path = ""
        result.kernel_duration = 0.0
    else:
        from source_parsing.kaggle_runner import run_kaggle_and_get_events
        
        logger.info("source_parsing: calling Kaggle runner...")
        events_by_source, log_path, kernel_duration = await run_kaggle_and_get_events()
        logger.info(
            "source_parsing: Kaggle returned sources=%d duration=%.1fs",
            len(events_by_source),
            kernel_duration,
        )
        result.log_file_path = log_path
        result.kernel_duration = kernel_duration
    
    if not events_by_source:
        result.errors.append("No events received from sources")
        return result
    
    # Count total events
    total_count = sum(len(events) for events in events_by_source.values())
    result.total_events = total_count
    
    logger.info(
        "source_parsing: starting processing sources=%d total_events=%d",
        len(events_by_source),
        total_count,
    )
    
    # Process each source
    current_index = 0
    for source, events in events_by_source.items():
        stats = await process_source_events(
            db,
            bot,
            events,
            source,
            current_index,
            total_count,
        )
        result.stats_by_source[source] = stats
        current_index += len(events)
    
    result.processing_duration = time.time() - start_time
    
    logger.info(
        "source_parsing: complete total=%d kernel=%.1fs processing=%.1fs",
        total_count,
        result.kernel_duration,
        result.processing_duration,
    )
    
    return result


def format_parsing_report(result: SourceParsingResult) -> str:
    """Format parsing result as a human-readable report.
    
    Args:
        result: Parsing result
    
    Returns:
        Formatted report string
    """
    lines = [
        "📊 **Отчёт о парсинге источников**",
        "",
        f"⏱ Время Kaggle: {result.kernel_duration:.1f}с",
        f"⏱ Время обработки: {result.processing_duration:.1f}с",
        f"⏱ Общее время: {result.kernel_duration + result.processing_duration:.1f}с",
        "",
    ]
    
    total_new = 0
    total_updated = 0
    total_exists = 0
    total_failed = 0
    
    for source, stats in result.stats_by_source.items():
        source_label = {
            "dramteatr": "🎭 Драматический театр",
            "muzteatr": "🎵 Музыкальный театр",
            "sobor": "⛪ Кафедральный собор",
        }.get(source, source)
        
        lines.append(f"**{source_label}**")
        lines.append(f"  Получено: {stats.total_received}")
        lines.append(f"  ✅ Добавлено: {stats.new_added}")
        lines.append(f"  🔄 Обновлено: {stats.ticket_updated}")
        lines.append(f"  ⏭ Уже было: {stats.already_exists}")
        if stats.failed:
            lines.append(f"  ❌ Ошибок: {stats.failed}")
        lines.append("")
        
        total_new += stats.new_added
        total_updated += stats.ticket_updated
        total_exists += stats.already_exists
        total_failed += stats.failed
    
    lines.append("**📈 Итого:**")
    lines.append(f"  Получено событий: {result.total_events}")
    lines.append(f"  Добавлено новых: {total_new}")
    lines.append(f"  Обновлено статусов: {total_updated}")
    lines.append(f"  Уже существовало: {total_exists}")
    if total_failed:
        lines.append(f"  Ошибок: {total_failed}")
    
    if result.errors:
        lines.append("")
        lines.append("**⚠️ Ошибки:**")
        for error in result.errors:
            lines.append(f"  • {error}")
    
    return "\n".join(lines)
