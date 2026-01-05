"""Universal Festival Parser module.

Provides functionality to:
1. Parse festival websites via Kaggle notebook (Playwright + Gemma 3-27B)
2. Convert parsed UDS JSON to Festival model
3. Store results in Supabase Storage
4. Integrate with Telegraph pages and navigation

This module keeps logic out of main.py/main_part2.py for cleaner architecture.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from aiogram import Bot
    from db import Database
    from models import Festival

logger = logging.getLogger(__name__)

# Parser version for tracking
PARSER_VERSION = "1.0.0"

# Kaggle kernel folder for festival parser
FESTIVAL_PARSER_KERNEL_FOLDER = "UniversalFestivalParser"

# Supabase bucket for parser artifacts
SUPABASE_PARSER_BUCKET = os.getenv("SUPABASE_PARSER_BUCKET", "festival-parsing")

# Known official/canonical domains for Kaliningrad region festivals
OFFICIAL_DOMAINS = {
    "zimafestkld.ru",
    "www.zimafestkld.ru",
    # Add more known festival domains here
}

# Known aggregator/external domains
EXTERNAL_DOMAINS = {
    "afisha.yandex.ru",
    "kudago.com",
    "eventbrite.com",
    "timepad.ru",
    # Add more aggregator domains
}


def classify_source_type(url: str) -> Literal["canonical", "official", "external"]:
    """Classify URL source type.
    
    Args:
        url: Festival URL to classify
        
    Returns:
        "canonical" - official festival domain
        "official" - likely official based on heuristics
        "external" - third-party aggregator/news
        
    Classification rules (deterministic and testable):
    1. If domain in OFFICIAL_DOMAINS → "canonical"
    2. If domain in EXTERNAL_DOMAINS → "external"
    3. If domain looks like festival-specific (contains "fest", "festival", etc.) → "official"
    4. Otherwise → "external"
    """
    if not url:
        return "external"
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        return "external"
    
    # Check known domains
    if domain in OFFICIAL_DOMAINS or f"www.{domain}" in OFFICIAL_DOMAINS:
        return "canonical"
    
    if domain in EXTERNAL_DOMAINS:
        return "external"
    
    # Heuristics for official domains
    official_patterns = [
        r"fest",
        r"festival",
        r"фест",
        r"фестиваль",
    ]
    
    for pattern in official_patterns:
        if re.search(pattern, domain, re.IGNORECASE):
            return "official"
    
    # Default to external for unknown sources
    return "external"


def generate_run_id(url: str) -> str:
    """Generate unique run ID for parser execution.
    
    Format: YYYYMMDDTHHMMSSZ_<short_hash>
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"{timestamp}_{url_hash}"


def is_valid_url(text: str) -> bool:
    """Check if text looks like a valid URL."""
    if not text:
        return False
    
    text = text.strip()
    if not text.lower().startswith(("http://", "https://")):
        return False
    
    try:
        result = urlparse(text)
        return bool(result.netloc)
    except Exception:
        return False


def generate_festival_slug(name: str) -> str:
    """Generate URL-safe slug from festival name."""
    # Transliterate Cyrillic
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    
    slug = name.lower()
    for cyrillic, latin in translit_map.items():
        slug = slug.replace(cyrillic, latin)
    
    # Replace non-alphanumeric with dashes
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    
    return slug or 'festival'


async def run_festival_parser_kernel(
    url: str,
    run_id: str,
    dataset_sources: list[str],
    timeout_minutes: int = 30,
    poll_interval: int = 30,
    status_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> tuple[str, list[str], float]:
    """Run Kaggle kernel for festival parsing.
    
    Args:
        url: Festival URL to parse
        run_id: Unique run identifier
        dataset_sources: List of private Kaggle datasets (for Gemma API key)
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
        status_callback: Optional async callback for status updates
        
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """
    from source_parsing.kaggle_runner import run_kaggle_kernel
    
    # Prepare run config with festival URL
    run_config = {
        "festival_url": url,
        "run_id": run_id,
        "parser_version": PARSER_VERSION,
    }
    
    async def _notify(phase: str, status: dict | None = None) -> None:
        if status_callback:
            status_str = status.get("status", "unknown") if status else phase
            await status_callback(phase, status_str)
    
    result = await run_kaggle_kernel(
        kernel_folder=FESTIVAL_PARSER_KERNEL_FOLDER,
        timeout_minutes=timeout_minutes,
        poll_interval=poll_interval,
        status_callback=_notify,
        run_config=run_config,
        dataset_sources=dataset_sources,
    )
    
    return result


def parse_uds_output(file_paths: list[str]) -> dict | None:
    """Parse UDS JSON from kernel output files.
    
    Args:
        file_paths: List of downloaded file paths
        
    Returns:
        Parsed UDS dict or None if not found
    """
    for path in file_paths:
        if path.endswith("uds.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to parse UDS JSON %s: %s", path, e)
    
    # Try any .json file as fallback
    for path in file_paths:
        if path.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "festival" in data:
                        return data
            except Exception:
                continue
    
    return None


async def upsert_festival_from_uds(
    db: "Database",
    uds: dict,
    run_id: str,
    source_url: str,
) -> "Festival":
    """Create or update Festival from UDS JSON.
    
    Args:
        db: Database instance
        uds: Parsed UDS JSON
        run_id: Parser run ID
        source_url: Original URL
        
    Returns:
        Created or updated Festival
    """
    from models import Festival
    from sqlalchemy import select, update
    
    festival_data = uds.get("festival", {})
    
    # Extract festival name (required)
    name = festival_data.get("title_short") or festival_data.get("title_full", "")
    if not name:
        raise ValueError("Festival name is required in UDS")
    
    source_type = classify_source_type(source_url)
    
    async with db.get_session() as session:
        # Try to find existing festival by source_url or name
        existing = None
        
        if source_url:
            result = await session.execute(
                select(Festival).where(Festival.source_url == source_url)
            )
            existing = result.scalar_one_or_none()
        
        if not existing:
            # Try by name (case-insensitive)
            result = await session.execute(
                select(Festival).where(Festival.name.ilike(name))
            )
            existing = result.scalar_one_or_none()
        
        now = datetime.now(timezone.utc)
        
        # Prepare field values from UDS
        field_updates = {
            "name": name,
            "full_name": festival_data.get("title_full"),
            "description": festival_data.get("description_short"),
            "website_url": _extract_link(festival_data, "website"),
            "vk_url": _extract_social(festival_data, "vk.com"),
            "tg_url": _extract_social(festival_data, "t.me"),
            "program_url": _extract_document(uds, "pdf"),
            "ticket_url": _extract_registration(festival_data),
            "start_date": _parse_uds_date(festival_data.get("dates", {}).get("start")),
            "end_date": _parse_uds_date(festival_data.get("dates", {}).get("end")),
            "location_name": _extract_location_name(uds),
            "city": _extract_city(uds),
            "photo_urls": uds.get("images_festival", []),
            "photo_url": (uds.get("images_festival") or [None])[0],
            "activities_json": uds.get("program", []),
            "source_url": source_url,
            "source_type": source_type,
            "parser_run_id": run_id,
            "parser_version": PARSER_VERSION,
            "last_parsed_at": now,
            "contacts_phone": _extract_contact(festival_data, "phone"),
            "contacts_email": _extract_contact(festival_data, "email"),
            "is_annual": festival_data.get("is_annual"),
            "audience": festival_data.get("audience"),
        }
        
        # Filter out None values for updates (keep existing data)
        field_updates = {k: v for k, v in field_updates.items() if v is not None}
        
        if existing:
            # Update existing festival
            for key, value in field_updates.items():
                setattr(existing, key, value)
            await session.commit()
            await session.refresh(existing)
            logger.info("Updated festival %s from UDS", existing.name)
            return existing
        else:
            # Create new festival
            fest = Festival(**field_updates)
            session.add(fest)
            await session.commit()
            await session.refresh(fest)
            logger.info("Created festival %s from UDS", fest.name)
            return fest


def _extract_link(data: dict, key: str) -> str | None:
    """Extract link from UDS links structure."""
    links = data.get("links", {})
    return links.get(key)


def _extract_social(data: dict, domain: str) -> str | None:
    """Extract social link containing domain."""
    links = data.get("links", {})
    socials = links.get("socials", [])
    for url in socials:
        if domain in url:
            return url
    return None


def _extract_document(uds: dict, doc_type: str) -> str | None:
    """Extract document URL by type."""
    festival = uds.get("festival", {})
    documents = festival.get("documents", [])
    for doc in documents:
        if doc.get("type") == doc_type or (doc_type in (doc.get("url") or "")):
            return doc.get("url")
    return None


def _extract_registration(data: dict) -> str | None:
    """Extract registration/ticket URL."""
    reg = data.get("registration", {})
    return reg.get("common_url")


def _parse_uds_date(iso_str: str | None) -> str | None:
    """Parse ISO 8601 date to YYYY-MM-DD format."""
    if not iso_str:
        return None
    try:
        # Handle various ISO formats
        if "T" in iso_str:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.date().isoformat()
        return iso_str[:10]  # Already YYYY-MM-DD
    except Exception:
        return None


def _extract_location_name(uds: dict) -> str | None:
    """Extract location name from venues."""
    venues = uds.get("venues", [])
    if venues:
        return venues[0].get("title")
    return None


def _extract_city(uds: dict) -> str | None:
    """Extract city from venues."""
    venues = uds.get("venues", [])
    if venues:
        return venues[0].get("city")
    return None


def _extract_contact(data: dict, contact_type: str) -> str | None:
    """Extract contact info."""
    contacts = data.get("contacts", {})
    return contacts.get(contact_type)


async def save_to_supabase_storage(
    run_id: str,
    festival_slug: str,
    uds_json: dict,
    llm_log_json: dict | None = None,
    debug_artifacts: dict[str, bytes] | None = None,
) -> tuple[str | None, str | None]:
    """Save UDS, LLM log, and debug artifacts to Supabase Storage.
    
    Args:
        run_id: Parser run ID
        festival_slug: URL-safe festival name
        uds_json: UDS JSON to save
        llm_log_json: Optional LLM log JSON (all requests/responses)
        debug_artifacts: Optional dict of artifact_name -> bytes
        
    Returns:
        Tuple of (UDS public URL, LLM log public URL) or (None, None) if failed
    """
    # Import Supabase client lazily
    try:
        from main import get_supabase_client
    except ImportError:
        logger.warning("Supabase client not available")
        return None, None
    
    client = get_supabase_client()
    if not client:
        logger.warning("Supabase client not configured")
        return None, None
    
    base_path = f"festival_parsing/{festival_slug}/{run_id}"
    uds_public_url = None
    llm_log_public_url = None
    
    try:
        # Upload UDS JSON
        uds_bytes = json.dumps(uds_json, ensure_ascii=False, indent=2).encode("utf-8")
        uds_path = f"{base_path}/uds.json"
        
        client.storage.from_(SUPABASE_PARSER_BUCKET).upload(
            path=uds_path,
            file=uds_bytes,
            file_options={"content-type": "application/json"},
        )
        uds_public_url = client.storage.from_(SUPABASE_PARSER_BUCKET).get_public_url(uds_path)
        logger.info("Saved UDS to Supabase: %s", uds_public_url)
        
        # Upload LLM log JSON
        if llm_log_json:
            llm_log_bytes = json.dumps(llm_log_json, ensure_ascii=False, indent=2).encode("utf-8")
            llm_log_path = f"{base_path}/llm_log.json"
            
            client.storage.from_(SUPABASE_PARSER_BUCKET).upload(
                path=llm_log_path,
                file=llm_log_bytes,
                file_options={"content-type": "application/json"},
            )
            llm_log_public_url = client.storage.from_(SUPABASE_PARSER_BUCKET).get_public_url(llm_log_path)
            logger.info("Saved LLM log to Supabase: %s", llm_log_public_url)
        
        # Upload debug artifacts
        if debug_artifacts:
            for name, data in debug_artifacts.items():
                artifact_path = f"{base_path}/debug/{name}"
                content_type = "application/octet-stream"
                if name.endswith(".json"):
                    content_type = "application/json"
                elif name.endswith(".html"):
                    content_type = "text/html"
                elif name.endswith(".png"):
                    content_type = "image/png"
                
                client.storage.from_(SUPABASE_PARSER_BUCKET).upload(
                    path=artifact_path,
                    file=data,
                    file_options={"content-type": content_type},
                )
        
        return uds_public_url, llm_log_public_url
        
    except Exception as e:
        logger.error("Failed to save to Supabase: %s", e)
        return None, None


async def process_festival_url(
    db: "Database",
    bot: "Bot",
    chat_id: int,
    url: str,
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> tuple["Festival", str | None, str | None]:
    """Full pipeline: URL -> Kaggle -> UDS -> DB -> Telegraph -> Storage.
    
    Args:
        db: Database instance
        bot: Telegram bot instance
        chat_id: Chat ID for status updates
        url: Festival URL to parse
        status_callback: Optional callback for status updates
        
    Returns:
        Tuple of (Festival, UDS JSON URL or None, LLM log URL or None)
    """
    run_id = generate_run_id(url)
    logger.info("Starting festival parser: run_id=%s url=%s", run_id, url)
    
    # Get Kaggle datasets from settings
    # These should be configured in environment
    dataset_sources = [
        os.getenv("KAGGLE_GEMMA_CIPHER_DATASET", ""),
        os.getenv("KAGGLE_GEMMA_KEY_DATASET", ""),
    ]
    dataset_sources = [ds for ds in dataset_sources if ds]
    
    if len(dataset_sources) < 2:
        logger.warning("Gemma API key datasets not configured, parser may fail")
    
    async def _status_update(phase: str, status: str) -> None:
        if status_callback:
            await status_callback(f"{phase}: {status}")
    
    # Run Kaggle kernel
    final_status, output_files, duration = await run_festival_parser_kernel(
        url=url,
        run_id=run_id,
        dataset_sources=dataset_sources,
        status_callback=_status_update,
    )
    
    if final_status != "complete":
        raise RuntimeError(f"Kaggle job failed: {final_status} (run_id={run_id})")
    
    # Parse UDS output
    uds = parse_uds_output(output_files)
    if not uds:
        raise RuntimeError(f"No valid UDS output found (run_id={run_id})")
    
    # Parse LLM log output (for debugging/analysis)
    llm_log = parse_llm_log_output(output_files)
    
    # Upsert festival to database
    festival = await upsert_festival_from_uds(db, uds, run_id, url)
    
    # Save to Supabase Storage (UDS + LLM log)
    slug = generate_festival_slug(festival.name)
    uds_url, llm_log_url = await save_to_supabase_storage(
        run_id=run_id,
        festival_slug=slug,
        uds_json=uds,
        llm_log_json=llm_log,
    )
    
    # Update festival with storage path
    if uds_url:
        async with db.get_session() as session:
            from models import Festival
            from sqlalchemy import update
            
            await session.execute(
                update(Festival)
                .where(Festival.id == festival.id)
                .values(uds_storage_path=f"festival_parsing/{slug}/{run_id}/uds.json")
            )
            await session.commit()
    
    # Sync Telegraph page and navigation
    try:
        from main import sync_festival_page, rebuild_fest_nav_if_changed
        await sync_festival_page(db, festival.name)
        await rebuild_fest_nav_if_changed(db)
    except Exception as e:
        logger.error("Failed to sync pages: %s", e)
    
    # Set menu highlight for 3 days
    try:
        from main import set_setting_value
        from datetime import timedelta
        highlight_until = datetime.now(timezone.utc) + timedelta(days=3)
        await set_setting_value(db, "festival_highlight_until", highlight_until.isoformat())
    except Exception as e:
        logger.error("Failed to set menu highlight: %s", e)
    
    return festival, uds_url, llm_log_url


def parse_llm_log_output(file_paths: list[str]) -> dict | None:
    """Parse LLM log JSON from kernel output files.
    
    Args:
        file_paths: List of downloaded file paths
        
    Returns:
        Parsed LLM log dict or None if not found
    """
    for path in file_paths:
        if path.endswith("llm_log.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to parse LLM log %s: %s", path, e)
    return None

