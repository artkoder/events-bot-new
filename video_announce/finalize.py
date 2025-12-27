from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from sqlalchemy import select

from db import Database
from main import HTTP_SEMAPHORE, ask_4o, get_http_session
from models import EventPoster, VideoAnnounceItem
from poster_ocr import recognize_posters

from .about import normalize_about_text, normalize_about_with_fallback
from .prompts import FINAL_TEXT_RESPONSE_FORMAT, finalize_prompt
from .types import FinalizedItem, PosterEnrichment, RankedEvent

logger = logging.getLogger(__name__)

DESCRIPTION_EXCERPT_LIMIT = 400


@dataclass
class _PromptItem:
    event_id: int
    title: str
    description: str | None
    date: str
    time: str
    city: str | None
    location: str | None
    topics: list[str]
    is_free: bool
    poster_text: str | None
    ocr_title: str | None
    search_digest: str | None
    promoted: bool


async def _load_posters(db: Database, event_ids: Iterable[int]) -> dict[int, list[EventPoster]]:
    if not event_ids:
        return {}
    async with db.get_session() as session:
        result = await session.execute(
            select(EventPoster).where(EventPoster.event_id.in_(list(event_ids)))
        )
        posters = list(result.scalars().all())
    grouped: dict[int, list[EventPoster]] = defaultdict(list)
    for poster in posters:
        grouped[poster.event_id].append(poster)
    return grouped


async def _download_missing_posters(posters: Sequence[EventPoster]) -> list[tuple[bytes, EventPoster]]:
    payloads: list[tuple[bytes, EventPoster]] = []
    if not posters:
        return payloads
    session = get_http_session()
    for poster in posters:
        if poster.ocr_text or not poster.catbox_url:
            continue
        try:
            async with HTTP_SEMAPHORE:
                resp = await session.get(poster.catbox_url)
                resp.raise_for_status()
                data = await resp.read()
        except Exception:  # pragma: no cover - network failures
            logger.warning("video_announce: failed to fetch poster bytes url=%s", poster.catbox_url)
            continue
        payloads.append((data, poster))
    return payloads


async def _ensure_ocr_cached(db: Database, grouped: dict[int, list[EventPoster]]) -> None:
    missing: list[tuple[bytes, EventPoster]] = []
    for posters in grouped.values():
        missing.extend(await _download_missing_posters(posters))
    if not missing:
        return
    payloads = [{"data": data} for data, _ in missing]
    results, _, _ = await recognize_posters(
        db, payloads, log_context={"feature": "video_announce.finalize"}
    )
    cache_map = {row.hash: row for row in results}
    async with db.get_session() as session:
        for data, poster in missing:
            cache = cache_map.get(poster.poster_hash)
            if cache is None:
                continue
            poster.ocr_text = cache.text
            poster.ocr_title = cache.title
            poster.prompt_tokens = cache.prompt_tokens
            poster.completion_tokens = cache.completion_tokens
            poster.total_tokens = cache.total_tokens
            session.add(poster)
        await session.commit()



def _build_enrichments(grouped: dict[int, list[EventPoster]]) -> dict[int, PosterEnrichment]:
    enrichments: dict[int, PosterEnrichment] = {}
    for event_id, posters in grouped.items():
        title = None
        text = None
        source = None
        for poster in posters:
            raw_text = (poster.ocr_text or "").strip()
            raw_title = (poster.ocr_title or "").strip()
            if raw_text or raw_title:
                text = raw_text
                title = raw_title
                source = poster.catbox_url or "ocr_cache"
                break
        enrichments[event_id] = PosterEnrichment(event_id=event_id, title=title, text=text, source=source)
    return enrichments


def _build_prompt_items(
    ranked: Sequence[RankedEvent],
    enrichments: dict[int, PosterEnrichment],
    descriptions: dict[int, str],
) -> list[_PromptItem]:
    items: list[_PromptItem] = []
    for r in ranked:
        e = r.event
        enrich = enrichments.get(e.id)
        description = descriptions.get(e.id)
        items.append(
            _PromptItem(
                event_id=e.id,
                title=e.title,
                description=description,
                date=e.date,
                time=e.time,
                city=e.city,
                location=e.location_name,
                topics=list(getattr(e, "topics", []) or []),
                is_free=bool(getattr(e, "is_free", False)),
                poster_text=enrich.text if enrich else None,
                ocr_title=enrich.title if enrich else None,
                search_digest=getattr(e, "search_digest", None),
                promoted=bool((e.video_include_count or 0) > 3),
            )
        )
    return items


def _description_excerpt(text: str | None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    return raw[:DESCRIPTION_EXCERPT_LIMIT]





def _normalize_title(title: str | None, limit: int = 12) -> str:
    collapsed = " ".join(str(title or "").replace("\n", " ").split())
    trimmed = collapsed.strip("«»\"' <>.,!?:;")
    if len(trimmed) > limit:
        trimmed = trimmed[:limit].rstrip()
    return trimmed





def _parse_final_response(raw: str, known_ids: set[int]) -> list[FinalizedItem]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("video_announce: failed to parse LLM final text JSON")
        return []
    items = data.get("items") if isinstance(data, dict) else None
    parsed: list[FinalizedItem] = []
    if not isinstance(items, list):
        return parsed
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id")
        if not isinstance(event_id, int) or event_id not in known_ids:
            continue
        title = _normalize_title(item.get("final_title") or item.get("title"))
        about = normalize_about_text(item.get("about"))
        description = str(item.get("description") or "").strip()
        if not title or not description:
            continue
        parsed.append(
            FinalizedItem(
                event_id=event_id,
                title=title,
                about=about,
                description=description,
                use_ocr=bool(item.get("use_ocr")) if "use_ocr" in item else False,
                poster_source=item.get("poster_source"),
            )
        )
    return parsed


async def prepare_final_texts(
    db: Database, session_id: int, ranked: Sequence[RankedEvent]
) -> list[FinalizedItem]:
    if not ranked:
        return []
    event_ids = [r.event.id for r in ranked]
    events = [r.event for r in ranked]
    event_map = {ev.id: ev for ev in events}
    grouped_posters = await _load_posters(db, event_ids)
    await _ensure_ocr_cached(db, grouped_posters)
    enrichments = _build_enrichments(grouped_posters)
    descriptions = {
        ev.id: excerpt
        for ev in events
        if (excerpt := _description_excerpt(getattr(ev, "description", None)))
    }
    prompt_items = _build_prompt_items(ranked, enrichments, descriptions)
    payload = json.dumps([item.__dict__ for item in prompt_items], ensure_ascii=False)
    try:
        raw = await ask_4o(
            payload,
            system_prompt=finalize_prompt(),
            response_format=FINAL_TEXT_RESPONSE_FORMAT,
            meta={"source": "video_announce.finalize"},
        )
    except Exception:  # pragma: no cover - remote failure
        logger.exception("video_announce: finalize prompt failed")
        return []
    finalized = _parse_final_response(raw, set(event_ids))
    for fin in finalized:
        event = event_map.get(fin.event_id)
        enrichment = enrichments.get(fin.event_id)
        # Use ocr_title (enrichment.title) as filter key for deduplication
        ocr_key = (enrichment.title if enrichment else None) or (enrichment.text if enrichment else None)
        fin.about = normalize_about_with_fallback(
            fin.about,
            ocr_text=ocr_key,
        )
    enrich_map = {en.event_id: en for en in enrichments.values()}
    async with db.get_session() as session:
        result = await session.execute(
            select(VideoAnnounceItem).where(VideoAnnounceItem.session_id == session_id)
        )
        item_map = {it.event_id: it for it in result.scalars().all()}
        for fin in finalized:
            item = item_map.get(fin.event_id)
            if not item:
                continue
            enrichment = enrich_map.get(fin.event_id)
            item.final_title = fin.title
            item.final_about = fin.about
            item.final_description = fin.description
            item.use_ocr = fin.use_ocr or bool(enrichment and enrichment.text)
            # Store enrichment.title (ocr_title) in poster_text for reuse in render
            item.poster_text = enrichment.title if enrichment else None
            item.poster_source = fin.poster_source or (enrichment.source if enrichment else None)
            session.add(item)
        await session.commit()
    return finalized
