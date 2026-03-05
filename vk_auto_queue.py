from __future__ import annotations

import asyncio
import logging
import os
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, Literal

from db import Database
from ops_run import finish_ops_run, start_ops_run

import vk_intake
import vk_review

logger = logging.getLogger(__name__)

_vk_auto_import_cancel_requests: set[tuple[int, int]] = set()

def _timings_enabled() -> bool:
    raw = (os.getenv("PIPELINE_TIMINGS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

_VK_CANCEL_RE = re.compile(
    r"(?i)\b("
    r"отмен\w*|"
    r"не\s+состо\w*|"
    r"перенос\w*|"
    # Avoid false positives like "иллюстрации перенесут вас..." (transport you),
    # while still catching reschedule notices: "перенесено/перенесён/перенесли/перенесём".
    r"перенес(?:ен(?:а|о)?|ена|ено|ены|ён(?:а|о)?|ёна|ёно|ёны|ли|ем|ём)\b|"
    r"сдвинул\w*\s+срок\w*|"
    r"отложен\w*|"
    r"показ\s+не\s+состо\w*"
    r")\b"
)


def _looks_like_cancellation_notice(text: str | None) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if not _VK_CANCEL_RE.search(raw):
        return False
    # Require at least one anchor to avoid silencing on generic news posts.
    # Prefer using the same extraction helpers as the cancellation flow.
    try:
        year_hint = datetime.now(timezone.utc).year
    except Exception:
        year_hint = None
    if year_hint and _parse_ru_date_from_text(raw, year_hint=year_hint):
        return True
    if _extract_title_hint(raw):
        return True
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", raw):
        return True
    if re.search(r"\b\d{1,2}\.\d{1,2}\b", raw):
        return True
    return False


def _parse_ru_date_from_text(text: str, *, year_hint: int | None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})(?:\.(20\d{2}))?\b", raw)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3)) if (m.group(3) or "").strip() else year_hint
        if not year:
            return None
        try:
            return datetime(year, month, day).date().isoformat()
        except Exception:
            return None
    m = re.search(
        r"(?i)\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
        raw,
    )
    if not m:
        return None
    day = int(m.group(1))
    month_word = (m.group(2) or "").casefold()
    try:
        from smart_event_update import _RU_MONTHS_GENITIVE  # type: ignore
    except Exception:
        _RU_MONTHS_GENITIVE = {}
    month = int(_RU_MONTHS_GENITIVE.get(month_word) or 0)
    if not month:
        return None
    year = year_hint
    if not year:
        return None
    try:
        return datetime(year, month, day).date().isoformat()
    except Exception:
        return None


def _extract_title_hint(text: str | None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # Common pattern: "кинофестиваля <Title>"
    m = re.search(
        r"(?i)\b(?:кинофестивал\w*|фестивал\w*|мероприят\w*|показ)\s+([A-Za-zА-Яа-яЁё][^\\n\\r.,!?:;]{3,80})",
        raw,
    )
    if m:
        value = (m.group(1) or "").strip().strip("«»\"'()[]")
        value = re.sub(r"\s+", " ", value).strip()
        if 4 <= len(value) <= 90:
            return value
    # Otherwise: prefer 2+ capitalized latin words (e.g. "Manhattan Short Online")
    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){1,4}\b",
        raw,
    )
    if candidates:
        # Prefer the longest phrase (more specific).
        best = max(candidates, key=lambda s: len(s))
        best = re.sub(r"\s+", " ", best).strip()
        if 4 <= len(best) <= 80:
            return best
    return None


def _title_tokens(title: str | None) -> set[str]:
    if not title:
        return set()
    words = re.findall(r"[a-zа-яё0-9]{4,}", title.lower(), flags=re.IGNORECASE)
    return {w for w in words if w and not w.isdigit()}


async def _cancel_matching_event_from_notice(
    db: Database,
    *,
    notice_text: str,
    source_url: str,
    source_name: str | None,
    location_hint: str | None,
    published_at: datetime | None,
) -> tuple[int | None, str | None]:
    """Try to find a matching event and mark it as cancelled/postponed (inactive)."""
    from sqlalchemy import select
    from models import Event, EventSource, EventSourceFact
    import main as main_mod

    year_hint = None
    if published_at is not None:
        try:
            year_hint = int(published_at.astimezone(timezone.utc).year)
        except Exception:
            year_hint = None
    if year_hint is None:
        year_hint = datetime.now(timezone.utc).year

    date_hint = _parse_ru_date_from_text(notice_text, year_hint=year_hint)
    time_hint = None
    m_time = re.search(r"\b(\d{1,2})[:.](\d{2})\b", notice_text or "")
    if m_time:
        try:
            hh = int(m_time.group(1))
            mm = int(m_time.group(2))
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                time_hint = f"{hh:02d}:{mm:02d}"
        except Exception:
            time_hint = None
    title_hint = _extract_title_hint(notice_text)
    is_postponed = bool(
        re.search(
            r"(?i)\b(?:перенос\w*|перенес(?:ен(?:а|о)?|ена|ено|ены|ён(?:а|о)?|ёна|ёно|ёны|ли|ем|ём)|сдвинул\w*)\b",
            notice_text or "",
        )
    )
    kind = "перенос" if is_postponed else "отмена"
    next_status = "postponed" if is_postponed else "cancelled"

    async with db.get_session() as session:
        stmt = select(Event).where(Event.lifecycle_status == "active")
        if date_hint:
            stmt = stmt.where(Event.date.like(f"{date_hint}%"))
        res = await session.execute(stmt)
        events = list(res.scalars().all())

        if not events:
            return None, f"no_events_for_date={date_hint or 'any'}"

        loc_norm = (location_hint or "").strip().casefold()
        title_tokens = _title_tokens(title_hint)

        scored: list[tuple[int, Event]] = []
        for ev in events:
            score = 0
            if time_hint and (ev.time or "").strip() == time_hint:
                score += 2
            if loc_norm:
                ev_loc = (ev.location_name or "").strip().casefold()
                if ev_loc == loc_norm:
                    score += 4
                elif loc_norm and (loc_norm in ev_loc or ev_loc in loc_norm):
                    score += 2
            if title_hint:
                ev_title = (ev.title or "").strip()
                if title_hint.casefold() in ev_title.casefold() or ev_title.casefold() in title_hint.casefold():
                    score += 5
                ev_tokens = _title_tokens(ev_title)
                overlap = len(title_tokens & ev_tokens) if title_tokens else 0
                score += min(6, overlap * 2)
            scored.append((score, ev))

        scored.sort(key=lambda x: (x[0], int(getattr(x[1], "id", 0) or 0)), reverse=True)
        best_score, best = scored[0]
        # Guardrail: require at least some matching signal.
        if best_score < 4 and title_hint:
            return None, f"low_confidence score={best_score} title_hint={title_hint!r} date={date_hint or ''}"
        if best_score < 2 and not title_hint:
            return None, f"low_confidence score={best_score} date={date_hint or ''}"

        best.lifecycle_status = next_status
        session.add(best)
        await session.flush()

        # Source URLs are unique per event (ux_event_source_event_url); cancellation notices can
        # arrive as an edit of the original post. Reuse existing source row when present.
        src = (
            (
                await session.execute(
                    select(EventSource).where(
                        EventSource.event_id == int(best.id),
                        EventSource.source_url == str(source_url),
                    )
                )
            )
            .scalars()
            .first()
        )
        if src is None:
            src = EventSource(
                event_id=int(best.id),
                source_type="vk_cancel",
                source_url=str(source_url),
                source_text=(notice_text or "")[:4000],
            )
            session.add(src)
            await session.flush()
        else:
            src.source_type = "vk_cancel"
            src.source_text = (notice_text or "")[:4000]
            session.add(src)
            await session.flush()
        note = f"❌ {kind}: событие помечено как {next_status} по источнику VK"
        if source_name:
            note += f" ({source_name})"
        session.add(
            EventSourceFact(
                event_id=int(best.id),
                source_id=int(src.id),
                fact=note,
                status="note",
            )
        )
        await session.commit()

        try:
            await main_mod.schedule_event_update_tasks(db, best, skip_vk_sync=True)
        except Exception:
            logger.warning("vk_auto: failed to schedule rebuild after cancel", exc_info=True)

        return int(best.id), None


def request_vk_auto_import_cancel(*, chat_id: int, operator_id: int) -> None:
    """Request cancellation of the currently running VK auto import for this chat/operator."""
    try:
        key = (int(chat_id), int(operator_id))
    except Exception:
        return
    _vk_auto_import_cancel_requests.add(key)


def _vk_auto_import_cancelled(*, chat_id: int, operator_id: int) -> bool:
    try:
        key = (int(chat_id), int(operator_id))
    except Exception:
        return False
    return key in _vk_auto_import_cancel_requests


def _clear_vk_auto_import_cancel(*, chat_id: int, operator_id: int) -> None:
    try:
        key = (int(chat_id), int(operator_id))
    except Exception:
        return
    _vk_auto_import_cancel_requests.discard(key)


def _vk_wall_url(group_id: int, post_id: int) -> str:
    return f"https://vk.com/wall-{int(group_id)}_{int(post_id)}"


def _best_url(sizes: Sequence[Mapping[str, Any]]) -> str:
    if not sizes:
        return ""
    best = max(
        sizes,
        key=lambda s: (s.get("width", 0) or 0) * (s.get("height", 0) or 0),
    )
    return str(best.get("url") or best.get("src") or "")


def _extract_media_urls(item: Mapping[str, Any], *, limit: int = 12) -> list[str]:
    """Extract image URLs from a VK wall item (photos + some common thumbnails)."""
    urls: list[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        u = (url or "").strip()
        if not u or u in seen:
            return
        seen.add(u)
        urls.append(u)

    def _process_atts(atts: Sequence[Mapping[str, Any]] | None) -> None:
        for att in atts or []:
            if len(urls) >= limit:
                return
            url = ""
            if att.get("type") == "photo":
                photo = att.get("photo") or {}
                url = _best_url(photo.get("sizes") or [])
            elif att.get("type") == "link":
                link = att.get("link") or {}
                url = _best_url(((link.get("photo") or {}).get("sizes") or []))
            elif att.get("type") == "video":
                video = att.get("video") or {}
                url = _best_url(video.get("first_frame") or video.get("image") or [])
            elif att.get("type") == "doc":
                sizes = (
                    ((att.get("doc") or {}).get("preview") or {})
                    .get("photo", {})
                    .get("sizes", [])
                )
                url = _best_url(sizes or [])
            if url:
                _add(url)

    _process_atts(item.get("attachments") or [])
    copy_history = item.get("copy_history") or []
    if copy_history and isinstance(copy_history, list):
        first = copy_history[0] if copy_history else None
        if isinstance(first, Mapping):
            _process_atts(first.get("attachments") or [])

    return urls


@dataclass(frozen=True)
class VkFetchStatus:
    ok: bool
    kind: Literal["ok", "not_found", "access_denied", "vk_api_error", "network_error"]
    error_code: int | None = None
    error: str | None = None


def _vk_auto_allow_stale_inbox_text() -> bool:
    raw = (os.getenv("VK_AUTO_IMPORT_ALLOW_STALE_INBOX_TEXT_ON_FETCH_FAIL") or "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


async def fetch_vk_post_text_and_photos(
    group_id: int,
    post_id: int,
    *,
    db: Database | None = None,
    bot: Any | None = None,
    limit: int = 12,
) -> tuple[str, list[str], datetime | None, dict[str, Any] | None, VkFetchStatus]:
    """Fetch VK wall post (text + image URLs) via VK API.

    Uses `main.vk_api` so it can read via service token when configured.
    """
    import main as main_mod

    try:
        resp = await main_mod.vk_api("wall.getById", posts=f"-{int(group_id)}_{int(post_id)}")
    except Exception as exc:
        # NOTE: VKAPIError is defined in main.py. We inspect it dynamically to avoid
        # a hard import cycle and still keep error codes for decision-making.
        code = None
        msg = str(exc or "").strip()
        kind: VkFetchStatus["kind"] = "network_error"
        try:
            VKAPIError = getattr(main_mod, "VKAPIError", None)
            if VKAPIError is not None and isinstance(exc, VKAPIError):
                code = getattr(exc, "code", None)
                low = (getattr(exc, "message", None) or msg or "").casefold()
                # wall.getById may fail when a post is deleted/unavailable.
                if any(tok in low for tok in ("post was deleted", "post deleted", "has been deleted", "пост удал")):
                    kind = "not_found"
                elif int(code or 0) in {100, 113}:
                    kind = "not_found"
                elif int(code or 0) in {15, 30} or "access denied" in low:
                    kind = "access_denied"
                else:
                    kind = "vk_api_error"
        except Exception:
            kind = "network_error"
        logger.warning(
            "vk_auto: wall.getById failed -%s_%s kind=%s code=%s err=%s",
            group_id,
            post_id,
            kind,
            code,
            msg,
        )
        return "", [], None, None, VkFetchStatus(False, kind, error_code=code, error=msg or None)

    # `main.vk_api()` already returns the unwrapped VK "response" payload.
    # Keep compatibility with legacy callers that may still pass {"response": ...}.
    raw: Any = resp
    if isinstance(resp, Mapping) and "response" in resp:
        raw = resp.get("response")
    items: list[Mapping[str, Any]] = []
    if isinstance(raw, dict):
        raw_items = raw.get("items")
        if isinstance(raw_items, list):
            items = [it for it in raw_items if isinstance(it, Mapping)]
        elif any(k in raw for k in ("text", "attachments", "date")):
            items = [raw]
    elif isinstance(raw, list):
        items = [it for it in raw if isinstance(it, Mapping)]

    text = ""
    published_at: datetime | None = None
    photos: list[str] = []
    metrics: dict[str, Any] | None = None
    for it in items:
        candidate_text = it.get("text") if isinstance(it.get("text"), str) else ""
        repost_text = ""
        copy_history = it.get("copy_history")
        if isinstance(copy_history, list) and copy_history:
            first = copy_history[0]
            if isinstance(first, Mapping):
                rt = first.get("text")
                if isinstance(rt, str) and rt.strip():
                    repost_text = rt.strip()
        base = candidate_text.strip() if isinstance(candidate_text, str) else ""
        combined = base
        if repost_text:
            if not combined:
                combined = repost_text
            elif repost_text not in combined:
                combined = f"{combined}\n\n[Репост]\n{repost_text}".strip()
        if combined:
            text = combined
        ts = it.get("date")
        if isinstance(ts, (int, float)):
            try:
                published_at = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            except Exception:
                published_at = None
        if metrics is None:
            m: dict[str, Any] = {}
            try:
                v = (it.get("views") or {}).get("count")
                if isinstance(v, int) and v >= 0:
                    m["views"] = v
            except Exception:
                pass
            try:
                l = (it.get("likes") or {}).get("count")
                if isinstance(l, int) and l >= 0:
                    m["likes"] = l
            except Exception:
                pass
            metrics = m or None
        photos.extend(_extract_media_urls(it, limit=limit))
        if text:
            break

    if not items:
        return "", [], None, None, VkFetchStatus(False, "not_found", error="empty_response")

    # Deduplicate photos while preserving order.
    out_photos: list[str] = []
    seen: set[str] = set()
    for u in photos:
        if u and u not in seen:
            seen.add(u)
            out_photos.append(u)
        if len(out_photos) >= limit:
            break

    return text, out_photos, published_at, metrics, VkFetchStatus(True, "ok")


async def _load_festival_hints(db: Database) -> tuple[list[str], list[tuple[str, int]]]:
    """Load festival names + alias pairs in the format expected by vk_intake.build_event_drafts()."""
    from sqlalchemy import select
    from models import Festival
    from main import normalize_alias

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        festivals = res_f.scalars().all()

    festival_names = sorted(
        {
            (fest.name or "").strip()
            for fest in festivals
            if (fest.name or "").strip()
        }
    )
    alias_pairs: list[tuple[str, int]] = []
    if not festival_names:
        return [], []

    index_map = {name: idx for idx, name in enumerate(festival_names)}
    for fest in festivals:
        name = (fest.name or "").strip()
        if not name:
            continue
        idx = index_map.get(name)
        if idx is None:
            continue
        base_norm = normalize_alias(name)
        for alias in getattr(fest, "aliases", None) or []:
            norm = normalize_alias(alias)
            if not norm or norm == base_norm:
                continue
            alias_pairs.append((norm, idx))

    if alias_pairs:
        seen_pairs: set[tuple[str, int]] = set()
        deduped: list[tuple[str, int]] = []
        for pair in alias_pairs:
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            deduped.append(pair)
        alias_pairs = deduped
    return festival_names, alias_pairs


@dataclass
class VkAutoImportReport:
    batch_id: str
    inbox_processed: int = 0
    inbox_imported: int = 0
    inbox_rejected: int = 0
    inbox_failed: int = 0
    inbox_deferred: int = 0
    skipped_requeued: int = 0
    cancelled: bool = False
    created_event_ids: list[int] = field(default_factory=list)
    updated_event_ids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class VkInboxPrefetch:
    source_url: str
    source_name: str | None
    location_hint: str | None
    default_time: str | None
    default_ticket_link: str | None
    text: str
    photos: list[str]
    publish_ts: datetime | int | float | None
    published_at: datetime | None
    source_is_festival: bool = False
    metrics: dict[str, Any] | None = None
    vk_fetch: VkFetchStatus | None = None
    drafts: Any | None = None
    stage_sec: dict[str, float] = field(default_factory=dict)
    error: str | None = None


async def _resolve_bot_username(bot: Any | None) -> str | None:
    if not bot or not hasattr(bot, "get_me"):
        return None
    try:
        me = await bot.get_me()
    except Exception:
        return None
    username = getattr(me, "username", None)
    if not username:
        return None
    return str(username).lstrip("@") or None


def _log_deeplink(bot_username: str | None, event_id: int) -> str | None:
    if not bot_username:
        return None
    return f"https://t.me/{bot_username}?start=log_{int(event_id)}"


def _shorten_reason(value: str | None, *, limit: int = 220) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _render_progress_text(
    icon: str,
    *,
    current_no: int,
    total_txt: str,
    source_url: str,
    extra_lines: Sequence[str] | None = None,
) -> str:
    header = f"{icon} Разбираю VK пост {current_no}/{total_txt}: {source_url}"
    lines = [header]
    for line in (extra_lines or []):
        line = (line or "").strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


async def _update_progress_message(
    bot: Any,
    *,
    chat_id: int,
    message_id: int | None,
    text: str,
) -> None:
    """Best-effort: edit an existing progress message, fallback to sending a new message."""
    payload = (text or "").strip()
    if not payload:
        return
    if message_id and hasattr(bot, "edit_message_text"):
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id),
                text=payload,
                disable_web_page_preview=True,
            )
            return
        except Exception:
            logger.warning("vk_auto: progress_edit_failed", exc_info=True)
    try:
        await bot.send_message(chat_id, payload, disable_web_page_preview=True)
    except Exception:
        logger.warning("vk_auto: progress_send_failed", exc_info=True)


async def _send_unified_event_report(
    db: Database,
    bot: Any,
    chat_id: int,
    *,
    created: list[int],
    updated: list[int],
    source_url: str,
    added_posters_by_event_id: Mapping[int, int] | None = None,
    post_metrics: Mapping[str, Any] | None = None,
    post_popularity: str | None = None,
) -> bool:
    from source_parsing.handlers import build_added_event_info, build_updated_event_info
    import html

    bot_username = await _resolve_bot_username(bot)
    lines: list[str] = []

    created = list(dict.fromkeys([int(eid) for eid in (created or []) if eid]))
    updated = list(dict.fromkeys([int(eid) for eid in (updated or []) if eid]))
    overlap = set(created) & set(updated)
    if overlap:
        # When one VK post yields multiple drafts that merge into the same event,
        # the event can be both "created" and then immediately "updated" within
        # the same run. Facts/logs reflect the LAST merge, so show such events in
        # the updated bucket to keep the report consistent.
        created = [eid for eid in created if eid not in overlap]

    ctx = None
    try:
        from source_parsing.smart_update_report import build_smart_update_report_context

        ctx = await build_smart_update_report_context(
            db,
            event_ids=(created + updated),
            source_urls=[source_url],
        )
    except Exception:
        ctx = None

    tz = getattr(ctx, "tz", None)
    sources_by_eid = getattr(ctx, "sources_by_event_id", None) or {}
    video_counts = getattr(ctx, "video_count_by_event_id", None) or {}
    ticket_queue_by_eid = getattr(ctx, "ticket_queue_by_event_id", None) or {}
    fest_queue_by_src = getattr(ctx, "festival_queue_by_source_url", None) or {}

    def _ics_line(url: str | None, *, has_time: bool) -> str:
        value = (url or "").strip()
        if value:
            safe = html.escape(value, quote=True)
            return f'ICS: <a href="{safe}">ics</a>'
        return "ICS: ⏳" if has_time else "ICS: —"

    def _sources_lines(eid: int) -> list[str]:
        rows = list(sources_by_eid.get(int(eid)) or [])
        if not rows or not tz:
            return []
        from source_parsing.smart_update_report import format_dt_compact, short_url_label

        out: list[str] = ["  Источники:"]
        limit = 24
        shown = rows[:limit]
        for imported_at, url in shown:
            stamp = format_dt_compact(imported_at, tz)
            label = short_url_label(url) or url
            if str(url).strip().startswith(("http://", "https://")):
                safe_href = html.escape(str(url).strip(), quote=True)
                safe_label = html.escape(label)
                out.append(f"  {stamp} <a href=\"{safe_href}\">{safe_label}</a>")
            else:
                out.append(f"  {stamp} {html.escape(label)}")
        if len(rows) > limit:
            out.append(f"  … ещё {len(rows) - limit}")
        return out

    def _queue_lines(eid: int) -> list[str]:
        out: list[str] = []
        fest = fest_queue_by_src.get((source_url or "").strip())
        if fest:
            name = (getattr(fest, "festival_name", None) or getattr(fest, "festival_full", None) or "").strip()
            ctx2 = (getattr(fest, "festival_context", None) or "").strip()
            status = (getattr(fest, "status", None) or "").strip()
            fid = getattr(fest, "id", None)
            tail = name or ctx2
            extra = f" {tail}" if tail else ""
            id_part = f" (id={int(fid)})" if isinstance(fid, int) and fid > 0 else ""
            st_part = f" {status}" if status else ""
            out.append(f"  🎪 festival_queue:{st_part}{extra}{id_part}".strip())

        tickets = list(ticket_queue_by_eid.get(int(eid)) or [])
        if tickets:
            first = tickets[0]
            href = html.escape(str(getattr(first, 'url', '') or '').strip(), quote=True)
            label = html.escape(str(getattr(first, 'site_kind', '') or 'tickets').strip() or "tickets")
            extra = f" +{len(tickets)}" if len(tickets) > 1 else ""
            if href:
                out.append(f'  🎟 ticket_site_queue:{extra} <a href="{href}">{label}</a>')
            else:
                out.append(f"  🎟 ticket_site_queue:{extra}".strip())
        return out

    def _render_fact_stats(stats: Mapping[str, Any] | None) -> str:
        data = stats or {}
        if not data:
            return "Факты: —"
        added = int(data.get("added") or 0)
        dup = int(data.get("duplicate") or 0)
        conf = int(data.get("conflict") or 0)
        note = int(data.get("note") or 0)
        return f"Факты: ✅{added} ↩️{dup} ⚠️{conf} ℹ️{note}"

    def _render_facts_and_photos(info: Any, *, eid: int) -> str:
        stats_text = _render_fact_stats(getattr(info, "fact_stats", None))
        added_posters = getattr(info, "added_posters", None)
        try:
            added_posters_int = int(added_posters) if added_posters is not None else None
        except Exception:
            added_posters_int = None
        photo_count = getattr(info, "photo_count", None)
        try:
            photos = int(photo_count or 0)
        except Exception:
            photos = 0
        try:
            videos_total = int(video_counts.get(int(eid), 0) or 0)
        except Exception:
            videos_total = 0
        if added_posters_int is None:
            photos_label = f"Иллюстрации: {'⚠️0' if photos == 0 else photos}"
        else:
            photos_label = f"Иллюстрации: +{added_posters_int}, всего {'⚠️0' if photos == 0 else photos}"
        videos_label = f" | Видео: {videos_total}" if videos_total > 0 else ""
        return f"{stats_text} | {photos_label}{videos_label}"

    def _render_meta(date_value: str | None, time_value: str | None) -> str:
        meta: list[str] = []
        if date_value:
            meta.append(str(date_value))
        if time_value:
            meta.append(str(time_value))
        return f" — {' '.join(meta)}" if meta else ""

    def _render_source(source_url: str, info: Any) -> str:
        ord_value = getattr(info, "source_ordinal", None)
        total_value = getattr(info, "source_total", None)
        if isinstance(ord_value, int) and ord_value > 0:
            if isinstance(total_value, int) and total_value > 0:
                return f"Источник #{ord_value}/{total_value}: {source_url}"
            return f"Источник #{ord_value}: {source_url}"
        return f"Источник: {source_url}"

    if created or updated:
        lines.append("<b>Smart Update (детали событий):</b>")
        if post_metrics:
            parts: list[str] = []
            v = post_metrics.get("views") if isinstance(post_metrics, Mapping) else None
            l = post_metrics.get("likes") if isinstance(post_metrics, Mapping) else None
            if isinstance(v, int) and v >= 0:
                parts.append(f"views={v}")
            if isinstance(l, int) and l >= 0:
                parts.append(f"likes={l}")
            if parts:
                lines.append(f"Метрики поста: {html.escape(' '.join(parts))}")
    if created:
        lines.append(f"✅ Созданные события: {len(created)}")
        for eid in created[:12]:
            info = await build_added_event_info(db, int(eid), "vk", source_url=source_url)
            if not info:
                continue
            if added_posters_by_event_id is not None:
                info.added_posters = int(added_posters_by_event_id.get(int(eid), 0) or 0)
            title = html.escape(info.title or "Без названия")
            if (post_popularity or "").strip():
                title = f"{html.escape(str(post_popularity).strip())} {title}"
            tg_url = html.escape(info.telegraph_url or "", quote=True)
            meta = _render_meta(info.date, info.time)
            if info.telegraph_url:
                lines.append(f"• <a href=\"{tg_url}\">{title}</a> (id={info.event_id}){meta}")
            else:
                lines.append(f"• {title} (id={info.event_id}){meta}")
            lines.append(f"  {html.escape(_render_source(source_url, info))}")
            if not info.telegraph_url:
                lines.append("  Telegraph: ⏳ в очереди")
            lines.extend(_sources_lines(int(info.event_id)))
            if info.log_cmd:
                href = _log_deeplink(bot_username, int(info.event_id))
                if href:
                    lines.append(
                        f"  Лог: <a href=\"{html.escape(href, quote=True)}\">{html.escape(info.log_cmd)}</a>"
                    )
                else:
                    lines.append(f"  Лог: {html.escape(info.log_cmd)}")
            lines.append(f"  {_ics_line(info.ics_url, has_time=bool((info.time or '').strip()))}")
            lines.append(f"  {_render_facts_and_photos(info, eid=int(info.event_id))}")
            lines.extend(_queue_lines(int(info.event_id)))
            lines.append("")
        if len(created) > 12:
            lines.append(f"... ещё {len(created) - 12}")
    if updated:
        lines.append(f"🔄 Обновлённые события: {len(updated)}")
        for eid in updated[:12]:
            info = await build_updated_event_info(
                db, int(eid), "vk", "full_update", source_url=source_url
            )
            if not info:
                continue
            if added_posters_by_event_id is not None:
                info.added_posters = int(added_posters_by_event_id.get(int(eid), 0) or 0)
            title = html.escape(info.title or "Без названия")
            if (post_popularity or "").strip():
                title = f"{html.escape(str(post_popularity).strip())} {title}"
            tg_url = html.escape(info.telegraph_url or "", quote=True)
            meta = _render_meta(info.date, info.time)
            if info.telegraph_url:
                lines.append(f"• <a href=\"{tg_url}\">{title}</a> (id={info.event_id}){meta}")
            else:
                lines.append(f"• {title} (id={info.event_id}){meta}")
            lines.append(f"  {html.escape(_render_source(source_url, info))}")
            if not info.telegraph_url:
                lines.append("  Telegraph: ⏳ в очереди")
            lines.extend(_sources_lines(int(info.event_id)))
            if info.log_cmd:
                href = _log_deeplink(bot_username, int(info.event_id))
                if href:
                    lines.append(
                        f"  Лог: <a href=\"{html.escape(href, quote=True)}\">{html.escape(info.log_cmd)}</a>"
                    )
                else:
                    lines.append(f"  Лог: {html.escape(info.log_cmd)}")
            lines.append(f"  {_ics_line(info.ics_url, has_time=bool((info.time or '').strip()))}")
            lines.append(f"  {_render_facts_and_photos(info, eid=int(info.event_id))}")
            lines.extend(_queue_lines(int(info.event_id)))
            lines.append("")
        if len(updated) > 12:
            lines.append(f"... ещё {len(updated) - 12}")

    if not lines:
        return True
    text = "\n".join(lines).strip()
    if not text:
        return True
    try:
        await asyncio.wait_for(
            bot.send_message(chat_id, text, parse_mode="HTML", disable_web_page_preview=True),
            timeout=30,
        )
        logger.info(
            "vk_auto: unified_report_sent chat_id=%s created=%s updated=%s source=%s",
            chat_id,
            len(created),
            len(updated),
            source_url,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "vk_auto: unified_report_timeout chat_id=%s created=%s updated=%s source=%s",
            chat_id,
            len(created),
            len(updated),
            source_url,
        )
        return False
    except Exception:
        logger.exception("vk_auto: failed to send unified report")
        return False


async def _prefetch_vk_inbox_row(
    db: Database,
    *,
    bot: Any | None,
    post: Any,
    source_url: str,
    festival_names: list[str] | None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None,
) -> VkInboxPrefetch:
    """Best-effort prefetch for VK auto-import pipelining (N+1).

    Prefetch has no write side-effects in the DB; it only prepares data that would
    otherwise be computed inside `_process_vk_inbox_row`.
    """
    import main as main_mod

    stage: dict[str, float] = {}

    # Fetch VK source defaults.
    source_name_val: str | None = None
    location_hint_val: str | None = None
    default_time_val: str | None = None
    default_ticket_link_val: str | None = None
    source_is_festival = False
    t0 = time.monotonic()
    try:
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT name, location, default_time, default_ticket_link, festival_source FROM vk_source WHERE group_id=?",
                (post.group_id,),
            )
            row = await cur.fetchone()
        if row:
            source_name_val, location_hint_val, default_time_val, default_ticket_link_val, source_is_festival = row
    except Exception as exc:
        logger.warning("vk_auto: prefetch db_source_defaults failed url=%s err=%s", source_url, exc)
    stage["db_source_defaults"] = float(time.monotonic() - t0)

    # Refresh text/photos from VK (best effort) to include attachments.
    t0 = time.monotonic()
    fetched_text, photos, published_at, metrics, vk_fetch = await fetch_vk_post_text_and_photos(
        post.group_id, post.post_id, db=db, bot=bot
    )
    stage["vk_fetch_post"] = float(time.monotonic() - t0)
    allow_stale = _vk_auto_allow_stale_inbox_text()
    text = ""
    if vk_fetch.ok:
        text = (fetched_text or post.text or "").strip()
    elif allow_stale and vk_fetch.kind != "not_found":
        text = (post.text or "").strip()
    publish_ts: datetime | int | float | None = getattr(post, "date", None)
    if published_at is not None:
        publish_ts = int(published_at.timestamp())

    # Normalize configured hints to canonical venue lines when possible.
    if not (location_hint_val or "").strip() and (source_name_val or "").strip():
        try:
            matcher = getattr(main_mod, "_match_known_venue", None)
            if callable(matcher):
                venue = matcher(source_name_val)
                if venue is not None:
                    location_hint_val = getattr(venue, "canonical_line", None) or location_hint_val
        except Exception:
            logger.warning("vk_auto: prefetch infer location_hint failed", exc_info=True)
    elif (location_hint_val or "").strip():
        try:
            matcher = getattr(main_mod, "_match_known_venue", None)
            if callable(matcher):
                venue = matcher(location_hint_val)
                if venue is not None:
                    location_hint_val = getattr(venue, "canonical_line", None) or location_hint_val
        except Exception:
            logger.warning("vk_auto: prefetch canonicalize location_hint failed", exc_info=True)

    parse_festival_names = festival_names if source_is_festival else None
    parse_festival_alias_pairs = festival_alias_pairs if source_is_festival else None
    drafts: Any | None = None
    err: str | None = None
    prefetch_drafts = (os.getenv("VK_AUTO_IMPORT_PREFETCH_DRAFTS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    t0 = time.monotonic()
    try:
        if (
            prefetch_drafts
            and vk_fetch.ok
            and text
            and (not _looks_like_cancellation_notice(text))
        ):
            drafts, _festival_info = await vk_intake.build_event_drafts(
                text,
                photos=photos,
                source_name=source_name_val,
                location_hint=location_hint_val,
                default_time=default_time_val,
                default_ticket_link=default_ticket_link_val,
                operator_extra=None,
                festival_names=parse_festival_names,
                festival_alias_pairs=parse_festival_alias_pairs or None,
                festival_hint=bool(source_is_festival),
                publish_ts=publish_ts,
                event_ts_hint=post.event_ts_hint,
                prefilter_obvious_non_events=True,
                db=db,
            )
    except Exception as exc:
        drafts = None
        err = str(exc)
    stage["build_drafts_total"] = float(time.monotonic() - t0)

    return VkInboxPrefetch(
        source_url=source_url,
        source_name=source_name_val,
        location_hint=location_hint_val,
        default_time=default_time_val,
        default_ticket_link=default_ticket_link_val,
        source_is_festival=bool(source_is_festival),
        text=text,
        photos=list(photos or []),
        publish_ts=publish_ts,
        published_at=published_at,
        metrics=metrics if isinstance(metrics, dict) else None,
        vk_fetch=vk_fetch,
        drafts=drafts,
        stage_sec=stage,
        error=err,
    )


async def run_vk_auto_import(
    db: Database,
    bot: Any,
    *,
    chat_id: int,
    limit: int = 25,
    operator_id: int = 0,
    include_skipped: bool = False,
    trigger: str = "manual",
    run_id: str | None = None,
) -> VkAutoImportReport:
    """Auto-import VK inbox queue sequentially via Smart Update (LLM).

    Intended usage:
    - scheduled job (admin chat)
    - manual command for E2E debugging
    """
    batch_id = f"auto:{int(time.time())}"
    report = VkAutoImportReport(batch_id=batch_id)
    ops_run_id = await start_ops_run(
        db,
        kind="vk_auto_import",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={
            "batch_id": batch_id,
            "run_id": run_id,
            "limit_requested": limit,
            "include_skipped": int(bool(include_skipped)),
        },
    )
    _clear_vk_auto_import_cancel(chat_id=chat_id, operator_id=operator_id)
    try:
        limit_int = int(limit)
    except Exception:
        limit_int = 25
    unbounded = limit_int <= 0

    await vk_review.release_stale_locks(db)

    def _env_enabled(name: str, default: bool) -> bool:
        raw = (os.getenv(name) or "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    send_progress = _env_enabled("VK_AUTO_IMPORT_SEND_PROGRESS", True)
    try:
        progress_every = int(os.getenv("VK_AUTO_IMPORT_PROGRESS_EVERY", "1") or "1")
    except Exception:
        progress_every = 1
    progress_every = max(1, min(progress_every, 50))

    # Optional: include previously skipped rows in the run. This is useful for
    # E2E over a prod DB snapshot where an operator may have skipped items
    # earlier, but we still want to validate Smart Update correctness.
    reject_cutoff = 0
    if include_skipped:
        reject_window_h = float(os.getenv("VK_REVIEW_REJECT_H", "2") or "2")
        reject_window_h = max(0.0, reject_window_h)
        reject_cutoff = int(time.time()) + int(reject_window_h * 3600)
        async with db.raw_conn() as conn:
            cur = await conn.execute("SELECT COUNT(1) FROM vk_inbox WHERE status='pending'")
            row = await cur.fetchone()
            pending_count = int((row[0] if row else 0) or 0)
        # Do not inflate queue for this run: requeue only enough skipped rows
        # to fill the remaining batch up to `limit`.
        if unbounded:
            requeue_limit = 10**9
        else:
            requeue_limit = max(0, int(limit_int) - pending_count)
        if requeue_limit <= 0:
            requeue_limit = 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                """
                SELECT id
                FROM vk_inbox
                WHERE status='skipped' AND (event_ts_hint IS NULL OR event_ts_hint >= ?)
                ORDER BY CASE WHEN event_ts_hint IS NULL THEN 1 ELSE 0 END,
                         event_ts_hint ASC,
                         date ASC,
                         id ASC
                LIMIT ?
                """,
                (reject_cutoff, requeue_limit),
            )
            ids = [int(r[0]) for r in (await cur.fetchall() or [])]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await conn.execute(
                    f"""
                    UPDATE vk_inbox
                    SET status='pending',
                        locked_by=NULL,
                        locked_at=NULL,
                        review_batch=NULL
                    WHERE id IN ({placeholders})
                    """,
                    tuple(ids),
                )
                await conn.commit()
                report.skipped_requeued = len(ids)
                logger.info(
                    "vk_auto: requeued_skipped=%s cutoff=%s pending=%s limit=%s",
                    len(ids),
                    reject_cutoff,
                    pending_count,
                    limit,
                )
    elif not reject_cutoff:
        reject_window_h = float(os.getenv("VK_REVIEW_REJECT_H", "2") or "2")
        reject_window_h = max(0.0, reject_window_h)
        reject_cutoff = int(time.time()) + int(reject_window_h * 3600)

    # Preload festival hints once per run.
    try:
        festival_names, festival_alias_pairs = await _load_festival_hints(db)
    except Exception as exc:
        festival_names, festival_alias_pairs = [], []
        report.errors.append(f"festival_hints_failed: {exc}")

    total_estimate = None
    try:
        statuses = ("pending", "skipped") if include_skipped else ("pending",)
        placeholders = ",".join(["?"] * len(statuses))
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                f"""
                SELECT COUNT(1)
                FROM vk_inbox
                WHERE status IN ({placeholders})
                  AND (event_ts_hint IS NULL OR event_ts_hint >= ?)
                """,
                (*statuses, int(reject_cutoff)),
            )
            row = await cur.fetchone()
            total_estimate = int((row[0] if row else 0) or 0)
            if not unbounded:
                total_estimate = min(int(total_estimate), int(limit_int))
    except Exception:
        total_estimate = None

    start = time.time()
    from heavy_ops import heavy_operation

    async with heavy_operation(
        kind="vk_auto_import",
        trigger=trigger,
        mode="wait",
        run_id=run_id,
        operator_id=operator_id,
        chat_id=chat_id,
    ):
        current_no = 0
        prefetch_enabled = _env_enabled("VK_AUTO_IMPORT_PREFETCH", True)

        async def _await_prefetch(task: asyncio.Task | None) -> VkInboxPrefetch | None:
            if task is None:
                return None
            try:
                res = await task
            except asyncio.CancelledError:
                return None
            except Exception:
                return None
            return res if isinstance(res, VkInboxPrefetch) else None

        def _start_prefetch(post_obj: Any) -> asyncio.Task | None:
            if not prefetch_enabled or not post_obj:
                return None
            next_url = _vk_wall_url(post_obj.group_id, post_obj.post_id)
            return asyncio.create_task(
                _prefetch_vk_inbox_row(
                    db,
                    bot=bot,
                    post=post_obj,
                    source_url=next_url,
                    festival_names=festival_names,
                    festival_alias_pairs=festival_alias_pairs,
                )
            )

        if unbounded:
            post = await vk_review.pick_next(
                db,
                operator_id,
                batch_id,
                requeue_skipped=False,
                prefer_oldest=True,
                strict_chronological=True,
            )
            prefetch_task = _start_prefetch(post) if post else None
            while post:
                if _vk_auto_import_cancelled(chat_id=chat_id, operator_id=operator_id):
                    report.cancelled = True
                    try:
                        await vk_review.mark_pending(db, int(post.id))
                    except Exception:
                        logger.warning("vk_auto: mark_pending failed on cancel", exc_info=True)
                    if prefetch_task:
                        prefetch_task.cancel()
                    break

                current_no += 1
                report.inbox_processed += 1
                source_url = _vk_wall_url(post.group_id, post.post_id)
                total_txt = str(int(total_estimate)) if isinstance(total_estimate, int) else "?"
                progress_mid: int | None = None
                if send_progress and (current_no % progress_every == 0):
                    try:
                        msg = await bot.send_message(
                            chat_id,
                            f"⏳ Разбираю VK пост {current_no}/{total_txt}: {source_url}",
                            disable_web_page_preview=True,
                        )
                        progress_mid = getattr(msg, "message_id", None) or getattr(msg, "id", None)
                        if progress_mid is not None:
                            progress_mid = int(progress_mid)
                    except Exception:
                        logger.warning("vk_auto: progress_send_failed", exc_info=True)

                prefetched_row = await _await_prefetch(prefetch_task)
                process_task = asyncio.create_task(
                    _process_vk_inbox_row(
                        db,
                        bot,
                        chat_id=chat_id,
                        operator_id=operator_id,
                        batch_id=batch_id,
                        post=post,
                        source_url=source_url,
                        report=report,
                        festival_names=festival_names,
                        festival_alias_pairs=festival_alias_pairs,
                        progress_message_id=progress_mid,
                        progress_current_no=current_no,
                        progress_total_txt=total_txt,
                        prefetched=prefetched_row,
                    )
                )

                next_post = await vk_review.pick_next(
                    db,
                    operator_id,
                    batch_id,
                    requeue_skipped=False,
                    prefer_oldest=True,
                    strict_chronological=True,
                    resume_locked=False,
                )
                next_prefetch_task = _start_prefetch(next_post) if next_post else None

                try:
                    await process_task
                except Exception as exc:
                    report.inbox_failed += 1
                    report.errors.append(f"unexpected_failed {source_url}: {exc}")
                    try:
                        await vk_review.mark_failed(db, int(post.id))
                    except Exception:
                        logger.warning("vk_auto: mark_failed failed after exception", exc_info=True)
                    logger.exception(
                        "vk_auto: unexpected exception in inbox row processing id=%s url=%s",
                        getattr(post, "id", None),
                        source_url,
                    )
                    try:
                        await bot.send_message(
                            chat_id,
                            f"❌ VK auto import: техническая ошибка при обработке поста\n{source_url}\n{exc}",
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        pass

                post = next_post
                prefetch_task = next_prefetch_task
        else:
            remaining = max(1, int(limit_int))
            post = await vk_review.pick_next(
                db,
                operator_id,
                batch_id,
                requeue_skipped=False,
                prefer_oldest=True,
                strict_chronological=True,
            )
            prefetch_task = _start_prefetch(post) if post else None
            while post and remaining > 0:
                if _vk_auto_import_cancelled(chat_id=chat_id, operator_id=operator_id):
                    report.cancelled = True
                    try:
                        await vk_review.mark_pending(db, int(post.id))
                    except Exception:
                        logger.warning("vk_auto: mark_pending failed on cancel", exc_info=True)
                    if prefetch_task:
                        prefetch_task.cancel()
                    break

                current_no += 1
                remaining -= 1
                report.inbox_processed += 1
                source_url = _vk_wall_url(post.group_id, post.post_id)
                total_txt = str(int(total_estimate)) if isinstance(total_estimate, int) else str(int(limit_int))
                progress_mid = None
                if send_progress and (current_no % progress_every == 0):
                    try:
                        msg = await bot.send_message(
                            chat_id,
                            f"⏳ Разбираю VK пост {current_no}/{total_txt}: {source_url}",
                            disable_web_page_preview=True,
                        )
                        progress_mid = getattr(msg, "message_id", None) or getattr(msg, "id", None)
                        if progress_mid is not None:
                            progress_mid = int(progress_mid)
                    except Exception:
                        logger.warning("vk_auto: progress_send_failed", exc_info=True)

                prefetched_row = await _await_prefetch(prefetch_task)
                process_task = asyncio.create_task(
                    _process_vk_inbox_row(
                        db,
                        bot,
                        chat_id=chat_id,
                        operator_id=operator_id,
                        batch_id=batch_id,
                        post=post,
                        source_url=source_url,
                        report=report,
                        festival_names=festival_names,
                        festival_alias_pairs=festival_alias_pairs,
                        progress_message_id=progress_mid,
                        progress_current_no=current_no,
                        progress_total_txt=total_txt,
                        prefetched=prefetched_row,
                    )
                )

                next_post = None
                next_prefetch_task = None
                if remaining > 0:
                    next_post = await vk_review.pick_next(
                        db,
                        operator_id,
                        batch_id,
                        requeue_skipped=False,
                        prefer_oldest=True,
                        strict_chronological=True,
                        resume_locked=False,
                    )
                    next_prefetch_task = _start_prefetch(next_post) if next_post else None

                try:
                    await process_task
                except Exception as exc:
                    report.inbox_failed += 1
                    report.errors.append(f"unexpected_failed {source_url}: {exc}")
                    try:
                        await vk_review.mark_failed(db, int(post.id))
                    except Exception:
                        logger.warning("vk_auto: mark_failed failed after exception", exc_info=True)
                    logger.exception(
                        "vk_auto: unexpected exception in inbox row processing id=%s url=%s",
                        getattr(post, "id", None),
                        source_url,
                    )
                    try:
                        await bot.send_message(
                            chat_id,
                            f"❌ VK auto import: техническая ошибка при обработке поста\n{source_url}\n{exc}",
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        pass

                post = next_post
                prefetch_task = next_prefetch_task

    took = time.time() - start
    total_txt = str(int(total_estimate)) if isinstance(total_estimate, int) else "?"
    summary = (
        "🏁 VK auto import завершён\n"
        f"batch: {batch_id}\n"
        f"limit: {'all' if unbounded else limit_int}\n"
        f"include_skipped: {1 if include_skipped else 0}\n"
        f"cancelled: {1 if report.cancelled else 0}\n"
        f"queue processed: {report.inbox_processed}/{total_txt}\n"
        f"inbox imported: {report.inbox_imported}\n"
        f"inbox rejected: {report.inbox_rejected}\n"
        f"inbox failed: {report.inbox_failed}\n"
        f"inbox deferred: {report.inbox_deferred}\n"
        f"events created: {len(set(report.created_event_ids))}\n"
        f"events updated: {len(set(report.updated_event_ids))}\n"
        f"took_sec: {took:.1f}"
    )
    try:
        await bot.send_message(chat_id, summary, disable_web_page_preview=True)
    except Exception:
        logger.exception("vk_auto: failed to send summary")

    _clear_vk_auto_import_cancel(chat_id=chat_id, operator_id=operator_id)
    await finish_ops_run(
        db,
        run_id=ops_run_id,
        status="canceled" if report.cancelled else "success",
        metrics={
            "inbox_processed": int(report.inbox_processed),
            "inbox_imported": int(report.inbox_imported),
            "inbox_rejected": int(report.inbox_rejected),
            "inbox_failed": int(report.inbox_failed),
            "inbox_deferred": int(report.inbox_deferred),
            "events_created": int(len(set(report.created_event_ids))),
            "events_updated": int(len(set(report.updated_event_ids))),
            "cancelled": int(bool(report.cancelled)),
            "skipped_requeued": int(report.skipped_requeued),
            "include_skipped": int(bool(include_skipped)),
            "limit": int(limit_int),
            "duration_sec": round(float(took), 3),
        },
        details={
            "batch_id": batch_id,
            "run_id": run_id,
            "errors": list(report.errors or [])[:40],
        },
    )
    return report


async def _process_vk_inbox_row(
    db: Database,
    bot: Any,
    *,
    chat_id: int,
    operator_id: int,
    batch_id: str,
    post: Any,
    source_url: str,
    report: VkAutoImportReport,
    festival_names: list[str] | None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None,
    progress_message_id: int | None,
    progress_current_no: int,
    progress_total_txt: str,
    prefetched: VkInboxPrefetch | None = None,
) -> None:
    import main as main_mod

    start_ts = time.monotonic()
    timings_on = _timings_enabled()
    t_stage: dict[str, float] = {}
    try:
        slow_log_sec = float(os.getenv("VK_AUTO_IMPORT_SLOW_ROW_LOG_SEC", "60") or "60")
    except Exception:
        slow_log_sec = 60.0
    slow_log_sec = max(0.0, min(slow_log_sec, 3600.0))

    def _tmark(name: str, elapsed: float) -> None:
        t_stage[name] = float(elapsed)

    def _log_row_timing(*, drafts_count: int, ok_value: bool) -> None:
        took_total = time.monotonic() - start_ts
        slow_log_due = took_total >= slow_log_sec if slow_log_sec > 0 else True
        if not (timings_on or slow_log_due or not ok_value):
            return
        try:
            logger.info(
                "timing vk_auto_import_row inbox_id=%s group_id=%s post_id=%s drafts=%s ok=%s took_sec=%.3f stages=%s",
                int(getattr(post, "id", 0) or 0),
                int(getattr(post, "group_id", 0) or 0),
                int(getattr(post, "post_id", 0) or 0),
                int(drafts_count),
                1 if ok_value else 0,
                float(took_total),
                {k: round(v, 3) for k, v in sorted(t_stage.items())},
            )
        except Exception:
            pass

    async def _emit_progress(icon: str, extra_lines: Sequence[str] | None = None) -> None:
        if not progress_message_id:
            return
        text = _render_progress_text(
            icon,
            current_no=int(progress_current_no),
            total_txt=str(progress_total_txt),
            source_url=source_url,
            extra_lines=extra_lines,
        )
        await _update_progress_message(
            bot,
            chat_id=chat_id,
            message_id=progress_message_id,
            text=text,
        )

    # Fetch VK source defaults.
    source_name_val: str | None = None
    location_hint_val: str | None = None
    default_time_val: str | None = None
    default_ticket_link_val: str | None = None
    source_is_festival = False
    pf = prefetched if (prefetched and prefetched.source_url == source_url) else None

    if pf is not None:
        source_name_val = pf.source_name
        location_hint_val = pf.location_hint
        default_time_val = pf.default_time
        default_ticket_link_val = pf.default_ticket_link
        source_is_festival = bool(pf.source_is_festival)
        _tmark("db_source_defaults", float(pf.stage_sec.get("db_source_defaults", 0.0) or 0.0))

        text = (pf.text or "").strip()
        photos = list(pf.photos or [])
        published_at = pf.published_at
        metrics = pf.metrics if isinstance(pf.metrics, dict) else None
        vk_fetch = pf.vk_fetch
        publish_ts = pf.publish_ts if pf.publish_ts is not None else getattr(post, "date", None)
        _tmark("vk_fetch_post", float(pf.stage_sec.get("vk_fetch_post", 0.0) or 0.0))
    else:
        t0 = time.monotonic()
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT name, location, default_time, default_ticket_link, festival_source FROM vk_source WHERE group_id=?",
                (post.group_id,),
            )
            row = await cur.fetchone()
        if row:
            source_name_val, location_hint_val, default_time_val, default_ticket_link_val, source_is_festival = row
        _tmark("db_source_defaults", time.monotonic() - t0)

        # Refresh text/photos from VK (best effort) to include attachments.
        t0 = time.monotonic()
        fetched_text, photos, published_at, metrics, vk_fetch = await fetch_vk_post_text_and_photos(
            post.group_id, post.post_id, db=db, bot=bot
        )
        _tmark("vk_fetch_post", time.monotonic() - t0)
        allow_stale = _vk_auto_allow_stale_inbox_text()
        if vk_fetch.ok:
            text = (fetched_text or post.text or "").strip()
        elif allow_stale and vk_fetch.kind != "not_found":
            text = (post.text or "").strip()
        else:
            text = ""
        publish_ts = post.date
        if published_at is not None:
            publish_ts = int(published_at.timestamp())

        # If VK source has no explicit location hint configured, try to map its name
        # to a canonical location from docs/reference/locations.md.
        if not (location_hint_val or "").strip() and (source_name_val or "").strip():
            try:
                matcher = getattr(main_mod, "_match_known_venue", None)
                if callable(matcher):
                    venue = matcher(source_name_val)
                    if venue is not None:
                        location_hint_val = getattr(venue, "canonical_line", None) or location_hint_val
            except Exception:
                logger.warning("vk_auto: failed to infer location_hint from reference", exc_info=True)
        elif (location_hint_val or "").strip():
            # Normalize configured hints to canonical venue lines when possible,
            # so LLM gets a stable "name, address, city" format.
            try:
                matcher = getattr(main_mod, "_match_known_venue", None)
                if callable(matcher):
                    venue = matcher(location_hint_val)
                    if venue is not None:
                        location_hint_val = getattr(venue, "canonical_line", None) or location_hint_val
            except Exception:
                logger.warning("vk_auto: failed to canonicalize location_hint", exc_info=True)

    if vk_fetch is not None and not vk_fetch.ok:
        allow_stale = _vk_auto_allow_stale_inbox_text()
        if vk_fetch.kind == "not_found":
            report.inbox_rejected += 1
            await vk_review.mark_rejected(db, int(post.id))
            await _emit_progress(
                "🗑️",
                [
                    "Результат: пост недоступен в VK (удалён/не найден)",
                    f"Причина: {vk_fetch.error_code or ''} {_shorten_reason(vk_fetch.error) or ''}".strip(),
                    f"took_sec: {(time.monotonic() - start_ts):.1f}",
                ],
            )
            return
        if not (allow_stale and (text or "").strip()):
            report.inbox_failed += 1
            report.errors.append(
                f"vk_fetch_failed {source_url}: kind={vk_fetch.kind} code={vk_fetch.error_code} err={vk_fetch.error}"
            )
            await vk_review.mark_failed(db, int(post.id))
            await _emit_progress(
                "❌",
                [
                    "Результат: не удалось загрузить пост из VK (wall.getById)",
                    f"Причина: kind={vk_fetch.kind} code={vk_fetch.error_code}",
                    f"took_sec: {(time.monotonic() - start_ts):.1f}",
                ],
            )
            return

    post_popularity: str | None = None
    if isinstance(metrics, dict) and metrics:
        try:
            from source_parsing.post_metrics import (
                compute_age_day,
                load_vk_popularity_baseline,
                normalize_age_day,
                popularity_marks,
                upsert_vk_post_metric,
            )

            collected_ts = int(time.time())
            published_ts: int | None = None
            if isinstance(publish_ts, datetime):
                dt = publish_ts
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                published_ts = int(dt.timestamp())
            elif isinstance(publish_ts, (int, float)):
                published_ts = int(publish_ts)

            age_day = normalize_age_day(compute_age_day(published_ts=published_ts, collected_ts=collected_ts))
            if isinstance(age_day, int) and age_day >= 0:
                views = metrics.get("views")
                likes = metrics.get("likes")
                await upsert_vk_post_metric(
                    db,
                    group_id=int(post.group_id),
                    post_id=int(post.post_id),
                    age_day=int(age_day),
                    source_url=source_url,
                    post_ts=published_ts,
                    views=int(views) if isinstance(views, int) else None,
                    likes=int(likes) if isinstance(likes, int) else None,
                    collected_ts=int(collected_ts),
                )
                baseline = await load_vk_popularity_baseline(
                    db,
                    group_id=int(post.group_id),
                    age_day=int(age_day),
                    now_ts=int(collected_ts),
                )
                marks = popularity_marks(
                    views=views if isinstance(views, int) else None,
                    likes=likes if isinstance(likes, int) else None,
                    baseline=baseline,
                )
                post_popularity = marks.text or None
        except Exception:
            logger.warning(
                "vk_auto: failed to persist/score post metrics gid=%s post_id=%s",
                getattr(post, "group_id", None),
                getattr(post, "post_id", None),
                exc_info=True,
            )

    # Cancellation/transfer notices: do not create new events. Instead, try to find the
    # matching existing event and mark it inactive (cancelled/postponed).
    if _looks_like_cancellation_notice(text):
        event_id, err = await _cancel_matching_event_from_notice(
            db,
            notice_text=text,
            source_url=source_url,
            source_name=source_name_val,
            location_hint=location_hint_val,
            published_at=published_at,
        )
        if event_id:
            # Fetch canonical date for batch month accounting.
            event_date_val: str | None = None
            try:
                async with db.raw_conn() as conn:
                    cur = await conn.execute("SELECT date FROM event WHERE id=?", (int(event_id),))
                    row = await cur.fetchone()
                    event_date_val = str(row[0]) if row and row[0] else None
            except Exception:
                event_date_val = None
            report.inbox_imported += 1
            report.updated_event_ids.append(int(event_id))
            # Link inbox row with the canceled event to keep queue idempotent.
            await vk_review.mark_imported_events(
                db,
                inbox_id=int(post.id),
                batch_id=batch_id,
                operator_id=operator_id,
                event_ids=[int(event_id)],
                event_dates=[event_date_val],
            )
            await _emit_progress(
                "🛑",
                [
                    "Результат: отмена/перенос — событие помечено неактивным",
                    f"event_id: {int(event_id)}",
                    f"took_sec: {(time.monotonic() - start_ts):.1f}",
                ],
            )
            # Unified report for operator (as an "updated" event).
            await _send_unified_event_report(
                db,
                bot,
                chat_id,
                created=[],
                updated=[int(event_id)],
                source_url=source_url,
                added_posters_by_event_id={int(event_id): 0},
                post_metrics=metrics,
                post_popularity=post_popularity,
            )
            return
        # Cancellation notices must not create new events.
        report.inbox_rejected += 1
        await vk_review.mark_rejected(db, post.id)
        await _emit_progress(
            "⛔",
            [
                "Результат: отмена/перенос — событие не найдено в базе",
                f"Причина: {_shorten_reason(err) or 'no_match'}",
                f"took_sec: {(time.monotonic() - start_ts):.1f}",
            ],
        )
        return

    if pf is not None and pf.drafts is not None and not (pf.error or "").strip():
        drafts = pf.drafts
        _tmark("build_drafts_total", float(pf.stage_sec.get("build_drafts_total", 0.0) or 0.0))
    else:
        try:
            parse_festival_names = festival_names if source_is_festival else None
            parse_festival_alias_pairs = festival_alias_pairs if source_is_festival else None
            rl_max_wait_sec = float(os.getenv("VK_AUTO_IMPORT_RATE_LIMIT_MAX_WAIT_SEC", "120") or "120")
            rl_max_wait_sec = max(0.0, min(rl_max_wait_sec, 1800.0))
            rl_deadline = time.monotonic() + rl_max_wait_sec
            t0 = time.monotonic()
            build_attempt = 1
            while True:
                try:
                    drafts, _festival_info = await vk_intake.build_event_drafts(
                        text,
                        photos=photos,
                        source_name=source_name_val,
                        location_hint=location_hint_val,
                        default_time=default_time_val,
                        default_ticket_link=default_ticket_link_val,
                        operator_extra=None,
                        festival_names=parse_festival_names,
                        festival_alias_pairs=parse_festival_alias_pairs or None,
                        festival_hint=bool(source_is_festival),
                        publish_ts=publish_ts,
                        event_ts_hint=post.event_ts_hint,
                        rate_limit_max_wait_sec=0,
                        prefilter_obvious_non_events=True,
                        db=db,
                    )
                    break
                except Exception as exc:
                    is_rate_limit = False
                    retry_after_ms = 0
                    try:
                        from google_ai.exceptions import (
                            RateLimitError as _RateLimitError,
                            ProviderError as _ProviderError,
                        )
                    except Exception:
                        _RateLimitError = None
                        _ProviderError = None
                    if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                        is_rate_limit = True
                        retry_after_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                    elif _ProviderError is not None and isinstance(exc, _ProviderError):
                        if int(getattr(exc, "status_code", 0) or 0) == 429:
                            is_rate_limit = True
                            retry_after_ms = int(getattr(exc, "retry_after_ms", 0) or 0)

                    if not is_rate_limit:
                        raise
                    if rl_max_wait_sec <= 0:
                        raise
                    if time.monotonic() >= rl_deadline:
                        raise

                    wait_sec = (
                        min(60.0, max(0.2, (retry_after_ms / 1000.0) + 0.2))
                        if retry_after_ms
                        else 1.0
                    )
                    logger.warning(
                        "vk_auto: rate_limited build_drafts attempt=%d retry_in=%.1fs url=%s err=%s",
                        build_attempt,
                        wait_sec,
                        source_url,
                        exc,
                    )
                    build_attempt += 1
                    await asyncio.sleep(wait_sec)
                    continue

            _tmark("build_drafts_total", time.monotonic() - t0)
        except Exception as exc:
            is_rate_limit = False
            retry_after_ms = 0
            try:
                from google_ai.exceptions import RateLimitError as _RateLimitError, ProviderError as _ProviderError
            except Exception:
                _RateLimitError = None
                _ProviderError = None
            if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                is_rate_limit = True
                retry_after_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
            elif _ProviderError is not None and isinstance(exc, _ProviderError):
                if int(getattr(exc, "status_code", 0) or 0) == 429:
                    is_rate_limit = True
                    retry_after_ms = int(getattr(exc, "retry_after_ms", 0) or 0)

            if is_rate_limit:
                report.inbox_deferred += 1
                report.errors.append(f"drafts_rate_limited {source_url}: {exc}")
                try:
                    rl_max_wait_sec = float(os.getenv("VK_AUTO_IMPORT_RATE_LIMIT_MAX_WAIT_SEC", "120") or "120")
                    rl_max_wait_sec = max(0.0, min(rl_max_wait_sec, 1800.0))
                    if rl_max_wait_sec <= 0:
                        await vk_review.mark_pending(db, int(post.id))
                    else:
                        # Keep the row locked to avoid immediately picking it again in the same
                        # unbounded run. The lock will auto-expire via vk_review._unlock_stale().
                        async with db.raw_conn() as conn:
                            await conn.execute(
                                "UPDATE vk_inbox SET status='locked', locked_at=CURRENT_TIMESTAMP WHERE id=?",
                                (int(post.id),),
                            )
                            await conn.commit()
                except Exception:
                    logger.warning("vk_auto: defer_lock_failed after rate limit", exc_info=True)
                retry_hint = (
                    f"Retry-after: {max(1, int(retry_after_ms / 1000))}s"
                    if retry_after_ms
                    else "Retry-after: —"
                )
                await _emit_progress(
                    "⏸️",
                    [
                        "Результат: лимит LLM — пост отложен",
                        f"Причина: {_shorten_reason(str(exc)) or '—'}",
                        retry_hint,
                        f"took_sec: {(time.monotonic() - start_ts):.1f}",
                    ],
                )
                return

            report.inbox_failed += 1
            report.errors.append(f"drafts_failed {source_url}: {exc}")
            logger.error(
                "vk_auto: build_event_drafts failed inbox_id=%s source=%s",
                int(post.id),
                source_url,
                exc_info=True,
            )
            await vk_review.mark_failed(db, int(post.id))
            await _emit_progress(
                "❌",
                [
                    "Результат: ошибка извлечения событий (drafts)",
                    f"Причина: {_shorten_reason(str(exc)) or '—'}",
                    f"took_sec: {(time.monotonic() - start_ts):.1f}",
                ],
            )
            _log_row_timing(drafts_count=0, ok_value=False)
            return

    if not drafts:
        report.inbox_rejected += 1
        await vk_review.mark_rejected(db, post.id)
        reason_line = None
        try:
            import re

            tzinfo = getattr(main_mod, "LOCAL_TZ", None) or timezone.utc
            now_dt = datetime.now(tzinfo)
            pub_dt = None
            if isinstance(publish_ts, (int, float)) and publish_ts:
                try:
                    pub_dt = datetime.fromtimestamp(float(publish_ts), tzinfo)
                except Exception:
                    pub_dt = None
            year = (pub_dt.year if pub_dt else now_dt.year)

            # Simple, explainable inference for operator messaging: dd.mm + optional HH:MM.
            m_date = re.search(r"\b(\d{1,2})\.(\d{1,2})\b", text or "")
            m_time = re.search(r"\b(\d{1,2})[:.](\d{2})\b", text or "")
            inferred_dt = None
            if m_date:
                day = int(m_date.group(1))
                month = int(m_date.group(2))
                hour = int(m_time.group(1)) if m_time else 0
                minute = int(m_time.group(2)) if m_time else 0
                try:
                    inferred_dt = datetime(year, month, day, hour, minute, tzinfo=tzinfo)
                except Exception:
                    inferred_dt = None

            if inferred_dt and inferred_dt < now_dt:
                reason_line = f"Причина: событие в прошлом: {inferred_dt.strftime('%Y-%m-%d %H:%M')}"
        except Exception:
            reason_line = None
        await _emit_progress(
            "⏭️",
            [
                "Результат: событий не найдено (LLM вернул 0)",
                reason_line or "",
                f"took_sec: {(time.monotonic() - start_ts):.1f}",
            ],
        )
        return

    # Filter low-confidence drafts (e.g. title likely copied from a recap of a past event).
    kept_drafts: list[Any] = []
    rejected_reasons: list[str] = []
    for d in list(drafts):
        reason = str(getattr(d, "reject_reason", "") or "").strip()
        if reason:
            rejected_reasons.append(reason)
            continue
        kept_drafts.append(d)

    if rejected_reasons and not kept_drafts:
        report.inbox_rejected += 1
        reason_short = _shorten_reason(rejected_reasons[0])
        report.errors.append(f"low_confidence {source_url}: {reason_short or 'low_confidence'}")
        await vk_review.mark_rejected(db, post.id)
        await _emit_progress(
            "⛔",
            [
                "Результат: низкая уверенность — событие пропущено",
                f"Причина: {reason_short or '—'}",
                f"took_sec: {(time.monotonic() - start_ts):.1f}",
            ],
        )
        return

    if rejected_reasons and kept_drafts:
        reason_short = _shorten_reason(rejected_reasons[0])
        report.errors.append(f"low_confidence_partial {source_url}: {reason_short or 'low_confidence'}")

    drafts = kept_drafts

    # If LLM returned drafts without location, use the source-level hint as a fallback.
    # This prevents Smart Update from rejecting otherwise valid events due to missing location.
    if (location_hint_val or "").strip():
        for draft in drafts:
            if not (getattr(draft, "venue", None) or "").strip():
                draft.venue = str(location_hint_val).strip()

    imported_event_ids: list[int] = []
    imported_event_dates: list[str | None] = []
    created_ids: list[int] = []
    updated_ids: list[int] = []
    added_posters_total = 0
    added_posters_by_event_id: dict[int, int] = {}
    partial_error: str | None = None
    inline_jobs_enabled = (os.getenv("VK_AUTO_IMPORT_INLINE_JOBS", "1") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    ok = True
    persist_total_sec = 0.0
    for draft in drafts:
        try:
            t0 = time.monotonic()
            res = await vk_intake.persist_event_and_pages(
                draft,
                photos,
                db,
                source_post_url=source_url,
                wait_for_telegraph_url=not inline_jobs_enabled,
            )
            took_one = time.monotonic() - t0
            persist_total_sec += float(took_one)

            imported_event_ids.append(int(res.event_id))
            imported_event_dates.append(res.event_date)
            if getattr(res, "smart_created", False) or getattr(res, "smart_status", "") == "created":
                created_ids.append(int(res.event_id))
            else:
                updated_ids.append(int(res.event_id))
            added = int(getattr(res, "smart_added_posters", 0) or 0)
            added_posters_total += added
            added_posters_by_event_id[int(res.event_id)] = added
        except Exception as exc:
            ok = False
            exc_txt = str(exc)
            if "smart_update rejected:" in exc_txt:
                report.inbox_rejected += 1
                report.errors.append(f"persist_rejected {source_url}: {exc_txt}")
                await vk_review.mark_rejected(db, post.id)
                await _emit_progress(
                    "⛔",
                    [
                        "Результат: Smart Update отклонил",
                        f"Причина: {_shorten_reason(exc_txt) or '—'}",
                        f"took_sec: {(time.monotonic() - start_ts):.1f}",
                    ],
                )
                _log_row_timing(drafts_count=len(drafts or []), ok_value=False)
                return
            if "smart_update returned no event_id:" in exc_txt:
                report.inbox_rejected += 1
                report.errors.append(f"persist_skipped {source_url}: {exc_txt}")
                await vk_review.mark_rejected(db, post.id)
                await _emit_progress(
                    "⏭️",
                    [
                        "Результат: Smart Update пропустил (нет event_id)",
                        f"Причина: {_shorten_reason(exc_txt) or '—'}",
                        f"took_sec: {(time.monotonic() - start_ts):.1f}",
                    ],
                )
                _log_row_timing(drafts_count=len(drafts or []), ok_value=False)
                return

            report.inbox_failed += 1
            report.errors.append(f"persist_failed {source_url}: {exc_txt}")
            if not imported_event_ids:
                await vk_review.mark_failed(db, post.id)
                await _emit_progress(
                    "❌",
                    [
                        "Результат: ошибка сохранения (persist)",
                        f"Причина: {_shorten_reason(exc_txt) or '—'}",
                        f"took_sec: {(time.monotonic() - start_ts):.1f}",
                    ],
                )
                _log_row_timing(drafts_count=len(drafts or []), ok_value=False)
                return
            # Partial success: keep already imported events linked to this inbox row.
            partial_error = exc_txt
            ok = True
            break
    if drafts:
        _tmark("persist_total", persist_total_sec)

    if not ok:
        _log_row_timing(drafts_count=len(drafts or []), ok_value=False)
        return

    t0 = time.monotonic()
    await vk_review.mark_imported_events(
        db,
        inbox_id=post.id,
        batch_id=batch_id,
        operator_id=operator_id,
        event_ids=imported_event_ids,
        event_dates=imported_event_dates,
    )
    _tmark("mark_imported_events", time.monotonic() - t0)
    report.inbox_imported += 1
    report.created_event_ids.extend(created_ids)
    report.updated_event_ids.extend(updated_ids)

    created_cnt = len(created_ids)
    updated_cnt = len(updated_ids)
    if created_cnt and not updated_cnt:
        icon = "✅"
    elif updated_cnt and not created_cnt:
        icon = "🔄"
    else:
        icon = "✅🔄"

    ids_preview = ", ".join(str(x) for x in (imported_event_ids[:5] or []))
    extra_lines = [
        f"Smart Update: ✅{created_cnt} 🔄{updated_cnt}",
        f"event_ids: {ids_preview}{'…' if len(imported_event_ids) > 5 else ''}",
        f"Иллюстрации: +{added_posters_total}",
        "Отчёт Smart Update: ⏳",
    ]
    if partial_error:
        extra_lines.insert(0, f"⚠️ Частично: {_shorten_reason(partial_error) or 'persist error'}")
    await _emit_progress(icon, extra_lines)

    if inline_jobs_enabled:
        timeout_sec = float(os.getenv("VK_AUTO_IMPORT_INLINE_JOBS_TIMEOUT_SEC", "90") or "90")
        t0 = time.monotonic()
        try:
            # Inline jobs exist only to make the operator report reflect the final
            # Telegraph URL right away. ICS publishing can be slow / flaky (and in
            # local E2E it may be intentionally misconfigured), so we do NOT wait
            # for it by default.
            allowed = {main_mod.JobTask.telegraph_build}

            include_ics_inline = (os.getenv("VK_AUTO_IMPORT_INLINE_INCLUDE_ICS") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            disable_ics_jobs = (os.getenv("DISABLE_ICS_JOBS") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if include_ics_inline and not disable_ics_jobs:
                allowed.add(main_mod.JobTask.ics_publish)
            for eid in imported_event_ids:
                await asyncio.wait_for(
                    main_mod.run_event_update_jobs(
                        db,
                        bot,
                        event_id=int(eid),
                        allowed_tasks=allowed,
                    ),
                    timeout=timeout_sec,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "vk_auto: inline event update jobs timeout source=%s events=%s timeout_sec=%s",
                source_url,
                imported_event_ids,
                timeout_sec,
            )
        except Exception:
            logger.exception(
                "vk_auto: inline event update jobs failed source=%s events=%s",
                source_url,
                imported_event_ids,
            )
        _tmark("inline_jobs", time.monotonic() - t0)

    # Send the unified report after inline Telegraph/ICS jobs so the operator sees
    # the final (potentially recreated) Telegraph URL, not the stale snapshot value.
    t0 = time.monotonic()
    report_sent = await _send_unified_event_report(
        db,
        bot,
        chat_id,
        created=created_ids,
        updated=updated_ids,
        source_url=source_url,
        added_posters_by_event_id=added_posters_by_event_id,
        post_metrics=metrics,
        post_popularity=post_popularity,
    )
    _tmark("send_unified_report", time.monotonic() - t0)
    extra_lines[-1] = f"Отчёт Smart Update: {'✅' if report_sent else '⚠️'}"
    extra_lines.append(f"took_sec: {(time.monotonic() - start_ts):.1f}")
    await _emit_progress(icon, extra_lines)
    _log_row_timing(drafts_count=len(drafts or []), ok_value=True)


async def vk_auto_import_scheduler(
    db: Database,
    bot: Any | None = None,
    *,
    run_id: str | None = None,
) -> None:
    """Scheduled job entrypoint: imports VK inbox queue when enabled.

    The report goes to ADMIN chat because there is no operator context.
    """
    if os.getenv("ENABLE_VK_AUTO_IMPORT", "").strip().lower() not in {"1", "true", "yes"}:
        return
    admin_chat = os.getenv("ADMIN_CHAT_ID")
    if not admin_chat or not bot:
        return
    try:
        chat_id = int(admin_chat)
    except (TypeError, ValueError):
        return
    limit = int(os.getenv("VK_AUTO_IMPORT_LIMIT", "25") or "25")
    await run_vk_auto_import(
        db,
        bot,
        chat_id=chat_id,
        limit=limit,
        operator_id=0,
        trigger="scheduled",
        run_id=run_id,
    )
