import logging
import os
import re
import asyncio
import time as _time
from datetime import date, timezone, datetime, timedelta
from dataclasses import dataclass, field
from typing import Callable, Iterable, Any, Sequence, List, Mapping, Optional, Dict, Tuple, Collection, Literal, Awaitable
from aiogram import Bot, types

from aiohttp import web
from telegraph import Telegraph
from markup import md_to_html, telegraph_br

from models import Event, Festival, WeekPage, WeekendPage, MonthPage, MonthPagePart, VkMissRecord, VkMissReviewSession, User
from poster_media import PosterMedia
from db import Database
from sqlalchemy import select, update, delete, text, func, or_, and_
from sqlmodel.ext.asyncio.session import AsyncSession
from scheduling import MONTHS_GEN
from event_utils import format_event_md, is_recent

if "LOCAL_TZ" not in globals():
    LOCAL_TZ = timezone.utc

if "dom_iskusstv_input_sessions" not in globals():
    dom_iskusstv_input_sessions = set()

if "format_day_pretty" not in globals():
    _MONTHS = [
        "января",
        "февраля",
        "марта",
        "апреля",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
    ]

    def format_day_pretty(day: date) -> str:
        return f"{day.day} {_MONTHS[day.month - 1]}"

if "is_long_event_type" not in globals():

    def is_long_event_type(event_type: str | None) -> bool:
        if not event_type:
            return False
        return event_type.strip().casefold() in {"выставка", "ярмарка"}


def clone_event_with_date(event: Event, day: date) -> Event:
    payload = event.model_dump()
    payload["date"] = day.isoformat()
    return Event(**payload)

if "month_name_prepositional" not in globals():
    _MONTHS_PREP = [
        "январе",
        "феврале",
        "марте",
        "апреле",
        "мае",
        "июне",
        "июле",
        "августе",
        "сентябре",
        "октябре",
        "ноябре",
        "декабре",
    ]

    def month_name_prepositional(month: str) -> str:
        y, m = month.split("-")
        return f"{_MONTHS_PREP[int(m) - 1]} {y}"

if "month_name_nominative" not in globals():
    _MONTHS_NOM = [
        "январь",
        "февраль",
        "март",
        "апрель",
        "май",
        "июнь",
        "июль",
        "август",
        "сентябрь",
        "октябрь",
        "ноябрь",
        "декабрь",
    ]

    def month_name_nominative(month: str) -> str:
        y, m = month.split("-")
        name = _MONTHS_NOM[int(m) - 1]
        if int(y) != datetime.now(LOCAL_TZ).year:
            return f"{name} {y}"
        return name


def _normalize_title_and_emoji(title: str, emoji: str | None) -> tuple[str, str]:
    """Normalize emoji placement so it appears only once per rendered line."""

    if not emoji:
        return title, ""

    trimmed_title = title.lstrip()
    if trimmed_title.startswith(emoji):
        trimmed_title = trimmed_title[len(emoji) :].lstrip()

    return trimmed_title or title.strip(), f"{emoji} "


def event_title_nodes(e: Event) -> list:
    nodes: list = []
    if is_recent(e):
        nodes.append("\U0001f6a9 ")
    title_text, emoji_part = _normalize_title_and_emoji(e.title, e.emoji)
    if emoji_part:
        nodes.append(emoji_part)
    url = e.telegraph_url
    if not url and e.telegraph_path:
        url = f"https://telegra.ph/{e.telegraph_path.lstrip('/')}"
    if not url and e.source_post_url:
        url = e.source_post_url
    if url:
        nodes.append({"tag": "a", "attrs": {"href": url}, "children": [title_text]})
    else:
        nodes.append(title_text)
    return nodes


def event_to_nodes(
    e: Event,
    festival: Festival | None = None,
    fest_icon: bool = False,
    log_fest_link: bool = False,
    *,
    show_festival: bool = True,
    include_ics: bool = True,
    include_details: bool = True,
    show_image: bool = False,
) -> list[dict]:
    md = format_event_md(
        e,
        festival if show_festival else None,
        include_ics=include_ics,
        include_details=include_details,
    )

    lines = md.split("\n")
    body_lines = lines[1:]
    if show_festival and festival and body_lines:
        body_lines = body_lines[1:]
    body_md = "\n".join(body_lines) if body_lines else ""
    from telegraph.utils import html_to_nodes

    nodes = []
    # Show 3D preview as main image if available, otherwise use first photo
    if show_image:
        preview_url = getattr(e, "preview_3d_url", None)
        if isinstance(preview_url, str):
            preview_url = preview_url.strip()
        if preview_url and isinstance(preview_url, str) and preview_url.startswith("http"):
            # Use 3D preview as main image
            nodes.append({
                "tag": "figure",
                "children": [{"tag": "img", "attrs": {"src": preview_url}, "children": []}]
            })
        elif e.photo_urls:
            # Fallback to first photo
            first_url = e.photo_urls[0]
            if isinstance(first_url, str):
                first_url = first_url.strip()
            if isinstance(first_url, str) and first_url.startswith("http"):
                nodes.append({
                    "tag": "figure",
                    "children": [{"tag": "img", "attrs": {"src": first_url}, "children": []}]
                })
    nodes.append({"tag": "h4", "children": event_title_nodes(e)})
    fest = festival if show_festival else None
    if fest is None and show_festival and e.festival:
        fest = getattr(e, "_festival", None)
    if log_fest_link and show_festival and e.festival:
        has_url = bool(getattr(fest, "telegraph_url", None))
        has_path = bool(getattr(fest, "telegraph_path", None))
        href = ""
        if has_url:
            href = fest.telegraph_url
        elif has_path:
            href = f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}"
        logging.info(
            "month_render_fest_link",
            extra={
                "event_id": e.id,
                "festival": e.festival,
                "has_url": has_url,
                "has_path": has_path,
                "href_used": href,
            },
        )
    if fest:
        prefix = "✨ " if fest_icon else ""
        url = fest.telegraph_url
        if not url and fest.telegraph_path:
            url = f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}"
        if url:
            children: list = []
            if prefix:
                children.append(prefix)
            children.append(
                {
                    "tag": "a",
                    "attrs": {"href": url},
                    "children": [fest.name],
                }
            )
            nodes.append({"tag": "p", "children": children})
        else:
            text = f"{prefix}{fest.name}" if prefix else fest.name
            nodes.append({"tag": "p", "children": [text]})
    if body_md:
        html_text = md_to_html(body_md)
        body_nodes = html_to_nodes(html_text)
        if (
            festival
            and not show_festival
            and not e.telegraph_url
            and not (e.description and e.description.strip())
            and body_nodes
        ):
            first = body_nodes[0]
            if first.get("tag") == "p":
                children = first.get("children") or []
                if children and isinstance(children[0], dict) and children[0].get("tag") == "br":
                    rest = children[1:]
                    if festival.program_url:
                        link_node = {
                            "tag": "p",
                            "children": [
                                {
                                    "tag": "a",
                                    "attrs": {"href": festival.program_url},
                                    "children": ["программа"],
                                }
                            ],
                        }
                        body_nodes = [link_node, {"tag": "p", "children": rest}]
                    else:
                        body_nodes = [{"tag": "p", "children": rest}]
        nodes.extend(body_nodes)
    nodes.extend(telegraph_br())
    return nodes


def exhibition_title_nodes(e: Event) -> list:
    nodes: list = []
    if is_recent(e):
        nodes.append("\U0001f6a9 ")
    title_text, emoji_part = _normalize_title_and_emoji(e.title, e.emoji)
    if emoji_part:
        nodes.append(emoji_part)
    url = e.telegraph_url
    if not url and e.telegraph_path:
        url = f"https://telegra.ph/{e.telegraph_path.lstrip('/')}"
    if not url and e.source_post_url:
        url = e.source_post_url
    if url:
        nodes.append({"tag": "a", "attrs": {"href": url}, "children": [title_text]})
    else:
        nodes.append(title_text)
    return nodes


def exhibition_to_nodes(e: Event) -> list[dict]:
    md = format_exhibition_md(e)
    lines = md.split("\n")
    body_md = "\n".join(lines[1:]) if len(lines) > 1 else ""
    from telegraph.utils import html_to_nodes

    nodes = [{"tag": "h4", "children": exhibition_title_nodes(e)}]
    if body_md:
        html_text = md_to_html(body_md)
        nodes.extend(html_to_nodes(html_text))
    nodes.extend(telegraph_br())
    return nodes


def add_day_sections(
    days: Iterable[date],
    by_day: dict[date, list[Event]],
    fest_map: dict[str, Festival],
    add_many: Callable[[Iterable[dict]], None],
    *,
    use_markers: bool = False,
    include_ics: bool = True,

    include_details: bool = True,
    show_images: bool = False,
):
    """Append event sections grouped by day to Telegraph content."""
    for d in days:
        events = by_day.get(d)
        if not events:
            continue
        if use_markers:
            add_many([DAY_START(d)])
        add_many(telegraph_br())
        if d.weekday() == 5:
            add_many([{ "tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"] }])
        elif d.weekday() == 6:
            add_many([{ "tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"] }])
        add_many([
            {"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"]},
            {"tag": "br"},
        ])
        add_many(telegraph_br())
        for ev in events:
            fest = fest_map.get((ev.festival or "").casefold())
            add_many(
                event_to_nodes(
                    ev,
                    fest,
                    fest_icon=True,
                    log_fest_link=use_markers,
                    include_ics=include_ics,
                    include_details=include_details,
                    show_image=show_images,
                )
            )
        if use_markers:
            add_many([DAY_END(d)])


def render_month_day_section(d: date, events: list[Event], show_images: bool = False) -> str:
    """Return HTML snippet for a single day on a month page."""
    from telegraph.utils import nodes_to_html

    nodes: list[dict] = []
    nodes.extend(telegraph_br())
    if d.weekday() == 5:
        nodes.append({"tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"]})
    elif d.weekday() == 6:
        nodes.append({"tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"]})
    nodes.append({"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"]})
    nodes.extend(telegraph_br())
    for ev in events:
        fest = getattr(ev, "_festival", None)
        nodes.extend(
            event_to_nodes(ev, fest, fest_icon=True, log_fest_link=True, show_image=show_images)
        )
    return nodes_to_html(nodes)

async def get_month_data(db: Database, month: str, *, fallback: bool = True):
    """Return events and exhibitions for the given month."""
    start = date.fromisoformat(month + "-01")
    next_start = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    today = datetime.now(LOCAL_TZ).date()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(Event.date >= start.isoformat(), Event.date < next_start.isoformat())
            .order_by(Event.date, Event.time)
        )
        events = result.scalars().all()
        events = [
            e for e in events if (e.event_type or "").casefold() != "ярмарка"
        ]

        ex_result = await session.execute(
            select(Event)
            .where(
                Event.end_date.is_not(None),
                Event.end_date >= start.isoformat(),
                Event.date <= next_start.isoformat(),
                Event.event_type == "выставка",
            )
            .order_by(Event.date)
        )
        exhibitions = ex_result.scalars().all()

    if month == today.strftime("%Y-%m"):
        today_str = today.isoformat()
        cutoff = (today - timedelta(days=30)).isoformat()
        events = [e for e in events if e.date.split("..", 1)[0] >= today_str]
        exhibitions = [
            e for e in exhibitions if e.end_date and e.end_date >= today_str
        ]

    if fallback and not events and not exhibitions:
        prev_month = (start - timedelta(days=1)).strftime("%Y-%m")
        if prev_month != month:
            prev_events, prev_exh = await get_month_data(
                db, prev_month, fallback=True
            )
            if not events:
                events.extend(prev_events)
            if not exhibitions:
                exhibitions.extend(prev_exh)

    return events, exhibitions


async def build_month_page_content(
    db: Database,
    month: str,
    events: list[Event] | None = None,
    exhibitions: list[Event] | None = None,
    continuation_url: str | None = None,
    size_limit: int | None = None,
    *,
    include_ics: bool = True,
    include_details: bool = True,
    page_number: int = 1,
    first_date: date | None = None,
    last_date: date | None = None,
) -> tuple[str, list, int]:
    if events is None or exhibitions is None:
        events, exhibitions = await get_month_data(db, month)

    async with span("db"):
        async with db.get_session() as session:
            res_f = await session.execute(select(Festival))
            fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}
    for e in events:
        fest = fest_map.get((e.festival or "").casefold())
        await ensure_event_telegraph_link(e, fest, db)
    for e in exhibitions:
        fest = fest_map.get((e.festival or "").casefold())
        await ensure_event_telegraph_link(e, fest, db)
    fest_index_url = await get_setting_value(db, "fest_index_url")

    async with span("render"):
        title, content, size = await asyncio.to_thread(
            _build_month_page_content_sync,
            month,
            events,
            exhibitions,
            fest_map,
            continuation_url,
            size_limit,
            fest_index_url,
            include_ics,
            include_details,
            page_number,
            first_date,
            last_date,
        )
    logging.info("build_month_page_content size=%d page=%d", size, page_number)
    return title, content, size


def _build_month_page_content_sync(
    month: str,
    events: list[Event],
    exhibitions: list[Event],
    fest_map: dict[str, Festival],
    continuation_url: str | None,
    size_limit: int | None,
    fest_index_url: str | None,
    include_ics: bool,
    include_details: bool,
    page_number: int = 1,
    first_date: date | None = None,
    last_date: date | None = None,
) -> tuple[str, list, int]:
    # Ensure festivals have full Telegraph URLs for easy linking
    for fest in fest_map.values():
        if not fest.telegraph_url and fest.telegraph_path:
            fest.telegraph_url = f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}"

    today = datetime.now(LOCAL_TZ).date()
    today_str = today.isoformat()
    cutoff = (today - timedelta(days=30)).isoformat()

    if month == today.strftime("%Y-%m"):
        events = [e for e in events if e.date.split("..", 1)[0] >= today_str]
        exhibitions = [e for e in exhibitions if e.end_date and e.end_date >= today_str]
    events = [
        e for e in events if not (e.event_type == "выставка" and e.date < today_str)
    ]
    exhibitions = [
        e for e in exhibitions if e.end_date and e.date <= today_str and e.end_date >= today_str
    ]

    by_day: dict[date, list[Event]] = {}
    for e in events:
        date_part = e.date.split("..", 1)[0]
        d = parse_iso_date(date_part)
        if not d:
            logging.error("Invalid date for event %s: %s", e.id, e.date)
            continue
        by_day.setdefault(d, []).append(e)

    content: list[dict] = []
    size = 0
    exceeded = False

    def add(node: dict):
        nonlocal size, exceeded
        size += rough_size((node,))
        if size_limit is not None and size > size_limit:
            exceeded = True
            return
        content.append(node)

    def add_many(nodes: Iterable[dict]):
        for n in nodes:
            if exceeded:
                break
            add(n)

    # Only add intro paragraph on page 1
    if page_number == 1:
        intro = (
            f"Планируйте свой месяц заранее: интересные мероприятия Калининграда и 39 региона в {month_name_prepositional(month)} — от лекций и концертов до культурных шоу. "
        )
        intro_nodes = [
            intro,
            {
                "tag": "a",
                "attrs": {"href": "https://t.me/kenigevents"},
                "children": ["Полюбить Калининград Анонсы"],
            },
        ]
        add({"tag": "p", "children": intro_nodes})

    add_day_sections(
        sorted(by_day),
        by_day,
        fest_map,
        add_many,
        use_markers=True,
        include_ics=include_ics,
        include_details=include_details,
        show_images=len(events) <= 30,
    )

    if exhibitions and not exceeded:
        add_many([PERM_START])
        add({"tag": "h3", "children": ["Постоянные выставки"]})
        add({"tag": "br"})
        add_many(telegraph_br())
        for ev in exhibitions:
            if exceeded:
                break
            add_many(exhibition_to_nodes(ev))
        add_many([PERM_END])


    if continuation_url and not exceeded:
        add_many(telegraph_br())
        add(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": continuation_url},
                        "children": [f"{month_name_nominative(month)} продолжение"],
                    }
                ],
            }
        )
        add({"tag": "br"})
        add_many(telegraph_br())

    if fest_index_url and not exceeded:
        add_many(telegraph_br())
        add(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest_index_url},
                        "children": ["Фестивали"],
                    }
                ],
            }
        )
        add_many(telegraph_br())

    # Generate title based on page number
    # Check DEV_MODE flag to add TEST prefix for separate dev pages
    import os
    is_dev_mode = os.getenv("DEV_MODE") == "1"
    test_prefix = "ТЕСТ " if is_dev_mode else ""
    
    if page_number == 1:
        title = (
            f"{test_prefix}События Калининграда в {month_name_prepositional(month)}: полный анонс от Полюбить Калининград Анонсы"
        )
    else:
        # For continuation pages, use date range in title
        year = int(month.split("-")[0])
        month_num = int(month.split("-")[1])
        month_gen = MONTHS_GEN[month_num]
        if first_date and last_date:
            title = f"{test_prefix}События Калининграда с {first_date.day} по {last_date.day} {month_gen} {year}"
        else:
            title = f"{test_prefix}События Калининграда в {month_name_prepositional(month)} (продолжение)"
    return title, content, size


async def _sync_month_page_inner(
    db: Database,
    month: str,
    update_links: bool = False,
    force: bool = False,
    progress: Any | None = None,
) -> bool:
    tasks: list[Awaitable[None]] = []
    async with HEAVY_SEMAPHORE:
        now = _time.time()
        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            and not force
            and now < _month_next_run[month]
        ):
            logging.debug("sync_month_page skipped, debounced")
            return False
        _month_next_run[month] = now + 60
        logging.info(
            "sync_month_page start: month=%s update_links=%s force=%s",
            month,
            update_links,
            force,
        )
        if DEBUG:
            mem_info("month page before")
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            raise RuntimeError("Telegraph token unavailable")
        tg = Telegraph(access_token=token)
        async with db.get_session() as session:
            page = await session.get(MonthPage, month)
            created = False
            if not page:
                page = MonthPage(month=month, url="", path="")
                session.add(page)
                await session.commit()
                created = True
            prev_hash = page.content_hash
            prev_hash2 = page.content_hash2

        async def commit_page() -> None:
            async with db.get_session() as s:
                db_page = await s.get(MonthPage, month)
                db_page.url = page.url
                db_page.path = page.path
                db_page.url2 = page.url2
                db_page.path2 = page.path2
                db_page.content_hash = page.content_hash
                db_page.content_hash2 = page.content_hash2
                await s.commit()

        if update_links and page.path:
            nav_block = await build_month_nav_block(db, month)
            nav_update_failed = False
            for path_attr, hash_attr in (("path", "content_hash"), ("path2", "content_hash2")):
                path = getattr(page, path_attr)
                if not path:
                    continue
                page_data = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page_data.get("content") or page_data.get("content_html") or ""
                html_content = unescape_html_comments(html_content)
                changed_any = False
                if "<!--DAY" not in html_content:
                    logging.warning("month_rebuild_markers_missing")
                    year = int(month.split("-")[0])
                    for m in re.finditer(
                        r"<h3>🟥🟥🟥 (\d{1,2} [^<]+) 🟥🟥🟥</h3>", html_content
                    ):
                        parsed = _parse_pretty_date(m.group(1), year)
                        if parsed:
                            html_content, changed = ensure_day_markers(
                                html_content, parsed
                            )
                            changed_any = changed_any or changed
                updated_html = ensure_footer_nav_with_hr(
                    html_content, nav_block, month=month, page=1 if path_attr == "path" else 2
                )
                if not changed_any and content_hash(updated_html) == content_hash(html_content):
                    continue
                if len(updated_html.encode()) > TELEGRAPH_LIMIT:
                    logging.warning(
                        "Updated navigation for %s (%s) exceeds limit, rebuilding",
                        month,
                        path,
                    )
                    nav_update_failed = True
                    break
                title = page_data.get("title") or month_name_prepositional(month)
                try:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        html_content=updated_html,
                        caller="month_build",
                    )
                except TelegraphException as e:
                    msg = str(e).lower()
                    if all(word in msg for word in ("content", "too", "big")):
                        logging.warning(
                            "Updated navigation for %s (%s) too big, rebuilding",
                            month,
                            path,
                        )
                        nav_update_failed = True
                        break
                    raise
                setattr(page, hash_attr, content_hash(updated_html))
            if not nav_update_failed:
                await commit_page()
                return False
            logging.info("Falling back to full rebuild for %s", month)

        events, exhibitions = await get_month_data(db, month)
        nav_block = await build_month_nav_block(db, month)

        from telegraph.utils import nodes_to_html

        title, content, _ = await build_month_page_content(
            db, month, events, exhibitions
        )
        html_full = unescape_html_comments(nodes_to_html(content))
        html_full = ensure_footer_nav_with_hr(html_full, nav_block, month=month, page=1)
        hash_full = content_hash(html_full)
        size = len(html_full.encode())

        try:
            if size <= TELEGRAPH_LIMIT:
                if page.path and page.content_hash == hash_full:
                    logging.debug("telegraph_update skipped (no changes)")
                else:
                    if not page.path:
                        logging.info("creating month page %s", month)
                        data = await telegraph_create_page(
                            tg,
                            title=title,
                            html_content=html_full,
                            caller="month_build",
                        )
                        page.url = normalize_telegraph_url(data.get("url"))
                        page.path = data.get("path")
                        created = True
                    else:
                        logging.info("updating month page %s", month)
                        start = _time.perf_counter()
                        await telegraph_edit_page(
                            tg,
                            page.path,
                            title=title,
                            html_content=html_full,
                            caller="month_build",
                        )
                        dur = (_time.perf_counter() - start) * 1000
                        logging.info("editPage %s done in %.0f ms", page.path, dur)
                    rough = rough_size(content)
                    logging.debug(
                        "telegraph_update page=%s nodes=%d bytes≈%d",
                        page.path,
                        len(content),
                        rough,
                    )
                    page.content_hash = hash_full
                    page.content_hash2 = None
                page.url2 = None
                page.path2 = None
                logging.info(
                    "%s month page %s", "Created" if created else "Edited", month
                )
                await commit_page()
            else:
                logging.info(
                    "sync_month_page: splitting %s (events=%d)",
                    month,
                    len(events),
                )
                had_path = bool(page.path)
                await split_month_until_ok(
                    db, tg, page, month, events, exhibitions, nav_block
                )
                if not had_path and page.path:
                    created = True
        except TelegraphException as e:
            msg = str(e).lower()
            if all(word in msg for word in ("content", "too", "big")):
                logging.warning("Month page %s too big, splitting", month)
                had_path = bool(page.path)
                await split_month_until_ok(
                    db, tg, page, month, events, exhibitions, nav_block
                )
                if not had_path and page.path:
                    created = True
            else:
                logging.error("Failed to sync month page %s: %s", month, e)
                raise
        except Exception as e:
            msg = str(e).lower()
            if all(word in msg for word in ("content", "too", "big")):
                logging.info(
                    "sync_month_page: splitting %s (events=%d)",
                    month,
                    len(events),
                )
                logging.warning("Month page %s too big, splitting", month)
                had_path = bool(page.path)
                await split_month_until_ok(
                    db, tg, page, month, events, exhibitions, nav_block
                )
                if not had_path and page.path:
                    created = True
            else:
                logging.error("Failed to sync month page %s: %s", month, e)

                if progress:
                    progress.mark(f"month_pages:{month}", "error", str(e))
                if update_links or created:
                    async with db.get_session() as session:
                        result = await session.execute(
                            select(MonthPage).order_by(MonthPage.month)
                        )
                        months = result.scalars().all()
                    for p in months:
                        if p.month != month:
                            await sync_month_page(db, p.month, update_links=False)
                            await asyncio.sleep(0)
                raise
        if page.path:
            await check_month_page_markers(tg, page.path)
        if page.path2:
            await check_month_page_markers(tg, page.path2)
        if DEBUG:
            mem_info("month page after")
        changed = (
            created
            or page.content_hash != prev_hash
            or page.content_hash2 != prev_hash2
        )
        if progress:
            status = "done" if changed else "skipped_nochange"
            progress.mark(f"month_pages:{month}", status, page.url or "")
        paths = ", ".join(p for p in [page.path, page.path2] if p)
        if changed:
            logging.info("month page %s: edited path=%s size=%d", month, paths, size)
        else:
            logging.info("month page %s: nochange", month)
    return True

async def sync_month_page(
    db: Database,
    month: str,
    update_links: bool = False,
    force: bool = False,
    progress: Any | None = None,
):
    async with _page_locks[f"month:{month}"]:
        needs_nav = await _sync_month_page_inner(
            db, month, update_links, force, progress
        )
    if needs_nav:
        await refresh_month_nav(db)


def week_start_for_date(d: date) -> date:
    return d - timedelta(days=d.weekday())


def next_week_start(d: date) -> date:
    w = week_start_for_date(d)
    if d <= w:
        return w
    return w + timedelta(days=7)


def weekend_start_for_date(d: date) -> date | None:
    if d.weekday() == 5:
        return d
    if d.weekday() == 6:
        return d - timedelta(days=1)
    return None


def next_weekend_start(d: date) -> date:
    w = weekend_start_for_date(d)
    if w and d <= w:
        return w
    days_ahead = (5 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


async def build_weekend_page_content(
    db: Database, start: str, size_limit: int | None = None
) -> tuple[str, list, int]:
    saturday = date.fromisoformat(start)
    sunday = saturday + timedelta(days=1)
    days = [saturday, sunday]
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(Event.date.in_([d.isoformat() for d in days]))
            .order_by(Event.date, Event.time)
        )
        events = result.scalars().all()

        ex_res = await session.execute(
            select(Event)
            .where(
                Event.event_type == "выставка",
                Event.end_date.is_not(None),
                Event.date <= sunday.isoformat(),
                Event.end_date >= saturday.isoformat(),
            )
            .order_by(Event.date)
        )
        exhibitions = ex_res.scalars().all()

        fair_res = await session.execute(
            select(Event)
            .where(
                Event.event_type == "ярмарка",
                Event.end_date.is_not(None),
                Event.date <= sunday.isoformat(),
                Event.end_date >= saturday.isoformat(),
            )
            .order_by(Event.date, Event.time)
        )
        fairs = fair_res.scalars().all()

        res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
        weekend_pages = res_w.scalars().all()
        res_m = await session.execute(select(MonthPage).order_by(MonthPage.month))
        month_pages = res_m.scalars().all()
        res_f = await session.execute(select(Festival))
        fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}

    today = datetime.now(LOCAL_TZ).date()
    events = [
        e
        for e in events
        if (
            (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        )
    ]
    fairs = [
        e
        for e in fairs
        if (
            (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        )
    ]

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}

    fest_index_url = await get_setting_value(db, "fest_index_url")

    for e in (*events, *fairs):
        fest = fest_map.get((e.festival or "").casefold())
        await ensure_event_telegraph_link(e, fest, db)

    by_day: dict[date, list[Event]] = {}
    for e in events:
        d = parse_iso_date(e.date)
        if not d:
            continue
        by_day.setdefault(d, []).append(e)

    day_ids: dict[date, set[int]] = {
        d: {e.id for e in by_day.get(d, []) if e.id is not None} for d in days
    }
    for fair in fairs:
        start = parse_iso_date(fair.date)
        end = parse_iso_date(fair.end_date) if fair.end_date else None
        if not start or not end:
            continue
        if end < start:
            start, end = end, start
        for day in days:
            if start <= day <= end:
                if fair.id is not None and fair.id in day_ids.get(day, set()):
                    continue
                by_day.setdefault(day, []).append(clone_event_with_date(fair, day))
                if fair.id is not None:
                    day_ids.setdefault(day, set()).add(fair.id)

    for day in by_day:
        by_day[day].sort(key=lambda e: e.time or "99:99")

    content: list[dict] = []
    size = 0
    exceeded = False

    def add(node: dict):
        nonlocal size, exceeded
        size += rough_size((node,))
        if size_limit is not None and size > size_limit:
            exceeded = True
            return
        content.append(node)

    def add_many(nodes: Iterable[dict]):
        for n in nodes:
            if exceeded:
                break
            add(n)
    # NEW: weekend cover
    cover_url = await get_setting_value(db, f"weekend_cover:{start}")
    if cover_url and not exceeded:
        add(
            {
                "tag": "figure",
                "children": [
                    {"tag": "img", "attrs": {"src": cover_url}, "children": []}
                ],
            }
        )
        add_many(telegraph_br())

    add(
        {
            "tag": "p",
            "children": [
                "Вот что рекомендуют ",
                {
                    "tag": "a",
                    "attrs": {"href": "https://t.me/kenigevents"},
                    "children": ["Полюбить Калининград Анонсы"],
                },
                " чтобы провести выходные ярко: события Калининградской области и 39 региона — концерты, спектакли, фестивали.",
            ],
        }
    )

    add_day_sections(days, by_day, fest_map, add_many, show_images=len(events) < 10)

    weekend_nav: list[dict] = []
    future_weekends = [w for w in weekend_pages if w.start >= start]
    if future_weekends:
        nav_children = []
        for idx, w in enumerate(future_weekends):
            s = date.fromisoformat(w.start)
            label = format_weekend_range(s)
            if w.start == start:
                nav_children.append(label)
            else:
                nav_children.append(
                    {"tag": "a", "attrs": {"href": w.url}, "children": [label]}
                )
            if idx < len(future_weekends) - 1:
                nav_children.append(" ")
        weekend_nav = [{"tag": "h4", "children": nav_children}]

    month_nav: list[dict] = []
    cur_month = start[:7]
    future_months = [m for m in month_pages if m.month >= cur_month]
    if future_months:
        nav_children = []
        for idx, p in enumerate(future_months):
            name = month_name_nominative(p.month)
            nav_children.append({"tag": "a", "attrs": {"href": p.url}, "children": [name]})
            if idx < len(future_months) - 1:
                nav_children.append(" ")
        month_nav = [{"tag": "h4", "children": nav_children}]

    if exhibitions and not exceeded:
        add({"tag": "h3", "children": ["Постоянные выставки"]})
        add_many(telegraph_br())
        for ev in exhibitions:
            if exceeded:
                break
            add_many(exhibition_to_nodes(ev))

    if weekend_nav and not exceeded:
        add_many(telegraph_br())
        add_many(weekend_nav)
        add_many(telegraph_br())
    if month_nav and not exceeded:
        add_many(telegraph_br())
        add_many(month_nav)
        add_many(telegraph_br())

    if fest_index_url and not exceeded:
        add_many(telegraph_br())
        add(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest_index_url},
                        "children": ["Фестивали"],
                    }
                ],
            }
        )
        add_many(telegraph_br())

    label = format_weekend_range(saturday)
    if saturday.month == sunday.month:
        label = f"{saturday.day}-{sunday.day} {MONTHS[saturday.month - 1]}"
    title = (
        "Чем заняться на выходных в Калининградской области "
        f"{label}"
    )
    if DEBUG:
        from telegraph.utils import nodes_to_html
        html = nodes_to_html(content)
        logging.debug(
            "weekend_html sizes: html=%d json=%d",
            len(html),
            len(json.dumps(content, ensure_ascii=False)),
        )
    return title, content, size


async def _sync_weekend_page_inner(
    db: Database,
    start: str,
    update_links: bool = True,
    post_vk: bool = True,
    force: bool = False,
    progress: Any | None = None,
):
    tasks: list[Awaitable[None]] = []
    async with HEAVY_SEMAPHORE:
        now = _time.time()
        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            and not force
            and now < _weekend_next_run[start]
        ):
            logging.debug("sync_weekend_page skipped, debounced")
            return
        _weekend_next_run[start] = now + 60
        logging.info(
            "sync_weekend_page start: start=%s update_links=%s post_vk=%s",
            start,
            update_links,
            post_vk,
        )
        if DEBUG:
            mem_info("weekend page before")
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            return
        tg = Telegraph(access_token=token)
        from telegraph.utils import nodes_to_html

        async with db.get_session() as session:
            page = await session.get(WeekendPage, start)
            if not page:
                page = WeekendPage(start=start, url="", path="")
                session.add(page)
                await session.commit()
                created = True
            else:
                created = False
            path = page.path
            prev_hash = page.content_hash

        try:
            title, content, _ = await build_weekend_page_content(db, start)
            html = nodes_to_html(content)
            hash_new = content_hash(html)
            if not path:
                title = re.sub(r"(\d+)-(\d+)", r"\1 - \2", title)
                data = await telegraph_create_page(
                    tg, title, content=content, caller="weekend_build"
                )
                page.url = normalize_telegraph_url(data.get("url"))
                page.path = data.get("path")
                created = True
                rough = rough_size(content)
                logging.debug(
                    "telegraph_update page=%s nodes=%d bytes≈%d",
                    page.path,
                    len(content),
                    rough,
                )
                page.content_hash = hash_new
                if update_links:
                    await telegraph_edit_page(
                        tg,
                        page.path,
                        title=title,
                        content=content,
                        caller="weekend_build",
                    )
            elif page.content_hash == hash_new and not update_links:
                logging.debug("telegraph_update skipped (no changes)")
            else:
                start_t = _time.perf_counter()
                await telegraph_edit_page(
                    tg, path, title=title, content=content, caller="weekend_build"
                )
                dur = (_time.perf_counter() - start_t) * 1000
                logging.info("editPage %s done in %.0f ms", path, dur)
                rough = rough_size(content)
                logging.debug(
                    "telegraph_update page=%s nodes=%d bytes≈%d",
                    path,
                    len(content),
                    rough,
                )
                page.content_hash = hash_new
            logging.info("%s weekend page %s", "Created" if created else "Edited", start)
        except Exception as e:
            logging.error("Failed to sync weekend page %s: %s", start, e)
            if progress:
                progress.mark(f"weekend_pages:{start}", "error", str(e))
            return

        async with db.get_session() as session:
            db_page = await session.get(WeekendPage, start)
            if db_page:
                db_page.url = page.url
                db_page.path = page.path
                db_page.content_hash = page.content_hash
                await session.commit()

        if post_vk:
            await sync_vk_weekend_post(db, start)
        if DEBUG:
            mem_info("weekend page after")

        changed = created or page.content_hash != prev_hash
        if progress:
            status = "done" if changed else "skipped_nochange"
            progress.mark(f"weekend_pages:{start}", status, page.url or "")
        size = len(html.encode())
        if changed:
            logging.info(
                "weekend page %s: edited path=%s size=%d", start, page.path, size
            )
        else:
            logging.info("weekend page %s: nochange", start)

        if update_links or created:
            d_start = date.fromisoformat(start)
            for d_adj in (d_start - timedelta(days=7), d_start + timedelta(days=7)):
                w_key = d_adj.isoformat()
                async with db.get_session() as session:
                    if await session.get(WeekendPage, w_key):
                        tasks.append(
                            sync_weekend_page(db, w_key, update_links=False, post_vk=False)
                        )
    for t in tasks:
        await t


async def sync_weekend_page(
    db: Database,
    start: str,
    update_links: bool = True,
    post_vk: bool = True,
    force: bool = False,
    progress: Any | None = None,
):
    async with _page_locks[f"week:{start}"]:
        await _sync_weekend_page_inner(db, start, update_links, post_vk, force, progress)


def _build_month_vk_nav_lines(week_pages: list[WeekPage], cur_month: str) -> list[str]:
    first_by_month: dict[str, WeekPage] = {}
    for w in week_pages:
        m = w.start[:7]
        if m not in first_by_month or w.start < first_by_month[m].start:
            first_by_month[m] = w
    parts: list[str] = []
    for m in sorted(first_by_month):
        if m < cur_month:
            continue
        w = first_by_month[m]
        label = month_name_nominative(m)
        if m == cur_month or not w.vk_post_url:
            parts.append(label)
        else:
            parts.append(f"[{w.vk_post_url}|{label}]")
    return parts


async def build_week_vk_message(db: Database, start: str) -> str:
    logging.info("build_week_vk_message start for %s", start)
    monday = date.fromisoformat(start)
    days = [monday + timedelta(days=i) for i in range(7)]
    async with span("db"):
        async with db.get_session() as session:
            result = await session.execute(
                select(Event)
                .where(Event.date.in_([d.isoformat() for d in days]))
                .order_by(Event.date, Event.time)
            )
            events = result.scalars().all()
            res_w = await session.execute(select(WeekPage).order_by(WeekPage.start))
            week_pages = res_w.scalars().all()

    async with span("render"):
        by_day: dict[date, list[Event]] = {}
        for e in events:
            if not e.source_vk_post_url:
                continue
            d = parse_iso_date(e.date)
            if not d:
                continue
            by_day.setdefault(d, []).append(e)

        lines = [f"{format_week_range(monday)} Афиша недели"]
        for d in days:
            evs = by_day.get(d)
            if not evs:
                continue
            lines.append(VK_BLANK_LINE)
            lines.append(f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥")
            for ev in evs:
                line = f"[{ev.source_vk_post_url}|{ev.title}]"
                if ev.time:
                    line = f"{ev.time} | {line}"
                lines.append(line)

                location_parts = [p for p in [ev.location_name, ev.city] if p]
                if location_parts:
                    lines.append(", ".join(location_parts))

        nav_weeks = [
            w
            for w in week_pages
            if w.start[:7] == start[:7] and (w.vk_post_url or w.start == start)
        ]
        if nav_weeks:
            parts = []
            for w in nav_weeks:
                label = format_week_range(date.fromisoformat(w.start))
                if w.start == start or not w.vk_post_url:
                    parts.append(label)
                else:
                    parts.append(f"[{w.vk_post_url}|{label}]")
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(parts))

        month_parts = _build_month_vk_nav_lines(week_pages, start[:7])
        if month_parts:
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(month_parts))

        message = "\n".join(lines)
    logging.info("build_week_vk_message built %d lines", len(lines))
    return message


async def build_weekend_vk_message(db: Database, start: str) -> str:
    logging.info("build_weekend_vk_message start for %s", start)
    saturday = date.fromisoformat(start)
    sunday = saturday + timedelta(days=1)
    days = [saturday, sunday]
    async with span("db"):
        async with db.get_session() as session:
            result = await session.execute(
                select(Event)
                .where(Event.date.in_([d.isoformat() for d in days]))
                .order_by(Event.date, Event.time)
            )
            events = result.scalars().all()
            res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
            weekend_pages = res_w.scalars().all()
            res_week = await session.execute(select(WeekPage).order_by(WeekPage.start))
            week_pages = res_week.scalars().all()

    async with span("render"):
        by_day: dict[date, list[Event]] = {}
        for e in events:
            if not e.source_vk_post_url:
                continue
            d = parse_iso_date(e.date)
            if not d:
                continue
            by_day.setdefault(d, []).append(e)

        lines = [f"{format_weekend_range(saturday)} Афиша выходных"]
        for d in days:
            evs = by_day.get(d)
            if not evs:
                continue
            lines.append(VK_BLANK_LINE)
            lines.append(f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥")
            for ev in evs:
                line = f"[{ev.source_vk_post_url}|{ev.title}]"
                if ev.time:
                    line = f"{ev.time} | {line}"
                lines.append(line)

                location_parts = [p for p in [ev.location_name, ev.city] if p]
                if location_parts:
                    lines.append(", ".join(location_parts))

        nav_pages = [
            w
            for w in weekend_pages
            if w.start >= start and (w.vk_post_url or w.start == start)
        ]
        if nav_pages:
            parts = []
            for w in nav_pages:
                label = format_weekend_range(date.fromisoformat(w.start))
                if w.start == start or not w.vk_post_url:
                    parts.append(label)
                else:
                    parts.append(f"[{w.vk_post_url}|{label}]")
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(parts))

        month_parts = _build_month_vk_nav_lines(week_pages, start[:7])
        if month_parts:
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(month_parts))

        message = "\n".join(lines)
    logging.info(
        "build_weekend_vk_message built %d lines", len(lines)
    )
    return message


async def sync_vk_weekend_post(db: Database, start: str, bot: Bot | None = None) -> None:
    lock = _weekend_vk_lock(start)
    async with lock:
        now = _time.time()
        if "PYTEST_CURRENT_TEST" not in os.environ and now < _vk_weekend_next_run[start]:
            logging.debug("sync_vk_weekend_post skipped, debounced")
            return
        _vk_weekend_next_run[start] = now + 60
        logging.info("sync_vk_weekend_post start for %s", start)
        group_id = VK_AFISHA_GROUP_ID
        if not group_id:
            logging.info("sync_vk_weekend_post: VK group not configured")
            return
        async with db.get_session() as session:
            page = await session.get(WeekendPage, start)
        if not page:
            logging.info("sync_vk_weekend_post: weekend page %s not found", start)
            return

        message = await build_weekend_vk_message(db, start)
        logging.info("sync_vk_weekend_post message len=%d", len(message))
        needs_new_post = not page.vk_post_url
        if page.vk_post_url:
            try:
                updated = await edit_vk_post(page.vk_post_url, message, db, bot)
                if updated:
                    logging.info("sync_vk_weekend_post updated %s", page.vk_post_url)
                else:
                    logging.info(
                        "sync_vk_weekend_post: no changes for %s", page.vk_post_url
                    )
            except Exception as e:
                if "post or comment deleted" in str(e) or "Пост удалён" in str(e):
                    logging.warning(
                        "sync_vk_weekend_post: original VK post missing, creating new"
                    )
                    needs_new_post = True
                else:
                    logging.error("VK post error for weekend %s: %s", start, e)
                    return
        if needs_new_post:
            url = await post_to_vk(group_id, message, db, bot)
            if url:
                async with db.get_session() as session:
                    obj = await session.get(WeekendPage, start)
                    if obj:
                        obj.vk_post_url = url
                        await session.commit()
                logging.info("sync_vk_weekend_post created %s", url)


async def sync_vk_week_post(db: Database, start: str, bot: Bot | None = None) -> None:
    lock = _week_vk_lock(start)
    async with lock:
        now = _time.time()
        if "PYTEST_CURRENT_TEST" not in os.environ and now < _vk_week_next_run[start]:
            logging.debug("sync_vk_week_post skipped, debounced")
            return
        _vk_week_next_run[start] = now + 60
        logging.info("sync_vk_week_post start for %s", start)
        group_id = VK_AFISHA_GROUP_ID
        if not group_id:
            logging.info("sync_vk_week_post: VK group not configured")
            return
        async with db.get_session() as session:
            page = await session.get(WeekPage, start)

        message = await build_week_vk_message(db, start)
        logging.info("sync_vk_week_post message len=%d", len(message))
        needs_new_post = not page or not page.vk_post_url
        if page and page.vk_post_url:
            try:
                updated = await edit_vk_post(page.vk_post_url, message, db, bot)
                if updated:
                    logging.info("sync_vk_week_post updated %s", page.vk_post_url)
                else:
                    logging.info(
                        "sync_vk_week_post: no changes for %s", page.vk_post_url
                    )
            except Exception as e:
                if "post or comment deleted" in str(e) or "Пост удалён" in str(e):
                    logging.warning(
                        "sync_vk_week_post: original VK post missing, creating new"
                    )
                    needs_new_post = True
                else:
                    logging.error("VK post error for week %s: %s", start, e)
                    return
        if needs_new_post:
            url = await post_to_vk(group_id, message, db, bot)
            if url:
                async with db.get_session() as session:
                    obj = await session.get(WeekPage, start)
                    if obj:
                        obj.vk_post_url = url
                    else:
                        session.add(WeekPage(start=start, vk_post_url=url))
                    await session.commit()
                logging.info("sync_vk_week_post created %s", url)


MAX_FEST_DESCRIPTION_LENGTH = 350
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _russian_plural(value: int, forms: tuple[str, str, str]) -> str:
    tail = value % 100
    if 10 < tail < 20:
        form = forms[2]
    else:
        tail = value % 10
        if tail == 1:
            form = forms[0]
        elif 1 < tail < 5:
            form = forms[1]
        else:
            form = forms[2]
    return f"{value} {form}"


async def generate_festival_description(
    fest: Festival, events: list[Event]
) -> str | None:
    """Use LLM to craft a short festival blurb."""

    name = fest.full_name or fest.name

    titles: list[str] = []
    seen_titles: set[str] = set()
    venues: list[str] = []
    seen_venues: set[str] = set()
    for event in events:
        title = (event.title or "").strip()
        if title:
            key = title.casefold()
            if key not in seen_titles and len(titles) < 10:
                seen_titles.add(key)
                titles.append(title)
        venue = (event.location_name or "").strip()
        if venue:
            key = venue.casefold()
            if key not in seen_venues and len(venues) < 5:
                seen_venues.add(key)
                venues.append(venue)

    start, end = festival_date_range(events)
    date_clause = ""
    if start and end:
        if start == end:
            date_clause = format_day_pretty(start)
        else:
            date_clause = (
                f"с {format_day_pretty(start)} по {format_day_pretty(end)}"
            )
    elif start:
        date_clause = format_day_pretty(start)

    city_values: list[str] = []
    if fest.city:
        city_values.append(fest.city)
    for event in events:
        if event.city:
            city_values.append(event.city)
    city_clause = ""
    city_counter = Counter(c.strip() for c in city_values if c and c.strip())
    if city_counter:
        most_common_city, _ = city_counter.most_common(1)[0]
        if most_common_city:
            city_clause = most_common_city

    duration_days = 0
    if start and end:
        duration_days = (end - start).days + 1

    event_count = len(events)

    fact_parts: list[str] = []
    if date_clause:
        fact_parts.append(f"период — {date_clause}")
    if city_clause:
        fact_parts.append(f"город — {city_clause}")
    if duration_days > 1:
        fact_parts.append(
            f"продолжительность — {_russian_plural(duration_days, ('день', 'дня', 'дней'))}"
        )
    if event_count:
        fact_parts.append(
            f"в программе {_russian_plural(event_count, ('событие', 'события', 'событий'))}"
        )
    if titles:
        fact_parts.append(f"сюжеты: {', '.join(titles)}")

    if not fact_parts:
        logging.warning(
            "generate_festival_description: insufficient data for %s", fest.name
        )
        return None

    context_sources: list[str] = []
    for candidate in (fest.source_text, fest.description):
        if candidate:
            context_sources.append(candidate)
    for event in events[:5]:
        if event.source_text:
            context_sources.append(event.source_text)

    context_snippet = ""
    for raw in context_sources:
        snippet = " ".join(raw.split()).strip()
        if snippet:
            context_snippet = snippet[:200]
            break

    facts_sentence = f"Исходные факты: {', '.join(fact_parts)}."
    third_segments: list[str] = []
    if venues:
        third_segments.append(f"площадки: {', '.join(venues)}")
    if context_snippet:
        third_segments.append(f"контекст: {context_snippet}")
    third_segments.append(
        "итоговый текст до 350 знаков, один абзац без списков и эмодзи"
    )
    third_sentence = "; ".join(third_segments) + "."

    prompt_sentences = [
        (
            "Ты — культурный журналист: напиши лаконичное описание фестиваля "
            f"{name} без выдуманных фактов."
        ),
        facts_sentence,
        third_sentence,
    ]
    prompt = " ".join(prompt_sentences)

    try:
        text = await ask_4o(prompt)
        logging.info("generated description for festival %s", fest.name)
    except Exception as e:
        logging.error("failed to generate festival description %s: %s", fest.name, e)
        return None

    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_FEST_DESCRIPTION_LENGTH:
        logging.warning(
            "festival description too long for %s: %d", fest.name, len(cleaned)
        )
        return None
    if _EMOJI_RE.search(cleaned):
        logging.warning(
            "festival description contains emoji for %s", fest.name
        )
        return None
    return cleaned


async def regenerate_festival_description(
    db: Database, fest_ref: Festival | int
) -> str | None:
    """Regenerate festival description using latest events."""

    async with db.get_session() as session:
        fest_obj: Festival | None = None
        if isinstance(fest_ref, Festival):
            if fest_ref.id is not None:
                fest_obj = await session.get(Festival, fest_ref.id)
            else:
                result = await session.execute(
                    select(Festival).where(Festival.name == fest_ref.name)
                )
                fest_obj = result.scalar_one_or_none()
        else:
            fest_obj = await session.get(Festival, fest_ref)

        if not fest_obj:
            return None

        events_query = (
            select(Event)
            .where(Event.festival == fest_obj.name)
            .order_by(Event.date, Event.time)
        )
        events_res = await session.execute(events_query)
        events = list(events_res.scalars().all())

        return await generate_festival_description(fest_obj, events)


async def merge_festivals(
    db: Database, src_id: int, dst_id: int, bot: Bot | None = None
) -> bool:
    """Merge two festivals moving events, media and metadata."""

    if src_id == dst_id:
        logging.warning("merge_festivals: identical ids src=%s dst=%s", src_id, dst_id)
        return False

    moved_events_count = 0
    aliases_added = 0
    photos_added = 0
    description_updated = False

    async with db.get_session() as session:
        src = await session.get(Festival, src_id)
        dst = await session.get(Festival, dst_id)

        if not src or not dst:
            logging.error(
                "merge_festivals: missing festivals src=%s dst=%s", src_id, dst_id
            )
            return False

        src_name = src.name
        dst_name = dst.name
        dst_pk = dst.id

        events_to_move_res = await session.execute(
            select(Event.id).where(Event.festival == src_name)
        )
        moved_events_count = len(list(events_to_move_res.scalars()))

        dst_photos_before = {url for url in list(dst.photo_urls or []) if url}
        dst_cover_before = dst.photo_url

        await session.execute(
            update(Event).where(Event.festival == src_name).values(festival=dst_name)
        )

        merged_photos: list[str] = []
        for url in list(dst.photo_urls or []) + list(src.photo_urls or []):
            if url and url not in merged_photos:
                merged_photos.append(url)
        if merged_photos:
            dst.photo_urls = merged_photos
            if not dst.photo_url or dst.photo_url not in merged_photos:
                dst.photo_url = merged_photos[0]
        elif src.photo_url and not dst.photo_url:
            dst.photo_url = src.photo_url

        if merged_photos:
            photos_added = sum(
                1 for url in merged_photos if url and url not in dst_photos_before
            )
        elif (
            dst.photo_url
            and dst.photo_url != dst_cover_before
            and dst.photo_url not in dst_photos_before
        ):
            photos_added = 1

        def _fill(field: str) -> None:
            dst_val = getattr(dst, field)
            src_val = getattr(src, field)
            if (dst_val is None or dst_val == "") and src_val:
                setattr(dst, field, src_val)

        existing_aliases = {
            normalized
            for alias in list(dst.aliases or [])
            if (normalized := normalize_alias(alias))
        }
        for field in (
            "full_name",
            "description",
            "website_url",
            "program_url",
            "vk_url",
            "tg_url",
            "ticket_url",
            "location_name",
            "location_address",
            "city",
            "source_text",
            "source_post_url",
            "source_chat_id",
            "source_message_id",
        ):
            _fill(field)

        skip_keys = {normalize_alias(dst_name)}
        if dst.full_name:
            dst_full_norm = normalize_alias(dst.full_name)
            if dst_full_norm:
                skip_keys.add(dst_full_norm)

        seen_aliases: set[str] = set()
        merged_aliases: list[str] = []

        def add_alias(raw: str | None) -> None:
            normalized = normalize_alias(raw)
            if not normalized or normalized in skip_keys or normalized in seen_aliases:
                return
            if len(merged_aliases) >= 8:
                return
            seen_aliases.add(normalized)
            merged_aliases.append(normalized)

        for alias in list(dst.aliases or []) + list(src.aliases or []):
            add_alias(alias)

        add_alias(src_name)
        add_alias(src.full_name)

        dst.aliases = merged_aliases
        aliases_added = sum(1 for alias in merged_aliases if alias not in existing_aliases)

        events_res = await session.execute(
            select(Event).where(Event.festival == dst_name)
        )
        all_events = list(events_res.scalars().all())
        start, end = festival_date_range(all_events)
        if not start and src.start_date:
            start = parse_iso_date(src.start_date)
        if not end and src.end_date:
            end = parse_iso_date(src.end_date)
        dst.start_date = start.isoformat() if start else None
        dst.end_date = end.isoformat() if end else None

        await session.delete(src)
        await session.commit()

        fest_ref: Festival | int
        if dst_pk is not None:
            fest_ref = dst_pk
        else:
            fest_ref = dst
        new_description = await regenerate_festival_description(db, fest_ref)
        if new_description and new_description != (dst.description or ""):
            dst.description = new_description
            await session.commit()
            description_updated = True

    await sync_festival_page(db, dst_name)
    await rebuild_fest_nav_if_changed(db)
    await sync_festival_vk_post(db, dst_name, bot)
    logging.info(
        "merge_festivals: merged src=%s dst=%s events_moved=%s aliases_added=%s photos_added=%s description_updated=%s",
        src_id,
        dst_id,
        moved_events_count,
        aliases_added,
        photos_added,
        description_updated,
    )
    return True


async def generate_festival_poll_text(fest: Festival) -> str:
    """Use LLM to craft poll question for VK."""
    base = (
        "Придумай короткий вопрос, приглашая читателей поделиться,"\
        f" пойдут ли они на фестиваль {fest.name}. "
        "Не повторяй дословно 'Друзья, а вы пойдёте на фестиваль'."
    )
    try:
        text = await ask_4o(base)
        text = text.strip()
    except Exception as e:
        logging.error("failed to generate poll text %s: %s", fest.name, e)
        text = f"Пойдёте ли вы на фестиваль {fest.name}?"
    if fest.vk_post_url:
        text += f"\n{fest.vk_post_url}"
    return text



async def build_festival_page_content(db: Database, fest: Festival) -> tuple[str, list]:
    logging.info("building festival page content for %s", fest.name)

    async with db.get_session() as session:
        today = date.today().isoformat()
        res = await session.execute(
            select(Event)
            .where(Event.festival == fest.name, Event.date >= today)
            .order_by(Event.date, Event.time)
        )
        events = res.scalars().all()

        logging.info("festival %s has %d events", fest.name, len(events))

        if not fest.description:
            desc = await generate_festival_description(fest, events)
            if desc:
                fest.description = desc
                await session.execute(
                    update(Festival)
                    .where(Festival.id == fest.id)
                    .values(description=desc)
                )
                await session.commit()

    nodes: list[dict] = []
    photo_urls = list(fest.photo_urls or [])
    cover = fest.photo_url or (photo_urls[0] if photo_urls else None)
    gallery_photos = [url for url in photo_urls if url != cover]
    if cover:
        nodes.append(
            {
                "tag": "figure",
                "children": [{"tag": "img", "attrs": {"src": cover}}],
            }
        )
        nodes.append({"tag": "p", "children": ["\u00a0"]})
    if fest.program_url:
        nodes.append({"tag": "h2", "children": ["ПРОГРАММА"]})
        links = [
            {
                "tag": "p",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest.program_url},
                        "children": ["Смотреть программу"],
                    }
                ],
            }
        ]
        if fest.website_url:
            links.append(
                {
                    "tag": "p",
                    "children": [
                        {
                            "tag": "a",
                            "attrs": {"href": fest.website_url},
                            "children": ["Сайт"],
                        }
                    ],
                }
            )
        nodes.extend(links)
        nodes.extend(telegraph_br())
    start, end = festival_dates(fest, events)
    if start:
        date_text = format_day_pretty(start)
        if end and end != start:
            date_text += f" - {format_day_pretty(end)}"
        nodes.append({"tag": "p", "children": [f"\U0001f4c5 {date_text}"]})
    loc_text = festival_location(fest, events)
    if loc_text:
        nodes.append({"tag": "p", "children": [f"\U0001f4cd {loc_text}"]})
    if fest.ticket_url:
        nodes.append({"tag": "p", "children": [f"\U0001f39f {fest.ticket_url}"]})
    if fest.description:
        nodes.append({"tag": "p", "children": [fest.description]})

    if fest.website_url or fest.vk_url or fest.tg_url:
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.append({"tag": "h3", "children": ["Контакты фестиваля"]})
        if fest.website_url:
            nodes.append(
                {
                    "tag": "p",
                    "children": [
                        "сайт: ",
                        {
                            "tag": "a",
                            "attrs": {"href": fest.website_url},
                            "children": [fest.website_url],
                        },
                    ],
                }
            )
        if fest.vk_url:
            nodes.append(
                {
                    "tag": "p",
                    "children": [
                        "вк: ",
                        {
                            "tag": "a",
                            "attrs": {"href": fest.vk_url},
                            "children": [fest.vk_url],
                        },
                    ],
                }
            )
        if fest.tg_url:
            nodes.append(
                {
                    "tag": "p",
                    "children": [
                        "телеграм: ",
                        {
                            "tag": "a",
                            "attrs": {"href": fest.tg_url},
                            "children": [fest.tg_url],
                        },
                    ],
                }
            )

    if events:
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.append({"tag": "h3", "children": ["Мероприятия фестиваля"]})
        for e in events:
            # Show 3D preview image if available for the event
            has_preview = bool(getattr(e, "preview_3d_url", None))
            nodes.extend(event_to_nodes(e, festival=fest, show_festival=False, show_image=has_preview))
    else:
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.append({"tag": "p", "children": ["Расписание скоро обновим"]})
    if gallery_photos:
        nodes.extend(telegraph_br())
        for url in gallery_photos:
            nodes.append({"tag": "img", "attrs": {"src": url}})
            nodes.append({"tag": "p", "children": ["\u00a0"]})
    nav_nodes, _ = await _build_festival_nav_block(db, exclude=fest.name)
    if nav_nodes:
        from telegraph.utils import nodes_to_html, html_to_nodes
        nav_html = nodes_to_html(nav_nodes)
        nav_block = f"{FEST_NAV_START}{nav_html}{FEST_NAV_END}"
        nodes.extend(html_to_nodes(nav_block))
    fest_index_url = await get_setting_value(db, "fest_index_url")
    if fest_index_url:
        logging.info(
            "festival_page_index_link",
            extra={"festival": fest.name, "fest_index_url": fest_index_url},
        )
        nodes.append(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest_index_url},
                        "children": ["Фестивали"],
                    }
                ],
            }
        )
        nodes.extend(telegraph_br())
    title = fest.full_name or fest.name
    return title, nodes


async def sync_festival_page(
    db: Database,
    name: str,
    *,
    refresh_nav_only: bool = False,
    items: list[tuple[date | None, date | None, Festival]] | None = None,
):
    async with HEAVY_SEMAPHORE:
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            return None
        tg = Telegraph(access_token=token)
        async with db.get_session() as session:
            result = await session.execute(
                select(Festival).where(Festival.name == name)
            )
            fest = result.scalar_one_or_none()
            if not fest:
                return None
            title = fest.full_name or fest.name
            path = fest.telegraph_path
            url = fest.telegraph_url

        changed = False
        try:
            created = False
            if refresh_nav_only and path:
                nav_html, _, _ = await build_festivals_nav_block(db)
                page = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page.get("content") or page.get("content_html") or ""
                new_html, changed, removed_blocks, markers_replaced = apply_festival_nav(
                    html_content, nav_html
                )
                extra = {
                    "target": "tg",
                    "path": path,
                    "removed_legacy_blocks": removed_blocks,
                    "legacy_markers_replaced": markers_replaced,
                }
                if changed:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        html_content=new_html,
                        caller="festival_build",
                    )
                    logging.info(
                        "updated festival page %s in Telegraph", name,
                        extra={"action": "edited", **extra},
                    )
                else:
                    logging.info(
                        "festival page %s navigation unchanged", name,
                        extra={"action": "skipped_nochange", **extra},
                    )
            else:
                title, content = await build_festival_page_content(db, fest)
                path = fest.telegraph_path
                url = fest.telegraph_url
                if path:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        content=content,
                        caller="festival_build",
                    )
                    changed = True
                    logging.info("updated festival page %s in Telegraph", name)
                else:
                    data = await telegraph_create_page(
                        tg, title, content=content, caller="festival_build"
                    )
                    url = normalize_telegraph_url(data.get("url"))
                    path = data.get("path")
                    created = True
                    changed = True
                    logging.info("created festival page %s: %s", name, url)
        except Exception as e:
            logging.error("Failed to sync festival %s: %s", name, e)
            return None

        async with db.get_session() as session:
            result = await session.execute(
                select(Festival).where(Festival.name == name)
            )
            fest_db = result.scalar_one_or_none()
            if fest_db:
                fest_db.telegraph_url = url
                fest_db.telegraph_path = path
                await session.commit()
                logging.info("synced festival page %s", name)
        return url if changed else None


async def refresh_nav_on_all_festivals(
    db: Database,
    bot: Bot | None = None,
    *,
    nav_html: str | None = None,
    nav_lines: list[str] | None = None,
) -> None:
    """Refresh navigation on all festival pages and VK posts."""
    async with db.get_session() as session:
        res = await session.execute(
            select(
                Festival.id,
                Festival.name,
                Festival.telegraph_path,
                Festival.vk_post_url,
            )
        )
        fests = res.all()

    if nav_html is None or nav_lines is None:
        nav_html, nav_lines, changed = await build_festivals_nav_block(db)
        if not changed:
            return
    token = get_telegraph_token()
    tg = Telegraph(access_token=token) if token else None
    for fid, name, path, vk_url in fests:
        if tg and path:
            try:
                page = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page.get("content") or page.get("content_html") or ""
                title = page.get("title") or name
                new_html, changed, removed_blocks, markers_replaced = apply_festival_nav(
                    html_content, nav_html
                )
                extra = {
                    "target": "tg",
                    "path": path,
                    "fest": name,
                    "removed_legacy_blocks": removed_blocks,
                    "legacy_markers_replaced": markers_replaced,
                }
                if changed:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        html_content=new_html,
                        caller="festival_build",
                    )
                    logging.info(
                        "updated festival page %s in Telegraph", name,
                        extra={"action": "edited", **extra},
                    )
                else:
                    logging.info(
                        "festival page %s navigation unchanged", name,
                        extra={"action": "skipped_nochange", **extra},
                    )
            except Exception as e:
                logging.error(
                    "Failed to update festival page %s: %s", name, e,
                    extra={"action": "error", "target": "tg", "path": path, "fest": name},
                )
        if vk_url:
            await sync_festival_vk_post(
                db, name, bot, nav_only=True, nav_lines=nav_lines
            )


async def build_festival_vk_message(db: Database, fest: Festival) -> str:
    async with db.get_session() as session:
        today = date.today().isoformat()
        res = await session.execute(
            select(Event)
            .where(Event.festival == fest.name, Event.date >= today)
            .order_by(Event.date, Event.time)
        )
        events = res.scalars().all()
        if not fest.description:
            desc = await generate_festival_description(fest, events)
            if desc:
                fest.description = desc
                await session.execute(
                    update(Festival)
                    .where(Festival.id == fest.id)
                    .values(description=desc)
                )
                await session.commit()
    lines = [fest.full_name or fest.name]
    start, end = festival_dates(fest, events)

    if start:
        date_text = format_day_pretty(start)
        if end and end != start:
            date_text += f" - {format_day_pretty(end)}"
        lines.append(f"\U0001f4c5 {date_text}")
    loc_text = festival_location(fest, events)
    if loc_text:
        lines.append(f"\U0001f4cd {loc_text}")
    if fest.ticket_url:
        lines.append(f"\U0001f39f {fest.ticket_url}")
    if fest.program_url:
        lines.append(f"программа: {fest.program_url}")
    if fest.description:
        lines.append(fest.description)
    if fest.website_url or fest.vk_url or fest.tg_url:
        lines.append(VK_BLANK_LINE)
        lines.append("Контакты фестиваля")
        if fest.website_url:
            lines.append(f"сайт: {fest.website_url}")
        if fest.vk_url:
            lines.append(f"вк: {fest.vk_url}")
        if fest.tg_url:
            lines.append(f"телеграм: {fest.tg_url}")
    if events:
        for ev in events:
            lines.append(VK_BLANK_LINE)
            lines.append(format_event_vk(ev))
    else:
        lines.append(VK_BLANK_LINE)
        lines.append("Расписание скоро обновим")
    _, nav_lines = await _build_festival_nav_block(db, exclude=fest.name)
    if nav_lines:
        lines.extend(nav_lines)
    return "\n".join(lines)


async def sync_festival_vk_post(
    db: Database,
    name: str,
    bot: Bot | None = None,
    *,
    nav_only: bool = False,
    nav_lines: list[str] | None = None,
    strict: bool = False,
) -> bool | None:
    group_id = await get_vk_group_id(db)
    if not group_id:
        return
    async with db.get_session() as session:
        res = await session.execute(select(Festival).where(Festival.name == name))
        fest = res.scalar_one_or_none()
        if not fest:
            return

    async def _try_edit(message: str, attachments: list[str] | None) -> bool | None:
        if not fest.vk_post_url:
            return False
        user_token = _vk_user_token()
        if not user_token:
            logging.error(
                "VK_USER_TOKEN missing",
                extra={
                    "action": "error",
                    "target": "vk",
                    "url": fest.vk_post_url,
                    "fest": name,
                },
            )
            return None
        for attempt in range(1, 4):
            try:
                await edit_vk_post(fest.vk_post_url, message, db, bot, attachments)
                return True
            except VKAPIError as e:
                logging.warning(
                    "Ошибка VK при редактировании (попытка %d из 3, код %s): %s actor=%s token=%s",
                    attempt,
                    e.code,
                    e.message,
                    e.actor,
                    e.token,
                )
                if e.code in {213, 214} or "edit time expired" in e.message.lower():
                    return False
                if attempt == 3:
                    return None
                await asyncio.sleep(2 ** (attempt - 1))
        return None

    async def _try_post(message: str, attachments: list[str] | None) -> str | None:
        for attempt in range(1, 4):
            try:
                url = await post_to_vk(group_id, message, db, bot, attachments)
                if url:
                    return url
            except VKAPIError as e:
                logging.warning(
                    "Ошибка VK при публикации (попытка %d из 3, код %s): %s actor=%s token=%s",
                    attempt,
                    e.code,
                    e.message,
                    e.actor,
                    e.token,
                )
            if attempt == 3:
                return None
            await asyncio.sleep(2 ** (attempt - 1))
        return None

    can_edit = True
    if nav_only and fest.vk_post_url:
        nav_lines_local = nav_lines
        if nav_lines_local is None:
            _, nav_lines_local = await _build_festival_nav_block(db, exclude=fest.name)
        if nav_lines_local:
            ids = _vk_owner_and_post_id(fest.vk_post_url)
            if not ids:
                logging.error(
                    "invalid VK post url %s",
                    fest.vk_post_url,
                    extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
                )
                return
            owner_id, post_id = ids
            try:
                response = await vk_api(
                    "wall.getById", posts=f"{owner_id}_{post_id}"
                )
                if isinstance(response, dict):
                    items = response.get("response") or (
                        response["response"] if "response" in response else response
                    )
                else:
                    items = response or []
                if not isinstance(items, list):
                    items = [items] if items else []
                text = items[0].get("text", "") if items else ""
            except Exception as e:
                logging.error(
                    "Не удалось получить пост VK для %s: %s",
                    name,
                    e,
                    extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
                )
                return
            lines = text.split("\n")
            idx = None
            for i, line in enumerate(lines):
                if line == "Ближайшие фестивали":
                    idx = i
                    if i > 0 and lines[i - 1] == VK_BLANK_LINE:
                        idx -= 1
                    break
            base = lines[:idx] if idx is not None else lines
            message = "\n".join(base + nav_lines_local)
            if message == text:
                logging.info(
                    "festival post %s navigation unchanged", name,
                    extra={
                        "action": "skipped_nochange",
                        "target": "vk",
                        "url": fest.vk_post_url,
                        "fest": name,
                    },
                )
                return False
            res_edit = await _try_edit(message, None)
            if res_edit is True:
                logging.info(
                    "updated festival post %s on VK", name,
                    extra={
                        "action": "edited",
                        "target": "vk",
                        "url": fest.vk_post_url,
                        "fest": name,
                    },
                )
                return True
            if res_edit is None:
                logging.error(
                    "VK post error for festival %s", name,
                    extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
                )
                if strict:
                    raise RuntimeError("vk edit failed")
                return False
            if os.getenv("VK_NAV_FALLBACK") == "skip":
                logging.info(
                    "festival post %s skipping VK edit", name,
                    extra={
                        "action": "vk_nav_skip_edit",
                        "target": "vk",
                        "url": fest.vk_post_url,
                        "fest": name,
                    },
                )
                return False
            can_edit = False  # editing not possible, create new post

    message = await build_festival_vk_message(db, fest)
    attachments: list[str] | None = None
    if fest.photo_url:
        if VK_PHOTOS_ENABLED:
            photo_id = await upload_vk_photo(group_id, fest.photo_url, db, bot)
            if photo_id:
                attachments = [photo_id]
        else:
            logging.info("VK photo posting disabled")

    if fest.vk_post_url and can_edit:
        res_edit = await _try_edit(message, attachments)
        if res_edit is True:
            logging.info(
                "updated festival post %s on VK", name,
                extra={
                    "action": "edited",
                    "target": "vk",
                    "url": fest.vk_post_url,
                    "fest": name,
                },
            )
            return True
        if res_edit is None:
            logging.error(
                "VK post error for festival %s", name,
                extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
            )
            if strict:
                raise RuntimeError("vk edit failed")
            return False

    url = await _try_post(message, attachments)
    if url:
        async with db.get_session() as session:
            fest_db = (
                await session.execute(select(Festival).where(Festival.name == name))
            ).scalar_one()
            fest_db.vk_post_url = url
            await session.commit()
        logging.info(
            "created festival post %s: %s", name, url,
            extra={"action": "created", "target": "vk", "url": url, "fest": name},
        )
        return True
    logging.error(
        "VK post error for festival %s", name,
        extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
    )
    if strict:
        raise RuntimeError("vk post failed")
    return False


async def send_festival_poll(
    db: Database,
    fest: Festival,
    group_id: str,
    bot: Bot | None = None,
) -> None:
    question = await generate_festival_poll_text(fest)
    url = await post_vk_poll(group_id, question, VK_POLL_OPTIONS, db, bot)
    if url:

        async with db.get_session() as session:
            obj = await session.get(Festival, fest.id)
            if obj:
                obj.vk_poll_url = url
                await session.commit()
        if bot:
            await notify_superadmin(db, bot, f"poll created {url}")




async def build_daily_posts(
    db: Database,
    tz: timezone,
    now: datetime | None = None,
) -> list[tuple[str, types.InlineKeyboardMarkup | None]]:
    from models import Event, WeekendPage, MonthPage, Festival

    if now is None:
        now = datetime.now(tz)
    today = now.date()
    yesterday_utc = recent_cutoff(tz, now)
    fest_map: dict[str, Festival] = {}
    recent_festival_entries: list[str] = []
    partner_creator_ids: set[int] = set()
    async with db.get_session() as session:
        res_today = await session.execute(
            select(Event)
            .where(Event.date == today.isoformat())
            .order_by(Event.time)
        )
        events_today = res_today.scalars().all()
        if len(events_today) < 6:
            res_fairs = await session.execute(
                select(Event)
                .where(
                    Event.event_type == "ярмарка",
                    Event.end_date.is_not(None),
                    Event.end_date >= today.isoformat(),
                    Event.date <= today.isoformat(),
                )
                .order_by(Event.date, Event.time)
            )
            fairs_today = res_fairs.scalars().all()
            if fairs_today:
                seen_ids = {e.id for e in events_today if e.id is not None}
                fairs_today = [e for e in fairs_today if e.id not in seen_ids]
                if fairs_today:
                    events_today.extend(
                        clone_event_with_date(e, today) for e in fairs_today
                    )
                    events_today.sort(key=lambda e: e.time or "99:99")
        res_new = await session.execute(
            select(Event)
            .where(
                Event.date > today.isoformat(),
                Event.added_at.is_not(None),
                Event.added_at >= yesterday_utc,
                Event.silent.is_(False),
            )
            .order_by(Event.date, Event.time)
        )
        events_new = res_new.scalars().all()

        w_start = next_weekend_start(today)
        wpage = await session.get(WeekendPage, w_start.isoformat())
        res_w_all = await session.execute(select(WeekendPage))
        weekend_map = {w.start: w for w in res_w_all.scalars().all()}
        cur_month = today.strftime("%Y-%m")
        mp_cur = await session.get(MonthPage, cur_month)
        mp_next = await session.get(MonthPage, next_month(cur_month))

        new_events = (
            await session.execute(
                select(Event).where(
                    Event.added_at.is_not(None),
                    Event.added_at >= yesterday_utc,
                )
            )
        ).scalars().all()

        res_fests = await session.execute(select(Festival))
        festivals = res_fests.scalars().all()
        fest_map = {f.name: f for f in festivals}
        recent_festivals: list[tuple[datetime, str]] = []
        for fest in festivals:
            url = _festival_telegraph_url(fest)
            if not url:
                continue
            created_at = _ensure_utc(getattr(fest, "created_at", None))
            if created_at and created_at >= yesterday_utc:
                recent_festivals.append(
                    (
                        created_at,
                        f'<a href="{url}">✨ {html.escape(fest.name)}</a>',
                    )
                )
        recent_festivals.sort(key=lambda item: item[0])
        recent_festival_entries = [entry for _, entry in recent_festivals]

        creator_ids = {
            e.creator_id
            for e in (*events_today, *events_new, *new_events)
            if e.creator_id is not None
        }
        if creator_ids:
            res_partners = await session.execute(
                select(User.user_id).where(
                    User.user_id.in_(creator_ids),
                    User.is_partner.is_(True),
                )
            )
            partner_creator_ids = set(res_partners.scalars().all())

        weekend_count = 0
        if wpage:
            sat = w_start
            sun = w_start + timedelta(days=1)
            weekend_new = [
                e
                for e in new_events
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    is_long_event_type(e.event_type)
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_today = [
                e
                for e in events_today
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    is_long_event_type(e.event_type)
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_count = max(0, len(weekend_new) - len(weekend_today))

        cur_count = 0
        next_count = 0
        for e in new_events:
            m = e.date[:7]
            if m == cur_month:
                cur_count += 1
            elif m == next_month(cur_month):
                next_count += 1

    processed_short_ids: set[int] = set()
    for candidate in (*events_today, *events_new):
        if (
            candidate.ticket_link
            and candidate.id is not None
            and candidate.id not in processed_short_ids
            and not (candidate.vk_ticket_short_url and candidate.vk_ticket_short_key)
        ):
            await ensure_vk_short_ticket_link(
                candidate,
                db,
                vk_api_fn=_vk_api,
            )
            processed_short_ids.add(candidate.id)

    tag = f"{today.day}{MONTHS[today.month - 1]}"
    lines1 = [
        f"<b>АНОНС на {format_day_pretty(today)} {today.year} #ежедневныйанонс</b>",
        DAYS_OF_WEEK[today.weekday()],
        "",
        "<b><i>НЕ ПРОПУСТИТЕ СЕГОДНЯ</i></b>",
    ]
    for e in events_today:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.url
        lines1.append("")
        lines1.append(
            format_event_daily(
                e,
                highlight=True,
                weekend_url=w_url,
                festival=fest_map.get((e.festival or "").casefold()),
                partner_creator_ids=partner_creator_ids,
            )
        )
    lines1.append("")
    lines1.append(
        f"#Афиша_Калининград #Калининград #концерт #{tag} #{today.day}_{MONTHS[today.month - 1]}"
    )
    section1 = "\n".join(lines1)

    lines2 = [f"<b><i>+{len(events_new)} ДОБАВИЛИ В АНОНС</i></b>"]
    if len(events_new) > 9:
        grouped: dict[str, list[Event]] = {}
        for e in events_new:
            raw_city = (e.city or "Калининград").strip()
            city = raw_city or "Калининград"
            grouped.setdefault(city, []).append(e)
        for city, events in grouped.items():
            lines2.append("")
            lines2.append(html.escape(city.upper()))
            for e in events:
                lines2.append(
                    format_event_daily_inline(
                        e,
                        partner_creator_ids=partner_creator_ids,
                    )
                )
        lines2.append("")
        lines2.append("ℹ️ Нажмите на название мероприятия, чтобы открыть подробности")
    else:
        for e in events_new:
            w_url = None
            d = parse_iso_date(e.date)
            if d and d.weekday() == 5:
                w = weekend_map.get(d.isoformat())
                if w:
                    w_url = w.url
            lines2.append("")
            lines2.append(
                format_event_daily(
                    e,
                    weekend_url=w_url,
                    festival=fest_map.get((e.festival or "").casefold()),
                    partner_creator_ids=partner_creator_ids,
                )
            )
    if recent_festival_entries:
        lines2.append("")
        lines2.append("ФЕСТИВАЛИ")
        lines2.append(" ".join(recent_festival_entries))
    section2 = "\n".join(lines2)

    fest_index_url = await get_setting_value(db, "fest_index_url")

    buttons = []
    if wpage:
        sunday = w_start + timedelta(days=1)
        prefix = f"(+{weekend_count}) " if weekend_count else ""
        text = (
            f"{prefix}Мероприятия на выходные {w_start.day} {sunday.day} {MONTHS[w_start.month - 1]}"
        )
        buttons.append(types.InlineKeyboardButton(text=text, url=wpage.url))
    if mp_cur:
        prefix = f"(+{cur_count}) " if cur_count else ""
        buttons.append(
            types.InlineKeyboardButton(
                text=f"{prefix}Мероприятия на {month_name_nominative(cur_month)}",
                url=mp_cur.url,
            )
        )
    if mp_next:
        prefix = f"(+{next_count}) " if next_count else ""
        buttons.append(
            types.InlineKeyboardButton(
                text=f"{prefix}Мероприятия на {month_name_nominative(next_month(cur_month))}",
                url=mp_next.url,
            )
        )
    if fest_index_url:
        buttons.append(
            types.InlineKeyboardButton(text="Фестивали", url=fest_index_url)
        )
    markup = None
    if buttons:
        markup = types.InlineKeyboardMarkup(inline_keyboard=[[b] for b in buttons])

    combined = section1 + "\n\n\n" + section2
    if len(combined) <= 4096:
        return [(combined, markup)]

    posts: list[tuple[str, types.InlineKeyboardMarkup | None]] = []
    for part in split_text(section1):
        posts.append((part, None))
    section2_parts = split_text(section2)
    for part in section2_parts[:-1]:
        posts.append((part, None))
    posts.append((section2_parts[-1], markup))
    return posts


async def build_daily_sections_vk(
    db: Database,
    tz: timezone,
    now: datetime | None = None,
) -> tuple[str, str]:
    from models import User

    if now is None:
        now = datetime.now(tz)
    today = now.date()
    yesterday_utc = recent_cutoff(tz, now)
    fest_map: dict[str, Festival] = {}
    async with db.get_session() as session:
        res_today = await session.execute(
            select(Event)
            .where(Event.date == today.isoformat())
            .order_by(Event.time)
        )
        events_today = res_today.scalars().all()
        if len(events_today) < 6:
            res_fairs = await session.execute(
                select(Event)
                .where(
                    Event.event_type == "ярмарка",
                    Event.end_date.is_not(None),
                    Event.end_date >= today.isoformat(),
                    Event.date <= today.isoformat(),
                )
                .order_by(Event.date, Event.time)
            )
            fairs_today = res_fairs.scalars().all()
            if fairs_today:
                seen_ids = {e.id for e in events_today if e.id is not None}
                fairs_today = [e for e in fairs_today if e.id not in seen_ids]
                if fairs_today:
                    events_today.extend(
                        clone_event_with_date(e, today) for e in fairs_today
                    )
                    events_today.sort(key=lambda e: e.time or "99:99")
        res_new = await session.execute(
            select(Event)
            .where(
                Event.date > today.isoformat(),
                Event.added_at.is_not(None),
                Event.added_at >= yesterday_utc,
                Event.silent.is_(False),
            )
            .order_by(Event.date, Event.time)
        )
        events_new = res_new.scalars().all()

        w_start = next_weekend_start(today)
        wpage = await session.get(WeekendPage, w_start.isoformat())
        res_w_all = await session.execute(select(WeekendPage))
        weekend_map = {w.start: w for w in res_w_all.scalars().all()}
        res_week_all = await session.execute(select(WeekPage))
        week_pages = res_week_all.scalars().all()
        cur_month = today.strftime("%Y-%m")

        def closest_week_page(month: str, ref: date) -> WeekPage | None:
            candidates = [w for w in week_pages if w.start[:7] == month and w.vk_post_url]
            if not candidates:
                return None
            return min(candidates, key=lambda w: abs(date.fromisoformat(w.start) - ref))

        week_cur = closest_week_page(cur_month, today)
        next_month_str = next_month(cur_month)
        week_next = closest_week_page(next_month_str, date.fromisoformat(f"{next_month_str}-01"))

        new_events = (
            await session.execute(
                select(Event).where(
                    Event.added_at.is_not(None),
                    Event.added_at >= yesterday_utc,
                )
            )
        ).scalars().all()

        creator_ids = {
            e.creator_id
            for e in (*events_today, *events_new, *new_events)
            if e.creator_id is not None
        }
        partner_creator_ids: set[int] = set()
        if creator_ids:
            res_partners = await session.execute(
                select(User.user_id).where(
                    User.user_id.in_(creator_ids),
                    User.is_partner.is_(True),
                )
            )
            partner_creator_ids = set(res_partners.scalars().all())

        weekend_count = 0
        if wpage:
            sat = w_start
            sun = w_start + timedelta(days=1)
            weekend_new = [
                e
                for e in new_events
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    is_long_event_type(e.event_type)
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_today = [
                e
                for e in events_today
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    is_long_event_type(e.event_type)
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_count = max(0, len(weekend_new) - len(weekend_today))

        cur_count = 0
        next_count = 0
        for e in new_events:
            m = e.date[:7]
            if m == cur_month:
                cur_count += 1
            elif m == next_month(cur_month):
                next_count += 1

    processed_short_ids: set[int] = set()
    for candidate in (*events_today, *events_new):
        if (
            candidate.ticket_link
            and candidate.id is not None
            and candidate.id not in processed_short_ids
            and not (
                candidate.vk_ticket_short_url and candidate.vk_ticket_short_key
            )
        ):
            await ensure_vk_short_ticket_link(
                candidate,
                db,
                vk_api_fn=_vk_api,
            )
            processed_short_ids.add(candidate.id)

    lines1 = [
        f"\U0001f4c5 АНОНС на {format_day_pretty(today)} {today.year}",
        DAYS_OF_WEEK[today.weekday()],
        "",
        "НЕ ПРОПУСТИТЕ СЕГОДНЯ",
    ]
    for e in events_today:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.vk_post_url
        lines1.append(
            format_event_vk(
                e,
                highlight=True,
                weekend_url=w_url,
                festival=fest_map.get((e.festival or "").casefold()),
                partner_creator_ids=partner_creator_ids,
                prefer_vk_repost=True,
            )
        )
        lines1.append(VK_EVENT_SEPARATOR)
    if events_today:
        lines1.pop()
    link_lines: list[str] = []
    if wpage and wpage.vk_post_url:
        label = f"выходные {format_weekend_range(w_start)}"
        prefix = f"(+{weekend_count}) " if weekend_count else ""
        link_lines.append(f"{prefix}[{wpage.vk_post_url}|{label}]")
    if week_cur:
        label = month_name_nominative(cur_month)
        prefix = f"(+{cur_count}) " if cur_count else ""
        link_lines.append(f"{prefix}[{week_cur.vk_post_url}|{label}]")
    if week_next:
        label = month_name_nominative(next_month_str)
        prefix = f"(+{next_count}) " if next_count else ""
        link_lines.append(f"{prefix}[{week_next.vk_post_url}|{label}]")
    if link_lines:
        lines1.append(VK_EVENT_SEPARATOR)
        lines1.extend(link_lines)
    lines1.append(VK_EVENT_SEPARATOR)
    lines1.append(
        f"#Афиша_Калининград #кудапойти_Калининград #Калининград #39region #концерт #{today.day}{MONTHS[today.month - 1]}"
    )
    section1 = "\n".join(lines1)

    lines2 = [f"+{len(events_new)} ДОБАВИЛИ В АНОНС", VK_BLANK_LINE]
    for e in events_new:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.vk_post_url
        lines2.append(
            format_event_vk(
                e,
                weekend_url=w_url,
                festival=fest_map.get((e.festival or "").casefold()),
                partner_creator_ids=partner_creator_ids,
                prefer_vk_repost=True,
            )
        )
        lines2.append(VK_EVENT_SEPARATOR)
    if events_new:
        lines2.pop()
    if link_lines:
        lines2.append(VK_EVENT_SEPARATOR)
        lines2.extend(link_lines)
    lines2.append(VK_EVENT_SEPARATOR)
    lines2.append(
        f"#события_Калининград #Калининград #39region #новое #фестиваль #{today.day}{MONTHS[today.month - 1]}"
    )
    section2 = "\n".join(lines2)

    return section1, section2


async def post_to_vk(
    group_id: str,
    message: str,
    db: Database | None = None,
    bot: Bot | None = None,
    attachments: list[str] | None = None,
) -> str | None:
    if not group_id:
        return None
    logging.info(
        "post_to_vk start: group=%s len=%d attachments=%d",
        group_id,
        len(message),
        len(attachments or []),
    )
    owner_id = -int(group_id.lstrip("-"))
    params_base = {"owner_id": f"-{group_id.lstrip('-')}", "message": message}
    if attachments:
        params_base["attachments"] = ",".join(attachments)
    actors = choose_vk_actor(owner_id, "wall.post")
    if not actors:
        raise VKAPIError(None, "VK token missing", method="wall.post")
    for idx, actor in enumerate(actors, start=1):
        params = params_base.copy()
        if actor.kind == "user" and owner_id < 0:
            params["from_group"] = 1
        logging.info(
            "vk.call method=wall.post owner_id=%s try=%d/%d actor=%s",
            owner_id,
            idx,
            len(actors),
            actor.label,
        )
        token = actor.token if actor.kind == "group" else VK_USER_TOKEN
        try:
            if DEBUG:
                mem_info("VK post before")
            data = await _vk_api(
                "wall.post",
                params,
                db,
                bot,
                token=token,
                token_kind=actor.kind,
                skip_captcha=(actor.kind == "group"),
            )
            if DEBUG:
                mem_info("VK post after")
            post_id = data.get("response", {}).get("post_id")
            if post_id:
                url = f"https://vk.com/wall-{group_id.lstrip('-')}_{post_id}"
                logging.info(
                    "post_to_vk ok group=%s post_id=%s len=%d attachments=%d",
                    group_id,
                    post_id,
                    len(message),
                    len(attachments or []),
                )
                return url
            err_code = (
                data.get("error", {}).get("error_code") if isinstance(data, dict) else None
            )
            logging.error(
                "post_to_vk fail group=%s code=%s len=%d attachments=%d",
                group_id,
                err_code,
                len(message),
                len(attachments or []),
            )
            return None
        except VKAPIError as e:
            logging.warning(
                "post_to_vk error code=%s msg=%s actor=%s token=%s",
                e.code,
                e.message,
                e.actor,
                e.token,
            )
            msg_l = (e.message or "").lower()
            perm = (
                e.code in VK_FALLBACK_CODES
                or "method is unavailable with group auth" in msg_l
                or "access denied" in msg_l
            )
            if idx < len(actors) and perm:
                logging.info(
                    "vk.retry reason=%s actor_next=%s",
                    e.code or e.message,
                    actors[idx].label,
                )
                continue
            raise
    return None


async def create_vk_poll(
    group_id: str,
    question: str,
    options: list[str],
    db: Database | None = None,
    bot: Bot | None = None,
) -> str | None:
    """Create poll and return attachment id."""
    logging.info(
        "create_vk_poll start: group=%s question=%s", group_id, question
    )
    params = {
        "owner_id": f"-{group_id.lstrip('-')}",
        "question": question,
        "is_anonymous": 0,
        "add_answers": json.dumps(options, ensure_ascii=False),
    }
    data = await _vk_api("polls.create", params, db, bot)
    poll = data.get("response") or {}
    p_id = poll.get("id")
    owner = poll.get("owner_id", f"-{group_id.lstrip('-')}")
    if p_id is not None:
        attachment = f"poll{owner}_{p_id}"
        logging.info("create_vk_poll success: %s", attachment)
        return attachment
    logging.error("create_vk_poll failed for group %s", group_id)
    return None


async def post_vk_poll(
    group_id: str,
    question: str,
    options: list[str],
    db: Database | None = None,
    bot: Bot | None = None,
) -> str | None:
    """Create poll and post it to group wall."""
    logging.info("post_vk_poll start for group %s", group_id)
    attachment = await create_vk_poll(group_id, question, options, db, bot)
    if not attachment:
        logging.error("post_vk_poll: poll creation failed for group %s", group_id)
        return None
    # ensure non-empty message to satisfy VK API
    return await post_to_vk(group_id, question or "?", db, bot, [attachment])



def _vk_owner_and_post_id(url: str) -> tuple[str, str] | None:
    m = re.search(r"wall(-?\d+)_(\d+)", url)
    if not m:
        return None
    return m.group(1), m.group(2)




def build_vk_source_header(event: Event, festival: Festival | None = None) -> list[str]:
    """Build header lines for VK source post with general event info."""

    lines: list[str] = [event.title]

    if festival:
        link = festival.vk_url or festival.vk_post_url
        prefix = "✨ "
        if link:
            lines.append(f"{prefix}[{link}|{festival.name}]")
        else:
            lines.append(f"{prefix}{festival.name}")

    lines.append(VK_BLANK_LINE)

    date_part = event.date.split("..", 1)[0]
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        logging.error("Invalid event date: %s", event.date)
        day = event.date
    lines.append(f"\U0001f4c5 {day} {event.time}")

    loc = event.location_name
    addr = event.location_address
    if addr and event.city:
        addr = strip_city_from_address(addr, event.city)
    if addr:
        loc += f", {addr}"
    if event.city:
        loc += f", #{event.city}"
    lines.append(f"\U0001f4cd {loc}")

    if event.pushkin_card:
        lines.append("\u2705 Пушкинская карта")

    ticket_link_display = (
        format_vk_short_url(event.vk_ticket_short_url)
        if event.vk_ticket_short_url
        else event.ticket_link
    )

    if event.is_free:
        lines.append("🟡 Бесплатно")
        if event.ticket_link:
            lines.append(f"\U0001f39f по регистрации {ticket_link_display}")
    elif event.ticket_link and (
        event.ticket_price_min is not None or event.ticket_price_max is not None
    ):
        if event.ticket_price_max is not None and event.ticket_price_max != event.ticket_price_min:
            price = f"от {event.ticket_price_min} до {event.ticket_price_max} руб."
        else:
            val = (
                event.ticket_price_min
                if event.ticket_price_min is not None
                else event.ticket_price_max
            )
            price = f"{val} руб." if val is not None else ""
        info = f"Билеты в источнике {price}".strip()
        lines.append(f"\U0001f39f {info} {ticket_link_display}".strip())
    elif event.ticket_link:
        lines.append(f"\U0001f39f по регистрации {ticket_link_display}")
    else:
        price = ""
        if (
            event.ticket_price_min is not None
            and event.ticket_price_max is not None
            and event.ticket_price_min != event.ticket_price_max
        ):
            price = f"от {event.ticket_price_min} до {event.ticket_price_max} руб."
        elif event.ticket_price_min is not None:
            price = f"{event.ticket_price_min} руб."
        elif event.ticket_price_max is not None:
            price = f"{event.ticket_price_max} руб."
        if price:
            lines.append(f"\U0001f39f Билеты {price}")

    lines.append(VK_BLANK_LINE)
    return lines


def build_vk_source_message(
    event: Event,
    text: str,
    festival: Festival | None = None,
    *,
    calendar_url: str | None = None,
) -> str:
    """Build detailed VK post for an event including original source text."""

    text = sanitize_for_vk(text)
    lines = build_vk_source_header(event, festival)
    lines.extend(text.strip().splitlines())
    lines.append(VK_BLANK_LINE)
    if calendar_url:
        lines.append(f"Добавить в календарь {calendar_url}")
    lines.append(VK_SOURCE_FOOTER)
    return "\n".join(lines)


async def sync_vk_source_post(
    event: Event,
    text: str,

    db: Database | None,
    bot: Bot | None,
    *,
    ics_url: str | None = None,
    append_text: bool = True,
) -> str | None:
    """Create or update VK source post for an event."""
    if not VK_AFISHA_GROUP_ID:
        return None
    logging.info("sync_vk_source_post start for event %s", event.id)
    festival = None
    if event.festival and db:
        async with db.get_session() as session:
            res = await session.execute(
                select(Festival).where(Festival.name == event.festival)
            )
            festival = res.scalars().first()

    attachments: list[str] | None = None
    if VK_PHOTOS_ENABLED and event.photo_urls:
        token = VK_TOKEN_AFISHA or VK_TOKEN
        if token:
            ids: list[str] = []
            for url in event.photo_urls[:VK_MAX_ATTACHMENTS]:
                photo_id = await upload_vk_photo(
                    VK_AFISHA_GROUP_ID, url, db, bot, token=token, token_kind="group"
                )
                if photo_id:
                    ids.append(photo_id)
                elif not VK_USER_TOKEN:
                    logging.info(
                        "VK photo upload skipped: user token required",
                        extra={"eid": event.id},
                    )
                    break
            if ids:
                attachments = ids
        else:
            logging.info("VK photo upload skipped: no group token")

    calendar_line_value: str | None = None
    previous_ics_url = event.ics_url
    calendar_source_url = event.ics_url if ics_url is None else ics_url
    if calendar_source_url != previous_ics_url or calendar_source_url is None:
        if event.vk_ics_short_url or event.vk_ics_short_key:
            event.vk_ics_short_url = None
            event.vk_ics_short_key = None
    event.ics_url = calendar_source_url
    if calendar_source_url:
        short_ics = await ensure_vk_short_ics_link(
            event,
            db,
            bot=bot,
            vk_api_fn=_vk_api,
        )
        if short_ics:
            calendar_line_value = format_vk_short_url(short_ics[0])
        else:
            calendar_line_value = calendar_source_url

    if event.source_vk_post_url:
        await ensure_vk_short_ticket_link(
            event, db, bot=bot, vk_api_fn=_vk_api
        )
        existing = ""
        try:
            ids = _vk_owner_and_post_id(event.source_vk_post_url)
            if ids:
                response = await vk_api(
                    "wall.getById", posts=f"{ids[0]}_{ids[1]}"
                )
                if isinstance(response, dict):
                    items = response.get("response") or (
                        response["response"] if "response" in response else response
                    )
                else:
                    items = response or []
                if not isinstance(items, list):
                    items = [items] if items else []
                if items:
                    existing = items[0].get("text", "")
        except Exception as e:
            logging.error("failed to fetch existing VK post: %s", e)

        # Extract previous text versions
        existing_main = existing.split(VK_SOURCE_FOOTER)[0].rstrip()
        segments = (
            existing_main.split(f"\n{CONTENT_SEPARATOR}\n") if existing_main else []
        )
        texts: list[str] = []
        for seg in segments:
            lines = seg.splitlines()
            blanks = 0
            i = 0
            while i < len(lines):
                if lines[i] == VK_BLANK_LINE:
                    blanks += 1
                    if blanks == 2:
                        i += 1
                        break
                i += 1
            lines = lines[i:]
            if lines and lines[-1].startswith("Добавить в календарь"):
                lines.pop()
            while lines and lines[-1] == VK_BLANK_LINE:
                lines.pop()
            texts.append("\n".join(lines).strip())

        text_clean = sanitize_for_vk(text).strip()
        if texts:
            if append_text:
                texts.append(text_clean)
            else:
                texts[-1] = text_clean
        else:
            texts = [text_clean]

        header_lines = build_vk_source_header(event, festival)
        new_lines = header_lines[:]
        for idx, t in enumerate(texts):
            if t:
                new_lines.extend(t.splitlines())
            new_lines.append(VK_BLANK_LINE)
            if idx < len(texts) - 1:
                new_lines.append(CONTENT_SEPARATOR)
        if calendar_line_value:
            new_lines.append(f"Добавить в календарь {calendar_line_value}")
        new_lines.append(VK_SOURCE_FOOTER)
        new_message = "\n".join(new_lines)
        await edit_vk_post(
            event.source_vk_post_url,
            new_message,
            db,
            bot,
            attachments,
        )
        url = event.source_vk_post_url
        logging.info("sync_vk_source_post updated %s", url)
    else:
        _short_link_result = await ensure_vk_short_ticket_link(
            event, db, vk_api_fn=_vk_api, bot=bot
        )
        message = build_vk_source_message(
            event, text, festival=festival, calendar_url=calendar_line_value
        )
        url = await post_to_vk(
            VK_AFISHA_GROUP_ID,
            message,
            db,
            bot,
            attachments,
        )
        if url:
            logging.info("sync_vk_source_post created %s", url)
    return url


async def edit_vk_post(
    post_url: str,
    message: str,
    db: Database | None = None,
    bot: Bot | None = None,
    attachments: list[str] | None = None,
) -> bool:
    """Edit an existing VK post.

    Returns ``True`` if the post was changed and ``False`` if the current
    content already matches ``message`` and ``attachments``.
    """
    logging.info("edit_vk_post start: %s", post_url)
    ids = _vk_owner_and_post_id(post_url)
    if not ids:
        logging.error("invalid VK post url %s", post_url)
        return
    owner_id, post_id = ids
    params = {
        "owner_id": owner_id,
        "post_id": post_id,
        "message": message,
        "from_group": 1,
    }
    owner_id_num: int | None = None
    try:
        owner_id_num = int(owner_id)
    except (TypeError, ValueError):
        owner_id_num = None

    def normalize_group_id(group_id: str | None) -> int | None:
        if not group_id:
            return None
        try:
            value = int(group_id)
        except (TypeError, ValueError):
            return None
        return -abs(value) if value else value

    main_owner_id = normalize_group_id(VK_MAIN_GROUP_ID)
    afisha_owner_id = normalize_group_id(VK_AFISHA_GROUP_ID)

    use_internal_api = False
    edit_token: str | None = None
    edit_token_kind = "group"

    if owner_id_num is not None:
        if owner_id_num == main_owner_id:
            use_internal_api = True
            if VK_TOKEN:
                edit_token = VK_TOKEN
        elif owner_id_num == afisha_owner_id:
            use_internal_api = True
            if VK_TOKEN_AFISHA:
                edit_token = VK_TOKEN_AFISHA

    if edit_token is None:
        edit_token = _vk_user_token()
        edit_token_kind = "user"
    else:
        edit_token_kind = "group"
    current: list[str] = []
    post_text = ""
    old_attachments: list[str] = []
    edit_allowed = True
    edit_block_reason: str | None = None
    try:
        if use_internal_api:
            response = await _vk_api(
                "wall.getById",
                {"posts": f"{owner_id}_{post_id}"},
                db,
                bot,
                token=edit_token,
                token_kind=edit_token_kind,
                skip_captcha=True,
            )
        else:
            response = await vk_api("wall.getById", posts=f"{owner_id}_{post_id}")
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
        if items:
            post = items[0]
            post_text = post.get("text") or ""
            can_edit_flag = post.get("can_edit")
            if can_edit_flag is not None and not bool(can_edit_flag):
                edit_allowed = False
                edit_block_reason = "can_edit=0"
            else:
                ts = post.get("date")
                if ts is not None:
                    try:
                        post_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    except (ValueError, OSError, OverflowError):
                        pass
                    else:
                        if datetime.now(timezone.utc) - post_dt > VK_POST_MAX_EDIT_AGE:
                            edit_allowed = False
                            edit_block_reason = "post too old"
            for att in post.get("attachments", []):
                if att.get("type") == "photo":
                    p = att.get("photo") or {}
                    o_id = p.get("owner_id")
                    p_id = p.get("id")
                    if o_id is not None and p_id is not None:
                        current.append(f"photo{o_id}_{p_id}")
            old_attachments = current.copy()
    except Exception as e:
        logging.error("failed to fetch VK post attachments: %s", e)
    if attachments is not None:
        current = attachments[:]
    else:
        current = old_attachments.copy()
    if not edit_allowed:
        logging.warning(
            "edit_vk_post: skipping %s, edit unavailable (%s)",
            post_url,
            edit_block_reason or "unknown reason",
        )
        if db is not None and bot is not None:
            try:
                await notify_superadmin(
                    db,
                    bot,
                    f"Не удалось отредактировать пост {post_url}: окно редактирования истекло",
                )
            except Exception:  # pragma: no cover - best effort
                logging.exception("edit_vk_post notify_superadmin failed")
        return False
    if post_text == message and current == old_attachments:
        logging.info("edit_vk_post: no changes for %s", post_url)
        return False
    if attachments is not None:
        params["attachments"] = ",".join(current) if current else ""
    elif current:
        params["attachments"] = ",".join(current)
    if not edit_token:
        raise VKAPIError(None, "VK_USER_TOKEN missing", method="wall.edit")
    await _vk_api(
        "wall.edit",
        params,
        db,
        bot,
        token=edit_token,
        token_kind=edit_token_kind,
    )
    logging.info("edit_vk_post done: %s", post_url)
    return True


async def delete_vk_post(
    post_url: str,
    db: Database | None = None,
    bot: Bot | None = None,
    token: str | None = None,
) -> None:
    """Delete a VK post given its URL."""
    logging.info("delete_vk_post start: %s", post_url)
    ids = _vk_owner_and_post_id(post_url)
    if not ids:
        logging.error("invalid VK post url %s", post_url)
        return
    owner_id, post_id = ids
    params = {"owner_id": owner_id, "post_id": post_id}
    try:
        await _vk_api("wall.delete", params, db, bot, token=token)
    except Exception as e:
        logging.error("failed to delete VK post %s: %s", post_url, e)
        return
    logging.info("delete_vk_post done: %s", post_url)


async def send_daily_announcement_vk(
    db: Database,
    group_id: str,
    tz: timezone,
    *,
    section: str,
    now: datetime | None = None,
    bot: Bot | None = None,
):
    # сборка постов/текста — вне семафоров
    async with span("db"):
        section1, section2 = await build_daily_sections_vk(db, tz, now)
    if section == "today":
        async with span("vk-send"):
            await post_to_vk(group_id, section1, db, bot)
    elif section == "added":
        async with span("vk-send"):
            await post_to_vk(group_id, section2, db, bot)
    else:
        async with span("vk-send"):
            await post_to_vk(group_id, section1, db, bot)
        async with span("vk-send"):
            await post_to_vk(group_id, section2, db, bot)


async def send_daily_announcement(
    db: Database,
    bot: Bot,
    channel_id: int,
    tz: timezone,
    *,
    record: bool = True,
    now: datetime | None = None,
):
    # 1) Собираем контент вне любых семафоров
    posts = await build_daily_posts(db, tz, now)
    if not posts:
        logging.info("daily: no posts for channel=%s; skip last_daily", channel_id)
        return
    # 2) Отправляем с «узким» шлюзом TG, чтобы не блокировать систему целиком
    sent = 0
    for text, markup in posts:
        try:
            async with TG_SEND_SEMAPHORE:
                async with span("tg-send"):
                    await bot.send_message(
                        channel_id,
                        text,
                        reply_markup=markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
            sent += 1
        except Exception as e:
            logging.error("daily send failed for %s: %s", channel_id, e)
            if "message is too long" in str(e):
                continue
            raise
    # 3) Отмечаем только если что-то реально ушло
    if record and now is None and sent > 0:
        async with db.raw_conn() as conn:
            await conn.execute(
                "UPDATE channel SET last_daily=? WHERE channel_id=?",
                ((now or datetime.now(tz)).date().isoformat(), channel_id),
            )
            await conn.commit()


async def daily_scheduler(db: Database, bot: Bot):
    import asyncio, logging, datetime as dt

    log = logging.getLogger(__name__)
    while True:
        log.info("daily_scheduler: start")
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = dt.datetime.now(tz)
        now_time = now.time().replace(second=0, microsecond=0)
        async with db.raw_conn() as conn:
            conn.row_factory = __import__("sqlite3").Row
            rows = await conn.execute_fetchall(
                """
                SELECT channel_id, daily_time, last_daily
                FROM channel
                WHERE daily_time IS NOT NULL
                """
            )
        for r in rows:
            if not r["daily_time"]:
                continue
            try:
                target_time = dt.datetime.strptime(r["daily_time"], "%H:%M").time()
            except ValueError:
                continue
            due = (r["last_daily"] or "") != now.date().isoformat() and now_time >= target_time
            logging.info(
                "daily_scheduler: channel=%s due=%s last_daily=%s now=%s target=%s",
                r["channel_id"], due, r["last_daily"], now_time, target_time
            )
            if due:
                try:
                    # не блокируем цикл планировщика — отправляем в фоне
                    asyncio.create_task(send_daily_announcement(db, bot, r["channel_id"], tz))
                except Exception as e:
                    log.exception(
                        "daily_scheduler: channel %s failed: %s",
                        r["channel_id"],
                        e,
                    )
        log.info("daily_scheduler: done")
        await asyncio.sleep(seconds_to_next_minute(dt.datetime.now(tz)))


async def vk_scheduler(db: Database, bot: Bot, run_id: str | None = None):
    if not (VK_TOKEN or os.getenv("VK_USER_TOKEN")):
        return
    async with span("db"):
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        group_id = await get_vk_group_id(db)
        if not group_id:
            return
        now_time = now.time().replace(second=0, microsecond=0)
        today_time = datetime.strptime(await get_vk_time_today(db), "%H:%M").time()
        added_time = datetime.strptime(await get_vk_time_added(db), "%H:%M").time()
        last_today = await get_vk_last_today(db)
        last_added = await get_vk_last_added(db)

    if (last_today or "") != now.date().isoformat() and now_time >= today_time:
        try:
            await send_daily_announcement_vk(db, group_id, tz, section="today", bot=bot)
            async with span("db"):
                await set_vk_last_today(db, now.date().isoformat())
        except Exception as e:
            logging.error("vk daily today failed: %s", e)

    if (last_added or "") != now.date().isoformat() and now_time >= added_time:
        try:
            await send_daily_announcement_vk(db, group_id, tz, section="added", bot=bot)
            async with span("db"):
                await set_vk_last_added(db, now.date().isoformat())
        except Exception as e:
            logging.error("vk daily added failed: %s", e)


async def vk_crawl_cron(db: Database, bot: Bot, run_id: str | None = None) -> None:
    """Scheduled VK crawl according to ``VK_CRAWL_TIMES_LOCAL``."""
    now = datetime.now(LOCAL_TZ).strftime("%H:%M")
    logging.info("vk.crawl.cron.fire time=%s", now)
    delay = max(0, random.uniform(-VK_CRAWL_JITTER_SEC, VK_CRAWL_JITTER_SEC))
    if delay:
        await asyncio.sleep(delay)
    try:
        await vk_intake.crawl_once(db, broadcast=True, bot=bot)
    except Exception:
        logging.exception("vk.crawl.cron.error")


async def cleanup_old_events(db: Database, now_utc: datetime | None = None) -> int:
    """Delete events that finished more than a week ago."""
    cutoff = (now_utc or datetime.now(timezone.utc)) - timedelta(days=7)
    cutoff_str = cutoff.date().isoformat()
    async with db.get_session() as session:
        async with session.begin():
            start_t = _time.perf_counter()
            res1 = await session.execute(
                delete(Event).where(
                    Event.end_date.is_not(None), Event.end_date < cutoff_str
                )
            )
            res2 = await session.execute(
                delete(Event).where(
                    Event.end_date.is_(None), Event.date < cutoff_str
                )
            )
        dur = (_time.perf_counter() - start_t) * 1000
        logging.debug("db cleanup_old_events took %.1f ms", dur)
    deleted = (res1.rowcount or 0) + (res2.rowcount or 0)
    return deleted


async def cleanup_scheduler(
    db: Database, bot: Bot, run_id: str | None = None
) -> None:
    retries = [0.8, 2.0]
    attempt = 0
    while True:
        try:
            start = _time.perf_counter()
            async with db.ensure_connection():
                deleted = await cleanup_old_events(db)
            db_took_ms = (_time.perf_counter() - start) * 1000
            logging.info(
                "cleanup_ok run_id=%s deleted_count=%s scanned=%s db_took_ms=%.0f commit_ms=0",
                run_id,
                deleted,
                0,
                db_took_ms,
            )
            try:
                await notify_superadmin(
                    db, bot, f"Cleanup finished, deleted={deleted}"
                )
            except Exception as e:
                logging.warning("cleanup notify failed: %s", e)
            break
        except (sqlite3.ProgrammingError, sqlite3.OperationalError) as e:
            msg = str(e)
            if "Connection closed" in msg or "database is locked" in msg:
                if attempt < len(retries):
                    delay = retries[attempt]
                    attempt += 1
                    logging.warning(
                        "cleanup_retry run_id=%s delay=%.1fs error=%s",
                        run_id,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
            logging.error("Cleanup failed: %s", e)
            break
        except Exception as e:
            logging.error("Cleanup failed: %s", e)
            break

async def rebuild_pages(
    db: Database,
    months: list[str],
    weekends: list[str],
    *,
    force: bool = False,
) -> dict[str, dict[str, dict[str, list[str] | str]]]:
    logging.info(
        "pages_rebuild start months=%s weekends=%s force=%s",
        months,
        weekends,
        force,
    )
    months_updated: dict[str, list[str]] = {}
    weekends_updated: dict[str, list[str]] = {}
    months_failed: dict[str, str] = {}
    weekends_failed: dict[str, str] = {}
    for month in months:
        logging.info("rebuild month start %s", month)
        async with db.get_session() as session:
            prev = await session.get(MonthPage, month)
            prev_hash = prev.content_hash if prev else None
            prev_hash2 = prev.content_hash2 if prev else None
        try:
            if force:
                await sync_month_page(db, month, update_links=False, force=True)
            else:
                await sync_month_page(db, month, update_links=False)
        except Exception as e:  # pragma: no cover
            logging.error("update month %s failed %s", month, e)
            months_failed[month] = str(e)
            continue
        async with db.get_session() as session:
            page = await session.get(MonthPage, month)
        if page and (force or prev is None or page.content_hash != prev_hash or page.content_hash2 != prev_hash2):
            urls = [u for u in [page.url, page.url2] if u]
            months_updated[month] = urls
            for idx, u in enumerate(urls, start=1):
                logging.info("update month %s part%d done %s", month, idx, u)
        else:
            logging.info("update month %s no changes", month)
        logging.info(
            "rebuild month finish %s updated=%s failed=%s",
            month,
            month in months_updated,
            month in months_failed,
        )
    for start in weekends:
        logging.info("rebuild weekend start %s", start)
        async with db.get_session() as session:
            prev = await session.get(WeekendPage, start)
            prev_hash = prev.content_hash if prev else None
        try:
            if force:
                await sync_weekend_page(
                    db, start, update_links=False, post_vk=False, force=True
                )
            else:
                await sync_weekend_page(db, start, update_links=False, post_vk=False)
        except Exception as e:  # pragma: no cover
            logging.error("update weekend %s failed %s", start, e)
            weekends_failed[start] = str(e)
            continue
        async with db.get_session() as session:
            page = await session.get(WeekendPage, start)
        if page and (force or prev is None or page.content_hash != prev_hash):
            urls = [page.url] if page.url else []
            weekends_updated[start] = urls
            for u in urls:
                logging.info("update weekend %s done %s", start, u)
        else:
            logging.info("update weekend %s no changes", start)
        logging.info(
            "rebuild weekend finish %s updated=%s failed=%s",
            start,
            start in weekends_updated,
            start in weekends_failed,
        )
    logging.info("rebuild finished")
    return {
        "months": {"updated": months_updated, "failed": months_failed},
        "weekends": {"updated": weekends_updated, "failed": weekends_failed},
    }


async def nightly_page_sync(db: Database, run_id: str | None = None) -> None:
    """Rebuild all stored month and weekend pages once per night."""
    async with span("db"):
        async with db.get_session() as session:
            res = await session.execute(select(Event.date))
            dates = [d for (d,) in res.all()]
    months: set[str] = set()
    weekends: set[str] = set()
    for dt_str in dates:
        start = dt_str.split("..", 1)[0]
        d = parse_iso_date(start)
        if not d:
            continue
        months.add(d.strftime("%Y-%m"))
        w = weekend_start_for_date(d)
        if w:
            weekends.add(w.isoformat())
    months_list = sorted(months)
    weekends_list = sorted(weekends)
    logging.info(
        "nightly_page_sync start months=%s weekends=%s",
        months_list,
        weekends_list,
    )
    await rebuild_pages(db, months_list, weekends_list, force=True)
    logging.info("nightly_page_sync finish")


async def partner_notification_scheduler(db: Database, bot: Bot, run_id: str | None = None):
    """Remind partners who haven't added events for a week."""
    async with span("db"):
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        last_run = await get_partner_last_run(db)
    if now.time() >= time(9, 0) and (last_run is None or last_run != now.date()):
        try:
            async with span("db-query"):
                async with db.get_session() as session:
                    stream = await session.stream(
                        text(
                            "SELECT id, title FROM event "
                            "WHERE festival IS NOT NULL "
                            "AND date BETWEEN :start AND :end "
                            "ORDER BY date"
                        ),
                        {
                            "start": now.date().isoformat(),
                            "end": (now.date() + timedelta(days=30)).isoformat(),
                        },
                    )
                    async for _ in stream:
                        pass
            await asyncio.sleep(0)
            notified = await notify_inactive_partners(db, bot, tz)
            if notified:
                names = ", ".join(
                    f"@{u.username}" if u.username else str(u.user_id)
                    for u in notified
                )
                async with span("tg-send"):
                    await notify_superadmin(
                        db, bot, f"Partner reminders sent to: {names}"
                    )
            else:
                logging.info("Partner reminders: none")
            await set_partner_last_run(db, now.date())
        except Exception as e:
            logging.error("partner reminder failed: %s", e)
            await notify_superadmin(db, bot, f"Partner reminder failed: {e}")


async def vk_poll_scheduler(db: Database, bot: Bot, run_id: str | None = None):
    if not (VK_TOKEN or os.getenv("VK_USER_TOKEN")):
        return
    async with span("db"):
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        group_id = await get_vk_group_id(db)
        if not group_id:
            return
        ev_map: dict[str, list[Event]] = {}
        async with db.get_session() as session:
            stream = await session.stream_scalars(
                select(Event).execution_options(yield_per=500)
            )
            async for e in stream:
                if e.festival:
                    ev_map.setdefault(e.festival, []).append(e)

            LIMIT = 1000
            off = 0
            while True:
                result = await session.execute(
                    select(
                        Festival.id,
                        Festival.name,
                        Festival.start_date,
                        Festival.end_date,
                        Festival.vk_poll_url,
                        Festival.vk_post_url,
                    )
                    .order_by(Festival.id)
                    .limit(LIMIT)
                    .offset(off)
                )
                rows = result.all()
                if not rows:
                    break
                for (
                    fest_id,
                    name,
                    start_date,
                    end_date,
                    vk_poll_url,
                    vk_post_url,
                ) in rows:
                    if vk_poll_url:
                        continue
                    fest = Festival(
                        id=fest_id,
                        name=name,
                        start_date=start_date,
                        end_date=end_date,
                        vk_poll_url=vk_poll_url,
                        vk_post_url=vk_post_url,
                    )
                    evs = ev_map.get(name, [])
                    start_dt, end_dt = festival_dates(fest, evs)
                    if not start_dt:
                        continue
                    first_time: time | None = None
                    for ev in evs:
                        if ev.date != start_dt.isoformat():
                            continue
                        tr = parse_time_range(ev.time)
                        if tr and (first_time is None or tr[0] < first_time):
                            first_time = tr[0]
                    if first_time is None:
                        first_time = time(0, 0)
                    if first_time >= time(17, 0):
                        sched = datetime.combine(start_dt, time(13, 0), tz)
                    else:
                        sched = datetime.combine(start_dt - timedelta(days=1), time(21, 0), tz)
                    if now >= sched and now.date() <= (end_dt or start_dt):
                        try:
                            await send_festival_poll(db, fest, group_id, bot)
                        except Exception as e:
                            logging.error("VK poll send failed for %s: %s", name, e)
                del rows
                await asyncio.sleep(0)
                off += LIMIT


async def init_db_and_scheduler(
    app: web.Application, db: Database, bot: Bot, webhook: str | None
) -> None:
    logging.info("Initializing database")
    await db.init()
    await get_tz_offset(db)
    global CATBOX_ENABLED
    CATBOX_ENABLED = await get_catbox_enabled(db)
    logging.info("CATBOX_ENABLED resolved to %s", CATBOX_ENABLED)
    global VK_PHOTOS_ENABLED
    VK_PHOTOS_ENABLED = await get_vk_photos_enabled(db)
    
    # Only set webhook if webhook URL is provided (production mode)
    if webhook:
        hook = webhook.rstrip("/") + "/webhook"
        logging.info("Setting webhook to %s", hook)
        try:
            await bot.set_webhook(
                hook,
                allowed_updates=["message", "callback_query", "my_chat_member"],
            )
        except Exception as e:
            logging.error("Failed to set webhook: %s", e)
    else:
        logging.info("No webhook URL provided, skipping webhook setup (dev mode)")
    
    try:
        scheduler_startup(db, bot)
    except Exception:
        logging.exception("scheduler_startup failed; continuing without scheduler")
    app["daily_scheduler"] = asyncio.create_task(daily_scheduler(db, bot))
    app["add_event_worker"] = asyncio.create_task(add_event_queue_worker(db, bot))
    app["add_event_watch"] = asyncio.create_task(_watch_add_event_worker(app, db, bot))
    app["job_outbox_worker"] = asyncio.create_task(job_outbox_worker(db, bot))
    gc.collect()
    logging.info("BOOT_OK pid=%s", os.getpid())


def _topic_labels_for_display(topics: Sequence[str] | None) -> list[str]:
    labels: list[str] = []
    if not topics:
        return labels

    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, str):
            continue
        raw = topic.strip()
        if not raw:
            continue
        canonical = normalize_topic_identifier(raw)
        if canonical:
            if canonical in seen:
                continue
            seen.add(canonical)
            labels.append(TOPIC_LABELS.get(canonical, canonical))
        else:
            dedup_key = raw.casefold()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            labels.append(raw)
    return labels


def _format_topics_line(topics: Sequence[str] | None, manual: bool) -> str:
    labels = _topic_labels_for_display(topics)
    content = ", ".join(labels) if labels else "—"
    suffix = " (ручной режим)" if manual else ""
    return f"Темы: {content}{suffix}"


def _format_topic_badges(topics: Sequence[str] | None) -> str | None:
    labels = _topic_labels_for_display(topics)
    if not labels:
        return None
    return " ".join(f"[{label}]" for label in labels)


async def build_events_message(db: Database, target_date: date, tz: timezone, creator_id: int | None = None):
    async with db.get_session() as session:
        stmt = select(Event).where(
            (Event.date == target_date.isoformat())
            | (Event.end_date == target_date.isoformat())
        )
        if creator_id is not None:
            stmt = stmt.where(Event.creator_id == creator_id)
        result = await session.execute(stmt.order_by(Event.time))
        events = result.scalars().all()

    lines = []
    for e in events:
        prefix = ""
        if e.end_date and e.date == target_date.isoformat():
            prefix = "(Открытие) "
        elif (
            e.end_date
            and e.end_date == target_date.isoformat()
            and e.end_date != e.date
        ):
            prefix = "(Закрытие) "
        title = f"{e.emoji} {e.title}" if e.emoji else e.title
        lines.append(f"{e.id}. {prefix}{title}")
        badges = _format_topic_badges(getattr(e, "topics", None))
        if badges:
            lines.append(badges)
        loc = f"{e.time} {e.location_name}"
        if e.city:
            loc += f", #{e.city}"
        lines.append(loc)
        if e.is_free:
            lines.append("Бесплатно")
        else:
            price_parts = []
            if e.ticket_price_min is not None:
                price_parts.append(str(e.ticket_price_min))
            if (
                e.ticket_price_max is not None
                and e.ticket_price_max != e.ticket_price_min
            ):
                price_parts.append(str(e.ticket_price_max))
            if price_parts:
                lines.append("-".join(price_parts))
        if e.telegraph_url:
            lines.append(f"исходное: {e.telegraph_url}")
        if e.vk_ticket_short_key:
            lines.append(
                f"Статистика VK: https://vk.com/cc?act=stats&key={e.vk_ticket_short_key}"
            )
        lines.append("")
    if not lines:
        lines.append("No events")

    keyboard = []
    for e in events:
        icon = "✂️" if not e.vk_repost_url else "✅"
        row = [
            types.InlineKeyboardButton(
                text=f"\u274c {e.id}",
                callback_data=f"del:{e.id}:{target_date.isoformat()}",
            ),
            types.InlineKeyboardButton(
                text=f"\u270e {e.id}", callback_data=f"edit:{e.id}"
            ),
            types.InlineKeyboardButton(
                text=f"{icon} Рерайт {e.id}",
                callback_data=f"vkrev:shortpost:{e.id}",
            ),
            types.InlineKeyboardButton(
                text=f"\ud83c\udfac {e.video_include_count}",
                callback_data=f"vidcnt:{e.id}",
            ),
        ]
        keyboard.append(row)

    today = datetime.now(tz).date()
    prev_day = target_date - timedelta(days=1)
    next_day = target_date + timedelta(days=1)
    row = []
    if target_date > today:
        row.append(
            types.InlineKeyboardButton(
                text="\u25c0", callback_data=f"nav:{prev_day.isoformat()}"
            )
        )
    row.append(
        types.InlineKeyboardButton(
            text="\u25b6", callback_data=f"nav:{next_day.isoformat()}"
        )
    )
    keyboard.append(row)

    text = f"Events on {format_day(target_date, tz)}\n" + "\n".join(lines)
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    return text, markup


async def build_events_message_compact(db: Database, target_date: date, tz: timezone, creator_id: int | None = None):
    """Compact version of events message - only ID and title with link.
    
    Used as fallback when full message exceeds Telegram's 4096 char limit.
    """
    async with db.get_session() as session:
        stmt = select(Event).where(
            (Event.date == target_date.isoformat())
            | (Event.end_date == target_date.isoformat())
        )
        if creator_id is not None:
            stmt = stmt.where(Event.creator_id == creator_id)
        result = await session.execute(stmt.order_by(Event.time))
        events = result.scalars().all()

    lines = [f"📋 Events on {format_day(target_date, tz)} (compact mode)\n"]
    
    for e in events:
        title = f"{e.emoji} {e.title}" if e.emoji else e.title
        if e.telegraph_url:
            # Clickable link to telegraph page
            lines.append(f'{e.id}. <a href="{e.telegraph_url}">{title}</a>')
        else:
            lines.append(f"{e.id}. {title}")
    
    if len(events) == 0:
        lines.append("No events")
    else:
        lines.append(f"\n<i>Total: {len(events)} events</i>")

    # Simplified navigation keyboard (no per-event buttons)
    today = datetime.now(tz).date()
    prev_day = target_date - timedelta(days=1)
    next_day = target_date + timedelta(days=1)
    nav_row = []
    if target_date > today:
        nav_row.append(
            types.InlineKeyboardButton(
                text="◀", callback_data=f"nav:{prev_day.isoformat()}"
            )
        )
    nav_row.append(
        types.InlineKeyboardButton(
            text="▶", callback_data=f"nav:{next_day.isoformat()}"
        )
    )
    keyboard = [nav_row]

    text = "\n".join(lines)
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    return text, markup

async def build_exhibitions_message(
    db: Database, tz: timezone
) -> tuple[list[str], types.InlineKeyboardMarkup | None]:
    today = datetime.now(tz).date()
    today_iso = today.isoformat()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                Event.event_type == "выставка",
                or_(
                    Event.end_date.is_not(None),
                    and_(Event.end_date.is_(None), Event.date >= today_iso),
                ),
                or_(Event.end_date >= today_iso, Event.end_date.is_(None)),
            )
            .order_by(Event.date)
        )
        events = result.scalars().all()

    lines = []
    for e in events:
        start = parse_iso_date(e.date)
        if not start:
            if ".." in e.date:
                start = parse_iso_date(e.date.split("..", 1)[0])
        if not start:
            logging.error("Bad start date %s for event %s", e.date, e.id)
            continue
        end = None
        if e.end_date:
            end = parse_iso_date(e.end_date)

        period = ""
        if end:
            period = f"c {format_day_pretty(start)} по {format_day_pretty(end)}"
        title = f"{e.emoji} {e.title}" if e.emoji else e.title
        if period:
            lines.append(f"{e.id}. {title} ({period})")
        else:
            lines.append(f"{e.id}. {title}")
        badges = _format_topic_badges(getattr(e, "topics", None))
        if badges:
            lines.append(badges)
        loc = f"{e.time} {e.location_name}"
        if e.city:
            loc += f", #{e.city}"
        lines.append(loc)
        if e.is_free:
            lines.append("Бесплатно")
        else:
            price_parts = []
            if e.ticket_price_min is not None:
                price_parts.append(str(e.ticket_price_min))
            if (
                e.ticket_price_max is not None
                and e.ticket_price_max != e.ticket_price_min
            ):
                price_parts.append(str(e.ticket_price_max))
            if price_parts:
                lines.append("-".join(price_parts))
        if e.telegraph_url:
            lines.append(f"исходное: {e.telegraph_url}")
        if e.vk_ticket_short_key:
            lines.append(
                f"Статистика VK: https://vk.com/cc?act=stats&key={e.vk_ticket_short_key}"
            )
        lines.append("")

    if not lines:
        lines.append("No exhibitions")

    while lines and lines[-1] == "":
        lines.pop()

    keyboard = []
    for e in events:
        row = [
            types.InlineKeyboardButton(
                text=f"\u274c {e.id}", callback_data=f"del:{e.id}:exh"
            ),
            types.InlineKeyboardButton(
                text=f"\u270e {e.id}", callback_data=f"edit:{e.id}"
            ),
        ]
        keyboard.append(row)
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if events else None
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_lines, current_len
        if current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0

    def add_line(line: str) -> None:
        nonlocal current_len
        while True:
            newline_cost = 1 if current_lines else 0
            projected = current_len + newline_cost + len(line)
            if projected <= TELEGRAM_MESSAGE_LIMIT:
                if newline_cost:
                    current_len += 1
                current_lines.append(line)
                current_len += len(line)
                return
            if current_lines:
                flush_current()
                continue
            truncated = _truncate_with_indicator(line, TELEGRAM_MESSAGE_LIMIT)
            if truncated:
                chunks.append(truncated)
            return

    for line in ["Exhibitions", *lines]:
        add_line(line)

    flush_current()

    if not chunks:
        chunks.append("")

    return chunks, markup


TELEGRAM_MESSAGE_LIMIT = 4096
POSTER_TRUNCATION_INDICATOR = "… (обрезано)"
POSTER_PREVIEW_UNAVAILABLE = "Poster OCR: превью недоступно — сообщение слишком длинное."


def _truncate_with_indicator(
    text: str, limit: int, indicator: str = POSTER_TRUNCATION_INDICATOR
) -> str:
    if limit <= 0:
        return ""
    if limit <= len(indicator):
        return indicator[:limit]
    return text[: limit - len(indicator)] + indicator


def _fit_poster_preview_lines(
    lines: Sequence[str], budget: int, indicator: str = POSTER_TRUNCATION_INDICATOR
) -> list[str]:
    if budget <= 0:
        return []

    fitted: list[str] = []
    for line in lines:
        candidate = fitted + [line]
        if len("\n".join(candidate)) <= budget:
            fitted.append(line)
            continue

        used_len = len("\n".join(fitted))
        newline_cost = 1 if fitted else 0
        remaining_for_content = budget - used_len - newline_cost
        if remaining_for_content <= 0:
            if not fitted:
                truncated = _truncate_with_indicator("", budget, indicator)
                return [truncated] if truncated else []

            prefix = fitted[:-1]
            last_line = fitted[-1]
            prefix_len = len("\n".join(prefix))
            if prefix:
                prefix_len += 1
            allowed_for_last = max(0, budget - prefix_len)
            fitted[-1] = _truncate_with_indicator(last_line, allowed_for_last, indicator)
            return fitted

        truncated_line = _truncate_with_indicator(line, remaining_for_content, indicator)
        if truncated_line:
            fitted.append(truncated_line)
        return fitted

    return fitted


async def show_edit_menu(
    user_id: int,
    event: Event,
    bot: Bot,
    db_obj: Database | None = None,
):
    data: dict[str, Any]
    try:
        data = event.model_dump()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        data = event.dict()

    database = db_obj or globals().get("db")
    poster_lines: list[str] = []
    if database and event.id:
        from sqlmodel import select
        from models import EventPoster

        async with database.get_session() as session:
            posters = (
                await session.execute(
                    select(EventPoster)
                    .where(EventPoster.event_id == event.id)
                    .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
                )
            ).scalars().all()
        if posters:
            poster_lines.append("Poster OCR:")
            for idx, poster in enumerate(posters[:3], 1):
                token_parts: list[str] = []
                if poster.prompt_tokens:
                    token_parts.append(f"prompt={poster.prompt_tokens}")
                if poster.completion_tokens:
                    token_parts.append(f"completion={poster.completion_tokens}")
                if poster.total_tokens:
                    token_parts.append(f"total={poster.total_tokens}")
                token_info = f" ({', '.join(token_parts)})" if token_parts else ""
                hash_display = poster.poster_hash[:10]
                poster_lines.append(f"{idx}. hash={hash_display}{token_info}")
                poster_lines.append(f"    ocr_title: {poster.ocr_title or ''}")
                poster_lines.append("    ocr_text:")
                raw_lines = (poster.ocr_text or "").splitlines()
                cleaned_lines = [line.strip() for line in raw_lines if line.strip()]
                if not cleaned_lines:
                    cleaned_lines = ["<пусто>"]
                for text_line in cleaned_lines:
                    poster_lines.append(f"        {text_line}")
                if poster.catbox_url:
                    url = poster.catbox_url
                    if len(url) > 120:
                        url = url[:117] + "..."
                    poster_lines.append(f"    catbox_url: {url}")
            poster_lines.append("---")

    lines = []
    topics_manual_flag = bool(data.get("topics_manual"))
    for key, value in data.items():
        if key == "topics":
            topics_value: Sequence[str] | None = None
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                topics_value = [str(item) for item in value]
            lines.append(_format_topics_line(topics_value, topics_manual_flag))
            continue
        if value is None:
            val = ""
        elif isinstance(value, str):
            val = value if len(value) <= 1000 else value[:1000] + "..."
        else:
            val = str(value)
        lines.append(f"{key}: {val}")

    fields = [k for k in data.keys() if k not in {"id", "added_at", "silent"}]
    keyboard = []
    row = []
    for idx, field in enumerate(fields, 1):
        row.append(
            types.InlineKeyboardButton(
                text=field, callback_data=f"editfield:{event.id}:{field}"
            )
        )
        if idx % 3 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append(
        [
            types.InlineKeyboardButton(
                text=(
                    "\U0001f6a9 Переключить на тихий режим"
                    if not event.silent
                    else "\U0001f910 Тихий режим"
                ),
                callback_data=f"togglesilent:{event.id}",
            )
        ]
    )
    keyboard.append(
        [
            types.InlineKeyboardButton(
                text=("\u2705 Бесплатно" if event.is_free else "\u274c Бесплатно"),
                callback_data=f"togglefree:{event.id}",
            )
        ]
    )
    if event.ics_url:
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="Delete ICS",
                    callback_data=f"delics:{event.id}",
                )
            ]
        )
    else:
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="Create ICS",
                    callback_data=f"createics:{event.id}",
                )
            ]
        )
    if event.id and not event.festival:
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="\U0001f3aa Сделать фестиваль",
                    callback_data=f"makefest:{event.id}",
                )
            ]
        )
    keyboard.append(
        [types.InlineKeyboardButton(text="Done", callback_data=f"editdone:{event.id}")]
    )
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)

    base_text = "\n".join(lines)
    base_len = len(base_text)
    poster_block: list[str] = []
    if poster_lines:
        newline_between_blocks = 1 if lines else 0
        poster_budget = TELEGRAM_MESSAGE_LIMIT - base_len - newline_between_blocks
        if poster_budget <= 0:
            notice_budget = TELEGRAM_MESSAGE_LIMIT - base_len - newline_between_blocks
            poster_block = _fit_poster_preview_lines(
                [POSTER_PREVIEW_UNAVAILABLE], notice_budget
            )
        else:
            poster_block = _fit_poster_preview_lines(poster_lines, poster_budget)
            if not poster_block and poster_budget > 0:
                poster_block = _fit_poster_preview_lines(
                    [POSTER_PREVIEW_UNAVAILABLE], poster_budget
                )

    message_lines = poster_block + lines if poster_block else lines
    message_text = "\n".join(message_lines)
    if len(message_text) > TELEGRAM_MESSAGE_LIMIT:
        message_text = _truncate_with_indicator(
            message_text, TELEGRAM_MESSAGE_LIMIT, POSTER_TRUNCATION_INDICATOR
        )

    await bot.send_message(user_id, message_text, reply_markup=markup)


async def show_festival_edit_menu(user_id: int, fest: Festival, bot: Bot):
    """Send festival fields with edit options."""
    lines = [
        f"name: {fest.name}",
        f"full: {fest.full_name or ''}",
        f"description: {fest.description or ''}",
        f"start: {fest.start_date or ''}",
        f"end: {fest.end_date or ''}",
        f"site: {fest.website_url or ''}",
        f"program: {fest.program_url or ''}",
        f"vk: {fest.vk_url or ''}",
        f"tg: {fest.tg_url or ''}",
        f"ticket: {fest.ticket_url or ''}",
        # New parser fields
        f"source_url: {fest.source_url or ''}",
        f"source_type: {fest.source_type or ''}",
        f"phone: {fest.contacts_phone or ''}",
        f"email: {fest.contacts_email or ''}",
        f"audience: {fest.audience or ''}",
        f"annual: {'✓' if fest.is_annual else '✗' if fest.is_annual is False else ''}",
    ]
    if fest.last_parsed_at:
        lines.append(f"last_parsed: {fest.last_parsed_at.strftime('%Y-%m-%d %H:%M')}")
    
    keyboard = [
        [
            types.InlineKeyboardButton(
                text="Edit short name",
                callback_data=f"festeditfield:{fest.id}:name",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Edit full name",
                callback_data=f"festeditfield:{fest.id}:full",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Edit description",
                callback_data=f"festeditfield:{fest.id}:description",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete start" if fest.start_date else "Add start"),
                callback_data=f"festeditfield:{fest.id}:start",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete end" if fest.end_date else "Add end"),
                callback_data=f"festeditfield:{fest.id}:end",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete site" if fest.website_url else "Add site"),
                callback_data=f"festeditfield:{fest.id}:site",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete program" if fest.program_url else "Add program"),
                callback_data=f"festeditfield:{fest.id}:program",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete VK" if fest.vk_url else "Add VK"),
                callback_data=f"festeditfield:{fest.id}:vk",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete TG" if fest.tg_url else "Add TG"),
                callback_data=f"festeditfield:{fest.id}:tg",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete ticket" if fest.ticket_url else "Add ticket"),
                callback_data=f"festeditfield:{fest.id}:ticket",
            )
        ],
        # New parser fields buttons
        [
            types.InlineKeyboardButton(
                text=("Delete phone" if fest.contacts_phone else "Add phone"),
                callback_data=f"festeditfield:{fest.id}:phone",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete email" if fest.contacts_email else "Add email"),
                callback_data=f"festeditfield:{fest.id}:email",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete audience" if fest.audience else "Add audience"),
                callback_data=f"festeditfield:{fest.id}:audience",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Обновить обложку из Telegraph",
                callback_data=f"festcover:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Добавить иллюстрацию",
                callback_data=f"festimgadd:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Иллюстрации / обложка",
                callback_data=f"festimgs:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="🧩 Склеить с…",
                callback_data=f"festmerge:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="🔄 Обновить события",
                callback_data=f"festsyncevents:{fest.id}",
            )
        ],
    ]
    # Add re-parse button only if source_url exists
    if fest.source_url:
        keyboard.append([
            types.InlineKeyboardButton(
                text="🔄 Перепарсить с сайта",
                callback_data=f"festreparse:{fest.id}",
            )
        ])
    keyboard.append([types.InlineKeyboardButton(text="Done", callback_data="festeditdone")])
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    await bot.send_message(user_id, "\n".join(lines), reply_markup=markup)


def add_festival_photo(fest: Festival, url: str, *, make_cover: bool = False) -> bool:
    """Append a photo URL to the festival album and optionally make it the cover."""

    url = (url or "").strip()
    if not url:
        return False
    existing_urls = list(fest.photo_urls or [])
    changed = False
    if url not in existing_urls:
        existing_urls.append(url)
        fest.photo_urls = existing_urls
        changed = True
    else:
        fest.photo_urls = existing_urls
    if make_cover or not fest.photo_url:
        if fest.photo_url != url:
            fest.photo_url = url
            changed = True
    return changed


FEST_MERGE_PAGE_SIZE = 12


_FEST_MERGE_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _festival_merge_tokens(fest: Festival) -> set[str]:
    tokens: set[str] = set()
    if fest.name:
        tokens.update(token.lower() for token in _FEST_MERGE_TOKEN_RE.findall(fest.name))
    if fest.city:
        tokens.add(fest.city.lower())
    return tokens


def _sort_festival_merge_targets(
    source: Festival, targets: Sequence[Festival]
) -> list[Festival]:
    source_tokens = _festival_merge_tokens(source)
    scored: list[tuple[int, str, int, Festival]] = []
    for index, target in enumerate(targets):
        target_tokens = _festival_merge_tokens(target)
        overlap = len(source_tokens & target_tokens)
        name_key = (target.name or "").lower()
        scored.append((overlap, name_key, index, target))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def _format_festival_merge_line(fest: Festival) -> str:
    parts = [f"{fest.id} {fest.name}"]
    if fest.start_date and fest.end_date:
        if fest.start_date == fest.end_date:
            parts.append(fest.start_date)
        else:
            parts.append(f"{fest.start_date}..{fest.end_date}")
    elif fest.start_date:
        parts.append(fest.start_date)
    elif fest.end_date:
        parts.append(fest.end_date)
    if fest.city:
        parts.append(f"#{fest.city}")
    return " · ".join(parts)


def build_festival_merge_selection(
    source: Festival, targets: Sequence[Festival], page: int
) -> tuple[str, types.InlineKeyboardMarkup]:
    sorted_targets = _sort_festival_merge_targets(source, targets)
    total = len(sorted_targets)
    total_pages = max(1, (total + FEST_MERGE_PAGE_SIZE - 1) // FEST_MERGE_PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * FEST_MERGE_PAGE_SIZE
    visible = sorted_targets[start : start + FEST_MERGE_PAGE_SIZE]

    heading = (
        f"🧩 Склеить фестиваль «{source.name}» (ID {source.id}).\n"
        f"Выберите фестиваль-цель (страница {page}/{total_pages}):"
    )
    lines = [heading]
    keyboard: list[list[types.InlineKeyboardButton]] = []

    if not visible:
        lines.append("Нет других фестивалей для объединения.")
    else:
        for target in visible:
            lines.append(_format_festival_merge_line(target))
            button_text = target.name
            if target.city:
                button_text = f"{target.name} · #{target.city}"
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text=button_text,
                        callback_data=f"festmerge_to:{source.id}:{target.id}:{page}",
                    )
                ]
            )

    nav_row: list[types.InlineKeyboardButton] = []
    if total_pages > 1 and page > 1:
        nav_row.append(
            types.InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=f"festmergep:{source.id}:{page-1}",
            )
        )
    if total_pages > 1 and page < total_pages:
        nav_row.append(
            types.InlineKeyboardButton(
                text="Вперёд ➡️",
                callback_data=f"festmergep:{source.id}:{page+1}",
            )
        )
    if nav_row:
        keyboard.append(nav_row)

    keyboard.append(
        [types.InlineKeyboardButton(text="Отмена", callback_data=f"festedit:{source.id}")]
    )

    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    return "\n".join(lines), markup


async def handle_festmerge_callback(
    callback: types.CallbackQuery, db: Database, _bot: Bot
) -> None:
    """Show merge selection for a festival."""

    try:
        fid = int((callback.data or "").split(":")[1])
    except (IndexError, ValueError):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        res = await session.execute(
            select(Festival).where(Festival.id != fid).order_by(Festival.name)
        )
        targets = list(res.scalars().all())

    if not targets:
        await callback.answer("Нет других фестивалей", show_alert=True)
        return

    text, markup = build_festival_merge_selection(fest, targets, page=1)
    await callback.message.answer(text, reply_markup=markup)
    await callback.answer()


async def handle_festmerge_page_callback(
    callback: types.CallbackQuery, db: Database, _: Bot
) -> None:
    """Handle pagination inside merge selection."""

    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer("Некорректный запрос", show_alert=True)
        return
    _, src_raw, page_raw = parts
    try:
        src_id = int(src_raw)
        page = int(page_raw)
    except ValueError:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        src = await session.get(Festival, src_id)
        if not src:
            await callback.answer("Festival not found", show_alert=True)
            return
        res = await session.execute(
            select(Festival).where(Festival.id != src_id).order_by(Festival.name)
        )
        targets = list(res.scalars().all())

    text, markup = build_festival_merge_selection(src, targets, page)
    await callback.message.edit_text(text, reply_markup=markup)
    await callback.answer()


async def handle_festmerge_to_callback(
    callback: types.CallbackQuery, db: Database, _: Bot
) -> None:
    """Confirm merge target selection."""

    parts = (callback.data or "").split(":")
    if len(parts) != 4:
        await callback.answer("Некорректный запрос", show_alert=True)
        return
    _, src_raw, dst_raw, page_raw = parts
    try:
        src_id = int(src_raw)
        dst_id = int(dst_raw)
        page = int(page_raw)
    except ValueError:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        src = await session.get(Festival, src_id)
        dst = await session.get(Festival, dst_id)

    if not src or not dst:
        await callback.answer("Festival not found", show_alert=True)
        return

    confirm_lines = [
        "Вы уверены, что хотите склеить фестивали?",
        f"Источник: {src.id} {src.name}",
        f"Цель: {dst.id} {dst.name}",
        "Все события и данные источника будут перенесены, сам фестиваль будет удалён.",
    ]
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="✅ Склеить",
                    callback_data=f"festmerge_do:{src_id}:{dst_id}",
                )
            ],
            [
                types.InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data=f"festmergep:{src_id}:{page}",
                ),
                types.InlineKeyboardButton(
                    text="Отмена",
                    callback_data=f"festedit:{src_id}",
                ),
            ],
        ]
    )
    await callback.message.edit_text("\n".join(confirm_lines), reply_markup=keyboard)
    await callback.answer()


async def handle_festmerge_do_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    """Execute merge and report result."""

    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer("Некорректный запрос", show_alert=True)
        return
    _, src_raw, dst_raw = parts
    try:
        src_id = int(src_raw)
        dst_id = int(dst_raw)
    except ValueError:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        src = await session.get(Festival, src_id)
        dst = await session.get(Festival, dst_id)

    if not src or not dst:
        await callback.answer("Festival not found", show_alert=True)
        return

    src_name = src.name
    dst_name = dst.name
    ok = await merge_festivals(db, src_id, dst_id, bot)
    if not ok:
        await callback.answer("Не удалось объединить", show_alert=True)
        return

    async with db.get_session() as session:
        dest = await session.get(Festival, dst_id)

    if dest:
        festival_edit_sessions[callback.from_user.id] = (dst_id, None)
        await show_festival_edit_menu(callback.from_user.id, dest, bot)

    await callback.message.edit_text(
        f"Фестиваль «{src_name}» объединён с «{dst_name}»."
    )
    await callback.answer("Склеено")


async def handle_events(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)

    if len(parts) == 2:
        day = parse_events_date(parts[1], tz)
        if not day:
            await bot.send_message(message.chat.id, "Usage: /events <date>")
            return
    else:
        day = datetime.now(tz).date()

    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
        creator_filter = user.user_id if user.is_partner else None

    text, markup = await build_events_message(db, day, tz, creator_filter)
    
    # Try sending full message, fallback to compact format if too long
    try:
        await bot.send_message(message.chat.id, text, reply_markup=markup)
    except TelegramBadRequest as e:
        if "message is too long" in str(e).lower():
            # Fallback to compact format
            text, markup = await build_events_message_compact(db, day, tz, creator_filter)
            await bot.send_message(message.chat.id, text, reply_markup=markup, parse_mode="HTML")


async def show_digest_menu(message: types.Message, db: Database, bot: Bot) -> None:
    if not (message.text or "").startswith("/digest"):
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return

    digest_id = uuid.uuid4().hex
    keyboard = [
        [
            types.InlineKeyboardButton(
                text="✅ Лекции",
                callback_data=f"digest:select:lectures:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Мастер-классы",
                callback_data=f"digest:select:masterclasses:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Выставки",
                callback_data=f"digest:select:exhibitions:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Психология",
                callback_data=f"digest:select:psychology:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Научпоп",
                callback_data=f"digest:select:science_pop:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Краеведение",
                callback_data=f"digest:select:kraevedenie:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Нетворкинг",
                callback_data=f"digest:select:networking:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Развлечения",
                callback_data=f"digest:select:entertainment:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Маркеты",
                callback_data=f"digest:select:markets:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Кинопоказы",
                callback_data=f"digest:select:movies:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Классический театр",
                callback_data=f"digest:select:theatre_classic:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Современный театр",
                callback_data=f"digest:select:theatre_modern:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Встречи и клубы",
                callback_data=f"digest:select:meetups:{digest_id}",
            )
        ],
        [
            types.InlineKeyboardButton(text="⏳ Выходные", callback_data="digest:disabled"),
            types.InlineKeyboardButton(text="⏳ Популярное за неделю", callback_data="digest:disabled"),
        ],
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    sent = await bot.send_message(
        message.chat.id, "Выберите тип дайджеста:", reply_markup=markup
    )
    logging.info(
        "digest.menu.shown digest_id=%s chat_id=%s user_id=%s message_id=%s",
        digest_id,
        message.chat.id,
        message.from_user.id,
        getattr(sent, "message_id", None),
    )


DEFAULT_PANEL_TEXT = (
    "Управление дайджестом\nВыключите лишнее и нажмите «Обновить превью»."
)


def _build_digest_panel_markup(digest_id: str, session: dict) -> types.InlineKeyboardMarkup:
    buttons: List[types.InlineKeyboardButton] = []
    excluded: set[int] = session.get("excluded", set())
    for idx, item in enumerate(session["items"]):
        mark = "✅" if idx not in excluded else "❌"
        buttons.append(
            types.InlineKeyboardButton(
                text=f"{mark} {item['index']}",
                callback_data=f"dg:t:{digest_id}:{item['index']}",
            )
        )
    rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    rows.append(
        [types.InlineKeyboardButton(text="🔄 Обновить превью", callback_data=f"dg:r:{digest_id}")]
    )
    for ch in session.get("channels", []):
        rows.append(
            [
                types.InlineKeyboardButton(
                    text=f"🚀 Отправить в «{ch['name']}»",
                    callback_data=f"dg:s:{digest_id}:{ch['channel_id']}",
                )
            ]
        )
    rows.append(
        [types.InlineKeyboardButton(text="🗑 Скрыть панель", callback_data=f"dg:x:{digest_id}")]
    )
    logging.info(
        "digest.controls.render digest_id=%s count=%s excluded=%s",
        digest_id,
        len(session["items"]),
        sorted(excluded),
    )
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


async def _compose_from_session(
    session: dict, digest_id: str
) -> tuple[str, List[str], bool, int, List[int]]:
    excluded: set[int] = session.get("excluded", set())
    indices = [i for i in range(len(session["items"])) if i not in excluded]
    lines_html = [session["items"][i]["line_html"] for i in indices]
    caption, used_lines = await compose_digest_caption(
        session["intro_html"],
        lines_html,
        session["footer_html"],
        digest_id=digest_id,
    )
    used_indices = indices[: len(used_lines)]
    media_urls: List[str] = []
    seen_urls: set[str] = set()
    for i in used_indices:
        url = session["items"][i]["cover_url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        media_urls.append(url)
    media = [types.InputMediaPhoto(media=url) for url in media_urls]
    attach, _ = attach_caption_if_fits(media, caption)
    vis_len = visible_caption_len(caption)
    logging.info(
        "digest.caption.visible_len digest_id=%s visible=%s attach=%s",
        digest_id,
        vis_len,
        int(attach),
    )
    return caption, media_urls, attach, vis_len, used_indices


async def _send_preview(session: dict, digest_id: str, bot: Bot):
    caption, media_urls, attach, vis_len, used_indices = await _compose_from_session(
        session, digest_id
    )
    session["current_caption_html"] = caption
    session["current_media_urls"] = media_urls
    session["current_attach"] = attach
    session["current_visible_len"] = vis_len
    session["current_used_indices"] = used_indices

    msg_ids: List[int] = []
    media: List[types.InputMediaPhoto] = []
    for i, url in enumerate(media_urls):
        if i == 0 and attach:
            media.append(
                types.InputMediaPhoto(
                    media=url, caption=caption, parse_mode="HTML"
                )
            )
        else:
            media.append(types.InputMediaPhoto(media=url))
    if media:
        sent = await bot.send_media_group(session["chat_id"], media)
        msg_ids.extend(m.message_id for m in sent)
    if not attach or not media:
        msg = await bot.send_message(
            session["chat_id"],
            caption,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        msg_ids.append(msg.message_id)
    panel = await bot.send_message(
        session["chat_id"],
        session.get("panel_text", DEFAULT_PANEL_TEXT),
        reply_markup=_build_digest_panel_markup(digest_id, session),
    )
    session["preview_msg_ids"] = msg_ids
    session["panel_msg_id"] = panel.message_id
    return caption, attach, vis_len, len(used_indices)




async def handle_digest_toggle(callback: types.CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    _, _, digest_id, idx_str = parts
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return
    index = int(idx_str) - 1
    excluded: set[int] = session.setdefault("excluded", set())
    if 0 <= index < len(session["items"]):
        if index in excluded:
            excluded.remove(index)
            active = True
        else:
            excluded.add(index)
            active = False
    else:
        active = False
    markup = _build_digest_panel_markup(digest_id, session)
    try:
        await bot.edit_message_reply_markup(
            session["chat_id"], session["panel_msg_id"], reply_markup=markup
        )
    except Exception:
        logging.exception(
            "digest.panel.toggle edit_error digest_id=%s message_id=%s",
            digest_id,
            session.get("panel_msg_id"),
        )
    logging.info(
        "digest.controls.toggle digest_id=%s idx=%s active=%s",
        digest_id,
        index,
        str(active).lower(),
    )
    await callback.answer()


async def handle_digest_refresh(callback: types.CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 3:
        return
    _, _, digest_id = parts
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return
    excluded: set[int] = session.get("excluded", set())
    remaining = [i for i in range(len(session["items"])) if i not in excluded]
    if not remaining:
        noun = session.get("items_noun", "лекций")
        await callback.answer(f"Нет выбранных {noun}", show_alert=True)
        return

    logging.info(
        "digest.preview.recompose.start digest_id=%s kept=%s excluded=%s",
        digest_id,
        len(remaining),
        len(excluded),
    )

    digest_type = session.get("digest_type", "lectures")
    horizon_days = session.get("horizon_days", 0)
    start = _time.monotonic()
    logging.info(
        "digest.intro.llm.request digest_id=%s type=%s items=%s",
        digest_id,
        digest_type,
        len(remaining),
    )
    try:
        if digest_type == "masterclasses":
            payload = []
            for idx in remaining:
                item = session["items"][idx]
                payload.append(
                    {
                        "title": item.get("norm_title") or item.get("title", ""),
                        "description": item.get("norm_description", ""),
                    }
                )
            intro = await compose_masterclasses_intro_via_4o(
                len(payload), horizon_days, payload
            )
        elif digest_type == "exhibitions":
            payload = []
            for idx in remaining:
                item = session["items"][idx]
                start_date = item.get("date", "")
                end_date = item.get("end_date") or start_date
                payload.append(
                    {
                        "title": item.get("norm_title") or item.get("title", ""),
                        "description": item.get("norm_description", ""),
                        "date_range": {"start": start_date, "end": end_date},
                    }
                )
            intro = await compose_exhibitions_intro_via_4o(
                len(payload), horizon_days, payload
            )
        elif digest_type == "psychology":
            payload = []
            for idx in remaining:
                item = session["items"][idx]
                payload.append(
                    {
                        "title": item.get("norm_title") or item.get("title", ""),
                        "description": item.get("norm_description", ""),
                        "topics": item.get("norm_topics", []),
                    }
                )
            intro = await compose_psychology_intro_via_4o(
                len(payload), horizon_days, payload
            )
        else:
            titles = [session["items"][i]["norm_title"] for i in remaining]
            intro = await compose_digest_intro_via_4o(
                len(remaining),
                horizon_days,
                titles,
                event_noun=session.get("items_noun", "лекций"),
            )
    except Exception as e:
        logging.error(
            "digest.intro.llm.error digest_id=%s err=\"%s\"",
            digest_id,
            e,
        )
    else:
        duration_ms = int((_time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response digest_id=%s type=%s text_len=%s took_ms=%s",
            digest_id,
            digest_type,
            len(intro),
            duration_ms,
        )
        session["intro_html"] = intro

    for mid in session.get("preview_msg_ids", []):
        try:
            await bot.delete_message(session["chat_id"], mid)
        except Exception:
            logging.error(
                "digest.panel.refresh delete_error digest_id=%s message_id=%s",
                digest_id,
                mid,
            )
    if session.get("panel_msg_id"):
        try:
            await bot.delete_message(session["chat_id"], session["panel_msg_id"])
        except Exception:
            logging.error(
                "digest.panel.refresh delete_error digest_id=%s message_id=%s",
                digest_id,
                session["panel_msg_id"],
            )

    caption, attach, vis_len, kept = await _send_preview(
        session, digest_id, bot
    )
    logging.info(
        "digest.preview.recompose.done digest_id=%s media=%s caption_attached=%s",
        digest_id,
        len(session.get("current_media_urls", [])),
        int(attach),
    )
    logging.info(
        "digest.panel.refresh.sent digest_id=%s panel_msg_id=%s",
        digest_id,
        session.get("panel_msg_id"),
    )
    await callback.answer()


async def handle_digest_send(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    _, _, digest_id, ch_id_str = parts
    channel_id = int(ch_id_str)
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return

    excluded: set[int] = session.get("excluded", set())
    if len(session["items"]) - len(excluded) == 0:
        noun = session.get("items_noun", "лекций")
        await callback.answer(f"Нет выбранных {noun}", show_alert=True)
        return

    caption = session.get("current_caption_html", "")
    media_urls = session.get("current_media_urls", [])
    attach = session.get("current_attach", False)

    album_msg_ids: List[int] = []
    caption_msg_id: int | None = None
    media = [types.InputMediaPhoto(media=url) for url in media_urls]
    if attach and media:
        media[0].parse_mode = "HTML"
        media[0].caption = caption
    if media:
        sent = await bot.send_media_group(channel_id, media)
        album_msg_ids = [m.message_id for m in sent]
    if not attach or not media:
        msg = await bot.send_message(
            channel_id,
            caption,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        caption_msg_id = msg.message_id
    else:
        caption_msg_id = album_msg_ids[0] if album_msg_ids else None

    ch_info = next(
        (c for c in session.get("channels", []) if c["channel_id"] == channel_id),
        None,
    )
    if ch_info and caption_msg_id is not None:
        ch_obj = SimpleNamespace(
            channel_id=ch_info["channel_id"],
            title=ch_info.get("name"),
            username=ch_info.get("username"),
        )
        link = build_channel_post_url(ch_obj, caption_msg_id)
        await bot.send_message(session["chat_id"], link)

        used_indices = session.get("current_used_indices", [])
        items = session.get("items", [])
        event_ids = [
            items[i]["event_id"]
            for i in used_indices
            if i < len(items) and items[i].get("event_id")
        ]
        creator_ids = {
            items[i].get("creator_id")
            for i in used_indices
            if i < len(items) and items[i].get("creator_id")
        }
        draft_key = f"draft:digest:{digest_id}"
        raw = await get_setting_value(db, draft_key)
        data = json.loads(raw) if raw else {}
        published_to = data.setdefault("published_to", {})
        published_to[str(channel_id)] = {
            "message_url": link,
            "event_ids": event_ids,
            "notified_partner_ids": [],
        }
        await set_setting_value(db, draft_key, json.dumps(data))

        partners: list[User] = []
        if creator_ids:
            async with db.get_session() as session_db:
                result = await session_db.execute(
                    select(User).where(
                        User.user_id.in_(creator_ids), User.is_partner == True
                    )
                )
                partners = result.scalars().all()
        if partners:
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        types.InlineKeyboardButton(
                            text="Уведомить партнёров",
                            callback_data=f"dg:np:{digest_id}:{channel_id}",
                        )
                    ]
                ]
            )
            await bot.send_message(
                session["chat_id"],
                "В дайджесте есть события, добавленные партнёрами.",
                reply_markup=markup,
            )
        else:
            await bot.send_message(
                session["chat_id"],
                "В дайджесте нет событий, добавленных партнёрами.",
            )

    logging.info(
        "digest.publish digest_id=%s channel_id=%s message_id=%s attached=%s kept=%s",
        digest_id,
        channel_id,
        caption_msg_id,
        int(attach),
        len(media_urls),
    )
    await callback.answer("Опубликовано", show_alert=False)


async def handle_digest_notify_partners(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    _, _, digest_id, ch_id_str = parts
    channel_id = int(ch_id_str)
    draft_key = f"draft:digest:{digest_id}"
    raw = await get_setting_value(db, draft_key)
    if not raw:
        await callback.answer("Сначала опубликуйте дайджест", show_alert=True)
        return
    data = json.loads(raw)
    published_to = data.get("published_to", {})
    entry = published_to.get(str(channel_id))
    if not entry:
        await callback.answer("Сначала опубликуйте дайджест", show_alert=True)
        return
    event_ids = entry.get("event_ids", [])
    notified_ids = set(entry.get("notified_partner_ids", []))
    async with db.get_session() as session_db:
        if event_ids:
            res_ev = await session_db.execute(
                select(Event).where(Event.id.in_(event_ids))
            )
            events = res_ev.scalars().all()
            creator_ids = {ev.creator_id for ev in events if ev.creator_id}
        else:
            creator_ids = set()
        if creator_ids:
            res_users = await session_db.execute(
                select(User).where(
                    User.user_id.in_(creator_ids), User.is_partner == True
                )
            )
            partners = res_users.scalars().all()
        else:
            partners = []
    to_notify = [u for u in partners if u.user_id not in notified_ids]
    if not to_notify:
        await callback.answer("Уже уведомлено", show_alert=False)
        return
    notified_now: list[int] = []
    for u in to_notify:
        try:
            await bot.send_message(
                u.user_id, f"Ваше событие попало в дайджест: {entry.get('message_url')}"
            )
            notified_now.append(u.user_id)
        except Exception as e:
            logging.error("digest.notify_partner failed user_id=%s err=%s", u.user_id, e)
    usernames: list[str] = []
    for u in to_notify:
        if u.username:
            usernames.append(f"@{u.username}")
        else:
            usernames.append(f'<a href="tg://user?id={u.user_id}">Партнёр</a>')
    if callback.message:
        await bot.send_message(
            callback.message.chat.id,
            f"Уведомлено: {', '.join(usernames)}",
            parse_mode="HTML",
        )
    entry["notified_partner_ids"] = list(notified_ids | set(notified_now))
    published_to[str(channel_id)] = entry
    data["published_to"] = published_to
    await set_setting_value(db, draft_key, json.dumps(data))
    await callback.answer()


async def handle_digest_hide(callback: types.CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 3:
        return
    _, _, digest_id = parts
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return
    if session.get("panel_msg_id"):
        try:
            await bot.delete_message(session["chat_id"], session["panel_msg_id"])
        except Exception:
            logging.error(
                "digest.panel.hide delete_error digest_id=%s message_id=%s",
                digest_id,
                session["panel_msg_id"],
            )
        session["panel_msg_id"] = None
    await callback.answer()

    draft_key = f"draft:digest:{digest_id}"
    raw = await get_setting_value(db, draft_key)
    if not raw:
        await callback.answer("Черновик не найден", show_alert=False)
        return
    data = json.loads(raw)
    published_to = data.setdefault("published_to", {})
    if str(channel_id) in published_to:
        await callback.answer("Уже отправлено", show_alert=False)
        logging.info(
            "digest.publish.skip digest_id=%s channel_id=%s reason=already_sent",
            digest_id,
            channel_id,
        )
        return

    image_urls = data.get("image_urls") or []
    caption_text = data.get("caption_text") or ""

async def handle_ask_4o(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2:
        await bot.send_message(message.chat.id, "Usage: /ask4o <text>")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    try:
        answer = await ask_4o(parts[1])
    except Exception as e:
        await bot.send_message(message.chat.id, f"LLM error: {e}")
        return
    await bot.send_message(message.chat.id, answer)


async def handle_exhibitions(message: types.Message, db: Database, bot: Bot):
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)

    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Not authorized")
            return

    chunks, markup = await build_exhibitions_message(db, tz)
    if not chunks:
        return
    first, *rest = chunks
    sent_messages: list[tuple[int, str]] = []
    first_msg = await bot.send_message(message.chat.id, first, reply_markup=markup)
    sent_messages.append((first_msg.message_id, first))
    for chunk in rest:
        msg = await bot.send_message(message.chat.id, chunk)
        sent_messages.append((msg.message_id, chunk))
    exhibitions_message_state[message.chat.id] = sent_messages


async def handle_pages(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(MonthPage).order_by(MonthPage.month))
        months = result.scalars().all()
        res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
        weekends = res_w.scalars().all()
    lines = ["Months:"]
    for p in months:
        lines.append(f"{p.month}: {p.url}")
    if weekends:
        lines.append("")
        lines.append("Weekends:")
        for w in weekends:
            lines.append(f"{w.start}: {w.url}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_weekendimg_cmd(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            return
    today = datetime.now(LOCAL_TZ).date()
    first = next_weekend_start(today)
    dates = [first + timedelta(days=7 * i) for i in range(5)]
    rows = [
        [
            types.InlineKeyboardButton(
                text=f"Выходные {format_weekend_range(d)}",
                callback_data=f"weekimg:{d.isoformat()}",
            )
        ]
        for d in dates
    ]
    rows.append(
        [
            types.InlineKeyboardButton(
                text="Обложка лендинга фестивалей",
                callback_data=f"weekimg:{FESTIVALS_INDEX_MARKER}",
            )
        ]
    )
    kb = types.InlineKeyboardMarkup(inline_keyboard=rows)
    await message.answer("Выберите выходные для обложки:", reply_markup=kb)


async def handle_weekendimg_cb(callback: types.CallbackQuery, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, callback.from_user.id)
        if not user or not user.is_superadmin:
            await callback.answer("Недостаточно прав", show_alert=True)
            return
    if not callback.data:
        return
    start = callback.data.split(":", 1)[1]
    weekend_img_wait[callback.from_user.id] = start
    await callback.message.edit_reply_markup(reply_markup=None)
    if start == FESTIVALS_INDEX_MARKER:
        await callback.message.answer(
            "Выбрана обложка лендинга фестивалей.\n"
            "Пришлите обложку одним сообщением (фото или файл).",
        )
    else:
        try:
            start_date = date.fromisoformat(start)
        except ValueError:
            await callback.message.answer(
                "Не удалось распознать дату. Попробуйте выбрать вариант ещё раз."
            )
            weekend_img_wait.pop(callback.from_user.id, None)
            await callback.answer()
            return
        await callback.message.answer(
            f"Выбраны выходные {format_weekend_range(start_date)}.\n"
            "Пришлите обложку одним сообщением (фото или файл).",
        )
    await callback.answer()


async def handle_weekendimg_photo(message: types.Message, db: Database, bot: Bot):
    start = weekend_img_wait.get(message.from_user.id)
    if not start:
        return

    images = (await extract_images(message, bot))[:1]
    if not images:
        await message.reply("Не вижу изображения. Пришлите одно фото/файл в ответ.")
        return

    urls, _ = await upload_images(images, limit=1, force=True)
    if not urls:
        await message.reply("Не удалось загрузить в Catbox. Попробуйте другое фото.")
        return

    cover = urls[0]
    if start == FESTIVALS_INDEX_MARKER:
        landing_url = ""
        try:
            await set_setting_value(db, "festivals_index_cover", cover)
            await sync_festivals_index_page(db)
            _, landing_url = await rebuild_festivals_index_if_needed(
                db, force=True
            )
        except Exception:
            logging.exception("Failed to update festivals index cover")
            await message.reply(
                "Обложка сохранена, но страницу не удалось обновить. Попробуйте ещё раз."
            )
        else:
            if landing_url:
                await message.reply(
                    f"Готово! Обложка лендинга обновлена.\n{landing_url}"
                )
            else:
                await message.reply(
                    "Обложка сохранена, но ссылку на лендинг получить не удалось."
                )
        finally:
            weekend_img_wait.pop(message.from_user.id, None)
        return

    await set_setting_value(db, f"weekend_cover:{start}", cover)
    await sync_weekend_page(db, start, update_links=True, post_vk=False, force=True)

    async with db.get_session() as session:
        page = await session.get(WeekendPage, start)
    if page and page.url:
        await message.reply(f"Готово! Обложка добавлена.\n{page.url}")
    else:
        await message.reply(
            "Обложка сохранена, но страницу не удалось обновить. Попробуйте ещё раз."
        )

    weekend_img_wait.pop(message.from_user.id, None)


def _shift_month(d: date, offset: int) -> date:
    year = d.year + (d.month - 1 + offset) // 12
    month = (d.month - 1 + offset) % 12 + 1
    return date(year, month, 1)


def _expand_months(months: list[str], past: int, future: int) -> list[str]:
    if months:
        return sorted(months)
    today = date.today().replace(day=1)
    start = _shift_month(today, -past)
    total = past + future + 1
    res = []
    for i in range(total):
        m = _shift_month(start, i)
        res.append(m.strftime("%Y-%m"))
    return res


async def _future_months_with_events(db: Database) -> list[str]:
    today = date.today().replace(day=1)
    rows = await db.exec_driver_sql("SELECT date FROM event")
    months: set[str] = set()
    for (raw_date,) in rows:
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except Exception:
            continue
        if dt >= today:
            months.add(dt.strftime("%Y-%m"))
    return sorted(months)


def _weekends_for_months(months: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    weekends: set[str] = set()
    mapping: dict[str, list[str]] = defaultdict(list)
    for m in months:
        year, mon = map(int, m.split("-"))
        d = date(year, mon, 1)
        while d.month == mon:
            w = weekend_start_for_date(d)
            if w and w.month == mon:
                key = w.isoformat()
                if key not in mapping[m]:
                    mapping[m].append(key)
                    weekends.add(key)
            d += timedelta(days=1)
    return sorted(weekends), mapping


async def _perform_pages_rebuild(db: Database, months: list[str], force: bool = False) -> str:
    weekends, mapping = _weekends_for_months(months)
    global DISABLE_EVENT_PAGE_UPDATES
    DISABLE_EVENT_PAGE_UPDATES = True
    try:
        result = await rebuild_pages(db, months, weekends, force=force)
    finally:
        DISABLE_EVENT_PAGE_UPDATES = False

    lines = ["Telegraph month rebuild:"]
    for m in months:
        failed = result["months"]["failed"].get(m)
        urls = result["months"]["updated"].get(m, [])
        if failed:
            lines.append(f"❌ {m} — ошибка: {failed}")
        else:
            lines.append(f"✅ {m} — обновлено:")
            if len(urls) == 2:
                lines.append(f"  • Часть 1: {urls[0]}")
                lines.append(f"  • Часть 2: {urls[1]}")
            elif len(urls) == 1:
                lines.append(f"  • {urls[0]}")
            else:
                lines.append("  • отсутствует")

    lines.append("\nTelegraph weekends rebuild:")
    for m in months:
        w_list = mapping.get(m, [])
        total = len(w_list)
        success = 0
        month_lines: list[str] = []
        for w in w_list:
            label = format_weekend_range(date.fromisoformat(w))
            urls = result["weekends"]["updated"].get(w, [])
            err = result["weekends"]["failed"].get(w)
            if urls:
                success += 1
                month_lines.append(f"  • {label}: ✅ {urls[0]}")
            elif err:
                status = "⏳ перенесено" if "flood" in err.lower() else "❌"
                month_lines.append(f"  • {label}: {status} {err}")
            else:
                month_lines.append(f"  • {label}: ❌ неизвестно")
        if success == total:
            lines.append(f"✅ {m} — обновлено: {total} страниц")
        elif success == 0:
            lines.append(f"❌ {m} — ошибка:")
        else:
            lines.append(f"☑️ {success} из {total} обновлены:")
        lines.extend(month_lines)

    return "\n".join(lines)


def _parse_pages_rebuild_args(text: str) -> tuple[list[str], int, int, bool]:
    parts = text.split()[1:]
    months: list[str] = []
    past = 0
    future = 2
    force = False
    for p in parts:
        if p.startswith("--past="):
            try:
                past = int(p.split("=", 1)[1])
            except ValueError:
                pass
        elif p.startswith("--future="):
            try:
                future = int(p.split("=", 1)[1])
            except ValueError:
                pass
        elif p == "--force":
            force = True
        else:
            months.append(p)
    return months, past, future, force


async def handle_pages_rebuild(message: types.Message, db: Database, bot: Bot):
    months, past, future, _ = _parse_pages_rebuild_args(message.text or "")
    if not months and (message.text or "").strip() == "/pages_rebuild":
        options = await _future_months_with_events(db)
        if not options:
            options = _expand_months([], past, future)
        buttons = [
            [types.InlineKeyboardButton(text=m, callback_data=f"pages_rebuild:{m}")]
            for m in options
        ]
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=buttons
            + [[types.InlineKeyboardButton(text="Все", callback_data="pages_rebuild:ALL")]]
        )
        msg_text = "Выберите месяц для пересборки или «Все»"
        if os.getenv("DEV_MODE") == "1":
            msg_text = "⚠️ DEV MODE: Страницы будут созданы с префиксом ТЕСТ!\n\n" + msg_text

        await bot.send_message(
            message.chat.id,
            msg_text,
            reply_markup=markup,
        )
        return
    months_list = _expand_months(months, past, future)
    report = await _perform_pages_rebuild(db, months_list, force=True)
    await bot.send_message(message.chat.id, report)


async def handle_pages_rebuild_cb(
    callback: types.CallbackQuery, db: Database, bot: Bot
):
    await callback.answer()
    val = callback.data.split(":", 1)[1]
    if val.upper() == "ALL":
        months = await _future_months_with_events(db)
        if not months:
            months = _expand_months([], 0, 2)
    else:
        months = [val]
    report = await _perform_pages_rebuild(db, months, force=True)
    await bot.send_message(callback.message.chat.id, report)


async def handle_fest(message: types.Message, db: Database, bot: Bot):
    archive = False
    page = 1
    text = message.text or ""
    parts = text.split()
    for part in parts[1:]:
        if part.lower() == "archive":
            archive = True
        else:
            try:
                page = int(part)
            except ValueError:
                continue
    await send_festivals_list(
        message,
        db,
        bot,
        user_id=message.from_user.id if message.from_user else None,
        edit=False,
        page=page,
        archive=archive,
    )





async def fetch_views(path: str, url: str | None = None) -> int | None:
    token = get_telegraph_token()
    if not token:
        return None
    domain = "telegra.ph"
    if url:
        try:
            domain = url.split("//", 1)[1].split("/", 1)[0]
        except Exception:
            pass
    tg = Telegraph(access_token=token, domain=domain)

    try:
        data = await telegraph_call(tg.get_views, path)
        return int(data.get("views", 0))
    except Exception as e:
        logging.error("Failed to fetch views for %s: %s", path, e)
        return None


async def collect_page_stats(db: Database) -> list[str]:
    today = datetime.now(LOCAL_TZ).date()
    prev_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    prev_month = prev_month_start.strftime("%Y-%m")

    prev_weekend = next_weekend_start(today - timedelta(days=7))
    cur_month = today.strftime("%Y-%m")
    cur_weekend = next_weekend_start(today)

    async with db.get_session() as session:
        mp_prev = await session.get(MonthPage, prev_month)
        wp_prev = await session.get(WeekendPage, prev_weekend.isoformat())

        res_months = await session.execute(
            select(MonthPage)
            .where(MonthPage.month >= cur_month)
            .order_by(MonthPage.month)
        )
        future_months = res_months.scalars().all()

        res_weekends = await session.execute(
            select(WeekendPage)
            .where(WeekendPage.start >= cur_weekend.isoformat())
            .order_by(WeekendPage.start)
        )
        future_weekends = res_weekends.scalars().all()

    lines: list[str] = []

    async def month_views(mp: MonthPage) -> int | None:
        paths: list[tuple[str, str | None]] = []
        if mp.path:
            paths.append((mp.path, mp.url))
        if mp.path2:
            paths.append((mp.path2, mp.url2))
        if not paths:
            return None

        total = 0
        has_value = False
        for path, url in paths:
            views = await fetch_views(path, url)
            if views is None:
                continue
            total += views
            has_value = True

        return total if has_value else None

    if mp_prev:

        views = await month_views(mp_prev)

        if views is not None:
            month_dt = date.fromisoformat(mp_prev.month + "-01")
            lines.append(f"{MONTHS_NOM[month_dt.month - 1]}: {views} просмотров")

    if wp_prev and wp_prev.path:

        views = await fetch_views(wp_prev.path, wp_prev.url)

        if views is not None:
            label = format_weekend_range(prev_weekend)
            lines.append(f"{label}: {views} просмотров")

    for wp in future_weekends:
        if not wp.path:
            continue

        views = await fetch_views(wp.path, wp.url)

        if views is not None:
            label = format_weekend_range(date.fromisoformat(wp.start))
            lines.append(f"{label}: {views} просмотров")

    for mp in future_months:
        if not (mp.path or mp.path2):
            continue

        views = await month_views(mp)

        if views is not None:
            month_dt = date.fromisoformat(mp.month + "-01")
            lines.append(f"{MONTHS_NOM[month_dt.month - 1]}: {views} просмотров")


    return lines


async def collect_event_stats(db: Database) -> list[str]:
    today = datetime.now(LOCAL_TZ).date()
    prev_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    async with db.get_session() as session:
        result = await session.execute(
            select(Event).where(
                Event.telegraph_path.is_not(None),
                Event.date >= prev_month_start.isoformat(),
            )
        )
        events = result.scalars().all()
    stats = []
    for e in events:
        if not e.telegraph_path:
            continue

        views = await fetch_views(e.telegraph_path, e.telegraph_url)

        if views is not None:
            stats.append((e.telegraph_url or e.telegraph_path, views))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{url}: {v}" for url, v in stats]


async def collect_vk_shortlink_click_stats(db: Database) -> list[str]:
    """Return aggregated vk.cc click statistics for active events."""

    today = datetime.now(LOCAL_TZ).date()
    week_ago = today - timedelta(days=7)
    async with db.get_session() as session:
        result = await session.execute(
            select(Event).where(Event.vk_ticket_short_key.is_not(None))
        )
        events = result.scalars().all()

    entries: list[tuple[str, int, int]] = []

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    for event in events:
        key = (event.vk_ticket_short_key or "").strip()
        if not key:
            continue

        date_raw = event.date or ""
        start = parse_iso_date(date_raw)
        if not start and ".." in date_raw:
            start = parse_iso_date(date_raw.split("..", 1)[0])
        if not start:
            logger.warning(
                "shortlink stats: skip event %s due to malformed start date %r",
                event.id,
                event.date,
            )
            continue

        end: date | None = None
        if event.end_date:
            end = parse_iso_date(event.end_date)
            if not end and ".." in event.end_date:
                end = parse_iso_date(event.end_date.split("..", 1)[-1])
        if not end and ".." in date_raw:
            end_part = date_raw.split("..", 1)[1]
            end = parse_iso_date(end_part)
        if end is None:
            end = start

        if not (start >= today or end >= week_ago):
            continue

        try:
            data = await vk_api(
                "utils.getLinkStats",
                key=key,
                interval="forever",
                intervals_count=1,
            )
        except VKAPIError as exc:
            logger.warning(
                "shortlink stats: vk api error for event %s key %s: %s",
                event.id,
                key,
                exc,
            )
            continue

        payload: Any = data
        if isinstance(payload, dict):
            payload = payload.get("response", payload)
        if isinstance(payload, dict):
            stats_list = payload.get("stats") or []
        elif isinstance(payload, list):
            stats_list = payload
        else:
            stats_list = []

        clicks_total = 0
        views_total = 0
        for item in stats_list:
            if not isinstance(item, dict):
                continue
            clicks_value = item.get("clicks")
            if clicks_value is None:
                clicks_value = item.get("visitors")
            if clicks_value is None:
                clicks_value = item.get("count")
            clicks_total += _as_int(clicks_value)
            views_total += _as_int(item.get("views"))

        entries.append((event.title, clicks_total, views_total))

    entries.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [f"{title}: {clicks}" for title, clicks, _ in entries]


async def collect_festivals_landing_stats(db: Database) -> str | None:
    """Return Telegraph view count for the festivals landing page."""
    path = await get_setting_value(db, "festivals_index_path") or await get_setting_value(
        db, "fest_index_path"
    )
    url = await get_setting_value(db, "festivals_index_url") or await get_setting_value(
        db, "fest_index_url"
    )
    if not path and url:
        try:
            path = url.split("//", 1)[1].split("/", 1)[1]
        except Exception:
            path = None
    if not path:
        return None
    views = await fetch_views(path, url)
    if views is None:
        return None
    return f"Лендинг фестивалей: {views} просмотров"


async def collect_festival_telegraph_stats(db: Database) -> list[str]:
    """Return Telegraph view counts for upcoming and recent festivals."""
    today = datetime.now(LOCAL_TZ).date()
    week_ago = today - timedelta(days=7)
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.telegraph_path.is_not(None))
        )
        fests = result.scalars().all()
    stats: list[tuple[str, int]] = []
    for f in fests:
        start = parse_iso_date(f.start_date) if f.start_date else None
        end = parse_iso_date(f.end_date) if f.end_date else start
        if not ((start and start >= today) or (end and end >= week_ago)):
            continue
        views = await fetch_views(f.telegraph_path, f.telegraph_url)
        if views is not None:
            stats.append((f.name, views))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}: {views}" for name, views in stats]


async def send_job_status(chat_id: int, event_id: int, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        stmt = select(JobOutbox).where(JobOutbox.event_id == event_id).order_by(JobOutbox.id)
        jobs = (await session.execute(stmt)).scalars().all()
    if not jobs:
        await bot.send_message(chat_id, "Нет задач")
        return
    lines = ["task | status | attempts | updated | result"]
    for j in jobs:
        link = await _job_result_link(j.task, event_id, db)
        result = link if j.status == JobStatus.done else (j.last_error or "")
        lines.append(
            f"{TASK_LABELS[j.task.value]} | {j.status.value} | {j.attempts} | {j.updated_at:%H:%M:%S} | {result}"
        )
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="🔁 Перезапустить невыполненные",
                    callback_data=f"requeue:{event_id}",
                )
            ]
        ]
    )
    await bot.send_message(chat_id, "\n".join(lines), reply_markup=markup)


async def fetch_vk_post_stats(
    post_url: str, db: Database | None = None, bot: Bot | None = None
) -> tuple[int | None, int | None]:
    """Return view and reach counts for a VK post."""
    ids = _vk_owner_and_post_id(post_url)
    if not ids:
        logging.error("invalid VK post url %s", post_url)
        return None, None
    owner_id, post_id = ids
    views: int | None = None
    reach: int | None = None
    try:
        response = await vk_api("wall.getById", posts=f"{owner_id}_{post_id}")
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
        if items:
            views = (items[0].get("views") or {}).get("count")
    except Exception as e:
        logging.error("VK views fetch error for %s: %s", post_url, e)
    try:
        data = await _vk_api(
            "stats.getPostReach",
            {"owner_id": owner_id, "post_id": post_id},
            db,
            bot,
        )
        items = data.get("response") or []
        if items:
            reach = items[0].get("reach_total")
    except Exception as e:
        logging.error("VK reach fetch error for %s: %s", post_url, e)
    return views, reach


async def collect_festival_vk_stats(db: Database) -> list[str]:
    """Return VK view and reach counts for upcoming and recent festivals."""
    today = datetime.now(LOCAL_TZ).date()
    week_ago = today - timedelta(days=7)
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.vk_post_url.is_not(None))
        )
        fests = result.scalars().all()
    stats: list[tuple[str, int, int]] = []
    for f in fests:
        start = parse_iso_date(f.start_date) if f.start_date else None
        end = parse_iso_date(f.end_date) if f.end_date else start
        if not ((start and start >= today) or (end and end >= week_ago)):
            continue
        views, reach = await fetch_vk_post_stats(f.vk_post_url, db)
        if views is not None or reach is not None:
            stats.append((f.name, views or 0, reach or 0))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}: {views}, {reach}" for name, views, reach in stats]


async def handle_status(
    message: types.Message, db: Database, bot: Bot, app: web.Application
):
    parts = (message.text or "").split()
    if len(parts) > 1 and parts[1].isdigit():
        await send_job_status(message.chat.id, int(parts[1]), db, bot)
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    uptime = _time.time() - START_TIME
    qlen = add_event_queue.qsize()
    worker_task = app.get("add_event_worker")
    alive = "no"
    if isinstance(worker_task, asyncio.Task) and not worker_task.done():
        alive = "yes"
    last_dequeue = (
        f"{int(_time.monotonic() - _ADD_EVENT_LAST_DEQUEUE_TS)}s"
        if _ADD_EVENT_LAST_DEQUEUE_TS
        else "never"
    )
    jobs = list(JOB_HISTORY)[-5:]
    lines = [
        f"uptime: {int(uptime)}s",
        f"queue_len: {qlen}",
        f"worker_alive: {alive}",
        f"last_dequeue_ago: {last_dequeue}",
    ]
    if jobs:
        lines.append("last_jobs:")
        for j in reversed(jobs):
            when = j["when"].strftime("%H:%M:%S")
            lines.append(f"- {j['id']} {when} {j['status']} {j['took_ms']}ms")
    if LAST_RUN_ID:
        lines.append(f"last_run_id: {LAST_RUN_ID}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_festivals_fix_nav(
    message: types.Message, db: Database, bot: Bot
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    run_id = uuid.uuid4().hex
    logging.info(
        "festivals_fix_nav start", extra={"run_id": run_id, "user": message.from_user.id}
    )
    async with page_lock["festivals-index"]:
        await message.answer("Пересобираю навигацию и лендинг…")
        pages = changed = duplicates_removed = 0
        try:
            pages, changed, duplicates_removed, _ = await festivals_fix_nav(db, bot)
            status, url = await rebuild_festivals_index_if_needed(db, force=True)
        except Exception as e:
            status = f"ошибка: {e}"
            url = ""
        landing_line = f"Лендинг: {status}" + (f" {url}" if url else "")
        await message.answer(
            f"Готово. pages:{pages}, changed:{changed}, duplicates_removed:{duplicates_removed}\n{landing_line}"
        )


async def handle_ics_fix_nav(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = (message.text or "").split()
    month = parts[1] if len(parts) > 1 else None
    count = await ics_fix_nav(db, month)
    await message.answer(f"Готово. события: {count}")


async def handle_trace(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await bot.send_message(message.chat.id, "Usage: /trace <run_id>")
        return
    run_id = parts[1].strip()
    lines = []
    for ts, level, msg in LOG_BUFFER:
        if run_id in msg:
            lines.append(f"{ts.strftime('%H:%M:%S')} {level[0]} {msg}")
    await bot.send_message(message.chat.id, "\n".join(lines) or "No trace")


async def handle_last_errors(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = message.text.split()
    count = 5
    if len(parts) > 1:
        try:
            count = min(int(parts[1]), len(ERROR_BUFFER))
        except Exception:
            pass
    errs = list(ERROR_BUFFER)[-count:]
    lines = []
    for e in reversed(errs):
        lines.append(
            f"{e['time'].strftime('%H:%M:%S')} {e['type']} {e['where']}"
        )
        await bot.send_message(message.chat.id, "\n".join(lines) or "No errors")


async def handle_debug(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = (message.text or "").split()
    if len(parts) < 2 or parts[1] != "queue":
        await bot.send_message(message.chat.id, "Usage: /debug queue")
        return
    async with db.get_session() as session:
        task_rows = await session.execute(
            select(JobOutbox.task, func.count()).group_by(JobOutbox.task)
        )
        event_rows = await session.execute(
            select(JobOutbox.event_id, func.count()).group_by(JobOutbox.event_id)
        )
    lines = ["tasks:"]
    for task, cnt in task_rows.all():
        lines.append(f"{task.value}: {cnt}")
    lines.append("events:")
    for eid, cnt in event_rows.all():
        lines.append(f"{eid}: {cnt}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_backfill_topics(
    message: types.Message, db: Database, bot: Bot
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return

    parts = (message.text or "").split()
    days = 90
    if len(parts) > 1:
        try:
            days = int(parts[1])
        except Exception:
            await bot.send_message(message.chat.id, "Usage: /backfill_topics [days]")
            return
        if days < 0:
            await bot.send_message(message.chat.id, "Usage: /backfill_topics [days]")
            return

    today = date.today()
    end_date = today + timedelta(days=days)
    start_iso = today.isoformat()
    end_iso = end_date.isoformat()

    logging.info(
        "backfill_topics.start user_id=%s days=%s start=%s end=%s",
        message.from_user.id,
        days,
        start_iso,
        end_iso,
    )

    processed = 0
    updated = 0
    skipped = 0
    total = 0
    async with db.get_session() as session:
        stmt = (
            select(Event)
            .where(Event.date >= start_iso)
            .where(Event.date <= end_iso)
            .order_by(Event.date, Event.time, Event.id)
        )
        events = (await session.execute(stmt)).scalars().all()
        total = len(events)
        for event in events:
            if getattr(event, "topics_manual", False):
                skipped += 1
                continue

            processed += 1
            previous_topics = list(getattr(event, "topics", []) or [])
            original_manual = bool(getattr(event, "topics_manual", False))
            try:
                new_topics_raw = await classify_event_topics(event)
            except Exception:
                logging.exception(
                    "backfill_topics.classify_failed event_id=%s", getattr(event, "id", None)
                )
                continue

            seen: set[str] = set()
            normalized_topics: list[str] = []
            for topic in new_topics_raw:
                canonical = normalize_topic_identifier(topic)
                if canonical is None or canonical in seen:
                    continue
                seen.add(canonical)
                normalized_topics.append(canonical)

            event.topics = normalized_topics
            event.topics_manual = False

            if normalized_topics != previous_topics or original_manual:
                updated += 1
                session.add(event)

        if updated:
            await session.commit()

    logging.info(
        "backfill_topics.summary total=%s processed=%s updated=%s skipped=%s",
        total,
        processed,
        updated,
        skipped,
    )
    summary = (
        f"Backfilled topics {start_iso}..{end_iso} (days={days}): "
        f"processed={processed}, updated={updated}, skipped={skipped}"
    )
    await bot.send_message(message.chat.id, summary)


async def handle_queue_reap(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    args = shlex.split(message.text or "")[1:]
    parser = argparse.ArgumentParser(prog="/queue_reap", add_help=False)
    parser.add_argument("--type")
    parser.add_argument("--ym")
    parser.add_argument("--key-prefix")
    parser.add_argument("--status", default="running")
    parser.add_argument("--older-than")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--action", choices=["fail", "requeue"])
    parser.add_argument("--apply", action="store_true")
    try:
        opts = parser.parse_args(args)
    except Exception:
        await bot.send_message(message.chat.id, "Invalid arguments")
        return
    key_prefix = opts.key_prefix
    if not key_prefix and opts.type and opts.ym:
        key_prefix = f"{opts.type}:{opts.ym}"
    if not key_prefix and opts.type:
        key_prefix = f"{opts.type}:"
    older_sec = 0
    if opts.older_than:
        s = opts.older_than
        mult = 60
        if s.endswith("h"):
            mult = 3600
        elif s.endswith("d"):
            mult = 86400
        value = int(s[:-1]) if s[-1] in "mhd" else int(s)
        older_sec = value * mult
    now = datetime.now(timezone.utc)
    async with db.get_session() as session:
        stmt = select(JobOutbox).where(JobOutbox.status == JobStatus(opts.status))
        if key_prefix:
            stmt = stmt.where(JobOutbox.coalesce_key.like(f"{key_prefix}%"))
        if older_sec:
            thresh = now - timedelta(seconds=older_sec)
            stmt = stmt.where(JobOutbox.updated_at < thresh)
        stmt = stmt.order_by(JobOutbox.updated_at).limit(opts.limit)
        jobs = (await session.execute(stmt)).scalars().all()
    lines: list[str] = []
    header = "[DRY-RUN] " if not opts.apply else ""
    header += f"candidates={len(jobs)} status={opts.status}"
    if opts.older_than:
        header += f" older-than={opts.older_than}"
    if key_prefix:
        header += f" key-prefix={key_prefix}"
    lines.append(header)
    for idx, j in enumerate(jobs, 1):
        key = j.coalesce_key or f"{j.task.value}:{j.event_id}"
        started = j.updated_at.replace(microsecond=0).isoformat()
        delta = now - j.updated_at
        days, rem = divmod(int(delta.total_seconds()), 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if not days and minutes:
            parts.append(f"{minutes}m")
        age = "".join(parts) or "0m"
        lines.append(
            f"{idx}) id={j.id} key={key} owner_eid={j.event_id} started={started} age={age}"
        )
    if opts.apply and opts.action and jobs:
        async with db.get_session() as session:
            for j in jobs:
                obj = await session.get(JobOutbox, j.id)
                if not obj:
                    continue
                if opts.action == "fail":
                    obj.status = JobStatus.error
                    obj.last_error = "reaped_by_admin"
                else:
                    obj.status = JobStatus.pending
                    obj.attempts = 0
                    obj.last_error = None
                    obj.next_run_at = now
                obj.updated_at = now
                session.add(obj)
            await session.commit()
        lines.append(f"applied {opts.action} to {len(jobs)}")
    await bot.send_message(message.chat.id, "\n".join(lines))


def _coerce_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_token_usage_lines(
    model_totals: Mapping[str, int], *, total_override: int | None = None
) -> list[str]:
    totals: dict[str, int] = {model: int(value) for model, value in model_totals.items()}
    if not totals:
        totals = {model: 0 for model in FOUR_O_TRACKED_MODELS}
    for model in FOUR_O_TRACKED_MODELS:
        totals.setdefault(model, 0)
    lines = [f"Tokens {model}: {totals[model]}" for model in sorted(totals)]
    total_value = total_override if total_override is not None else sum(totals.values())
    lines.append(f"Tokens total: {total_value}")
    return lines


async def _collect_supabase_token_usage_lines() -> list[str] | None:
    client = get_supabase_client()
    if client is None:
        return None

    today = datetime.now(timezone.utc).date()
    day_start = datetime.combine(today, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    date_str = today.isoformat()

    def _fetch(table: str) -> list[Mapping[str, Any]]:
        query = client.table(table)
        if table == "token_usage_daily":
            result = (
                query.select("model,total_tokens")
                .eq("bot", BOT_CODE)
                .eq("date", date_str)
                .execute()
            )
            return list(result.data or [])
        result = (
            query.select("model,total_tokens,prompt_tokens,completion_tokens,at")
            .eq("bot", BOT_CODE)
            .gte("at", day_start.isoformat())
            .lt("at", day_end.isoformat())
            .execute()
        )
        return list(result.data or [])

    last_error: Exception | None = None
    for table in ("token_usage_daily", "token_usage"):
        try:
            records = await asyncio.to_thread(_fetch, table)
        except Exception as exc:  # pragma: no cover - network failure
            logging.debug(
                "stats.token_usage_query_failed table=%s error=%s",
                table,
                exc,
                exc_info=True,
            )
            last_error = exc
            continue

        totals: dict[str, int] = {model: 0 for model in FOUR_O_TRACKED_MODELS}
        for row in records:
            model = row.get("model")
            if not model:
                continue
            total_value = _coerce_optional_int(row.get("total_tokens"))
            if total_value is None:
                prompt = _coerce_optional_int(row.get("prompt_tokens")) or 0
                completion = _coerce_optional_int(row.get("completion_tokens")) or 0
                total_value = prompt + completion
            totals[model] = totals.get(model, 0) + int(total_value or 0)

        return _format_token_usage_lines(totals)

    if last_error is not None:
        raise last_error
    return None


def _parse_supabase_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            pass
        else:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
    return datetime.now(timezone.utc)


async def fetch_vk_miss_samples(limit: int) -> list[VkMissRecord]:
    if limit <= 0:
        return []
    client = get_supabase_client()
    if client is None:
        logging.info("vk_miss_review.supabase_disabled")
        return []

    def _query() -> list[Mapping[str, Any]]:
        result = (
            client.table("vk_misses_sample")
            .select("id,url,reason,matched_kw,ts,checked")
            .is_("checked", False)
            .order("ts", desc=True)
            .limit(limit)
            .execute()
        )
        return list(result.data or [])

    try:
        rows = await asyncio.to_thread(_query)
    except Exception as exc:  # pragma: no cover - network failure
        logging.exception("vk_miss_review.supabase_query_failed")
        return []

    records: list[VkMissRecord] = []
    for row in rows:
        raw_id = str(row.get("id") or "").strip()
        url = str(row.get("url") or "").strip()
        reason = row.get("reason")
        matched_kw = row.get("matched_kw")
        timestamp = _parse_supabase_ts(row.get("ts"))
        records.append(
            VkMissRecord(
                id=raw_id,
                url=url,
                reason=reason if isinstance(reason, str) else None,
                matched_kw=matched_kw if isinstance(matched_kw, str) else None,
                timestamp=timestamp,
            )
        )
    return records


async def _vk_miss_mark_checked(record_id: str) -> None:
    if not record_id:
        return
    client = get_supabase_client()
    if client is None:
        logging.info("vk_miss_review.supabase_disabled")
        return

    def _update() -> None:
        (
            client.table("vk_misses_sample")
            .update({"checked": True})
            .eq("id", record_id)
            .execute()
        )

    try:
        await asyncio.to_thread(_update)
    except Exception:  # pragma: no cover - network failure
        logging.exception("vk_miss_review.supabase_update_failed")


VK_MISS_CALLBACK_PREFIX = "vkmiss:"


async def handle_stats(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split()
    mode = parts[1] if len(parts) > 1 else ""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    if mode == "events":
        lines = await collect_event_stats(db)
    elif mode == "shortlinks":
        lines = await collect_vk_shortlink_click_stats(db)
        await bot.send_message(
            message.chat.id, "\n".join(lines) if lines else "No data"
        )
        return
    else:
        lines = await collect_page_stats(db)
        fest_landing = await collect_festivals_landing_stats(db)
        fest_tg = await collect_festival_telegraph_stats(db)
        fest_vk = await collect_festival_vk_stats(db)
        if fest_landing:
            lines.append("")
            lines.append(fest_landing)
        if fest_tg:
            lines.append("")
            lines.append("Фестивали (телеграм)")
            lines.extend(fest_tg)
        if fest_vk:
            lines.append("")
            lines.append("Фестивали (Вк) (просмотров, пользователи)")
            lines.extend(fest_vk)
    supabase_lines: list[str] | None = None
    try:
        supabase_lines = await _collect_supabase_token_usage_lines()
    except Exception:  # pragma: no cover - network failure
        logging.exception("stats.supabase_usage_failed")

    if not supabase_lines:
        usage_snapshot = _get_four_o_usage_snapshot()
        usage_models = dict(usage_snapshot.get("models", {}))

        def _coerce_int(value: Any) -> int:
            coerced = _coerce_optional_int(value)
            return int(coerced or 0)

        mini_snapshot = _coerce_int(usage_models.get("gpt-4o-mini", 0))
        ocr_tokens = 0
        today_key = poster_ocr._today_key()
        async with db.get_session() as session:
            ocr_usage = await session.get(OcrUsage, today_key)
            if ocr_usage and ocr_usage.spent_tokens:
                ocr_tokens = max(int(ocr_usage.spent_tokens), 0)
        new_mini_total = mini_snapshot + ocr_tokens
        usage_models["gpt-4o-mini"] = new_mini_total

        snapshot_total = _coerce_int(usage_snapshot.get("total", 0))
        tokens_total = snapshot_total - mini_snapshot + new_mini_total

        model_totals = {model: _coerce_int(value) for model, value in usage_models.items()}
        supabase_lines = _format_token_usage_lines(model_totals, total_override=tokens_total)

    lines.extend(supabase_lines)
    await bot.send_message(message.chat.id, "\n".join(lines) if lines else "No data")


async def handle_mem(message: types.Message, db: Database, bot: Bot):
    rss, _ = mem_info(update=False)
    await bot.send_message(message.chat.id, f"RSS: {rss / (1024**2):.1f} MB")


async def handle_usage_test(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user or not user.is_superadmin:
        await bot.send_message(message.chat.id, "Not authorized")
        return

    model_name = "gpt-4o-mini"
    try:
        await ask_4o("usage probe", model=model_name)
    except Exception as exc:  # pragma: no cover - network failure
        logging.exception("usage_test ask_4o failed")
        await bot.send_message(message.chat.id, f"ask_4o failed: {exc}")
        return

    request_id = get_last_ask_4o_request_id()
    if not request_id:
        await bot.send_message(message.chat.id, "No request ID returned")
        return

    client = get_supabase_client()
    if client is None:
        await bot.send_message(message.chat.id, "Supabase disabled")
        return

    deadline = _time.monotonic() + 1.0
    row: Mapping[str, object] | None = None

    try:
        while True:
            response = (
                client.table("token_usage")
                .select("prompt_tokens,completion_tokens,total_tokens")
                .eq("request_id", request_id)
                .order("at", desc=True)
                .limit(1)
                .execute()
            )
            records = getattr(response, "data", response)
            if records:
                row = records[0]
                break
            now = _time.monotonic()
            if now >= deadline:
                break
            await asyncio.sleep(min(0.1, deadline - now))
    except Exception as exc:  # pragma: no cover - supabase failure
        logging.exception("usage_test supabase query failed")
        await bot.send_message(message.chat.id, f"Supabase query failed: {exc}")
        return

    if row is None:
        await bot.send_message(
            message.chat.id,
            f"Token usage for request {request_id} was not yet available; please retry.",
        )
        return

    prompt_tokens = int(row.get("prompt_tokens") or 0)
    completion_tokens = int(row.get("completion_tokens") or 0)
    total_tokens = int(row.get("total_tokens") or (prompt_tokens + completion_tokens))

    bot_label = getattr(bot, "id", None)
    if bot_label is None:
        bot_label = bot.__class__.__name__
    logging.info(
        "usage_test trace bot=%s model=%s request_id=%s",
        bot_label,
        model_name,
        request_id,
    )

    payload = {
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": total_tokens,
    }
    await bot.send_message(
        message.chat.id,
        json.dumps(payload, ensure_ascii=False),
    )


def _coerce_report_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text:
                return int(float(text))
            return int(text)
        except ValueError:
            return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _collect_report_metrics(
    record: Mapping[str, Any],
    metric_defs: Sequence[tuple[str, Sequence[str]]],
) -> tuple[dict[str, int], set[str]]:
    values: dict[str, int] = {}
    used_columns: set[str] = set()
    for label, columns in metric_defs:
        for column in columns:
            if column not in record:
                continue
            coerced = _coerce_report_int(record.get(column))
            if coerced is None:
                continue
            values[label] = coerced
            used_columns.add(column)
            break
    return values, used_columns


def _collect_extra_numeric_fields(
    record: Mapping[str, Any],
    *,
    skip_fields: Collection[str],
    used_columns: Collection[str],
) -> dict[str, int]:
    extras: dict[str, int] = {}
    for key, value in record.items():
        if key in skip_fields or key in used_columns:
            continue
        coerced = _coerce_report_int(value)
        if coerced is None:
            continue
        extras[key] = coerced
    return extras


def _try_parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parsed_date = parse_iso_date(text)
        if parsed_date:
            return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        with contextlib.suppress(ValueError):
            parsed_dt = datetime.fromisoformat(text)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            return parsed_dt
    return None


def _guess_record_datetime(
    record: Mapping[str, Any],
    fields: Sequence[str],
) -> datetime | None:
    for field in fields:
        if field not in record:
            continue
        candidate = _try_parse_iso_datetime(record.get(field))
        if candidate is not None:
            return candidate
    return None


def _format_imp_groups_report(
    records: Sequence[Mapping[str, Any]],
    *,
    days: int,
    max_rows: int = 25,
) -> list[str]:
    metric_defs: Sequence[tuple[str, Sequence[str]]] = [
        ("Импорт", ("imported", "imported_count", "imports", "matches", "added")),
        (
            "На проверке",
            (
                "pending",
                "pending_count",
                "queue",
                "queued",
                "needs_review",
                "needs_review_count",
            ),
        ),
        (
            "Отклонено",
            ("rejected", "rejected_count", "missed", "misses", "total_rejected"),
        ),
        ("Дубликаты", ("duplicates", "duplicates_count")),
        ("Пропущено", ("skipped", "skipped_count", "auto_skipped")),
        ("Всего", ("total", "total_count", "posts", "events", "sum")),
    ]
    ts_fields = (
        "last_import_at",
        "last_imported_at",
        "last_at",
        "updated_at",
    )
    skip_fields: set[str] = {
        "window_days",
        "days",
        "period_days",
        "span_days",
        "group_name",
        "group_title",
        "name",
        "title",
        "group_id",
        "id",
        "screen_name",
        "vk_url",
        "group",
        "group_url",
    }
    entries: list[dict[str, Any]] = []
    totals: dict[str, int] = {label: 0 for label, _ in metric_defs}
    tz = ZoneInfo(VK_WEEK_EDIT_TZ)
    for record in records:
        metrics, used_columns = _collect_report_metrics(record, metric_defs)
        for label, value in metrics.items():
            totals[label] = totals.get(label, 0) + value
        name = ""
        for key in ("group_name", "group_title", "name", "title", "screen_name"):
            raw = record.get(key)
            if isinstance(raw, str) and raw.strip():
                name = raw.strip()
                break
        group_id = _coerce_report_int(record.get("group_id"))
        if group_id is None:
            group_id = _coerce_report_int(record.get("id"))
        display_name = name or (f"id={group_id}" if group_id is not None else "Без названия")
        if group_id is not None and display_name and str(group_id) not in display_name:
            display_name = f"{display_name} (id={group_id})"
        extras = _collect_extra_numeric_fields(
            record,
            skip_fields=skip_fields,
            used_columns=used_columns,
        )
        timestamp = _guess_record_datetime(record, ts_fields)
        entries.append(
            {
                "display": display_name,
                "metrics": metrics,
                "extras": extras,
                "timestamp": timestamp,
                "sort": metrics.get("Импорт", 0),
            }
        )

    entries.sort(
        key=lambda item: (
            item["sort"],
            item["display"].casefold() if isinstance(item["display"], str) else "",
        ),
        reverse=True,
    )

    lines = [f"Импорт из VK по группам за последние {days} дн.:"]
    if entries:
        lines.append(f"Групп в отчёте: {len(entries)}")
        summary_bits = [
            f"{label}: {value}" for label, value in totals.items() if value
        ]
        if summary_bits:
            lines.append("Итого: " + ", ".join(summary_bits))

    display_entries = entries[:max_rows]
    for idx, item in enumerate(display_entries, start=1):
        metrics_line = [
            f"{label}: {value}" for label, value in item["metrics"].items() if value or value == 0
        ]
        extras = dict(item["extras"])
        timestamp = item.get("timestamp")
        for label, value in extras.items():
            metrics_line.append(f"{label}: {value}")
        if timestamp is not None:
            metrics_line.append(
                "посл. импорт: "
                + timestamp.astimezone(tz).strftime("%Y-%m-%d %H:%M")
            )
        if not metrics_line:
            metrics_line.append("нет числовых данных")
        lines.append(f"{idx}. {item['display']}: {', '.join(metrics_line)}")

    if len(entries) > len(display_entries):
        lines.append(f"… и ещё {len(entries) - len(display_entries)} групп(ы).")
    if not entries:
        lines.append("Нет данных за указанный период.")
    return lines


def _format_imp_daily_report(
    records: Sequence[Mapping[str, Any]],
    *,
    days: int,
) -> list[str]:
    metric_defs: Sequence[tuple[str, Sequence[str]]] = [
        ("Импорт", ("imported", "imported_count", "imports", "matches", "added")),
        ("Отклонено", ("rejected", "rejected_count", "missed", "misses")),
        ("Дубликаты", ("duplicates", "duplicates_count")),
        ("Пропущено", ("skipped", "skipped_count", "auto_skipped")),
        (
            "На проверке",
            (
                "pending",
                "pending_count",
                "queue",
                "queued",
                "needs_review",
                "needs_review_count",
            ),
        ),
        ("Групп", ("groups", "groups_count", "group_count")),
    ]
    skip_fields: set[str] = {
        "window_days",
        "days",
        "period_days",
        "span_days",
        "bucket",
        "bucket_day",
        "day",
        "date",
        "ts",
        "at",
        "label",
    }
    time_fields = ("day", "date", "bucket_day", "bucket", "ts", "at", "label")
    tz = ZoneInfo(VK_WEEK_EDIT_TZ)
    entries: list[dict[str, Any]] = []
    totals: dict[str, int] = {label: 0 for label, _ in metric_defs}
    for record in records:
        metrics, used_columns = _collect_report_metrics(record, metric_defs)
        for label, value in metrics.items():
            totals[label] = totals.get(label, 0) + value
        extras = _collect_extra_numeric_fields(
            record,
            skip_fields=skip_fields,
            used_columns=used_columns,
        )
        sort_dt = _guess_record_datetime(record, time_fields)
        label_value = None
        if sort_dt is not None:
            label_value = sort_dt.astimezone(tz).date().isoformat()
        if not label_value:
            for field in time_fields:
                raw = record.get(field)
                if isinstance(raw, str) and raw.strip():
                    label_value = raw.strip()
                    break
        entries.append(
            {
                "label": label_value or "—",
                "metrics": metrics,
                "extras": extras,
                "sort": sort_dt,
            }
        )

    entries.sort(
        key=lambda item: (
            item["sort"] or datetime.min.replace(tzinfo=timezone.utc),
            item["label"],
        ),
        reverse=True,
    )

    lines = [f"Импорт из VK по дням за последние {days} дн.:"]
    if entries:
        summary_bits = [
            f"{label}: {value}" for label, value in totals.items() if value
        ]
        if summary_bits:
            lines.append("Итого: " + ", ".join(summary_bits))

    for item in entries:
        bits = [
            f"{label}: {value}" for label, value in item["metrics"].items() if value or value == 0
        ]
        for label, value in item["extras"].items():
            bits.append(f"{label}: {value}")
        if not bits:
            bits.append("нет числовых данных")
        lines.append(f"{item['label']}: {', '.join(bits)}")

    if not entries:
        lines.append("Нет данных за указанный период.")
    return lines


async def _fetch_vk_import_view(
    view: str,
    *,
    days: int,
    limit: int | None = None,
    client: Any | None = None,
) -> list[Mapping[str, Any]]:
    supabase_client = client if client is not None else get_supabase_client()
    candidate_filters: tuple[str, ...] = (
        "window_days",
        "days",
        "period_days",
        "span_days",
    )
    last_error: Exception | None = None
    if supabase_client is not None:
        for column in (*candidate_filters, None):
            try:
                def _query(column: str | None = column) -> list[Mapping[str, Any]]:
                    query = supabase_client.table(view).select("*")
                    if column:
                        query = query.eq(column, days)
                    if limit is not None:
                        query = query.limit(limit)
                    result = query.execute()
                    data = getattr(result, "data", result)
                    return list(data or [])

                return await asyncio.to_thread(_query)
            except Exception as exc:  # pragma: no cover - network failure
                logging.debug(
                    "vk_import_report.supabase_query_failed view=%s column=%s error=%s",
                    view,
                    column,
                    exc,
                    exc_info=True,
                )
                last_error = exc
                continue

    supabase_disabled = os.getenv("SUPABASE_DISABLED") == "1"
    base_url = _get_normalized_supabase_url()
    if supabase_disabled or not (base_url and SUPABASE_KEY):
        if last_error is not None:
            raise last_error
        raise RuntimeError("Supabase client unavailable")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    url = f"{base_url}/rest/v1/{view}"
    timeout = httpx.Timeout(HTTP_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        for column in (*candidate_filters, None):
            params: dict[str, str] = {"select": "*"}
            if column:
                params[column] = f"eq.{days}"
            if limit is not None:
                params["limit"] = str(limit)
            try:
                response = await http_client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
            except Exception as exc:  # pragma: no cover - network failure
                logging.debug(
                    "vk_import_report.http_query_failed view=%s column=%s error=%s",
                    view,
                    column,
                    exc,
                    exc_info=True,
                )
                last_error = exc
                continue

    if last_error is not None:
        raise last_error
    return []


async def handle_imp_groups_30d(
    message: types.Message, db: Database, bot: Bot
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user or not user.is_superadmin:
        await bot.send_message(message.chat.id, "Not authorized")
        return

    client = get_supabase_client()
    supabase_disabled = os.getenv("SUPABASE_DISABLED") == "1"
    if client is None and (supabase_disabled or not (SUPABASE_URL and SUPABASE_KEY)):
        logging.warning("imp_groups_30d.supabase_unavailable")
        await bot.send_message(
            message.chat.id,
            "Supabase отключён или не настроен.",
        )
        return

    try:
        records = await _fetch_vk_import_view(
            "vk_import_by_group",
            days=30,
            client=client,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logging.exception("imp_groups_30d.fetch_failed")
        await bot.send_message(
            message.chat.id,
            f"Не удалось получить отчёт из Supabase: {exc}",
        )
        return

    logger.info(
        "imp_groups_30d.success user=%s groups=%s",
        message.from_user.id,
        len(records),
    )

    lines = _format_imp_groups_report(records, days=30)
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_imp_daily_14d(
    message: types.Message, db: Database, bot: Bot
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user or not user.is_superadmin:
        await bot.send_message(message.chat.id, "Not authorized")
        return

    client = get_supabase_client()
    supabase_disabled = os.getenv("SUPABASE_DISABLED") == "1"
    if client is None and (supabase_disabled or not (SUPABASE_URL and SUPABASE_KEY)):
        logging.warning("imp_daily_14d.supabase_unavailable")
        await bot.send_message(
            message.chat.id,
            "Supabase отключён или не настроен.",
        )
        return

    try:
        records = await _fetch_vk_import_view(
            "vk_import_daily",
            days=14,
            client=client,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logging.exception("imp_daily_14d.fetch_failed")
        await bot.send_message(
            message.chat.id,
            f"Не удалось получить отчёт из Supabase: {exc}",
        )
        return

    logger.info(
        "imp_daily_14d.success user=%s rows=%s",
        message.from_user.id,
        len(records),
    )

    lines = _format_imp_daily_report(records, days=14)
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_dumpdb(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(Channel))
        channels = result.scalars().all()
        tz_setting = await session.get(Setting, "tz_offset")
        catbox_setting = await session.get(Setting, "catbox_enabled")

    data = await dump_database(db)
    file = types.BufferedInputFile(data, filename="dump.sql")
    await bot.send_document(message.chat.id, file)
    token_exists = os.path.exists(TELEGRAPH_TOKEN_FILE)
    if token_exists:
        with open(TELEGRAPH_TOKEN_FILE, "rb") as f:
            token_file = types.BufferedInputFile(
                f.read(), filename="telegraph_token.txt"
            )
        await bot.send_document(message.chat.id, token_file)

    lines = ["Channels:"]
    for ch in channels:
        roles: list[str] = []
        if ch.is_registered:
            roles.append("announcement")
        if ch.is_asset:
            roles.append("asset")
        if ch.daily_time:
            roles.append(f"daily {ch.daily_time}")
        title = ch.title or ch.username or str(ch.channel_id)
        lines.append(f"- {title}: {', '.join(roles) if roles else 'admin'}")

    lines.append("")
    lines.append("To restore on another server:")
    step = 1
    lines.append(f"{step}. Start the bot and send /restore with the dump file.")
    step += 1
    if tz_setting:
        lines.append(f"{step}. Current timezone: {tz_setting.value}")
        step += 1
    lines.append(f"{step}. Add the bot as admin to the channels listed above.")
    step += 1
    if token_exists:
        lines.append(
            f"{step}. Copy telegraph_token.txt to {TELEGRAPH_TOKEN_FILE} before first run."
        )
        step += 1
    if catbox_setting and catbox_setting.value == "1":
        lines.append(f"{step}. Run /images to enable photo uploads.")

    await bot.send_message(message.chat.id, "\n".join(lines))


def _coerce_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(stripped)
    return value


async def handle_tourist_export(message: types.Message, db: Database, bot: Bot) -> None:
    args = shlex.split(message.text or "")[1:]
    parser = argparse.ArgumentParser(prog="/tourist_export", add_help=False)
    parser.add_argument("--period")
    try:
        opts, extra = parser.parse_known_args(args)
    except SystemExit:
        await bot.send_message(message.chat.id, "Invalid arguments")
        return
    period_arg = opts.period
    if not period_arg and extra:
        token = extra[0]
        if token.startswith("period="):
            period_arg = token.split("=", 1)[1]
        else:
            period_arg = token
    start_date, end_date = (None, None)
    if period_arg:
        start_date, end_date = parse_period_range(period_arg)
        if not start_date and not end_date:
            await bot.send_message(message.chat.id, "Invalid period")
            return

    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not _user_can_label_event(user):
            await bot.send_message(message.chat.id, "Not authorized")
            return

        logger.info(
            "tourist_export.request user=%s period=%s",
            message.from_user.id,
            period_arg or "",
        )

        pragma_rows = await session.execute(text("PRAGMA table_info('event')"))
        event_columns = [row[1] for row in pragma_rows]
        base_fields = [
            "id",
            "title",
            "description",
            "festival",
            "date",
            "end_date",
            "time",
            "location_name",
            "location_address",
            "city",
            "ticket_price_min",
            "ticket_price_max",
            "ticket_link",
            "event_type",
            "emoji",
            "is_free",
            "pushkin_card",
            "telegraph_url",
            "source_post_url",
            "source_vk_post_url",
            "ics_url",
            "topics",
            "photo_urls",
        ]
        available_fields = [col for col in base_fields if col in event_columns]
        tourist_fields = [col for col in event_columns if col.startswith("tourist_")]
        seen: set[str] = set()
        selected_columns: list[str] = []
        for col in available_fields + tourist_fields:
            if col in seen:
                continue
            seen.add(col)
            selected_columns.append(col)
        if "id" not in seen and "id" in event_columns:
            selected_columns.insert(0, "id")
        if not selected_columns:
            await bot.send_message(message.chat.id, "No exportable fields")
            return

        query = text("SELECT {} FROM event".format(", ".join(selected_columns)))
        rows = await session.execute(query)
        records: list[dict[str, Any]] = []
        for row in rows:
            mapping = dict(row._mapping)
            start = parse_iso_date(str(mapping.get("date", "") or ""))
            end = parse_iso_date(str(mapping.get("end_date", "") or ""))
            if not end:
                end = start
            if start_date and end and end < start_date:
                continue
            if end_date and start and start > end_date:
                continue
            record: dict[str, Any] = {}
            for col in selected_columns:
                record[col] = _coerce_jsonish(mapping.get(col))
            records.append({
                "_sort": (
                    start or date.min,
                    mapping.get("id") or 0,
                ),
                "data": record,
            })

    logger.info(
        "tourist_export.start user=%s period=%s count=%s",
        message.from_user.id,
        period_arg or "",
        len(records),
    )

    if not records:
        await bot.send_message(message.chat.id, "No events found")
        return

    records.sort(key=lambda item: item["_sort"])
    lines = [json.dumps(item["data"], ensure_ascii=False) for item in records]
    payload = "\n".join(lines).encode("utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename_bits = ["tourist_export", timestamp]
    if start_date:
        filename_bits.insert(1, start_date.isoformat())
    if end_date and (not start_date or end_date != start_date):
        filename_bits.insert(2 if start_date else 1, end_date.isoformat())
    filename = "_".join(filename_bits) + ".jsonl"

    max_buffer = 45 * 1024 * 1024
    if len(payload) <= max_buffer:
        file = types.BufferedInputFile(payload, filename=filename)
        await bot.send_document(message.chat.id, file)
    else:
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".jsonl") as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            file = types.FSInputFile(tmp_path, filename=filename)
            await bot.send_document(message.chat.id, file)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(Exception):
                    os.remove(tmp_path)

    logger.info(
        "tourist_export.done user=%s count=%s bytes=%s",
        message.from_user.id,
        len(records),
        len(payload),
    )


async def handle_telegraph_fix_author(message: types.Message, db: Database, bot: Bot):
    await bot.send_message(
        message.chat.id,
        "Начинаю проставлять автора на всех Telegraph-страницах…",
    )
    token = get_telegraph_token()
    if not token:
        await bot.send_message(message.chat.id, "Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    pages: list[tuple[str, str]] = []
    async with db.get_session() as session:
        result = await session.execute(
            select(Event.title, Event.telegraph_path).where(
                Event.telegraph_path.is_not(None)
            )
        )
        pages.extend(result.all())
        result = await session.execute(
            select(Festival.name, Festival.telegraph_path).where(
                Festival.telegraph_path.is_not(None)
            )
        )
        pages.extend(result.all())
        result = await session.execute(select(MonthPage))
        for mp in result.scalars().all():
            pages.append((f"Month {mp.month}", mp.path))
            if mp.path2:
                pages.append((f"Month {mp.month} (2)", mp.path2))
        result = await session.execute(select(WeekendPage.start, WeekendPage.path))
        pages.extend([(f"Weekend {s}", p) for s, p in result.all()])

    updated: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    start = _time.perf_counter()
    for title, path in pages:
        try:
            await telegraph_edit_page(tg, path, title=title, caller="festival_build")
            updated.append((title, path))
        except Exception as e:  # pragma: no cover - network errors
            errors.append((title, str(e)))
        await asyncio.sleep(random.uniform(0.7, 1.2))
    dur = _time.perf_counter() - start
    lines = [
        f"Готово за {dur:.1f}с. Обновлено: {len(updated)}, ошибок: {len(errors)}"
    ]
    lines += [f"✓ {t} — https://telegra.ph/{p}" for t, p in updated[:50]]
    if errors:
        lines.append("\nОшибки:")
        lines += [f"✗ {t} — {err}" for t, err in errors[:50]]
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_restore(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    document = message.document
    if not document and message.reply_to_message:
        document = message.reply_to_message.document
    if not document:
        await bot.send_message(message.chat.id, "Attach dump file")
        return
    bio = BytesIO()
    await bot.download(document.file_id, destination=bio)
    await restore_database(bio.getvalue(), db)
    await bot.send_message(message.chat.id, "Database restored")
async def handle_edit_message(message: types.Message, db: Database, bot: Bot):
    state = editing_sessions.get(message.from_user.id)
    if not state:
        return
    eid, field = state
    if field is None:
        return
    value = (message.text or message.caption or "").strip()
    if field == "ticket_link" and value in {"", "-"}:
        value = ""
    if not value and field != "ticket_link":
        await bot.send_message(message.chat.id, "No text supplied")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        event = await session.get(Event, eid)
        if not event or (user and user.blocked) or (
            user and user.is_partner and event.creator_id != user.user_id
        ):
            await bot.send_message(message.chat.id, "Event not found" if not event else "Not authorized")
            del editing_sessions[message.from_user.id]
            return
        old_date = event.date.split("..", 1)[0]
        old_month = old_date[:7]
        old_fest = event.festival
        topics_meta: tuple[list[str], int, str | None, bool] | None = None
        if field in {"ticket_price_min", "ticket_price_max"}:
            try:
                setattr(event, field, int(value))
            except ValueError:
                await bot.send_message(message.chat.id, "Invalid number")
                return
        else:
            if field == "search_digest":
                event.search_digest = value.strip() or None
            elif field in {"is_free", "pushkin_card", "silent"}:
                bool_val = parse_bool_text(value)
                if bool_val is None:
                    await bot.send_message(message.chat.id, "Invalid boolean")
                    return
                setattr(event, field, bool_val)
            elif field == "ticket_link":
                if value == "":
                    setattr(event, field, None)
                elif is_tg_folder_link(value):
                    await bot.send_message(
                        message.chat.id,
                        "Это ссылка на папку Telegram, не на регистрацию",
                    )
                    return
                else:
                    setattr(event, field, value)
                event.vk_ticket_short_url = None
                event.vk_ticket_short_key = None
            else:
                setattr(event, field, value)
        if field in {"title", "description", "source_text"} and not event.topics_manual:
            topics_meta = await assign_event_topics(event)
        await session.commit()
        if topics_meta:
            topics_list, text_len, error_text, manual_flag = topics_meta
            if manual_flag:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s manual=True",
                    event.id,
                    text_len,
                    topics_list,
                )
            elif error_text:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s error=%s",
                    event.id,
                    text_len,
                    topics_list,
                    error_text,
                )
            else:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s",
                    event.id,
                    text_len,
                    topics_list,
                )
        new_date = event.date.split("..", 1)[0]
        new_month = new_date[:7]
        new_fest = event.festival
    results = await schedule_event_update_tasks(db, event)
    await publish_event_progress(event, db, bot, message.chat.id, results)
    if field == "city":
        page_url = None
        async with db.get_session() as session:
            page = await session.get(MonthPage, new_month)
            page_url = page.url if page else None
        markup = None
        if page_url:
            label = f"Открыть {month_name_prepositional(new_month)}"
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[[types.InlineKeyboardButton(text=label, url=page_url)]]
            )
        await bot.send_message(
            message.chat.id,
            f"Город обновлён: {event.city}",
            reply_markup=markup,
        )
    if old_fest:
        await sync_festival_page(db, old_fest)
        await sync_festival_vk_post(db, old_fest, bot)
    if new_fest and new_fest != old_fest:
        await sync_festival_page(db, new_fest)
        await sync_festival_vk_post(db, new_fest, bot)
    if event.source_vk_post_url:
        try:
            await sync_vk_source_post(
                event,
                event.source_text,
                db,
                bot,
                ics_url=event.ics_url,
                append_text=False,
            )
        except Exception as e:
            logging.error("failed to sync VK source post: %s", e)
    editing_sessions[message.from_user.id] = (eid, None)
    await show_edit_menu(message.from_user.id, event, bot)


async def handle_dom_iskusstv_start(message: types.Message, db: Database, bot: Bot):
    await bot.send_message(
        message.chat.id,
        "🎭 Отправьте ссылку на событие Дом искусств (domiskusstv.ru).\n"
        "Бот автоматически распознает ссылку и запустит парсер.\n"
        "Пример: https://дом-искусств.рф/events/...",
    )


async def handle_add_event_start(message: types.Message, db: Database, bot: Bot):
    """Initiate event creation via the menu."""
    logging.info("handle_add_event_start from user %s", message.from_user.id)
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    add_event_sessions[message.from_user.id] = "event"
    logging.info(
        "handle_add_event_start session opened for user %s", message.from_user.id
    )
    await bot.send_message(message.chat.id, "Send event text and optional photo")


async def handle_add_festival_start(message: types.Message, db: Database, bot: Bot):
    """Initiate festival creation via the menu."""
    logging.info("handle_add_festival_start from user %s", message.from_user.id)
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    add_event_sessions[message.from_user.id] = "festival"
    logging.info(
        "handle_add_festival_start session opened for user %s",
        message.from_user.id,
    )
    await bot.send_message(
        message.chat.id,
        "Пришлите текст фестиваля или ссылку на его сайт…",
    )


async def handle_vk_link_command(message: types.Message, db: Database, bot: Bot):
    parts = (message.text or "").split(maxsplit=2)
    logging.info("handle_vk_link_command start: user=%s", message.from_user.id)
    if len(parts) < 3:
        await bot.send_message(
            message.chat.id, "Usage: /vklink <event_id> <VK post link>"
        )
        return
    try:
        eid = int(parts[1])
    except ValueError:
        await bot.send_message(message.chat.id, "Invalid event id")
        return
    link = parts[2].strip()
    if not is_vk_wall_url(link):
        await bot.send_message(message.chat.id, "Invalid link")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        event = await session.get(Event, eid)
        if not event:
            await bot.send_message(message.chat.id, "Event not found")
            return
        if event.creator_id != message.from_user.id and not (user and user.is_superadmin):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        event.source_post_url = link
        await session.commit()
    await bot.send_message(message.chat.id, "Link saved")

async def handle_daily_time_message(message: types.Message, db: Database, bot: Bot):
    cid = daily_time_sessions.get(message.from_user.id)
    if not cid:
        return
    value = (message.text or "").strip()
    if not re.match(r"^\d{1,2}:\d{2}$", value):
        await bot.send_message(message.chat.id, "Invalid time")
        return
    if len(value.split(":")[0]) == 1:
        value = f"0{value}"
    async with db.get_session() as session:
        ch = await session.get(Channel, cid)
        if ch:
            ch.daily_time = value
            await session.commit()
    del daily_time_sessions[message.from_user.id]
    await bot.send_message(message.chat.id, f"Time set to {value}")


async def handle_vk_group_message(message: types.Message, db: Database, bot: Bot):
    if message.from_user.id not in vk_group_sessions:
        return
    value = (message.text or "").strip()
    if value.lower() == "off":
        await set_vk_group_id(db, None)
        await bot.send_message(message.chat.id, "VK posting disabled")
    else:
        await set_vk_group_id(db, value)
        await bot.send_message(message.chat.id, f"VK group set to {value}")
    vk_group_sessions.discard(message.from_user.id)


async def handle_vk_time_message(message: types.Message, db: Database, bot: Bot):
    typ = vk_time_sessions.get(message.from_user.id)
    if not typ:
        return
    value = (message.text or "").strip()
    if not re.match(r"^\d{1,2}:\d{2}$", value):
        await bot.send_message(message.chat.id, "Invalid time")
        return
    if len(value.split(":")[0]) == 1:
        value = f"0{value}"
    if typ == "today":
        await set_vk_time_today(db, value)
    else:
        await set_vk_time_added(db, value)
    vk_time_sessions.pop(message.from_user.id, None)
    await bot.send_message(message.chat.id, f"Time set to {value}")


async def handle_vk_command(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        await bot.send_message(message.chat.id, "Access denied")
        return
    if not (VK_USER_TOKEN or VK_TOKEN or VK_TOKEN_AFISHA):
        if os.getenv("DEV_MODE") != "1":
            await bot.send_message(message.chat.id, "VK token not configured")
            return
    buttons = [
        [types.KeyboardButton(text=VK_BTN_ADD_SOURCE)],
        [types.KeyboardButton(text=VK_BTN_LIST_SOURCES)],
        [
            types.KeyboardButton(text=VK_BTN_CHECK_EVENTS),
            types.KeyboardButton(text=VK_BTN_QUEUE_SUMMARY),
        ],
    ]
    markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
    await bot.send_message(message.chat.id, "VK мониторинг", reply_markup=markup)




async def handle_vk_add_start(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        await bot.send_message(message.chat.id, "Access denied")
        return
    vk_add_source_sessions.add(message.from_user.id)
    await bot.send_message(
        message.chat.id,
        "Отправьте ссылку или скриннейм, опционально локацию и время через |",
    )


async def handle_pyramida_start(message: types.Message, db: Database, bot: Bot) -> None:
    """Handle Pyramida button click - start waiting for URL input."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        await bot.send_message(message.chat.id, "Access denied")
        return
    pyramida_input_sessions.add(message.from_user.id)
    await bot.send_message(
        message.chat.id,
        "🔮 Отправьте текст со ссылками pyramida.info/tickets/...\n"
        "Я извлеку все ссылки и парсю события.",
    )


async def handle_pyramida_input(message: types.Message, db: Database, bot: Bot) -> None:
    """Handle text input with Pyramida URLs."""
    if message.from_user.id not in pyramida_input_sessions:
        return
    pyramida_input_sessions.discard(message.from_user.id)
    
    text = (message.text or "").strip()
    if not text:
        await bot.send_message(message.chat.id, "❌ Пустой ввод")
        return
    
    # Extract URLs
    from source_parsing.pyramida import (
        extract_pyramida_urls,
        run_pyramida_kaggle_kernel,
        parse_pyramida_output,
        process_pyramida_events,
    )
    
    urls = extract_pyramida_urls(text)
    if not urls:
        await bot.send_message(message.chat.id, "❌ Не найдены ссылки pyramida.info/tickets/")
        return
    
    status_msg = await bot.send_message(message.chat.id, f"🔮 Найдено {len(urls)} ссылок. Запускаю Kaggle...")
    
    async def _status_cb(text: str):
        try:
            await bot.edit_message_text(
                f"🔮 {text}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass
    
    # Run Kaggle
    try:
        status, output_files, duration = await run_pyramida_kaggle_kernel(urls, status_callback=_status_cb)
    except Exception as e:
        logging.exception("pyramida_input: kaggle failed")
        await bot.send_message(message.chat.id, f"❌ Ошибка Kaggle: {e}")
        return
    
    if status != "complete":
        await bot.send_message(message.chat.id, f"❌ Kaggle завершился с ошибкой: {status}")
        return
    
    await bot.send_message(message.chat.id, f"✅ Kaggle завершён за {duration:.1f}с. Обрабатываю...")
    
    # Send JSON files to chat
    for file_path in output_files:
        try:
            await bot.send_document(
                message.chat.id,
                types.FSInputFile(file_path),
                caption=f"📄 {os.path.basename(file_path)}"
            )
        except Exception as e:
            logging.error(f"Failed to send JSON file {file_path}: {e}")
    
    # Parse events
    try:
        events = parse_pyramida_output(output_files)
    except Exception as e:
        logging.exception("pyramida_input: parse failed")
        await bot.send_message(message.chat.id, f"❌ Ошибка парсинга: {e}")
        return
    
    if not events:
        await bot.send_message(message.chat.id, "⚠️ Не найдено событий в результатах Kaggle")
        return
    
    await bot.send_message(message.chat.id, f"📝 Обрабатываю {len(events)} событий...")
    
    # Process events
    try:
        stats = await process_pyramida_events(
            db,
            bot,
            events,
            chat_id=message.chat.id,
            skip_pages_rebuild=True,
        )
    except Exception as e:
        logging.exception("pyramida_input: processing failed")
        await bot.send_message(message.chat.id, f"❌ Ошибка обработки: {e}")
        return
    
    # Summary
    summary_lines = [
        "🔮 **Pyramida импорт завершён**",
        f"✅ Добавлено: {stats.new_added}",
    ]
    if stats.ticket_updated:
        summary_lines.append(f"🔄 Обновлено: {stats.ticket_updated}")
    if stats.failed:
        summary_lines.append(f"❌ Ошибок: {stats.failed}")
    
    await bot.send_message(message.chat.id, "\n".join(summary_lines), parse_mode="Markdown")


async def handle_dom_iskusstv_start(message: types.Message, db: Database, bot: Bot) -> None:
    """Handle Dom Iskusstv button click - start waiting for URL input."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not has_admin_access(user):
        await bot.send_message(message.chat.id, "Access denied")
        return
    dom_iskusstv_input_sessions.add(message.from_user.id)
    await bot.send_message(
        message.chat.id,
        "🏛 Отправьте ссылку на спецпроект Дом искусств\n"
        "(например: https://домискусств.рф/skazka или https://xn--b1admiilxbaki.xn--p1ai/aladdin)",
    )


async def handle_dom_iskusstv_input(message: types.Message, db: Database, bot: Bot) -> None:
    """Handle text input with Dom Iskusstv URLs."""
    if message.from_user.id not in dom_iskusstv_input_sessions:
        return
    dom_iskusstv_input_sessions.discard(message.from_user.id)
    
    text = (message.text or "").strip()
    if not text:
        await bot.send_message(message.chat.id, "❌ Пустой ввод")
        return
    
    # Extract URLs
    from source_parsing.dom_iskusstv import (
        extract_dom_iskusstv_urls,
        run_dom_iskusstv_kaggle_kernel,
        parse_dom_iskusstv_output,
        process_dom_iskusstv_events,
    )
    
    urls = extract_dom_iskusstv_urls(text)
    if not urls:
        await bot.send_message(message.chat.id, "❌ Не найдены ссылки на домискусств.рф")
        return
    
    status_msg = await bot.send_message(message.chat.id, f"🏛 Найдено {len(urls)} ссылок. Запускаю Kaggle...")
    
    async def _status_cb(text: str):
        try:
            await bot.edit_message_text(
                f"🏛 {text}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass
    
    # Run Kaggle
    try:
        status, output_files, duration = await run_dom_iskusstv_kaggle_kernel(urls, status_callback=_status_cb)
    except Exception as e:
        logging.exception("dom_iskusstv_input: kaggle failed")
        await bot.send_message(message.chat.id, f"❌ Ошибка Kaggle: {e}")
        return
    
    if status != "complete":
        await bot.send_message(message.chat.id, f"❌ Kaggle завершился с ошибкой: {status}")
        # Send logs if available
        for file_path in output_files:
            if file_path.endswith(".log"):
                try:
                    await bot.send_document(
                        message.chat.id,
                        types.FSInputFile(file_path),
                        caption=f"📝 Log: {os.path.basename(file_path)}"
                    )
                except Exception as e:
                    logging.error(f"Failed to send log file {file_path}: {e}")
        return
    
    await bot.send_message(message.chat.id, f"✅ Kaggle завершён за {duration:.1f}с. Обрабатываю...")
    
    # Send JSON files to chat
    for file_path in output_files:
        try:
            await bot.send_document(
                message.chat.id,
                types.FSInputFile(file_path),
                caption=f"📄 {os.path.basename(file_path)}"
            )
        except Exception as e:
            logging.error(f"Failed to send JSON file {file_path}: {e}")
    
    # Parse events
    try:
        events = parse_dom_iskusstv_output(output_files)
    except Exception as e:
        logging.exception("dom_iskusstv_input: parse failed")
        await bot.send_message(message.chat.id, f"❌ Ошибка парсинга: {e}")
        return
    
    if not events:
        await bot.send_message(message.chat.id, "⚠️ Не найдено событий в результатах Kaggle")
        return
    
    await bot.send_message(message.chat.id, f"📝 Обрабатываю {len(events)} событий...")
    
    # Process events
    try:
        stats = await process_dom_iskusstv_events(
            db,
            bot,
            events,
            chat_id=message.chat.id,
            skip_pages_rebuild=True,
        )
    except Exception as e:
        logging.exception("dom_iskusstv_input: processing failed")
        await bot.send_message(message.chat.id, f"❌ Ошибка обработки: {e}")
        return
    
    # Summary
    summary_lines = [
        "🏛 **Дом искусств импорт завершён**",
        f"✅ Добавлено: {stats.new_added}",
    ]
    if stats.ticket_updated:
        summary_lines.append(f"🔄 Обновлено: {stats.ticket_updated}")
    if stats.failed:
        summary_lines.append(f"❌ Ошибок: {stats.failed}")

    # Add Telegraph links
    all_event_ids = stats.added_event_ids + stats.updated_event_ids
    if all_event_ids:
        from source_parsing.handlers import build_added_event_info
        event_infos = []
        for eid in all_event_ids:
            # build_added_event_info is async
            info = await build_added_event_info(db, eid, "dom_iskusstv")
            if info:
                event_infos.append(info)
        
        if event_infos:
            summary_lines.append("")
            summary_lines.append("🔗 **Ссылки:**")
            for info in event_infos:
                # Escape for Legacy Markdown
                safe_title = info.title.replace("[", "\\[").replace("]", "\\]")
                summary_lines.append(f"• [{safe_title}]({info.telegraph_url})")
    
    await bot.send_message(message.chat.id, "\n".join(summary_lines), parse_mode="Markdown")


async def handle_vk_add_message(message: types.Message, db: Database, bot: Bot) -> None:
    if message.from_user.id not in vk_add_source_sessions:
        return
    vk_add_source_sessions.discard(message.from_user.id)
    text = (message.text or "").strip()
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if not parts:
        await bot.send_message(message.chat.id, "Пустой ввод")
        return
    screen = parts[-1]
    location = None
    default_time = None
    default_ticket_link = None
    for p in parts[:-1]:
        if re.match(r"^\d{1,2}:\d{2}$", p):
            default_time = p if len(p.split(":")[0]) == 2 else f"0{p}"
        elif p.startswith("http://") or p.startswith("https://"):
            default_ticket_link = p
        else:
            location = p
    try:
        gid, name, screen_name = await vk_resolve_group(screen)
    except Exception as e:
        logging.exception("vk_resolve_group failed")
        await bot.send_message(
            message.chat.id,
            "Не удалось определить сообщество.\n"
            "Проверьте ссылку/скриннейм (пример: https://vk.com/muzteatr39).\n"
            f"Технические детали: {e}.",
        )
        return
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (gid, screen_name, name, location, default_time, default_ticket_link),
        )
        await conn.commit()
    extra = []
    if location:
        extra.append(location)
    if default_time:
        extra.append(default_time)
    if default_ticket_link:
        extra.append(default_ticket_link)
    suffix = f" — {', '.join(extra)}" if extra else ""
    await bot.send_message(
        message.chat.id,
        f"Добавлено: {name} (vk.com/{screen_name}){suffix}",
    )


async def _fetch_vk_sources(
    db: Database,
) -> list[
    tuple[
        int,
        int,
        str,
        str,
        str | None,
        str | None,
        str | None,
        Any,
        Any,
    ]
]:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            """
            SELECT
                s.id,
                s.group_id,
                s.screen_name,
                s.name,
                s.location,
                s.default_time,
                s.default_ticket_link,
                c.updated_at,
                c.checked_at
            FROM vk_source AS s
            LEFT JOIN vk_crawl_cursor AS c ON c.group_id = s.group_id
            ORDER BY s.id
            """
        )
        rows = await cursor.fetchall()
    return rows


VK_SOURCES_PER_PAGE = 10
VK_STATUS_LABELS: Sequence[tuple[str, str]] = (
    ("pending", "Pending"),
    ("skipped", "Skipped"),
    ("imported", "Imported"),
    ("rejected", "Rejected"),
)


def _zero_vk_status_counts() -> dict[str, int]:
    return {key: 0 for key, _ in VK_STATUS_LABELS}


async def _fetch_vk_inbox_counts(db: Database) -> dict[int, dict[str, int]]:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT group_id, status, COUNT(*) FROM vk_inbox GROUP BY group_id, status"
        )
        rows = await cursor.fetchall()
    counts: dict[int, dict[str, int]] = {}
    for gid, status, amount in rows:
        bucket = counts.get(gid)
        if bucket is None:
            bucket = _zero_vk_status_counts()
            counts[gid] = bucket
        if status in bucket:
            bucket[status] = amount
    return counts


async def handle_vk_list(
    message: types.Message,
    db: Database,
    bot: Bot,
    edit: types.Message | None = None,
    page: int = 1,
) -> None:
    rows = await _fetch_vk_sources(db)
    if not rows:
        if edit:
            await edit.edit_text("Список пуст")
        else:
            await bot.send_message(message.chat.id, "Список пуст")
        return
    total_pages = max(1, (len(rows) + VK_SOURCES_PER_PAGE - 1) // VK_SOURCES_PER_PAGE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * VK_SOURCES_PER_PAGE
    end = start + VK_SOURCES_PER_PAGE
    page_rows = rows[start:end]
    inbox_counts = await _fetch_vk_inbox_counts(db)
    page_items: list[
        tuple[
            int,
            tuple[int, int, str, str, str | None, str | None, str | None, Any, Any],
            dict[str, int],
        ]
    ] = []
    for offset, row in enumerate(page_rows, start=start + 1):
        rid, gid, screen, name, loc, dtime, default_ticket_link, updated_at, checked_at = row
        counts = inbox_counts.get(gid)
        if counts is None:
            counts = _zero_vk_status_counts()
        else:
            counts = dict(counts)
        page_items.append((offset, row, counts))

    if page_items:
        count_widths = {}
        for key, label in VK_STATUS_LABELS:
            max_value_len = max(len(str(item[2][key])) for item in page_items)
            base_width = max(len(label), max_value_len)
            count_widths[key] = max(1, math.ceil(base_width * 1.87))
    else:
        count_widths = {}
        for key, label in VK_STATUS_LABELS:
            base_width = len(label)
            count_widths[key] = max(1, math.ceil(base_width * 1.87))

    status_header_parts = [f" {label} " for _, label in VK_STATUS_LABELS]
    status_header_line = "|".join(status_header_parts)

    lines: list[str] = []
    buttons: list[list[types.InlineKeyboardButton]] = []
    def _format_timestamp(value: Any) -> str:
        if value in (None, "", 0):
            return "-"
        try:
            if isinstance(value, (int, float)):
                if value <= 0:
                    raise ValueError("non-positive timestamp")
                parsed = datetime.fromtimestamp(value, tz=timezone.utc)
            else:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, OSError):
            return str(value)

    for offset, row, counts in page_items:
        rid, gid, screen, name, loc, dtime, default_ticket_link, updated_at, checked_at = row
        info_parts = [f"id={gid}"]
        if loc:
            info_parts.append(loc)
        if default_ticket_link:
            info_parts.append(f"билеты: {default_ticket_link}")
        info = ", ".join(info_parts)
        human_checked = _format_timestamp(checked_at)
        human_updated = _format_timestamp(updated_at)
        lines.append(
            f"{offset}. {name} (vk.com/{screen}) — {info}, типовое время: {dtime or '-'}, последнее сканирование: {human_checked}, последний найденный пост: {human_updated}"
        )
        value_parts = [
            f" {counts[key]:^{count_widths[key]}} "
            for key, _ in VK_STATUS_LABELS
        ]
        lines.append(status_header_line)
        lines.append("|".join(value_parts))
        buttons.append(
            [
                types.InlineKeyboardButton(
                    text=f"❌ {offset}", callback_data=f"vkdel:{rid}:{page}"
                ),
                types.InlineKeyboardButton(
                    text=f"⚙️ {offset}", callback_data=f"vkset:{rid}:{page}"
                ),
            ]
        )
        buttons.append(
            [
                types.InlineKeyboardButton(
                    text=f"🕒 {offset}", callback_data=f"vkdt:{rid}:{page}"
                ),
                types.InlineKeyboardButton(
                    text=f"🎟 {offset}", callback_data=f"vklink:{rid}:{page}"
                ),
                types.InlineKeyboardButton(
                    text=f"📍 {offset}", callback_data=f"vkloc:{rid}:{page}"
                ),
            ]
        )
        if counts["rejected"] > 0:
            buttons.append(
                [
                    types.InlineKeyboardButton(
                        text=f"🚫 Rejected: {counts['rejected']}",
                        callback_data=f"vkrejected:{gid}:{page}",
                    )
                ]
            )
    text = "\n".join(lines)
    if total_pages > 1:
        nav_row: list[types.InlineKeyboardButton] = []
        if page > 1:
            nav_row.append(
                types.InlineKeyboardButton(
                    text="◀️", callback_data=f"vksrcpage:{page - 1}"
                )
            )
        nav_row.append(
            types.InlineKeyboardButton(
                text=f"{page}/{total_pages}", callback_data=f"vksrcpage:{page}"
            )
        )
        if page < total_pages:
            nav_row.append(
                types.InlineKeyboardButton(
                    text="▶️", callback_data=f"vksrcpage:{page + 1}"
                )
            )
        buttons.append(nav_row)
    markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    if edit:
        await edit.edit_text(text, reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, text, reply_markup=markup)


async def handle_vk_list_page_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    try:
        page = int(callback.data.split(":", 1)[1])
    except Exception:
        await callback.answer()
        return
    await handle_vk_list(callback.message, db, bot, edit=callback.message, page=page)
    await callback.answer()


async def handle_vk_delete_callback(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    async with db.raw_conn() as conn:
        await conn.execute("DELETE FROM vk_source WHERE id=?", (vid,))
        await conn.commit()
    await callback.answer("Удалено")
    await handle_vk_list(callback.message, db, bot, edit=callback.message, page=page)


async def handle_vk_rejected_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":")
        group_id = int(parts[0])
    except Exception:
        await callback.answer()
        return

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, screen_name FROM vk_source WHERE group_id=?",
            (group_id,),
        )
        source_row = await cur.fetchone()
        await cur.close()
        cur = await conn.execute(
            """
            SELECT post_id
            FROM vk_inbox
            WHERE group_id=? AND status='rejected'
            ORDER BY
                COALESCE(created_at, '') DESC,
                id DESC
            LIMIT 30
            """,
            (group_id,),
        )
        rows = await cur.fetchall()
        await cur.close()

    if not rows:
        await callback.answer("Нет отклонённых постов", show_alert=True)
        return

    if source_row:
        name, screen = source_row
    else:
        name = None
        screen = None

    if name and screen:
        header = f"{name} (vk.com/{screen})"
    elif screen:
        header = f"vk.com/{screen}"
    elif name:
        header = name
    else:
        header = f"group {group_id}"

    links = [f"https://vk.com/wall-{group_id}_{post_id}" for (post_id,) in rows]
    message_text = "\n".join([f"🚫 Отклонённые посты — {header}"] + links)

    await bot.send_message(
        callback.message.chat.id,
        message_text,
        disable_web_page_preview=True,
    )
    await callback.answer()


async def handle_vk_settings_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, location, default_time, default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    if not row:
        await callback.answer("Источник не найден", show_alert=True)
        return
    name, location, default_time, default_ticket_link = row
    lines = [f"{name}"]
    lines.append(f"Локация: {location or 'не указана'}")
    lines.append(f"Типовое время: {default_time or 'не установлено'}")
    lines.append(f"Ссылка на билеты: {default_ticket_link or 'не указана'}")
    lines.append("Используйте кнопки 🕒, 🎟 и 📍 для изменений.")
    if callback.message:
        await bot.send_message(callback.message.chat.id, "\n".join(lines))
    await callback.answer()


async def handle_vk_ticket_link_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    vk_default_ticket_link_sessions[callback.from_user.id] = (
        VkDefaultTicketLinkSession(
            source_id=vid,
            page=page,
            message=callback.message,
        )
    )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    current = row[1] if row else None
    await bot.send_message(
        callback.message.chat.id,
        f"{name}: текущая ссылка на билеты — {current or 'не установлена'}. "
        "Отправьте ссылку, начинающуюся с http(s)://, или '-' для удаления.",
    )
    await callback.answer()


async def handle_vk_ticket_link_message(
    message: types.Message, db: Database, bot: Bot
) -> None:
    session = vk_default_ticket_link_sessions.pop(message.from_user.id, None)
    if not session:
        return
    vid = session.source_id
    text = (message.text or "").strip()
    if text in {"", "-"}:
        new_link: str | None = None
    else:
        if not re.match(r"^https?://", text, re.IGNORECASE):
            await bot.send_message(
                message.chat.id,
                "Неверный формат. Укажите ссылку, начинающуюся с http(s)://, или '-' для удаления.",
            )
            vk_default_ticket_link_sessions[message.from_user.id] = session
            return
        new_link = text
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_source SET default_ticket_link=? WHERE id=?",
            (new_link, vid),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT name FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    if new_link:
        msg = f"Ссылка на билеты для {name} установлена: {new_link}"
    else:
        msg = f"Ссылка на билеты для {name} удалена"
    await bot.send_message(message.chat.id, msg)
    if session.message:
        await handle_vk_list(
            session.message,
            db,
            bot,
            edit=session.message,
            page=session.page,
        )


async def handle_vk_location_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    vk_default_location_sessions[callback.from_user.id] = VkDefaultLocationSession(
        source_id=vid,
        page=page,
        message=callback.message,
    )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, location FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    current = row[1] if row else None
    await bot.send_message(
        callback.message.chat.id,
        f"{name}: текущая локация — {current or 'не указана'}. "
        "Отправьте новую локацию или '-' для удаления.",
    )
    await callback.answer()


async def handle_vk_location_message(
    message: types.Message, db: Database, bot: Bot
) -> None:
    session = vk_default_location_sessions.pop(message.from_user.id, None)
    if not session:
        return
    vid = session.source_id
    text = (message.text or "").strip()
    if text in {"", "-"}:
        new_location: str | None = None
    else:
        new_location = text
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_source SET location=? WHERE id=?",
            (new_location, vid),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT name FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    if new_location:
        msg = f"Локация для {name} установлена: {new_location}"
    else:
        msg = f"Локация для {name} удалена"
    await bot.send_message(message.chat.id, msg)
    if session.message:
        await handle_vk_list(
            session.message,
            db,
            bot,
            edit=session.message,
            page=session.page,
        )


async def handle_vk_dtime_callback(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    vk_default_time_sessions[callback.from_user.id] = VkDefaultTimeSession(
        source_id=vid,
        page=page,
        message=callback.message,
    )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, default_time FROM vk_source WHERE id=?", (vid,)
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    current = row[1] if row else None
    await bot.send_message(
        callback.message.chat.id,
        f"{name}: типовое время сейчас — {current or 'не установлено'}. "
        "Отправьте время в формате HH:MM или '-' для удаления.",
    )
    await callback.answer()


async def handle_vk_dtime_message(message: types.Message, db: Database, bot: Bot) -> None:
    session = vk_default_time_sessions.pop(message.from_user.id, None)
    if not session:
        return
    vid = session.source_id
    text = (message.text or "").strip()
    if text in {"", "-"}:
        new_time: str | None = None
    else:
        if not re.match(r"^\d{1,2}:\d{2}$", text):
            await bot.send_message(
                message.chat.id,
                "Неверный формат. Используйте HH:MM или '-' для удаления.",
            )
            return
        new_time = text if len(text.split(":")[0]) == 2 else f"0{text}"
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_source SET default_time=? WHERE id=?", (new_time, vid)
        )
        await conn.commit()
        cur = await conn.execute("SELECT name FROM vk_source WHERE id=?", (vid,))
        row = await cur.fetchone()
    name = row[0] if row else ""
    if new_time:
        msg = f"Типовое время для {name} установлено: {new_time}"
    else:
        msg = f"Типовое время для {name} удалено"
    await bot.send_message(message.chat.id, msg)
    if session.message:
        await handle_vk_list(
            session.message,
            db,
            bot,
            edit=session.message,
            page=session.page,
        )


async def send_vk_tmp_post(chat_id: int, batch: str, idx: int, total: int, db: Database, bot: Bot) -> None:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT text, photos, url FROM vk_tmp_post WHERE batch=? ORDER BY date DESC, post_id DESC LIMIT 1 OFFSET ?",
            (batch, idx),
        )
        row = await cursor.fetchone()
    if not row:
        await bot.send_message(chat_id, "Это был последний пост.")
        return
    text, photos_json, url = row
    photos = json.loads(photos_json) if photos_json else []
    if len(photos) >= 2:
        media = [types.InputMediaPhoto(media=p) for p in photos[:10]]
        try:
            await bot.send_media_group(chat_id, media)
        except Exception:
            await bot.send_photo(chat_id, photos[0])
    elif len(photos) == 1:
        await bot.send_photo(chat_id, photos[0])
    msg = (text or "").strip()
    if len(msg) > 3500:
        msg = msg[:3500] + "…"
    msg = msg + f"\n\n{url}"
    if idx + 1 < total:
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[[types.InlineKeyboardButton(text="Следующее ▶️", callback_data=f"vknext:{batch}:{idx+1}")]]
        )
        await bot.send_message(chat_id, msg, reply_markup=markup)
    else:
        await bot.send_message(chat_id, msg)
        await bot.send_message(chat_id, "Это был последний пост.")


async def handle_vk_check(message: types.Message, db: Database, bot: Bot) -> None:
    """Start VK inbox review."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user:
        await message.answer("Not authorized")
        return
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT batch_id FROM vk_review_batch
            WHERE operator_id=? AND finished_at IS NULL
            ORDER BY started_at DESC LIMIT 1
            """,
            (message.from_user.id,),
        )
        row = await cur.fetchone()
    if row:
        batch_id = row[0]
    else:
        batch_id = f"{int(unixtime.time())}:{message.from_user.id}"
        async with db.raw_conn() as conn:
            await conn.execute(
                "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
                (batch_id, message.from_user.id, ""),
            )
            await conn.commit()
    await _vkrev_show_next(message.chat.id, batch_id, message.from_user.id, db, bot)


async def handle_vk_crawl_now(message: types.Message, db: Database, bot: Bot) -> None:
    """Manually trigger VK crawling."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    text = message.text or ""
    try:
        tokens = shlex.split(text)
    except ValueError:
        await bot.send_message(
            message.chat.id,
            "Не удалось разобрать параметры команды",
        )
        return

    forced_backfill = False
    requested_backfill_days: int | None = None
    error_message: str | None = None

    for token in tokens:
        if not token.startswith("--backfill-days"):
            continue
        forced_backfill = True
        if token == "--backfill-days":
            error_message = "Используйте синтаксис --backfill-days=N"
            break
        if token.startswith("--backfill-days="):
            value = token.split("=", 1)[1]
            if not value:
                error_message = "Значение для --backfill-days отсутствует"
                break
            try:
                requested_backfill_days = int(value)
            except ValueError:
                error_message = "Значение --backfill-days должно быть целым числом"
                break
        else:
            error_message = "Используйте синтаксис --backfill-days=N"
            break

    if error_message:
        await bot.send_message(message.chat.id, error_message)
        return

    if forced_backfill and requested_backfill_days is not None and requested_backfill_days < 1:
        await bot.send_message(
            message.chat.id,
            "Значение --backfill-days должно быть положительным",
        )
        return

    stats = await vk_intake.crawl_once(
        db,
        broadcast=True,
        bot=bot,
        force_backfill=forced_backfill,
        backfill_days=requested_backfill_days if forced_backfill else None,
    )
    q = stats.get("queue", {})
    forced_note = ""
    if stats.get("forced_backfill"):
        used_days = stats.get("backfill_days_used") or vk_intake.VK_CRAWL_BACKFILL_DAYS
        requested_days = stats.get("backfill_days_requested")
        forced_note = f", режим: принудительный бэкафилл до {used_days} дн."
        if (
            requested_days is not None
            and requested_days != used_days
        ):
            forced_note += f" (запрошено {requested_days})"
    msg = (
        f"Проверено {stats['groups_checked']} сообществ, "
        f"просмотрено {stats['posts_scanned']} постов, "
        f"совпало {stats['matches']}, "
        f"дубликатов {stats['duplicates']}, "
        f"добавлено {stats['added']}, "
        f"всего постов {stats['inbox_total']} "
        f"(в очереди: {q.get('pending',0)}, locked: {q.get('locked',0)}, "
        f"skipped: {q.get('skipped',0)}, imported: {q.get('imported',0)}, "
        f"rejected: {q.get('rejected',0)})"
        f", страниц на группу: {'/'.join(str(p) for p in stats.get('pages_per_group', []))}, "
        f"перекрытие: {stats.get('overlap_sec')} сек"
        f"{forced_note}"
    )
    await bot.send_message(message.chat.id, msg)


async def handle_vk_queue(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user:
        await bot.send_message(message.chat.id, "Not authorized")
        return
    await vk_review.release_stale_locks(db)
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, COUNT(*) FROM vk_inbox GROUP BY status"
        )
        rows = await cur.fetchall()
    counts = {r[0]: r[1] for r in rows}
    lines = [
        f"pending: {counts.get('pending', 0)}",
        f"locked: {counts.get('locked', 0)}",
        f"skipped: {counts.get('skipped', 0)}",
        f"imported: {counts.get('imported', 0)}",
        f"rejected: {counts.get('rejected', 0)}",
    ]
    schedule_raw = os.getenv(
        "VK_CRAWL_TIMES_LOCAL", "05:15,09:15,13:15,17:15,21:15,22:45"
    )
    schedule_times = [part.strip() for part in schedule_raw.split(",") if part.strip()]
    schedule_line = ", ".join(schedule_times)
    schedule_tz = os.getenv("VK_CRAWL_TZ")
    if schedule_tz:
        schedule_line = f"{schedule_line} ({schedule_tz})"
    if schedule_line:
        lines.insert(0, f"Обновление базы: {schedule_line}")
    markup = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=VK_BTN_CHECK_EVENTS)]],
        resize_keyboard=True,
    )
    await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def handle_vk_requeue_imported(
    message: types.Message, db: Database, bot: Bot
) -> None:
    parts = message.text.split()
    n = 1
    if len(parts) > 1:
        try:
            n = int(parts[1])
        except ValueError:
            await bot.send_message(message.chat.id, "Usage: /vk_requeue_imported [N]")
            return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user:
        await bot.send_message(message.chat.id, "Not authorized")
        return
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT id FROM vk_inbox
            WHERE status='imported' AND review_batch IN (
                SELECT batch_id FROM vk_review_batch WHERE operator_id=?
            )
            ORDER BY id DESC LIMIT ?
            """,
            (message.from_user.id, n),
        )
        rows = await cur.fetchall()
        ids = [r[0] for r in rows]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            await conn.execute(
                f"""
                UPDATE vk_inbox
                SET status='pending', imported_event_id=NULL, review_batch=NULL,
                    locked_by=NULL, locked_at=NULL
                WHERE id IN ({placeholders})
                """,
                ids,
            )
            await conn.commit()
    await bot.send_message(message.chat.id, f"Requeued {len(ids)} item(s)")

async def _vkrev_queue_size(db: Database) -> int:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM vk_inbox WHERE status IN ('pending','skipped')",
        )
        (cnt,) = await cur.fetchone()
    return cnt


async def _vk_wall_get_items(
    group_id: int, post_id: int, db: Database, bot: Bot
) -> list[dict[str, Any]]:
    token: str | None = VK_SERVICE_TOKEN
    token_kind = "service"
    if not token:
        token = _vk_user_token()
        token_kind = "user"
    if not token:
        return []
    try:
        data = await _vk_api(
            "wall.getById",
            {"posts": f"-{group_id}_{post_id}"},
            db,
            bot,
            token=token,
            token_kind=token_kind,
            skip_captcha=True,
        )
    except VKAPIError as e:  # pragma: no cover
        logging.error(
            "wall.getById failed gid=%s post=%s actor=%s token=%s code=%s msg=%s",
            group_id,
            post_id,
            e.actor,
            e.token,
            e.code,
            e.message,
        )
        return []
    except Exception as e:  # pragma: no cover
        logging.error("wall.getById failed gid=%s post=%s: %s", group_id, post_id, e)
        return []
    response = data.get("response") if isinstance(data, dict) else data
    if isinstance(response, dict):
        items = response.get("items") or []
    else:
        items = response or []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def _vk_extract_photo_urls(
    items: Sequence[Mapping[str, Any]], limit: int = 10
) -> list[str]:
    def best_url(sizes: Sequence[Mapping[str, Any]]) -> str:
        if not sizes:
            return ""
        best = max(
            sizes,
            key=lambda s: (s.get("width", 0) or 0) * (s.get("height", 0) or 0),
        )
        return str(best.get("url") or best.get("src") or "")

    photos: list[str] = []
    seen: set[str] = set()

    def process_atts(atts: Sequence[Mapping[str, Any]], source: str) -> bool:
        counts = {"photo": 0, "link": 0, "video_thumbs": 0, "doc": 0}
        for att in atts or []:
            url = ""
            if att.get("type") == "photo":
                photo = att.get("photo") or {}
                sizes = photo.get("sizes") or []
                url = best_url(sizes)
                if url:
                    counts["photo"] += 1
            elif att.get("type") == "link":
                link = att.get("link") or {}
                sizes = (link.get("photo") or {}).get("sizes", [])
                url = best_url(sizes)
                if url:
                    counts["link"] += 1
            elif att.get("type") == "video":
                video = att.get("video") or {}
                images = video.get("first_frame") or video.get("image", [])
                url = best_url(images)
                if url:
                    counts["video_thumbs"] += 1
            elif att.get("type") == "doc":
                sizes = (
                    ((att.get("doc") or {}).get("preview") or {})
                    .get("photo", {})
                    .get("sizes", [])
                )
                url = best_url(sizes)
                if url:
                    counts["doc"] += 1
            if url and url not in seen:
                seen.add(url)
                photos.append(url)
                if len(photos) >= limit:
                    break
        total = sum(counts.values())
        logging.info(
            "found_photos=%s (photo=%s, link=%s, video_thumbs=%s, doc=%s) source=%s",
            total,
            counts["photo"],
            counts["link"],
            counts["video_thumbs"],
            counts["doc"],
            source,
        )
        return len(photos) >= limit

    for item in items:
        copy_history = item.get("copy_history") or []
        first_copy = copy_history[0] if copy_history and isinstance(copy_history[0], Mapping) else None
        copy_atts = first_copy.get("attachments") if isinstance(first_copy, Mapping) else None
        if copy_atts and process_atts(copy_atts, "copy_history"):
            break
        if len(photos) >= limit:
            break
        atts = item.get("attachments") or []
        if process_atts(atts, "attachments"):
            break

    return photos


async def _vkrev_fetch_photos(group_id: int, post_id: int, db: Database, bot: Bot) -> list[str]:
    items = await _vk_wall_get_items(group_id, post_id, db, bot)
    if not items:
        return []
    photos = _vk_extract_photo_urls(items)
    if not photos:
        logging.info("no media found for -%s_%s", group_id, post_id)
    return photos


async def fetch_vk_post_preview(
    group_id: int, post_id: int, db: Database, bot: Bot
) -> tuple[str, list[str], datetime | None]:
    items = await _vk_wall_get_items(group_id, post_id, db, bot)
    if not items:
        return "", [], None
    text = ""
    published_at: datetime | None = None

    def parse_date(source: Mapping[str, object] | None) -> datetime | None:
        if not source:
            return None
        raw = source.get("date")
        if isinstance(raw, datetime):
            if raw.tzinfo is not None:
                return raw
            return raw.replace(tzinfo=timezone.utc)
        timestamp: int | float | None
        if isinstance(raw, (int, float)):
            timestamp = raw
        elif isinstance(raw, str):
            try:
                timestamp = int(raw)
            except ValueError:
                return None
        else:
            return None
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None

    for item in items:
        item_mapping = item if isinstance(item, Mapping) else None
        item_date = parse_date(item_mapping)
        if published_at is None and item_date is not None:
            published_at = item_date
        candidate = item.get("text")
        if candidate:
            text = str(candidate)
            if item_date is not None:
                published_at = item_date
            break
        copy_history = item.get("copy_history") or []
        for copy in copy_history:
            if not isinstance(copy, Mapping):
                continue
            copy_date = parse_date(copy)
            if published_at is None and copy_date is not None:
                published_at = copy_date
            candidate = copy.get("text")
            if candidate:
                text = str(candidate)
                if copy_date is not None:
                    published_at = copy_date
                elif item_date is not None:
                    published_at = item_date
                break
        if text:
            break
    photos = _vk_extract_photo_urls(items)
    if not photos:
        logging.info("no media found for -%s_%s", group_id, post_id)
    return text, photos, published_at


_VK_POST_ID_RE = re.compile(r"(-?\d+)_(\d+)")


def _vk_miss_extract_ids(record: VkMissRecord) -> tuple[int, int] | None:
    candidates: list[str] = []
    if record.id:
        candidates.append(record.id)
    if record.url:
        candidates.append(record.url)
        try:
            parsed = urlparse(record.url)
        except Exception:
            parsed = None
        if parsed and parsed.path:
            candidates.append(parsed.path)
    for candidate in candidates:
        if not candidate:
            continue
        match = _VK_POST_ID_RE.search(candidate)
        if match:
            owner = int(match.group(1))
            post = int(match.group(2))
            group_id = abs(owner)
            if group_id > 0 and post > 0:
                return group_id, post
    return None


def _vk_miss_format_timestamp(value: datetime) -> str:
    try:
        localized = value.astimezone(LOCAL_TZ)
    except Exception:
        localized = value
    return localized.strftime("%Y-%m-%d %H:%M:%S %Z")


async def _vk_miss_send_card(
    bot: Bot,
    chat_id: int,
    session: VkMissReviewSession,
    record: VkMissRecord,
    text: str,
    photos: Sequence[str],
    published_at: datetime | None,
) -> None:
    session.last_text = text
    session.last_published_at = published_at
    position = session.index + 1
    total = len(session.queue)
    header = f"Карточка {position}/{total}"
    display_text = text.strip()
    if not display_text:
        display_text = "(текст поста не загружен)"
    timestamp = _vk_miss_format_timestamp(record.timestamp)
    if published_at is not None:
        published = _vk_miss_format_timestamp(published_at)
    else:
        published = "—"
    lines: list[str] = [header]
    if display_text:
        lines.append(display_text)
    if record.url:
        lines.append(record.url)
    else:
        lines.append("URL: —")
    lines.append("")
    lines.append(f"Причина фильтра: {record.reason or '—'}")
    lines.append(f"matched_kw: {record.matched_kw or '—'}")
    lines.append(f"Дата публикации: {published}")
    lines.append(f"Дата: {timestamp}")
    message_text = "\n".join(lines)

    media_urls = [url for url in photos if url]
    if media_urls:
        media = [types.InputMediaPhoto(media=url) for url in media_urls[:10]]
        try:
            await bot.send_media_group(chat_id, media)
        except TelegramBadRequest:
            if len(media) == 1:
                await bot.send_photo(chat_id, media[0].media)
            else:
                raise

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="Отклонено верно",
                    callback_data=f"{VK_MISS_CALLBACK_PREFIX}ok:{session.index}",
                ),
                types.InlineKeyboardButton(
                    text="На доработку",
                    callback_data=f"{VK_MISS_CALLBACK_PREFIX}redo:{session.index}",
                ),
            ]
        ]
    )

    parts = split_text(message_text, TELEGRAM_MESSAGE_LIMIT)
    for idx, part in enumerate(parts):
        markup = keyboard if idx == len(parts) - 1 else None
        await bot.send_message(
            chat_id,
            part,
            reply_markup=markup,
            disable_web_page_preview=True,
        )


async def _vk_miss_append_feedback(
    record: VkMissRecord, post_text: str, published_at: datetime | None = None
) -> None:
    path = VK_MISS_REVIEW_FILE
    if not path:
        return

    def _write() -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        now_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        body = post_text.strip()
        if not body:
            body = "(текст поста не загружен)"
        timestamp_str = _vk_miss_format_timestamp(record.timestamp)
        lines = [
            f"### {now_str}",
            f"- URL: {record.url or '—'}",
            f"- Причина: {record.reason or '—'}",
            "- Время фиксации пропуска: "
            f"{timestamp_str} (время попадания в отклонённые)",
        ]
        if published_at is not None:
            published = _vk_miss_format_timestamp(published_at)
            lines.append(f"- Дата публикации: {published}")
        if record.matched_kw:
            lines.append(f"- matched_kw: {record.matched_kw}")
        lines.append("")
        lines.append(body)
        lines.append("")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    await asyncio.to_thread(_write)


async def _vk_miss_offer_feedback_file(bot: Bot, chat_id: int) -> None:
    path = VK_MISS_REVIEW_FILE

    def _has_data() -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    if not await asyncio.to_thread(_has_data):
        return

    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="Скачать файл",
                    callback_data=f"{VK_MISS_CALLBACK_PREFIX}download",
                )
            ]
        ]
    )
    await bot.send_message(
        chat_id,
        "Файл с доработками готов к скачиванию",
        reply_markup=keyboard,
    )


async def _vk_miss_show_next(
    user_id: int, chat_id: int, db: Database, bot: Bot
) -> None:
    session = vk_miss_review_sessions.get(user_id)
    if not session:
        return
    if session.index >= len(session.queue):
        vk_miss_review_sessions.pop(user_id, None)
        await bot.send_message(chat_id, "Карточки закончились")
        await _vk_miss_offer_feedback_file(bot, chat_id)
        return
    record = session.queue[session.index]
    ids = _vk_miss_extract_ids(record)
    text = ""
    photos: list[str] = []
    published_at: datetime | None = None
    if ids is None:
        logging.warning(
            "vk_miss_review.unparsable_id id=%s url=%s", record.id, record.url
        )
    else:
        group_id, post_id = ids
        text, photos, published_at = await fetch_vk_post_preview(
            group_id, post_id, db, bot
        )
    await _vk_miss_send_card(
        bot, chat_id, session, record, text, photos, published_at
    )


async def _vk_miss_send_feedback_file(
    callback: types.CallbackQuery, bot: Bot
) -> None:
    path = VK_MISS_REVIEW_FILE

    def _file_size() -> int | None:
        try:
            return os.path.getsize(path)
        except FileNotFoundError:
            return None
        except OSError:
            return None

    size = await asyncio.to_thread(_file_size)
    if not size:
        await callback.answer("Файл отсутствует или пуст", show_alert=True)
        return

    chat_id = (
        callback.message.chat.id
        if callback.message is not None
        else callback.from_user.id
    )
    timestamp_suffix = datetime.now(LOCAL_TZ).strftime("%Y%m%d-%H%M%S")
    filename = f"vk_miss_review_{timestamp_suffix}.md"
    document = types.FSInputFile(path, filename=filename)

    await bot.send_document(chat_id, document)

    def _clear() -> None:
        try:
            with open(path, "w", encoding="utf-8"):
                pass
        except FileNotFoundError:
            pass

    await asyncio.to_thread(_clear)
    await callback.answer("Файл отправлен")


async def handle_vk_miss_review(
    message: types.Message, db: Database, bot: Bot
) -> None:
    parts = (message.text or "").split()
    limit = 10
    if len(parts) > 1:
        try:
            limit = int(parts[1])
        except ValueError:
            await bot.send_message(message.chat.id, "Usage: /vk_misses [N]")
            return
    limit = max(1, min(limit, 50))
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user or not user.is_superadmin:
        await bot.send_message(message.chat.id, "Not authorized")
        return

    records = await fetch_vk_miss_samples(limit)
    if not records:
        await bot.send_message(message.chat.id, "Нет пропусков для ревизии")
        return
    vk_miss_review_sessions[message.from_user.id] = VkMissReviewSession(queue=records)
    await bot.send_message(
        message.chat.id,
        f"Загружено карточек: {len(records)}",
        disable_web_page_preview=True,
    )
    await _vk_miss_show_next(message.from_user.id, message.chat.id, db, bot)


async def handle_vk_miss_review_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    data = callback.data or ""
    if not data.startswith(VK_MISS_CALLBACK_PREFIX):
        await callback.answer()
        return
    payload = data[len(VK_MISS_CALLBACK_PREFIX) :]
    if payload == "download":
        await _vk_miss_send_feedback_file(callback, bot)
        return
    try:
        action, idx_raw = payload.split(":", 1)
        idx = int(idx_raw)
    except ValueError:
        await callback.answer()
        return

    session = vk_miss_review_sessions.get(callback.from_user.id)
    if not session:
        await callback.answer("Сессия завершена", show_alert=True)
        return
    if idx != session.index:
        await callback.answer("Устаревшая карточка", show_alert=True)
        return
    if callback.message is None:
        await callback.answer()
        return

    record = session.queue[session.index]

    if action == "redo":
        await _vk_miss_mark_checked(record.id)
        await _vk_miss_append_feedback(
            record, session.last_text or "", session.last_published_at
        )
        await callback.answer("Отправлено на доработку")
    elif action == "ok":
        await _vk_miss_mark_checked(record.id)
        await callback.answer("Отмечено")
    else:
        await callback.answer()
        return

    session.index += 1
    session.last_text = None
    session.last_published_at = None
    await _vk_miss_show_next(
        callback.from_user.id,
        callback.message.chat.id,
        db,
        bot,
    )


def _vkrev_collect_photo_ids(items: list[dict], max_photos: int) -> list[str]:
    photos: list[str] = []
    seen: set[str] = set()

    def process(atts: list[dict]) -> bool:
        for att in atts or []:
            if att.get("type") != "photo":
                continue
            ph = att.get("photo", {})
            owner = ph.get("owner_id")
            pid = ph.get("id")
            if owner is None or pid is None:
                continue
            key = f"{owner}_{pid}"
            access = ph.get("access_key")
            if access:
                key = f"{key}_{access}"
            if key in seen:
                continue
            seen.add(key)
            photos.append("photo" + key)
            if len(photos) >= max_photos:
                return True
        return False

    for item in items:
        copy = (item.get("copy_history") or [{}])[0].get("attachments")
        if process(copy):
            break
        if process(item.get("attachments") or []):
            break
    return photos


async def build_short_vk_text(
    event: Event,
    source_text: str,
    max_sentences: int = 4,
    *,
    poster_texts: Sequence[str] | None = None,
) -> str:
    text = (source_text or "").strip()
    fallback_from_title = False
    if not text:
        desc = (event.description or "").strip()
        if desc:
            text = desc
        else:
            title = (event.title or "").strip()
            text = title
            fallback_from_title = True
    if not text:
        return ""
    if fallback_from_title:
        return text

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")

    def _truncate_sentences(source: str, limit: int) -> str:
        if limit <= 0:
            return ""
        paragraphs: list[str] = []
        remaining = limit
        for block in source.split("\n\n"):
            paragraph = block.strip()
            if not paragraph or remaining <= 0:
                continue
            sentences = [part.strip() for part in sentence_splitter.split(paragraph) if part.strip()]
            if not sentences:
                continue
            selected: list[str] = []
            for sentence in sentences:
                if remaining <= 0:
                    break
                selected.append(sentence)
                remaining -= 1
            if selected:
                paragraphs.append(" ".join(selected))
        return "\n\n".join(paragraphs).strip()

    def _fallback_summary() -> str:
        fallback = _truncate_sentences(text, min(max_sentences, 2))
        return fallback or text

    extra_blocks = [block.strip() for block in poster_texts or [] if block.strip()]
    prompt_text = text
    if extra_blocks:
        joined = "\n\n".join(extra_blocks)
        prompt_text = f"{prompt_text}\n\nДополнительный текст с афиш:\n{joined}"

    prompt = (
        "Сократи описание ниже без выдумок, сохраняя все важные детали "
        "и перечисленных ключевых участников, максимум до "
        f"{max_sentences} предложений. Разрешены эмодзи. "
        "Пиши дружелюбно и не добавляй прямых рекламных призывов (например, про покупку билетов). "
        "Сразу начинай с главной идеи — в первой строке не повторяй название события и не добавляй блок про дату, время, место или билеты. "
        "Название проекта или события можно упомянуть позже. "
        "Не повторяй дату, время и место события в абзацах — мы выводим их отдельными строками. "
        "Сделай первую фразу крючком, который вызывает любопытство: это может быть вопрос или интригующая деталь. "
        "Не используй фразу «Погрузитесь в мир» ни в каком виде. "
        f"Разбивай текст на абзацы для удобства чтения.\n\n{prompt_text}"
    )
    try:
        raw = await ask_4o(
            prompt,
            system_prompt=(
                "Ты сжимаешь текст фактически, без новых деталей и не упуская важные факты. "
                "Эмодзи допустимы. Делай текст читабельным и дружелюбным, разбивая его на абзацы. "
                "Не используй прямые рекламные формулировки, в том числе призывы покупать билеты. "
                "Сразу начинай с сути — в первой строке не повторяй название события и не добавляй блок про дату, время, место или билеты. "
                "Название проекта или события можно упомянуть позже. "
                "Не повторяй дату, время и место события в абзацах — они выводятся отдельно. "
                "Первая фраза должна быть крючком, вызывающим любопытство, и избегай фразы «Погрузитесь в мир»."
            ),
            max_tokens=400,
        )
    except Exception:
        return _fallback_summary()
    cleaned = raw.strip()
    if not cleaned:
        return _fallback_summary()

    banned_phrase_pattern = re.compile(r"погрузитесь в мир", re.IGNORECASE)

    def _remove_banned_sentences(value: str) -> str:
        if not banned_phrase_pattern.search(value):
            return value
        paragraphs: list[str] = []
        for block in value.split("\n\n"):
            sentences = [
                sentence.strip()
                for sentence in sentence_splitter.split(block)
                if sentence.strip()
            ]
            filtered = [
                sentence
                for sentence in sentences
                if not banned_phrase_pattern.search(sentence)
            ]
            if filtered:
                paragraphs.append(" ".join(filtered))
        return "\n\n".join(paragraphs).strip()

    def _ensure_curiosity_hook(value: str) -> str:
        stripped = value.lstrip()
        prefix = value[: len(value) - len(stripped)]
        if not stripped:
            return value
        match = re.search(r"^([^\n]*?[.!?])(\s|$)", stripped)
        if match:
            first_sentence = match.group(1).strip()
            separator = match.group(2) or ""
            remainder = separator + stripped[match.end():]
        else:
            first_sentence = stripped
            remainder = ""
        hook_prefixes = (
            "что если",
            "представьте",
            "знаете ли вы",
            "как насчет",
            "как насчёт",
            "готовы ли вы",
            "хотите узнать",
            "угадайте",
        )
        first_lower = first_sentence.casefold()
        has_hook = "?" in first_sentence or any(
            first_lower.startswith(prefix) for prefix in hook_prefixes
        )
        if has_hook:
            return prefix + stripped
        base = first_sentence.rstrip(".!?").strip()
        if not base:
            return prefix + stripped
        if len(base) > 1:
            body = base[0].lower() + base[1:]
        else:
            body = base.lower()
        new_first_sentence = f"Знаете ли вы, {body}?"
        remainder = remainder.lstrip()
        if remainder:
            if remainder.startswith("\n"):
                rebuilt = new_first_sentence + remainder
            else:
                rebuilt = new_first_sentence + " " + remainder
        else:
            rebuilt = new_first_sentence
        return prefix + rebuilt

    cleaned = _remove_banned_sentences(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return _fallback_summary()
    if banned_phrase_pattern.search(cleaned):
        cleaned = banned_phrase_pattern.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return _fallback_summary()
    cleaned = _ensure_curiosity_hook(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return _fallback_summary()
    cleaned_lower = cleaned.casefold()
    if not cleaned:
        return _fallback_summary()
    if ("предостав" in cleaned_lower and "текст" in cleaned_lower) or (
        "provide" in cleaned_lower and "text" in cleaned_lower
    ):
        return _fallback_summary()
    summary = _truncate_sentences(cleaned, max_sentences)
    return summary or _fallback_summary()


VK_LOCATION_TAG_OVERRIDES: dict[str, str] = {
    "ицаэ": "#ИЦАЭ",
    "кгту": "#КГТУ",
    "коихм": "#КОИХМ",
}


VK_TOPIC_HASHTAGS: Mapping[str, str] = {
    "STANDUP": "#стендап",
    "QUIZ_GAMES": "#квиз",
    "OPEN_AIR": "#openair",
    "PARTIES": "#вечеринка",
    "CONCERTS": "#музыка",
    "MOVIES": "#кино",
    "EXHIBITIONS": "#искусство",
    "THEATRE": "#театр",
    "THEATRE_CLASSIC": "#классика",
    "THEATRE_MODERN": "#перфоманс",
    "LECTURES": "#лекция",
    "MASTERCLASS": "#мастеркласс",
    "PSYCHOLOGY": "#здоровье",
    "SCIENCE_POP": "#научпоп",
    "HANDMADE": "#маркет",
    "NETWORKING": "#митап",
    "ACTIVE": "#спорт",
    "HISTORICAL_IMMERSION": "#история",
    "FASHION": "#мода",
    "KIDS_SCHOOL": "#детям",
    "FAMILY": "#семье",
    "URBANISM": "#урбанистика",
    "KRAEVEDENIE_KALININGRAD_OBLAST": "#калининград",
}


async def build_short_vk_tags(
    event: Event, summary: str, used_type_hashtag: str | None = None
) -> list[str]:
    """Generate 5-7 hashtags for the short VK post."""
    day = int(event.date.split("-")[2])
    month = int(event.date.split("-")[1])
    month_name = MONTHS[month - 1]
    current_year = date.today().year
    tags: list[str] = []
    seen: set[str] = set()

    used_type_hashtag_normalized: str | None = None
    if used_type_hashtag:
        used_tag_clean = used_type_hashtag.strip()
        if used_tag_clean:
            if not used_tag_clean.startswith("#"):
                used_tag_clean = "#" + used_tag_clean.lstrip("#")
            used_type_hashtag_normalized = used_tag_clean.lower()
            seen.add(used_type_hashtag_normalized)

    def add_tag(tag: str) -> None:
        tag_clean = (tag or "").strip()
        if not tag_clean:
            return
        if not tag_clean.startswith("#"):
            tag_clean = "#" + tag_clean.lstrip("#")
        tag_lower = tag_clean.lower()
        if tag_lower in seen:
            return
        years = re.findall(r"\d{4}", tag_lower)
        for year_text in years:
            try:
                if int(year_text) < current_year:
                    return
            except ValueError:  # pragma: no cover
                continue
        tags.append(tag_clean)
        seen.add(tag_lower)

    add_tag(f"#{day}_{month_name}")
    add_tag(f"#{day}{month_name}")
    city = (event.city or "").strip()
    if city:
        normalized_city = re.sub(r"[^0-9a-zа-яё]+", "", city.lower())
        if normalized_city:
            add_tag(f"#{normalized_city}")
    seen_location_tokens: set[str] = set()
    for source in (event.location_name or "", event.location_address or ""):
        if not source:
            continue
        for token in re.findall(r"[А-ЯЁ]{2,}", source):
            normalized_token = token.lower()
            if normalized_token in seen_location_tokens:
                continue
            seen_location_tokens.add(normalized_token)
            tag = VK_LOCATION_TAG_OVERRIDES.get(normalized_token, f"#{token}")
            add_tag(tag)
    if event.event_type:
        raw_event_type = event.event_type.strip()
        if raw_event_type:
            event_type_lower = raw_event_type.casefold()
            normalized_event_type = re.sub(
                r"[^0-9a-zа-яё]+", "_", event_type_lower
            ).strip("_")
            if normalized_event_type:
                add_tag(f"#{normalized_event_type}")
            if re.search(r"[-–—]", raw_event_type):
                hyphen_free_variant = re.sub(
                    r"[^0-9a-zа-яё]", "", event_type_lower
                )
                if hyphen_free_variant:
                    hyphen_tag = f"#{hyphen_free_variant}"
                    if not (
                        used_type_hashtag_normalized
                        and hyphen_tag.lower() == used_type_hashtag_normalized
                    ):
                        add_tag(hyphen_tag)
    topic_values = getattr(event, "topics", None) or []
    for topic in topic_values:
        if len(tags) >= 7:
            break
        normalized_topic = (topic or "").strip().upper()
        if not normalized_topic:
            continue
        topic_tag = VK_TOPIC_HASHTAGS.get(normalized_topic)
        if not topic_tag:
            continue
        topic_tag_clean = topic_tag.strip()
        if not topic_tag_clean:
            continue
        if not topic_tag_clean.startswith("#"):
            topic_tag_clean = "#" + topic_tag_clean.lstrip("#")
        if (
            used_type_hashtag_normalized
            and topic_tag_clean.lower() == used_type_hashtag_normalized
        ):
            continue
        add_tag(topic_tag_clean)
    needed = 7 - len(tags)
    if needed > 0:
        prompt = (
            "Подбери ещё {n} коротких и актуальных хештегов "
            "для поста о событии. Используй русский язык, "
            "начинай каждый хештег с #, не добавляй пояснений. "
            "Добавь хештег с форматом события (например, #спектакль, #мастеркласс, #лекция). "
            "Не предлагай хештеги со старыми годами (раньше {current_year}).\n"
            "Название: {title}\nОписание: {desc}"
        ).format(
            n=needed,
            current_year=current_year,
            title=event.title,
            desc=summary,
        )
        try:
            raw = await ask_4o(
                prompt,
                system_prompt=(
                    "Ты подбираешь хештеги к событию, отвечай только хештегами. "
                    "Избегай устаревших годов и рекламных формулировок."
                ),
                max_tokens=60,
            )
            extra = re.findall(r"#[^\s#]+", raw.lower())
            for t in extra:
                add_tag(t)
                if len(tags) >= 7:
                    break
        except Exception:
            pass
    if len(tags) < 5:
        fallback = ["#афиша", "#кудапойти", "#событие", "#выходные", "#калининград"]
        for t in fallback:
            if len(tags) >= 5:
                break
            add_tag(t)
            if len(tags) >= 5:
                break
    return tags[:7]


async def build_short_vk_location(parts: Sequence[str]) -> str:
    cleaned_parts = [part.strip() for part in parts if part and part.strip()]
    if not cleaned_parts:
        return ""
    joined = ", ".join(cleaned_parts)
    prompt = (
        "Собери короткую и понятную формулировку адреса события для поста ВКонтакте. "
        "Используй все важные части, убери дубли и лишние слова. Не пиши ничего, кроме адреса, "
        "не добавляй слово «Локация» и эмодзи.\n"
        f"Части адреса: {joined}"
    )
    try:
        raw = await ask_4o(
            prompt,
            system_prompt=(
                "Ты формируешь краткую строку с адресом события. "
                "Верни только сам адрес без вводных слов и эмодзи."
            ),
            max_tokens=60,
        )
    except Exception:
        return joined
    location = raw.strip()
    if not location:
        return joined
    location = re.sub(r"\s+", " ", location)
    location = location.replace("📍", "").strip()
    location = re.sub(r"^[Лл]окация[:\-\s]*", "", location).strip()
    return location or joined


async def _vkrev_show_next(chat_id: int, batch_id: str, operator_id: int, db: Database, bot: Bot) -> None:
    await get_tz_offset(db)
    post = await vk_review.pick_next(db, operator_id, batch_id)
    if not post:
        buttons = [
            [types.KeyboardButton(text=VK_BTN_ADD_SOURCE)],
            [types.KeyboardButton(text=VK_BTN_LIST_SOURCES)],
            [
                types.KeyboardButton(text=VK_BTN_CHECK_EVENTS),
                types.KeyboardButton(text=VK_BTN_QUEUE_SUMMARY),
            ],
        ]
        markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
        await bot.send_message(chat_id, "Очередь пуста", reply_markup=markup)
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                """
                SELECT batch_id, months_csv
                FROM vk_review_batch
                WHERE operator_id=? AND finished_at IS NULL AND months_csv<>''
                ORDER BY started_at DESC
                """,
                (operator_id,),
            )
            rows = await cur.fetchall()
        inline_buttons: list[list[types.InlineKeyboardButton]] = []
        summaries: list[str] = []
        for batch_id_db, months_csv in rows:
            months = [m for m in months_csv.split(',') if m]
            if not months:
                continue
            months_display = ", ".join(months)
            summaries.append(months_display)
            base_text = "🧹 Завершить и обновить страницы месяцев"
            suffix = f" ({months_display})" if months_display else ""
            button_text = base_text
            if suffix and len(base_text + suffix) <= 64:
                button_text = base_text + suffix
            inline_buttons.append(
                [
                    types.InlineKeyboardButton(
                        text=button_text,
                        callback_data=f"vkrev:finish:{batch_id_db}",
                    )
                ]
            )
        if inline_buttons:
            info_lines = ["Опубликованные события ждут обновления страниц месяцев."]
            if summaries:
                info_lines.append("; ".join(summaries))
            info_lines.append("Нажмите кнопку ниже, чтобы запустить обновление.")
            await bot.send_message(
                chat_id,
                "\n".join(info_lines),
                reply_markup=types.InlineKeyboardMarkup(inline_keyboard=inline_buttons),
            )
        return
    photos = await _vkrev_fetch_photos(post.group_id, post.post_id, db, bot)
    if photos:
        media = [types.InputMediaPhoto(media=p) for p in photos[:10]]
        with contextlib.suppress(Exception):
            await bot.send_media_group(chat_id, media)

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, default_time FROM vk_source WHERE group_id=?",
            (post.group_id,),
        )
        row = await cur.fetchone()
    group_name = f"group {post.group_id}"
    default_time_val: str | None = None
    if row:
        group_name = row[0] or group_name
        default_time_val = row[1]

    recomputed_hint: int | None = None
    try:
        recomputed_hint = vk_intake.extract_event_ts_hint(
            post.text or "",
            default_time_val,
            publish_ts=getattr(post, "date", None),
            allow_past=True,
        )
    except Exception:  # pragma: no cover - defensive
        logging.exception(
            "vkrev recompute_event_ts_hint_failed inbox_id=%s", getattr(post, "id", "?")
        )

    ts_hint = getattr(post, "event_ts_hint", None)
    if recomputed_hint and recomputed_hint > 0:
        if recomputed_hint != ts_hint:
            async with db.raw_conn() as conn:
                await conn.execute(
                    "UPDATE vk_inbox SET event_ts_hint=? WHERE id=?",
                    (recomputed_hint, post.id),
                )
                await conn.commit()
        ts_hint = recomputed_hint

    url = f"https://vk.com/wall-{post.group_id}_{post.post_id}"
    pending = await _vkrev_queue_size(db)
    if post.matched_kw == vk_intake.OCR_PENDING_SENTINEL:
        matched_kw_display = "ожидает OCR"
    elif post.matched_kw:
        matched_kw_display = post.matched_kw
    else:
        matched_kw_display = "-"
    status_line = (
        f"ключи: {matched_kw_display} | дата: {'да' if post.has_date else 'нет'} | в очереди: {pending}"
    )
    published_line: str | None = None
    post_date_raw = getattr(post, "date", None)
    if post_date_raw is not None:
        try:
            timestamp = int(post_date_raw)
        except (TypeError, ValueError):
            timestamp = None
        if timestamp and timestamp > 0:
            published_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(
                LOCAL_TZ
            )
            published_line = (
                "Опубликовано: "
                f"{published_dt.day} {MONTHS[published_dt.month - 1]} "
                f"{published_dt.year} {published_dt.strftime('%H:%M')}"
            )
    heading_line: str | None = None
    event_lines: list[str | tuple[str, str | None]] = []
    if ts_hint and ts_hint > 0:
        dt = datetime.fromtimestamp(ts_hint, tz=LOCAL_TZ)
        heading_line = f"{dt.day:02d} {MONTHS[dt.month - 1]} {dt.strftime('%H:%M')}"
        async with db.get_session() as session:
            result = await session.execute(
                select(Event).where(
                    Event.date == dt.date().isoformat(),
                    Event.time == dt.strftime("%H:%M"),
                )
            )
            matched_events = result.scalars().all()
        if matched_events:
            for event in matched_events:
                link = normalize_telegraph_url(event.telegraph_url)
                if not link and event.telegraph_path:
                    link = f"https://telegra.ph/{event.telegraph_path.lstrip('/')}"
                event_lines.append((event.title, link))
        else:
            event_lines.append("Совпадений нет")
    inline_keyboard = [
        [
            types.InlineKeyboardButton(text="✅ Добавить", callback_data=f"vkrev:accept:{post.id}"),
            types.InlineKeyboardButton(
                text="🎉 Добавить (+ фестиваль)",
                callback_data=f"vkrev:accept_fest:{post.id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="📝 Добавить с доп.инфо",
                callback_data=f"vkrev:accept_extra:{post.id}",
            ),
            types.InlineKeyboardButton(
                text="📝🎉 Добавить с доп.инфо (+ фестиваль)",
                callback_data=f"vkrev:accept_fest_extra:{post.id}",
            ),
        ],
        [
            types.InlineKeyboardButton(text="✖️ Отклонить", callback_data=f"vkrev:reject:{post.id}"),
            types.InlineKeyboardButton(text="⏭ Пропустить", callback_data=f"vkrev:skip:{post.id}"),
        ],
        [
            types.InlineKeyboardButton(
                text="Создать историю", callback_data=f"vkrev:story:{post.id}"
            )
        ],
    ]
    # Add Pyramida extraction button if post contains pyramida.info links
    from source_parsing.pyramida import extract_pyramida_urls
    pyramida_urls = extract_pyramida_urls(post.text or "")
    if pyramida_urls:
        inline_keyboard.append([
            types.InlineKeyboardButton(
                text=f"🔮 Извлечь из Pyramida ({len(pyramida_urls)})",
                callback_data=f"vkrev:pyramida:{post.id}",
            )
        ])
    # Add Dom Iskusstv extraction button (always visible)
    from source_parsing.dom_iskusstv import extract_dom_iskusstv_urls
    dom_iskusstv_urls = extract_dom_iskusstv_urls(post.text or "")
    logging.info(f"Adding Dom Iskusstv button. URLs found: {len(dom_iskusstv_urls)}")
    # Always add the button, showing count (0 if none)
    inline_keyboard.append([
        types.InlineKeyboardButton(
            text=f"🏛 Извлечь из Дом искусств ({len(dom_iskusstv_urls)})",
            callback_data=f"vkrev:domiskusstv:{post.id}",
        )
    ])
    inline_keyboard.extend([
        [types.InlineKeyboardButton(text="⏹ Стоп", callback_data=f"vkrev:stop:{batch_id}")],
        [
            types.InlineKeyboardButton(
                text="🧹 Завершить и обновить страницы месяцев",
                callback_data=f"vkrev:finish:{batch_id}",
            )
        ],
    ])
    imported_event_id = getattr(post, "imported_event_id", None)

    if imported_event_id:
        async with db.get_session() as session:
            event = await session.get(Event, imported_event_id)
        if event:
            inline_keyboard = append_tourist_block(inline_keyboard, event, "vk")
    markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
    post_text = post.text or ""
    def build_tail_lines(warning: str | None = None) -> list[str]:
        lines = [group_name, "", url]
        if heading_line:
            lines.extend(["", heading_line, *event_lines])
        lines.append("")
        if warning:
            lines.append(warning)
        if published_line:
            lines.append(published_line)
        lines.append(status_line)
        return lines

    TailLine = str | tuple[str, str | None]

    def format_blockquote(lines: Sequence[TailLine]) -> str:
        rendered_lines: list[str] = []
        for line in lines:
            if isinstance(line, tuple):
                title, link = line
                if link:
                    href = escape(link, quote=True)
                    rendered_lines.append(
                        f'<a href="{href}">{escape(title)}</a>'
                    )
                else:
                    rendered_lines.append(
                        escape(f"{title} — Telegraph отсутствует")
                    )
            else:
                rendered_lines.append(escape(line))
        return "<blockquote>" + "\n".join(rendered_lines) + "</blockquote>"

    def compose_message_html(post_body: str, tail_blockquote: str) -> str:
        if post_body:
            return f"{escape(post_body)}\n{tail_blockquote}"
        return tail_blockquote

    def truncate_text_for_html(text: str, max_escaped_len: int) -> str:
        if max_escaped_len <= 0:
            return ""
        ellipsis = "…"
        ellipsis_len = len(escape(ellipsis))
        escaped_lengths = [len(escape(ch)) for ch in text]
        prefix: list[int] = [0]
        for length in escaped_lengths:
            prefix.append(prefix[-1] + length)
        if prefix[-1] <= max_escaped_len:
            return text
        end_index = 0
        for idx in range(len(text)):
            if prefix[idx + 1] > max_escaped_len:
                end_index = idx
                break
        truncated = text[:end_index].rstrip()
        while truncated:
            truncated_len = len(truncated)
            escaped_len = prefix[truncated_len]
            if escaped_len + ellipsis_len <= max_escaped_len:
                return truncated + ellipsis
            if escaped_len <= max_escaped_len:
                return truncated
            truncated = truncated[:-1].rstrip()
        if ellipsis_len <= max_escaped_len:
            return ellipsis
        return ""

    tail_lines = build_tail_lines()
    tail_blockquote = format_blockquote(tail_lines)
    message_html = compose_message_html(post_text, tail_blockquote)

    if len(message_html) > TELEGRAM_MESSAGE_LIMIT:
        warning_line = (
            f"⚠️ Текст поста был обрезан до {TELEGRAM_MESSAGE_LIMIT} символов"
        )
        tail_lines = build_tail_lines(warning_line)
        tail_blockquote = format_blockquote(tail_lines)
        available = TELEGRAM_MESSAGE_LIMIT - len(tail_blockquote)
        if post_text:
            available -= 1
        available = max(0, available)
        if post_text:
            post_text = truncate_text_for_html(post_text, available)
        message_html = compose_message_html(post_text, tail_blockquote)
        if len(message_html) > TELEGRAM_MESSAGE_LIMIT and post_text:
            post_text = ""
            message_html = compose_message_html(post_text, tail_blockquote)

    await bot.send_message(
        chat_id,
        message_html,
        reply_markup=markup,
        parse_mode="HTML",
    )


async def _vkrev_import_flow(
    chat_id: int,
    operator_id: int,
    inbox_id: int,
    batch_id: str,
    db: Database,
    bot: Bot,
    operator_extra: str | None = None,
    festival_hint: bool | None = None,
    *,
    force_festival: bool = False,
) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, text, date, event_ts_hint FROM vk_inbox WHERE id=?",
            (inbox_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(chat_id, "Инбокс не найден")
        return
    group_id, post_id, text, publish_ts, event_ts_hint = row
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, location, default_time, default_ticket_link FROM vk_source WHERE group_id=?",
            (group_id,),
        )
        source = await cur.fetchone()
    photos = await _vkrev_fetch_photos(group_id, post_id, db, bot)
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
    festival_alias_pairs: list[tuple[str, int]] = []
    if festival_names:
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
                festival_alias_pairs.append((norm, idx))
        if festival_alias_pairs:
            seen_pairs: set[tuple[str, int]] = set()
            deduped: list[tuple[str, int]] = []
            for pair in festival_alias_pairs:
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                deduped.append(pair)
            festival_alias_pairs = deduped
    if festival_hint is None:
        festival_hint = force_festival
    source_name_val: str | None = None
    location_hint_val: str | None = None
    default_time_val: str | None = None
    default_ticket_link_val: str | None = None
    if source:
        source_name_val, location_hint_val, default_time_val, default_ticket_link_val = source

    drafts, festival_info_raw = await vk_intake.build_event_drafts(
        text,
        photos=photos,
        source_name=source_name_val,
        location_hint=location_hint_val,
        default_time=default_time_val,
        default_ticket_link=default_ticket_link_val,
        operator_extra=operator_extra,
        festival_names=festival_names,
        festival_alias_pairs=festival_alias_pairs or None,
        festival_hint=festival_hint,
        publish_ts=publish_ts,
        event_ts_hint=event_ts_hint,
        db=db,
    )
    source_post_url = f"https://vk.com/wall-{group_id}_{post_id}"
    if isinstance(festival_info_raw, str):
        festival_info_raw = {"name": festival_info_raw}

    poster_urls: list[str] = []
    first_draft = drafts[0] if drafts else None
    if first_draft and first_draft.poster_media:
        poster_urls = [
            media.catbox_url
            for media in first_draft.poster_media
            if getattr(media, "catbox_url", None)
        ]
    poster_urls = [url for url in poster_urls if url]

    festival_obj: Festival | None = None
    fest_created = False
    fest_updated = False
    fest_status_line: str | None = None
    fest_data: dict[str, Any] | None = None
    fest_start_date: str | None = None
    fest_end_date: str | None = None
    if isinstance(festival_info_raw, dict):
        fest_data = festival_info_raw
        fest_name = clean_optional_str(
            fest_data.get("name")
            or fest_data.get("festival")
            or fest_data.get("full_name")
        )
    else:
        fest_name = None

    if force_festival and not fest_name:
        await bot.send_message(
            chat_id,
            "❌ Не удалось распознать фестиваль, импорт остановлен.",
        )
        return

    if fest_name:
        start_raw = None
        end_raw = None
        location_name = None
        city = None
        location_address = None
        website_url = None
        program_url = None
        ticket_url = None
        full_name = None
        if fest_data:
            start_raw = clean_optional_str(fest_data.get("start_date"))
            if not start_raw:
                start_raw = clean_optional_str(fest_data.get("date"))
            end_raw = clean_optional_str(fest_data.get("end_date"))
            location_name = clean_optional_str(fest_data.get("location_name"))
            city = clean_optional_str(fest_data.get("city"))
            location_address = clean_optional_str(fest_data.get("location_address"))
            website_url = clean_optional_str(fest_data.get("website_url"))
            program_url = clean_optional_str(fest_data.get("program_url"))
            ticket_url = clean_optional_str(fest_data.get("ticket_url"))
            full_name = clean_optional_str(fest_data.get("full_name"))
        fest_start_date = canonicalize_date(start_raw)
        fest_end_date = canonicalize_date(end_raw)
        location_address = strip_city_from_address(location_address, city)
        source_text_value = text
        if first_draft and first_draft.source_text:
            source_text_value = first_draft.source_text
        festival_obj, fest_created, fest_updated = await ensure_festival(
            db,
            fest_name,
            full_name=full_name,
            photo_url=poster_urls[0] if poster_urls else None,
            photo_urls=poster_urls,
            website_url=website_url,
            program_url=program_url,
            ticket_url=ticket_url,
            start_date=fest_start_date,
            end_date=fest_end_date,
            location_name=location_name,
            location_address=location_address,
            city=city,
            source_text=source_text_value,
            source_post_url=source_post_url,
        )
        if festival_obj:
            for draft in drafts:
                draft.festival = festival_obj.name
            status = "создан" if fest_created else "обновлён" if fest_updated else "без изменений"
            fest_status_line = f"Фестиваль: {festival_obj.name} ({status})"

    async def _sync_festival_updates() -> None:
        if festival_obj and (fest_created or fest_updated):
            try:
                await sync_festival_page(db, festival_obj.name)
            except Exception:
                logging.exception("festival page sync failed for %s", festival_obj.name)
            try:
                await sync_festivals_index_page(db)
            except Exception:
                logging.exception("festival index sync failed")
            try:
                await sync_festival_vk_post(db, festival_obj.name, bot, strict=True)
            except Exception:
                logging.exception("festival vk sync failed for %s", festival_obj.name)

    persist_results: list[
        tuple[vk_intake.EventDraft, vk_intake.PersistResult, Event | None]
    ] = []
    admin_chat = os.getenv("ADMIN_CHAT_ID")

    async def _send_persist_summary_messages() -> None:
        if not persist_results:
            return
        link_lines: list[str] = []
        for idx, (_draft, res, _event_obj) in enumerate(persist_results, start=1):
            link_lines.append(f"Событие {idx}: ID {res.event_id}")
            link_lines.append(f"✅ Telegraph — {res.telegraph_url}")
            link_lines.append(f"✅ Календарь (ICS) — {res.ics_supabase_url}")
            link_lines.append(f"✅ ICS (Telegram) — {res.ics_tg_url}")
            if idx != len(persist_results):
                link_lines.append("")
        links = "\n".join(link_lines)
        if admin_chat:
            await bot.send_message(int(admin_chat), links)
        await bot.send_message(chat_id, links)

    async def _send_persist_detail_messages() -> None:
        for idx, (draft, res, event_obj) in enumerate(persist_results, start=1):
            base_keyboard = [
                [
                    types.InlineKeyboardButton(
                        text="↪️ Репостнуть в Vk",
                        callback_data=f"vkrev:repost:{res.event_id}",
                    ),
                    types.InlineKeyboardButton(
                        text="✂️ Сокращённый рерайт",
                        callback_data=f"vkrev:shortpost:{res.event_id}",
                    ),
                ],
                [
                    types.InlineKeyboardButton(
                        text="Редактировать",
                        callback_data=f"edit:{res.event_id}",
                    )
                ],
            ]
            if event_obj:
                inline_keyboard = append_tourist_block(base_keyboard, event_obj, "vk")
            else:
                inline_keyboard = base_keyboard
            markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)

            def _display(value: str | None) -> str:
                return value if value else "—"

            detail_lines = [
                f"Тип: {_display(res.event_type)}",
                f"Дата начала: {_display(res.event_date)}",
                f"Дата окончания: {_display(res.event_end_date)}",
                f"Время: {_display(res.event_time)}",
                f"Бесплатное: {'да' if res.is_free else 'нет'}",
            ]
            if event_obj:
                festival_line = f"Фестиваль/праздник: {_display(getattr(event_obj, 'festival', None))}"
            else:
                festival_line = f"Фестиваль/праздник: {_display(getattr(draft, 'festival', None))}"
            detail_lines.append(festival_line)
            if event_obj:
                detail_lines.append(
                    _format_topics_line(
                        getattr(event_obj, "topics", None),
                        bool(getattr(event_obj, "topics_manual", False)),
                    )
                )
            if fest_status_line:
                detail_lines.append(fest_status_line)
            if draft.poster_media and draft.ocr_tokens_remaining is not None:
                if getattr(draft, "ocr_limit_notice", None):
                    detail_lines.append(draft.ocr_limit_notice)
                detail_lines.append(
                    f"OCR: потрачено {draft.ocr_tokens_spent}, осталось {draft.ocr_tokens_remaining}"
                )
            if event_obj and getattr(event_obj, "search_digest", None):
                detail_lines.append(f"Дайджест: {event_obj.search_digest}")

            header = "Импортировано"
            if len(persist_results) > 1:
                header = f"Импортировано #{idx}"

            if event_obj:
                message_text = build_event_card_message(
                    header, event_obj, detail_lines
                )
            else:
                message_text = "\n".join([header, *detail_lines])

            await bot.send_message(chat_id, message_text, reply_markup=markup)

    total_drafts = len(drafts)
    for idx, draft in enumerate(drafts, start=1):
        tolerance_days: int | None = None
        if draft.festival:
            record = get_holiday_record(draft.festival)
            if record is not None:
                tolerance_days = record.tolerance_days
        persist_kwargs: dict[str, Any] = {"source_post_url": source_post_url}
        if tolerance_days is not None:
            persist_kwargs["holiday_tolerance_days"] = tolerance_days
        start_time = _time.monotonic()
        try:
            res = await vk_intake.persist_event_and_pages(
                draft,
                photos,
                db,
                **persist_kwargs,
            )
        except Exception:
            elapsed = _time.monotonic() - start_time
            logging.exception(
                "vkrev.persist_event.failed idx=%s title=%r elapsed=%.2fs successes=%s total=%s",
                idx,
                getattr(draft, "title", None),
                elapsed,
                len(persist_results),
                total_drafts,
            )
            if persist_results:
                await _send_persist_summary_messages()
                await _sync_festival_updates()
                await _send_persist_detail_messages()
            failure_title = (
                getattr(draft, "title", None)
                or getattr(draft, "name", None)
                or "—"
            )
            failure_lines = [
                f"❌ Импорт остановлен на событии {idx} из {total_drafts}.",
            ]
            if failure_title and failure_title != "—":
                failure_lines.append(f"Название: {failure_title}")
            failure_lines.append(
                f"Успешно импортировано: {len(persist_results)}."
            )
            failure_lines.append("Проверьте логи и попробуйте ещё раз.")
            failure_message = "\n".join(failure_lines)
            await bot.send_message(chat_id, failure_message)
            if admin_chat:
                await bot.send_message(int(admin_chat), failure_message)
            return
        async with db.get_session() as session:
            event_obj = await session.get(Event, res.event_id)
        persist_results.append((draft, res, event_obj))

    if not persist_results:
        if festival_obj:
            fest_month_hint = fest_start_date or fest_end_date or ""
            await vk_review.mark_imported(
                db, inbox_id, batch_id, operator_id, None, fest_month_hint
            )
            vk_review_actions_total["imported"] += 1
            message_lines = ["Импортировано только фестиваль"]
            if fest_status_line:
                message_lines.append(fest_status_line)
            message_lines.append("События не импортированы.")
            await bot.send_message(chat_id, "\n".join(message_lines))
            await _sync_festival_updates()
        else:
            await bot.send_message(chat_id, "LLM не вернул события")
        try:
            mark_vk_import_result(
                group_id=group_id,
                post_id=post_id,
                url=source_post_url,
                outcome="imported",
                event_id=None,
            )
        except Exception:
            logging.exception("vk_import_result.supabase_failed")
        return

    first_res = persist_results[0][1]
    await vk_review.mark_imported(
        db, inbox_id, batch_id, operator_id, first_res.event_id, first_res.event_date
    )
    try:
        mark_vk_import_result(
            group_id=group_id,
            post_id=post_id,
            url=source_post_url,
            outcome="imported",
            event_id=first_res.event_id,
        )
    except Exception:
        logging.exception("vk_import_result.supabase_failed")
    vk_review_actions_total["imported"] += 1

    await _send_persist_summary_messages()

    await _sync_festival_updates()

    await _send_persist_detail_messages()


_VK_STORY_LINK_RE = re.compile(r"\[([^\[\]]+)\]")
_VKREV_EDITOR_H3_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", re.IGNORECASE | re.DOTALL)


def _vk_story_link_label(match: re.Match[str]) -> str:
    content = match.group(1)
    if "|" not in content:
        return content
    parts = [part.strip() for part in content.split("|")]
    for candidate in parts[1:]:
        if candidate:
            return candidate
    return ""


def _vkrev_story_title(text: str | None, group_id: int, post_id: int) -> str:
    if text:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            cleaned = _VK_STORY_LINK_RE.sub(_vk_story_link_label, stripped)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                return cleaned[:64]
    return f"История VK {group_id}_{post_id}"


def _vkrev_extract_editor_title(editor_html: str | None) -> str | None:
    if not editor_html:
        return None
    match = _VKREV_EDITOR_H3_RE.search(editor_html)
    candidate: str | None = None
    if match:
        candidate = _strip_tags(match.group(1))
    if not candidate:
        plain = _strip_tags(editor_html)
        for line in plain.splitlines():
            line_clean = line.strip()
            if line_clean:
                candidate = line_clean
                break
    if not candidate:
        return None
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate or None


def _vkrev_extract_forbidden_phrases(instructions: str | None) -> list[str]:
    if not instructions:
        return []
    lowered = instructions.casefold()
    phrases: set[str] = set()
    patterns = [
        r"не\s+(?:используй|использовать|упоминать|упоминай|говорить|говори|писать|пиши|употреблять|вставлять)"
        r"(?:\s+(?:ничего|ни\s+слова))?(?:\s+(?:про|о|об))?\s+([^.!?\n]+)",
        r"без\s+([^.!?\n]+)",
        r"чтобы\s+не\s+было\s+([^.!?\n]+)",
        r"никаких\s+([^.!?\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            match_text = match.group(0)
            fragment = match.group(1)
            fragment = re.split(r"[,;]\s*", fragment, maxsplit=1)[0]
            fragment = fragment.strip()
            prefix_word: str | None = None
            if fragment:
                index = match_text.rfind(fragment)
                if index != -1:
                    before_fragment = match_text[:index]
                    prefix_match = re.search(r"(про|о|об|никаких)\s+$", before_fragment)
                    if prefix_match:
                        prefix_word = prefix_match.group(1)
            fragment = re.sub(r"^(слова?|word|emoji|эмодзи)\s+", "", fragment)
            fragment = re.sub(r"^(?:про|о|об)\s+", "", fragment)
            fragment = re.sub(r"^(?:ничего|ни\s+слова)\s+", "", fragment)
            fragment = re.sub(r"^никаких\s+", "", fragment)
            fragment = fragment.strip(" \"'«»“”„‹›‚‘’`()[]")
            if fragment:
                phrases.add(fragment)
                if prefix_word:
                    variant = f"{prefix_word} {fragment}".strip()
                    variant = variant.strip(" \"'«»“”„‹›‚‘’`()[]")
                    if variant:
                        phrases.add(variant)
    ordered = sorted(phrases, key=len, reverse=True)
    return ordered


def _vkrev_phrase_regex(phrase: str) -> re.Pattern[str]:
    words = [re.escape(part) for part in phrase.split() if part]
    if not words:
        return re.compile(r"^$")
    pattern = r"\s+".join(words)
    return re.compile(rf"(?i)\b{pattern}\b")


def _vkrev_apply_title_instructions(
    title: str | None, instructions: str | None
) -> str:
    candidate = (title or "").strip()
    if not candidate:
        return ""
    candidate = re.sub(r"\s+", " ", candidate)
    instructions_clean = (instructions or "").strip()
    if instructions_clean:
        lowered = instructions_clean.casefold()
        if any(token in lowered for token in ("эмодзи", "emoji", "смай")):
            candidate = _EMOJI_RE.sub("", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
        for phrase in _vkrev_extract_forbidden_phrases(instructions_clean):
            pattern = _vkrev_phrase_regex(phrase)
            candidate = pattern.sub("", candidate)
            candidate = re.sub(r"\s{2,}", " ", candidate)
            candidate = candidate.strip()
    candidate = candidate.strip()
    candidate = candidate.strip(_QUOTE_CHARS + " -–—,:;")
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    candidate = candidate.strip(_QUOTE_CHARS)
    if candidate and candidate[0].islower():
        candidate = candidate[0].upper() + candidate[1:]
    return candidate[:64]


def _vkrev_select_story_title(
    fallback_title: str,
    default_title: str,
    editor_title: str | None,
    instructions: str | None,
) -> str:
    candidates = [editor_title, fallback_title, default_title, "История"]
    for candidate in candidates:
        cleaned = _vkrev_apply_title_instructions(candidate, instructions)
        if cleaned:
            return cleaned
    return "История"


def _vkrev_story_placement_keyboard(inbox_id: int) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="В конце",
                    callback_data=f"vkrev:storypos:end:{inbox_id}",
                )
            ],
            [
                types.InlineKeyboardButton(
                    text="Посреди текста",
                    callback_data=f"vkrev:storypos:middle:{inbox_id}",
                )
            ],
        ]
    )


async def _vkrev_handle_story_choice(
    callback: types.CallbackQuery,
    placement: str,
    inbox_id_hint: int,
    db: Database,
    bot: Bot,
) -> None:
    operator_id = callback.from_user.id
    state = vk_review_story_sessions.get(operator_id)
    if not state:
        await bot.send_message(callback.message.chat.id, "Нет активной истории")
        return
    inbox_id = state.inbox_id
    if inbox_id_hint and inbox_id_hint != inbox_id:
        logging.info(
            "vk_review story inbox mismatch operator=%s stored=%s hint=%s",
            operator_id,
            inbox_id,
            inbox_id_hint,
        )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, text FROM vk_inbox WHERE id=?",
            (inbox_id,),
        )
        row = await cur.fetchone()
    if not row:
        vk_review_story_sessions.pop(operator_id, None)
        await bot.send_message(callback.message.chat.id, "Инбокс не найден")
        return
    group_id, post_id, text = row
    raw_title = _vkrev_story_title(text, group_id, post_id)
    default_title = f"История VK {group_id}_{post_id}"
    source_url = f"https://vk.com/wall-{group_id}_{post_id}"
    photos = await _vkrev_fetch_photos(group_id, post_id, db, bot)
    image_mode = "inline" if placement == "middle" else "tail"
    source_text = text or ""
    editor_html: str | None = None
    editor_title_candidate: str | None = None
    pitch_text = ""
    if source_text.strip():
        try:
            pitch_text = await compose_story_pitch_via_4o(
                source_text,
                title=raw_title,
                instructions=state.instructions,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "vk_review story pitch failed",  # pragma: no cover - logging only
                extra={
                    "operator": operator_id,
                    "inbox_id": inbox_id,
                    "error": str(exc),
                },
            )
            pitch_text = ""
        try:
            editor_candidate = await compose_story_editorial_via_4o(
                source_text,
                title=raw_title,
                instructions=state.instructions,
            )
        except Exception as exc:
            logging.warning(
                "vk_review story editor request failed",  # pragma: no cover - logging only
                extra={
                    "operator": operator_id,
                    "inbox_id": inbox_id,
                    "error": str(exc),
                },
            )
        else:
            cleaned = editor_candidate.strip()
            if cleaned:
                editor_html = cleaned
                editor_title_candidate = _vkrev_extract_editor_title(cleaned)
            else:
                logging.warning(
                    "vk_review story editor returned empty response",
                    extra={"operator": operator_id, "inbox_id": inbox_id},
                )
    pitch_text = (pitch_text or "").strip()
    if pitch_text:
        pitch_html = f"<p><i>{html.escape(pitch_text)}</i></p>"
        if editor_html:
            editor_html = pitch_html + "\n" + editor_html
        else:
            editor_html = pitch_html
    telegraph_title = _vkrev_select_story_title(
        raw_title,
        default_title,
        editor_title_candidate,
        state.instructions,
    )
    try:
        result = await create_source_page(
            telegraph_title,
            source_text,
            source_url,
            editor_html,
            db=None,
            catbox_urls=photos,
            image_mode=image_mode,
            page_mode="history",
        )
    except Exception as exc:  # pragma: no cover - network and external API
        logging.exception(
            "vk_review story creation failed",
            extra={"operator": operator_id, "inbox_id": inbox_id},
        )
        await bot.send_message(
            callback.message.chat.id, f"❌ Не удалось создать историю: {exc}"
        )
        return
    if not result:
        await bot.send_message(
            callback.message.chat.id, "❌ Не удалось создать историю"
        )
        return
    url, _path, catbox_msg, _uploaded = result
    if catbox_msg:
        logging.info("vkrev story catbox: %s", catbox_msg)
    if not url:
        await bot.send_message(
            callback.message.chat.id, "❌ Не удалось получить ссылку на Telegraph"
        )
        return
    placement_display = {
        "end": "в конце",
        "middle": "посреди текста",
    }.get(placement, placement or "неизвестно")
    with contextlib.suppress(Exception):
        await callback.message.edit_reply_markup()
    message_lines = [f"История готова ({placement_display}): {url}"]
    if pitch_text:
        message_lines.append(pitch_text)
    await bot.send_message(
        callback.message.chat.id,
        "\n".join(message_lines),
    )
    vk_review_story_sessions.pop(operator_id, None)


async def _handle_pyramida_extraction(
    chat_id: int,
    operator_id: int,
    inbox_id: int,
    batch_id: str,
    post_text: str,
    db: Database,
    bot: Bot,
) -> None:
    """Handle Pyramida event extraction from VK post.
    
    1. Extract pyramida.info URLs from post text
    2. Run Kaggle kernel to parse events
    3. Process events (add to DB without pages rebuild)
    4. Mark post as imported
    5. Show next post
    """
    from source_parsing.pyramida import (
        extract_pyramida_urls,
        run_pyramida_kaggle_kernel,
        parse_pyramida_output,
        process_pyramida_events,
    )
    
    # 1. Extract URLs
    urls = extract_pyramida_urls(post_text)
    if not urls:
        await bot.send_message(chat_id, "❌ Не найдены ссылки на pyramida.info")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    status_msg = await bot.send_message(chat_id, f"🔮 Найдено {len(urls)} ссылок. Запускаю Kaggle...")
    
    async def _status_cb(text: str):
        try:
            await bot.edit_message_text(
                f"🔮 {text}",
                chat_id=chat_id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass
            
    # 2. Run Kaggle kernel
    try:
        status, output_files, duration = await run_pyramida_kaggle_kernel(urls, status_callback=_status_cb)
    except Exception as e:
        logging.exception("pyramida_extraction: kaggle failed")
        await bot.send_message(chat_id, f"❌ Ошибка Kaggle: {e}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    if status != "complete":
        await bot.send_message(chat_id, f"❌ Kaggle завершился с ошибкой: {status}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    await bot.send_message(chat_id, f"✅ Kaggle завершён за {duration:.1f}с. Обрабатываю...")
    
    # Send JSON files to chat
    for file_path in output_files:
        try:
            await bot.send_document(
                chat_id,
                types.FSInputFile(file_path),
                caption=f"📄 {os.path.basename(file_path)}"
            )
        except Exception as e:
            logging.error(f"Failed to send JSON file {file_path}: {e}")
    
    # 3. Parse and process events
    try:
        events = parse_pyramida_output(output_files)
    except Exception as e:
        logging.exception("pyramida_extraction: parse failed")
        await bot.send_message(chat_id, f"❌ Ошибка парсинга: {e}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    if not events:
        await bot.send_message(chat_id, "⚠️ Не найдено событий в результатах Kaggle")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    await bot.send_message(chat_id, f"📝 Обрабатываю {len(events)} событий...")
    
    try:
        stats = await process_pyramida_events(
            db,
            bot,
            events,
            chat_id=chat_id,
            skip_pages_rebuild=True,
        )
    except Exception as e:
        logging.exception("pyramida_extraction: processing failed")
        await bot.send_message(chat_id, f"❌ Ошибка обработки: {e}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    # 4. Send summary
    summary_lines = [
        "🔮 **Pyramida импорт завершён**",
        f"✅ Добавлено: {stats.new_added}",
    ]
    if stats.ticket_updated:
        summary_lines.append(f"🔄 Обновлено: {stats.ticket_updated}")
    if stats.failed:
        summary_lines.append(f"❌ Ошибок: {stats.failed}")
    
    await bot.send_message(chat_id, "\n".join(summary_lines), parse_mode="Markdown")
    
    # 5. Mark post as imported (use first event date or today)
    from datetime import datetime, timezone
    event_date = None
    if events and events[0].parsed_date:
        event_date = events[0].parsed_date
    else:
        event_date = datetime.now(timezone.utc).date().isoformat()
    
    # Don't mark as imported - let user decide with accept/reject buttons
    # Just show next post
    await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)


async def _handle_dom_iskusstv_extraction(
    chat_id: int,
    operator_id: int,
    inbox_id: int,
    batch_id: str,
    post_text: str,
    db: Database,
    bot: Bot,
) -> None:
    """Handle Dom Iskusstv event extraction from VK post.
    
    1. Extract домискусств.рф URLs from post text
    2. Run Kaggle kernel to parse events
    3. Process events (add to DB without pages rebuild)
    4. Show next post
    """
    from source_parsing.dom_iskusstv import (
        extract_dom_iskusstv_urls,
        run_dom_iskusstv_kaggle_kernel,
        parse_dom_iskusstv_output,
        process_dom_iskusstv_events,
    )
    
    # 1. Extract URLs
    urls = extract_dom_iskusstv_urls(post_text)
    if not urls:
        await bot.send_message(chat_id, "❌ Не найдены ссылки на домискусств.рф")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    status_msg = await bot.send_message(chat_id, f"🏛 Найдено {len(urls)} ссылок. Запускаю Kaggle...")
    
    async def _status_cb(text: str):
        try:
            await bot.edit_message_text(
                f"🏛 {text}",
                chat_id=chat_id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass
            
    # 2. Run Kaggle kernel
    try:
        status, output_files, duration = await run_dom_iskusstv_kaggle_kernel(urls, status_callback=_status_cb)
    except Exception as e:
        logging.exception("dom_iskusstv_extraction: kaggle failed")
        await bot.send_message(chat_id, f"❌ Ошибка Kaggle: {e}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    if status != "complete":
        await bot.send_message(chat_id, f"❌ Kaggle завершился с ошибкой: {status}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    await bot.send_message(chat_id, f"✅ Kaggle завершён за {duration:.1f}с. Обрабатываю...")
    
    # Send JSON files to chat
    for file_path in output_files:
        try:
            await bot.send_document(
                chat_id,
                types.FSInputFile(file_path),
                caption=f"📄 {os.path.basename(file_path)}"
            )
        except Exception as e:
            logging.error(f"Failed to send JSON file {file_path}: {e}")
    
    # 3. Parse and process events
    try:
        events = parse_dom_iskusstv_output(output_files)
    except Exception as e:
        logging.exception("dom_iskusstv_extraction: parse failed")
        await bot.send_message(chat_id, f"❌ Ошибка парсинга: {e}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    if not events:
        await bot.send_message(chat_id, "⚠️ Не найдено событий в результатах Kaggle")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    await bot.send_message(chat_id, f"📝 Обрабатываю {len(events)} событий...")
    
    try:
        stats = await process_dom_iskusstv_events(
            db,
            bot,
            events,
            chat_id=chat_id,
            skip_pages_rebuild=True,
        )
    except Exception as e:
        logging.exception("dom_iskusstv_extraction: processing failed")
        await bot.send_message(chat_id, f"❌ Ошибка обработки: {e}")
        await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)
        return
    
    # 4. Send summary
    summary_lines = [
        "🏛 **Дом искусств импорт завершён**",
        f"✅ Добавлено: {stats.new_added}",
    ]
    if stats.ticket_updated:
        summary_lines.append(f"🔄 Обновлено: {stats.ticket_updated}")
    if stats.failed:
        summary_lines.append(f"❌ Ошибок: {stats.failed}")
    
    await bot.send_message(chat_id, "\n".join(summary_lines), parse_mode="Markdown")
    
    # Show next post
    await _vkrev_show_next(chat_id, batch_id, operator_id, db, bot)


async def handle_vk_review_cb(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:

    assert callback.data
    parts = callback.data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    answered = False
    if action in {
        "accept",
        "accept_extra",
        "accept_fest",
        "accept_fest_extra",
        "reject",
        "skip",
    }:
        inbox_id = int(parts[2]) if len(parts) > 2 else 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT review_batch FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
        batch_id = row[0] if row else ""
        if action in {"accept", "accept_fest"}:
            force_festival = action == "accept_fest"
            if action == "accept" and len(parts) > 3:
                force_arg = parts[3].strip().lower()
                force_festival = force_arg in {"1", "true", "fest", "festival", "force"}
            
            # Atomic lock: try to switch content to 'importing' state
            # This prevents double-clicks from spawning multiple import flows
            async with db.raw_conn() as conn:
                cur = await conn.execute(
                    "UPDATE vk_inbox SET status='importing' WHERE id=? AND status IN ('locked', 'pending') RETURNING id",
                    (inbox_id,),
                )
                locked_row = await cur.fetchone()
                await conn.commit()
            
            if not locked_row:
                # Already importing or processed
                await callback.answer("⏳ Уже в обработке или завершено")
                return

            await callback.answer("Запускаю импорт…")
            answered = True
            await bot.send_message(
                callback.message.chat.id,
                "⏳ Начинаю импорт события…",
            )
            await _vkrev_import_flow(
                callback.message.chat.id,
                callback.from_user.id,
                inbox_id,
                batch_id,
                db,
                bot,
                force_festival=force_festival,
            )
        elif action in {"accept_extra", "accept_fest_extra"}:
            force_festival = action == "accept_fest_extra"
            vk_review_extra_sessions[callback.from_user.id] = (
                inbox_id,
                batch_id,
                force_festival,
            )
            await bot.send_message(
                callback.message.chat.id,
                "Отправьте доп. информацию одним сообщением",
            )
        elif action == "reject":
            await vk_review.mark_rejected(db, inbox_id)
            vk_review_actions_total["rejected"] += 1
            post_url: str | None = None
            try:
                async with db.raw_conn() as conn:
                    cur = await conn.execute(
                        "SELECT group_id, post_id FROM vk_inbox WHERE id=?",
                        (inbox_id,),
                    )
                    row_ids = await cur.fetchone()
            except Exception:
                row_ids = None
            if row_ids:
                gid, pid = row_ids
                post_url = f"https://vk.com/wall-{gid}_{pid}"
                try:
                    mark_vk_import_result(
                        group_id=gid,
                        post_id=pid,
                        url=post_url,
                        outcome="rejected",
                        event_id=None,
                        reject_code=VkImportRejectCode.MANUAL_REVIEW,
                        reject_note=VkImportRejectCode.MANUAL_REVIEW.value,
                    )
                except Exception:
                    logging.exception("vk_import_result.supabase_failed")
            await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)
        elif action == "skip":
            await vk_review.mark_skipped(db, inbox_id)
            vk_review_actions_total["skipped"] += 1
            await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)
    elif action == "story":
        inbox_id = int(parts[2]) if len(parts) > 2 else 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT review_batch FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
        batch_id = row[0] if row else ""
        vk_review_story_sessions[callback.from_user.id] = VkReviewStorySession(
            inbox_id=inbox_id,
            batch_id=batch_id,
        )
        guidance_keyboard = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text="Да",
                        callback_data=f"vkrev:storyinstr:yes:{inbox_id}",
                    )
                ],
                [
                    types.InlineKeyboardButton(
                        text="Нет",
                        callback_data=f"vkrev:storyinstr:no:{inbox_id}",
                    )
                ],
            ]
        )
        await bot.send_message(
            callback.message.chat.id,
            "Нужны дополнительные инструкции редактору?",
            reply_markup=guidance_keyboard,
        )
        answered = True
        await callback.answer()
    elif action == "storyinstr":
        choice = parts[2] if len(parts) > 2 else ""
        inbox_id = int(parts[3]) if len(parts) > 3 else 0
        state = vk_review_story_sessions.get(callback.from_user.id)
        if not state:
            await callback.answer("Нет активной истории", show_alert=True)
            return
        if inbox_id and inbox_id != state.inbox_id:
            logging.info(
                "vk_review storyinstr inbox mismatch operator=%s stored=%s hint=%s",
                callback.from_user.id,
                state.inbox_id,
                inbox_id,
            )
        if choice == "yes":
            state.awaiting_instructions = True
            state.instructions = None
            await bot.send_message(
                callback.message.chat.id,
                "Отправьте дополнительные инструкции редактору одним сообщением. "
                "Если инструкций нет, отправьте «нет» или пустое сообщение.",
            )
        else:
            state.awaiting_instructions = False
            keyboard = _vkrev_story_placement_keyboard(state.inbox_id)
            await bot.send_message(
                callback.message.chat.id,
                "Где разместить иллюстрации?",
                reply_markup=keyboard,
            )
        answered = True
        await callback.answer()
    elif action == "storypos":
        placement = parts[2] if len(parts) > 2 else ""
        inbox_id = int(parts[3]) if len(parts) > 3 else 0
        answered = True
        await callback.answer("Создаю историю…")
        await _vkrev_handle_story_choice(callback, placement, inbox_id, db, bot)
    elif action == "pyramida":
        # Handle Pyramida event extraction
        inbox_id = int(parts[2]) if len(parts) > 2 else 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT review_batch, text FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
        if not row:
            await callback.answer("Пост не найден", show_alert=True)
            return
        batch_id, post_text = row
        await callback.answer("Извлекаю события из Pyramida…")
        answered = True
        await _handle_pyramida_extraction(
            callback.message.chat.id,
            callback.from_user.id,
            inbox_id,
            batch_id,
            post_text or "",
            db,
            bot,
        )
    elif action == "domiskusstv":
        # Handle Dom Iskusstv event extraction
        inbox_id = int(parts[2]) if len(parts) > 2 else 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT review_batch, text FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
        if not row:
            await callback.answer("Пост не найден", show_alert=True)
            return
        batch_id, post_text = row
        
        from source_parsing.dom_iskusstv import extract_dom_iskusstv_urls
        if not extract_dom_iskusstv_urls(post_text or ""):
            await callback.answer("Ссылки не найдены")
            await bot.send_message(
                callback.message.chat.id, 
                "⚠️ В тексте поста не найдено ссылок на домискусств.рф (или tickets.domiskusstv.ru)."
            )
            return

        await callback.answer("Извлекаю события из Дом искусств…")
        answered = True
        await _handle_dom_iskusstv_extraction(
            callback.message.chat.id,
            callback.from_user.id,
            inbox_id,
            batch_id,
            post_text or "",
            db,
            bot,
        )
    elif action == "stop":

        async with db.raw_conn() as conn:
            await conn.execute(
                "UPDATE vk_inbox SET status='pending', locked_by=NULL, locked_at=NULL WHERE locked_by=?",
                (callback.from_user.id,),
            )
            await conn.commit()
        vk_review_extra_sessions.pop(callback.from_user.id, None)
        buttons = [
            [types.KeyboardButton(text=VK_BTN_ADD_SOURCE)],
            [types.KeyboardButton(text=VK_BTN_LIST_SOURCES)],
            [
                types.KeyboardButton(text=VK_BTN_CHECK_EVENTS),
                types.KeyboardButton(text=VK_BTN_QUEUE_SUMMARY),
            ],
        ]
        markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
        await bot.send_message(callback.message.chat.id, "Остановлено", reply_markup=markup)
    elif action == "finish":
        batch_id = parts[2] if len(parts) > 2 else ""
        reports: list[tuple[str, str]] = []

        async def rebuild_cb(db_: Database, month: str) -> None:
            report = await _perform_pages_rebuild(db_, [month], force=True)
            reports.append((month, report))

        months = await vk_review.finish_batch(db, batch_id, rebuild_cb)
        if months:
            await bot.send_message(
                callback.message.chat.id,
                "Запущен rebuild для: " + ", ".join(months),
            )
            for _, report in reports:
                if report:
                    await bot.send_message(callback.message.chat.id, report)
        else:
            await bot.send_message(callback.message.chat.id, "Нет месяцев для обновления")
    elif action == "repost":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        await _vkrev_handle_repost(callback, event_id, db, bot)
    elif action == "shortpost":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        await _vkrev_handle_shortpost(callback, event_id, db, bot)
    elif action == "shortpost_pub":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        await _vkrev_publish_shortpost(
            event_id, db, bot, callback.message.chat.id, callback.from_user.id
        )
    elif action == "shortpost_edit":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        vk_shortpost_edit_sessions[callback.from_user.id] = (
            event_id,
            callback.message.message_id,
        )
        await bot.send_message(
            callback.message.chat.id,
            "Отправьте новый текст поста одной строкой/сообщением",
        )
    if not answered:
        await callback.answer()


async def _vkrev_handle_repost(callback: types.CallbackQuery, event_id: int, db: Database, bot: Bot) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, review_batch FROM vk_inbox WHERE imported_event_id=?",
            (event_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(callback.message.chat.id, "❌ Репост не удался: нет события")
        return
    group_id, post_id, batch_id = row
    vk_url = f"https://vk.com/wall-{group_id}_{post_id}"
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev:
        await bot.send_message(callback.message.chat.id, "❌ Репост не удался: нет события")
        return

    if VK_ALLOW_TRUE_REPOST:
        object_id = f"wall-{group_id}_{post_id}"
        target_group = int(VK_AFISHA_GROUP_ID.lstrip('-')) if VK_AFISHA_GROUP_ID else None
        params = {"object": object_id}
        if target_group:
            params["group_id"] = target_group
        global vk_repost_attempts_total, vk_repost_errors_total
        vk_repost_attempts_total += 1
        try:
            data = await _vk_api("wall.repost", params, db, bot, token=VK_TOKEN_AFISHA)
            post = data.get("response", {}).get("post_id")
            if not post:
                raise RuntimeError("no post_id")
            url = f"https://vk.com/wall-{VK_AFISHA_GROUP_ID.lstrip('-')}_{post}"
            await vk_review.save_repost_url(db, event_id, url)
            await bot.send_message(callback.message.chat.id, url)
        except Exception as e:  # pragma: no cover
            vk_repost_errors_total += 1
            logging.exception("vk repost failed")
            await bot.send_message(
                callback.message.chat.id,
                f"❌ Репост не удался: {getattr(e, 'code', getattr(e, 'message', str(e)))}",
            )
        await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)
        return

    try:
        response = await vk_api("wall.getById", posts=f"-{group_id}_{post_id}")
    except Exception:
        items: list[dict[str, Any]] = []
    else:
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
    photos = _vkrev_collect_photo_ids(items, VK_SHORTPOST_MAX_PHOTOS)
    attachments = ",".join(photos) if photos else vk_url
    message = f"Репост: {ev.title}\n\n[{vk_url}|Источник]"
    params = {
        "owner_id": f"-{VK_AFISHA_GROUP_ID.lstrip('-')}",
        "from_group": 1,
        "message": message,
        "attachments": attachments,
        "copyright": vk_url,
        "signed": 0,
    }
    try:
        data = await _vk_api(
            "wall.post",
            params,
            db,
            bot,
            token=VK_TOKEN_AFISHA,
            skip_captcha=True,
        )
        post = data.get("response", {}).get("post_id")
        if not post:
            raise RuntimeError("no post_id")
        url = f"https://vk.com/wall-{VK_AFISHA_GROUP_ID.lstrip('-')}_{post}"
        await vk_review.save_repost_url(db, event_id, url)
        await bot.send_message(callback.message.chat.id, url)
    except VKAPIError as e:
        logging.error(
            "vk.repost_failed actor=%s token=%s code=%s msg=%s",
            e.actor,
            e.token,
            e.code,
            e.message,
        )
        await bot.send_message(
            callback.message.chat.id,
            f"❌ Репост не удался: {e.message}",
        )
    except Exception as e:  # pragma: no cover
        await bot.send_message(
            callback.message.chat.id,
            f"❌ Репост не удался: {getattr(e, 'message', str(e))}",
        )
    await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)


def _normalize_location_part(part: str | None) -> str:
    if not part:
        return ""
    normalized = unicodedata.normalize("NFKC", part)
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


async def _vkrev_build_shortpost(
    ev: Event,
    vk_url: str,
    *,
    db: Database | None = None,
    session: AsyncSession | None = None,
    bot: Bot | None = None,
    for_preview: bool = False,
    poster_texts: Sequence[str] | None = None,
) -> tuple[str, str | None]:

    text_len = len(ev.source_text or "")
    if text_len < 200:
        max_sent = 1
    elif text_len < 500:
        max_sent = 2
    elif text_len < 800:
        max_sent = 3
    else:
        max_sent = 4
    summary = await build_short_vk_text(
        ev,
        ev.source_text or "",
        max_sent,
        poster_texts=poster_texts,
    )

    start_date: date | None = None
    try:
        start_date = date.fromisoformat(ev.date)
    except (TypeError, ValueError):
        start_date = None

    end_date_obj: date | None = None
    if ev.end_date:
        try:
            end_date_obj = date.fromisoformat(ev.end_date)
        except (TypeError, ValueError):
            end_date_obj = None

    if start_date:
        default_date_str = f"{start_date.day} {MONTHS[start_date.month - 1]}"
    else:
        try:
            parts = ev.date.split("-")
            day = int(parts[2])
            month = int(parts[1])
        except (AttributeError, IndexError, ValueError):
            default_date_str = ev.date
        else:
            default_date_str = f"{day} {MONTHS[month - 1]}"

    today = date.today()

    ongoing_exhibition = (
        ev.event_type == "выставка"
        and start_date is not None
        and end_date_obj is not None
        and start_date <= today <= end_date_obj
    )

    if ongoing_exhibition:
        end_month_name = MONTHS[end_date_obj.month - 1]
        year_suffix = ""
        if start_date.year != end_date_obj.year:
            year_suffix = f" {end_date_obj.year}"
        date_line = f"🗓 по {end_date_obj.day} {end_month_name}{year_suffix}"
    else:
        time_part = f" ⏰ {ev.time}" if ev.time and ev.time != "00:00" else ""
        date_line = f"🗓 {default_date_str}{time_part}"

    type_line: str | None = None
    type_line_used_tag: str | None = None
    raw_event_type = (ev.event_type or "").strip()
    if raw_event_type:
        event_type_lower = raw_event_type.casefold()
        normalized_event_type = re.sub(
            r"[^0-9a-zа-яё]+", "_", event_type_lower
        ).strip("_")
        if normalized_event_type:
            normalized_hashtag = f"#{normalized_event_type}"
            type_line = normalized_hashtag
            type_line_used_tag = normalized_hashtag
            if re.search(r"[-–—]", raw_event_type):
                hyphen_free = re.sub(r"[^0-9a-zа-яё]", "", event_type_lower)
                if hyphen_free:
                    type_line = f"#{hyphen_free}"
                    type_line_used_tag = type_line

    tags = await build_short_vk_tags(ev, summary, used_type_hashtag=type_line_used_tag)
    title_line = ev.title.upper() if ev.title else ""
    if getattr(ev, "is_free", False):
        title_line = f"🆓 {title_line}".strip()
    lines = [
        title_line,
        "",
    ]
    lines.append(date_line)
    if type_line:
        lines.append(type_line)
    ticket_url_for_message = (
        format_vk_short_url(ev.vk_ticket_short_url)
        if ev.vk_ticket_short_url
        else ev.ticket_link
    )
    if ev.ticket_link and not for_preview:
        short_result = await ensure_vk_short_ticket_link(
            ev,
            db,
            session=session,
            bot=bot,
            vk_api_fn=_vk_api,
        )
        if short_result:
            ticket_url_for_message = format_vk_short_url(short_result[0])
    if ev.ticket_link:
        if getattr(ev, "is_free", False):
            lines.append(
                f"🆓 Бесплатно, по регистрации {ticket_url_for_message}"
            )
        else:
            lines.append(f"🎟 Билеты: {ticket_url_for_message}")
    loc_parts: list[str] = []
    existing_normalized: set[str] = set()
    for part in (ev.location_name, ev.location_address):
        if part:
            loc_parts.append(part)
            normalized = _normalize_location_part(part)
            if normalized:
                existing_normalized.add(normalized)
    if ev.city:
        city_normalized = _normalize_location_part(ev.city)
        if not city_normalized or city_normalized not in existing_normalized:
            loc_parts.append(ev.city)
            if city_normalized:
                existing_normalized.add(city_normalized)
    location_text = await build_short_vk_location(loc_parts)
    if location_text:
        lines.append(f"📍 {location_text}")
    lines.append("")
    lines.append(summary)
    summary_idx = len(lines) - 1
    lines.append("")
    if for_preview:
        lines.append("Источник")
        lines.append(vk_url)
    else:
        lines.append(f"[{vk_url}|Источник]")
    lines.append("")
    lines.append(" ".join(tags))
    message = "\n".join(lines)
    if len(message) > 4096:
        excess = len(message) - 4096
        lines[summary_idx] = lines[summary_idx][: -excess]
        message = "\n".join(lines)
    link_attachment = ev.telegraph_url or vk_url
    return message, link_attachment


async def _vkrev_handle_shortpost(callback: types.CallbackQuery, event_id: int, db: Database, bot: Bot) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, review_batch FROM vk_inbox WHERE imported_event_id=?",
            (event_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(callback.message.chat.id, "❌ Не удалось: нет события")
        return
    group_id, post_id, batch_id = row
    vk_url = f"https://vk.com/wall-{group_id}_{post_id}"
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            await bot.send_message(
                callback.message.chat.id, "❌ Не удалось: нет события"
            )
            return

        poster_texts = await get_event_poster_texts(event_id, db)

        message, link_attachment = await _vkrev_build_shortpost(
            ev,
            vk_url,
            db=db,
            session=session,
            bot=bot,
            for_preview=True,
            poster_texts=poster_texts,
        )
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="Опубликовать",
                    callback_data=f"vkrev:shortpost_pub:{event_id}",
                ),
                types.InlineKeyboardButton(
                    text="Изменить",
                    callback_data=f"vkrev:shortpost_edit:{event_id}",
                ),
            ]
        ]
    )
    await bot.send_message(callback.message.chat.id, message, reply_markup=markup)
    logging.info(
        "shortpost_preview_sent",
        extra={"eid": event_id, "chat_id": callback.message.chat.id},
    )
    vk_shortpost_ops[event_id] = VkShortpostOpState(
        chat_id=callback.message.chat.id,
        preview_text=message,
        preview_link_attachment=link_attachment,
    )


async def _vkrev_publish_shortpost(
    event_id: int,
    db: Database,
    bot: Bot,
    actor_chat_id: int,
    operator_id: int,
    text: str | None = None,
    edited: bool = False,
) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, review_batch FROM vk_inbox WHERE imported_event_id=?",
            (event_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(actor_chat_id, "❌ Не удалось: нет события")
        return
    group_id, post_id, batch_id = row
    vk_url = f"https://vk.com/wall-{group_id}_{post_id}"
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            await bot.send_message(actor_chat_id, "❌ Не удалось: нет события")
            return
        op_state = vk_shortpost_ops.get(event_id)
        poster_texts = await get_event_poster_texts(event_id, db)
    def _ensure_publish_markup(message: str) -> str:
        lines = message.split("\n")
        markup = f"[{vk_url}|Источник]"
        for idx, line in enumerate(lines):
            if line.strip() == "Источник":
                if idx + 1 < len(lines):
                    del lines[idx + 1]
                lines[idx] = markup
                return "\n".join(lines)
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("|Источник]"):
                lines[idx] = markup
                return "\n".join(lines)
        return message

    short_ticket = None
    if text is None:
        if op_state and op_state.preview_text is not None:
            message = _ensure_publish_markup(op_state.preview_text)
            link_attachment = (
                op_state.preview_link_attachment
                if op_state.preview_link_attachment is not None
                else ev.telegraph_url or vk_url
            )
            if ev.ticket_link:
                short_ticket = await ensure_vk_short_ticket_link(
                    ev,
                    db,
                    bot=bot,
                    vk_api_fn=_vk_api,
                )
                if short_ticket:
                    message = message.replace(
                        ev.ticket_link, format_vk_short_url(short_ticket[0])
                    )
        else:
            message, link_attachment = await _vkrev_build_shortpost(
                ev,
                vk_url,
                db=db,
                bot=bot,
                poster_texts=poster_texts,
            )
    else:
        message = _ensure_publish_markup(text)
        link_attachment = ev.telegraph_url or vk_url
        if ev.ticket_link:
            short_ticket = await ensure_vk_short_ticket_link(
                ev,
                db,
                bot=bot,
                vk_api_fn=_vk_api,
            )
            if short_ticket:
                message = message.replace(
                    ev.ticket_link, format_vk_short_url(short_ticket[0])
                )

    photo_attachments: list[str] = []
    try:
        response = await vk_api("wall.getById", posts=f"-{group_id}_{post_id}")
    except Exception as exc:  # pragma: no cover - logging only
        logging.error(
            "shortpost_fetch_photos_failed gid=%s post=%s: %s",
            group_id,
            post_id,
            exc,
        )
    else:
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
        photo_attachments.extend(
            _vkrev_collect_photo_ids(items, VK_SHORTPOST_MAX_PHOTOS)
        )

    if not photo_attachments and VK_PHOTOS_ENABLED and ev.photo_urls:
        token = VK_TOKEN_AFISHA or VK_TOKEN
        if token:
            uploaded: list[str] = []
            for url in ev.photo_urls[:VK_SHORTPOST_MAX_PHOTOS]:
                photo_id = await upload_vk_photo(
                    VK_AFISHA_GROUP_ID,
                    url,
                    db,
                    bot,
                    token=token,
                    token_kind="group",
                )
                if photo_id:
                    uploaded.append(photo_id)
                elif not VK_USER_TOKEN:
                    logging.info(
                        "shortpost_photo_upload_skipped gid=%s post=%s reason=user_token_required",
                        group_id,
                        post_id,
                    )
                    break
            photo_attachments.extend(uploaded)
        else:
            logging.info(
                "shortpost_photo_upload_skipped gid=%s post=%s reason=no_token",
                group_id,
                post_id,
            )

    attachments: list[str] = []
    if photo_attachments:
        attachments.extend(photo_attachments)
        if link_attachment:
            attachments.append(link_attachment)

    attachments_str = ",".join(attachments) if attachments else None

    params = {
        "owner_id": f"-{VK_AFISHA_GROUP_ID.lstrip('-')}",
        "from_group": 1,
        "message": message,
        "copyright": vk_url,
        "signed": 0,
    }
    if attachments_str:
        params["attachments"] = attachments_str
    operator_chat = op_state.chat_id if op_state else None
    try:
        data = await _vk_api(
            "wall.post",
            params,
            db,
            bot,
            token=VK_TOKEN_AFISHA,
            skip_captcha=True,
        )
        post = data.get("response", {}).get("post_id")
        if not post:
            raise RuntimeError("no post_id")
        url = f"https://vk.com/wall-{VK_AFISHA_GROUP_ID.lstrip('-')}_{post}"
        await vk_review.save_repost_url(db, event_id, url)
        await bot.send_message(actor_chat_id, f"✅ Опубликовано: {url}")
        if operator_chat and operator_chat != actor_chat_id:
            await bot.send_message(operator_chat, f"✅ Опубликовано: {url}")
        logging.info("shortpost_publish", extra={"eid": event_id, "edited": edited})
        vk_shortpost_ops.pop(event_id, None)
        await _vkrev_show_next(actor_chat_id, batch_id, operator_id, db, bot)
    except VKAPIError as e:
        if e.code == 14:
            msg = "Капча, публикацию не делаем. Попробуйте позже"
        else:
            msg = f"❌ Не удалось: {e.message}"
        await bot.send_message(actor_chat_id, msg)
        if operator_chat and operator_chat != actor_chat_id:
            await bot.send_message(operator_chat, msg)
        logging.warning(
            "shortpost_publish_failed code=%s actor=%s token=%s",
            e.code,
            e.actor,
            e.token,
            extra={"eid": event_id},
        )
    except Exception as e:  # pragma: no cover
        msg = f"❌ Не удалось: {getattr(e, 'message', str(e))}"
        await bot.send_message(actor_chat_id, msg)
        if operator_chat and operator_chat != actor_chat_id:
            await bot.send_message(operator_chat, msg)
        logging.warning("shortpost_publish_failed", extra={"eid": event_id, "error": str(e)})


def extract_message_text_with_links(message: types.Message) -> str:
    """Return message text where hidden links are exposed for downstream use."""

    base_text = message.text or message.caption or ""
    if not base_text:
        return ""

    html_text, _mode = ensure_html_text(message)
    if not html_text:
        return base_text

    def escape_md_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    def escape_md_url(value: str) -> str:
        return value.replace("\\", "\\\\").replace(")", "\\)")

    def repl_anchor(match: re.Match[str]) -> str:
        href = match.group(1)
        label_html = match.group(2)
        label = re.sub(r"</?[^>]+>", "", label_html)
        label = html.unescape(label)
        label = label.replace("\xa0", " ")
        label_md = escape_md_label(label)
        if not label_md.strip():
            return href
        return f"[{label_md}]({escape_md_url(href)})"

    text = re.sub(r"(?is)<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", repl_anchor, html_text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?i)</div>", "\n", text)
    text = re.sub(r"(?i)</li>", "\n", text)
    text = re.sub(r"(?i)<li>", "• ", text)
    text = re.sub(r"</?[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")

    return text or base_text


async def handle_vk_extra_message(message: types.Message, db: Database, bot: Bot) -> None:
    info = vk_review_extra_sessions.pop(message.from_user.id, None)
    if not info:
        return
    inbox_id, batch_id, force_festival = info
    operator_extra = extract_message_text_with_links(message)
    await _vkrev_import_flow(
        message.chat.id,
        message.from_user.id,
        inbox_id,
        batch_id,
        db,
        bot,
        operator_extra=operator_extra,
        force_festival=force_festival,
    )


_VK_STORY_SKIP_TOKENS = {
    "нет",
    "нет.",
    "нет!",
    "нет,",
    "нету",
    "не надо",
    "не надо.",
    "не нужно",
    "не нужно.",
    "skip",
    "skip.",
    "/skip",
    "пропуск",
    "no",
}


async def handle_vk_story_instruction_message(
    message: types.Message, db: Database, bot: Bot
) -> None:
    state = vk_review_story_sessions.get(message.from_user.id)
    if not state or not state.awaiting_instructions:
        return
    raw_text = extract_message_text_with_links(message)
    cleaned = (raw_text or "").strip()
    if cleaned.casefold() in _VK_STORY_SKIP_TOKENS or not cleaned:
        instructions: str | None = None
    else:
        instructions = cleaned
    state.instructions = instructions
    state.awaiting_instructions = False
    if instructions:
        ack = "Получил инструкции, записал."
    else:
        ack = "Инструкций нет, продолжаем."
    await bot.send_message(message.chat.id, ack)
    keyboard = _vkrev_story_placement_keyboard(state.inbox_id)
    await bot.send_message(
        message.chat.id,
        "Где разместить иллюстрации?",
        reply_markup=keyboard,
    )


async def handle_tourist_note_message(message: types.Message, db: Database, bot: Bot) -> None:
    try:
        session_state = tourist_note_sessions.pop(message.from_user.id)
    except KeyError:
        await bot.send_message(
            message.chat.id,
            "Сессия для комментария истекла, нажмите кнопку заново.",
        )
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        event = await session.get(Event, session_state.event_id)
        if not event or not _user_can_label_event(user):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        note_text = (message.text or "").strip()
        is_trimmed = False
        if len(note_text) > 500:
            note_text = note_text[:500]
            is_trimmed = True
        event.tourist_note = note_text or None
        event.tourist_label_by = message.from_user.id
        event.tourist_label_at = datetime.now(timezone.utc)
        event.tourist_label_source = "operator"
        session.add(event)
        await session.commit()
        await session.refresh(event)
    logging.info(
        "tourist_note_saved",
        extra={
            "event_id": session_state.event_id,
            "user_id": message.from_user.id,
            "has_note": bool(event.tourist_note),
        },
    )
    confirmation_text = (
        "Комментарий сохранён (обрезан до 500 символов)."
        if is_trimmed
        else "Комментарий сохранён."
    )
    await bot.send_message(message.chat.id, confirmation_text)
    base_markup = session_state.markup
    new_markup = replace_tourist_block(
        base_markup, event, session_state.source, menu=session_state.menu
    )
    original_text = session_state.message_text
    if original_text is not None:
        updated_text = apply_tourist_status_to_text(original_text, event)
        try:
            await bot.edit_message_text(
                chat_id=session_state.chat_id,
                message_id=session_state.message_id,
                text=updated_text,
                reply_markup=new_markup,
            )
        except TelegramBadRequest as exc:  # pragma: no cover - Telegram quirks
            logging.warning(
                "tourist_note_message_update_failed",
                extra={"event_id": session_state.event_id, "error": exc.message},
            )
            with contextlib.suppress(Exception):
                await bot.edit_message_reply_markup(
                    chat_id=session_state.chat_id,
                    message_id=session_state.message_id,
                    reply_markup=new_markup,
                )
    else:
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(
                chat_id=session_state.chat_id,
                message_id=session_state.message_id,
                reply_markup=new_markup,
            )


async def handle_vk_shortpost_edit_message(message: types.Message, db: Database, bot: Bot) -> None:
    info = vk_shortpost_edit_sessions.pop(message.from_user.id, None)
    if not info:
        return
    event_id, _ = info
    await _vkrev_publish_shortpost(
        event_id,
        db,
        bot,
        message.chat.id,
        message.from_user.id,
        text=message.text or "",
        edited=True,
    )


async def handle_vk_next_callback(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    try:
        _, batch, idx = callback.data.split(":", 2)
        index = int(idx)
    except Exception:
        await callback.answer()
        return
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM vk_tmp_post WHERE batch=?", (batch,)
        )
        total = (await cursor.fetchone())[0]
    await send_vk_tmp_post(callback.message.chat.id, batch, index, total, db, bot)
    await callback.answer()


async def handle_partner_info_message(message: types.Message, db: Database, bot: Bot):
    uid = partner_info_sessions.get(message.from_user.id)
    if not uid:
        return
    text = (message.text or "").strip()
    if "," not in text:
        await bot.send_message(message.chat.id, "Please send 'Organization, location'")
        return
    org, loc = [p.strip() for p in text.split(",", 1)]
    async with db.get_session() as session:
        pending = await session.get(PendingUser, uid)
        if not pending:
            await bot.send_message(message.chat.id, "Pending user not found")
            partner_info_sessions.pop(message.from_user.id, None)
            return
        session.add(
            User(
                user_id=uid,
                username=pending.username,
                is_partner=True,
                organization=org,
                location=loc,
            )
        )
        await session.delete(pending)
        await session.commit()
    partner_info_sessions.pop(message.from_user.id, None)
    await bot.send_message(uid, "You are approved as partner")
    await bot.send_message(
        message.chat.id,
        f"User {uid} approved as partner at {org}, {loc}",
    )
    logging.info("approved user %s as partner %s, %s", uid, org, loc)


async def handle_festival_edit_message(message: types.Message, db: Database, bot: Bot):
    state = festival_edit_sessions.get(message.from_user.id)
    if not state:
        return
    fid, field = state
    if field is None:
        return
    if field == FESTIVAL_EDIT_FIELD_IMAGE:
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
            if not fest:
                await bot.send_message(message.chat.id, "Festival not found")
                festival_edit_sessions.pop(message.from_user.id, None)
                return
            images: list[tuple[bytes, str]] = []
            if message.photo:
                photo = message.photo[-1]
                bio = BytesIO()
                async with span("tg-send"):
                    await bot.download(photo.file_id, destination=bio)
                data, name = ensure_jpeg(bio.getvalue(), "photo.jpg")
                images.append((data, name))
            if message.document:
                mime = message.document.mime_type or ""
                if mime.startswith("image/"):
                    bio = BytesIO()
                    async with span("tg-send"):
                        await bot.download(message.document.file_id, destination=bio)
                    doc_name = message.document.file_name or "image.jpg"
                    data, doc_name = ensure_jpeg(bio.getvalue(), doc_name)
                    images.append((data, doc_name))
                else:
                    await bot.send_message(
                        message.chat.id,
                        "Документ должен быть изображением (image/*).",
                    )
                    return
            new_urls: list[str] = []
            catbox_msg = ""
            if images:
                poster_items, catbox_msg = await process_media(
                    images, need_catbox=True, need_ocr=False
                )
                new_urls = [item.catbox_url for item in poster_items if item.catbox_url]
            else:
                text_candidate = (message.text or message.caption or "").strip()
                if text_candidate.lower().startswith(("http://", "https://")):
                    new_urls = [text_candidate]
            if not new_urls:
                await bot.send_message(
                    message.chat.id,
                    "Не удалось получить изображение. Пришлите фото, документ или ссылку на картинку.",
                )
                return
            appended_count = 0
            cover_changed = False
            for idx, url in enumerate(new_urls):
                was_cover = fest.photo_url
                already_present = url in (fest.photo_urls or [])
                changed = add_festival_photo(fest, url, make_cover=(idx == 0))
                if not changed:
                    continue
                if not already_present and url in (fest.photo_urls or []):
                    appended_count += 1
                if fest.photo_url == url and was_cover != url:
                    cover_changed = True
            if not appended_count and not cover_changed:
                await bot.send_message(
                    message.chat.id,
                    "Эта иллюстрация уже есть в альбоме фестиваля.",
                )
                return
            await session.commit()
            await session.refresh(fest)
            fest_view = Festival(**fest.model_dump())  # type: ignore[arg-type]
            fest_name = fest.name
        festival_edit_sessions[message.from_user.id] = (fid, None)
        response_lines: list[str] = []
        if appended_count:
            if appended_count == 1:
                response_lines.append("Добавлена новая иллюстрация.")
            else:
                response_lines.append(f"Добавлено иллюстраций: {appended_count}.")
        if cover_changed:
            response_lines.append("Обложка обновлена.")
        if catbox_msg:
            response_lines.append(catbox_msg)
        if response_lines:
            await bot.send_message(message.chat.id, "\n".join(response_lines))
        await show_festival_edit_menu(message.from_user.id, fest_view, bot)
        await sync_festival_page(db, fest_name)
        await sync_festivals_index_page(db)
        return
    text = (message.text or "").strip()
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        if not fest:
            await bot.send_message(message.chat.id, "Festival not found")
            festival_edit_sessions.pop(message.from_user.id, None)
            return
        if field == "description":
            fest.description = None if text in {"", "-"} else text
        elif field == "name":
            if text:
                fest.name = text
        elif field == "full":
            fest.full_name = None if text in {"", "-"} else text
        elif field == "start":
            if text in {"", "-"}:
                fest.start_date = None
            else:
                d = parse_events_date(text, timezone.utc)
                if not d:
                    await bot.send_message(message.chat.id, "Invalid date")
                    return
                fest.start_date = d.isoformat()
        elif field == "end":
            if text in {"", "-"}:
                fest.end_date = None
            else:
                d = parse_events_date(text, timezone.utc)
                if not d:
                    await bot.send_message(message.chat.id, "Invalid date")
                    return
                fest.end_date = d.isoformat()
        elif field == "site":
            fest.website_url = None if text in {"", "-"} else text
        elif field == "program":
            fest.program_url = None if text in {"", "-"} else text
        elif field == "vk":
            fest.vk_url = None if text in {"", "-"} else text
        elif field == "tg":
            fest.tg_url = None if text in {"", "-"} else text
        elif field == "ticket":
            fest.ticket_url = None if text in {"", "-"} else text
        # New parser fields
        elif field == "phone":
            fest.contacts_phone = None if text in {"", "-"} else text
        elif field == "email":
            fest.contacts_email = None if text in {"", "-"} else text
        elif field == "audience":
            fest.audience = None if text in {"", "-"} else text
        await session.commit()
        logging.info("festival %s updated", fest.name)
    festival_edit_sessions[message.from_user.id] = (fid, None)
    await show_festival_edit_menu(message.from_user.id, fest, bot)

    await sync_festival_page(db, fest.name)
    await sync_festival_vk_post(db, fest.name, bot)
    await rebuild_fest_nav_if_changed(db)



@dataclass
class AlbumImage:
    data: bytes
    name: str
    seq: int
    file_unique_id: str | None = None


@dataclass
class AlbumState:
    images: list[AlbumImage]
    text: str | None = None
    html: str | None = None
    html_mode: str = "native"
    message: types.Message | None = None
    timer: asyncio.Task | None = None
    created: float = field(default_factory=_time.monotonic)


pending_albums: dict[str, AlbumState] = {}
processed_media_groups: set[str] = set()


async def _drop_album_after_ttl(gid: str) -> None:
    await asyncio.sleep(ALBUM_PENDING_TTL_S)
    state = pending_albums.get(gid)
    if state and not state.text:
        age = int(_time.monotonic() - state.created)
        logging.info(
            "album_drop_no_caption gid=%s buf_size=%d age_s=%d",
            gid,
            len(state.images),
            age,
        )
        pending_albums.pop(gid, None)


async def _finalize_album_after_delay(gid: str, db: Database, bot: Bot) -> None:
    await asyncio.sleep(ALBUM_FINALIZE_DELAY_MS / 1000)
    await finalize_album(gid, db, bot)


async def finalize_album(gid: str, db: Database, bot: Bot) -> None:
    state = pending_albums.pop(gid, None)
    if not state or not state.text or not state.message:
        return
    start = _time.monotonic()
    images_total = len(state.images)
    processed_media_groups.add(gid)
    images_sorted = sorted(state.images, key=lambda im: im.seq)
    uniq: list[AlbumImage] = []
    seen: set[str] = set()
    for im in images_sorted:
        uid = im.file_unique_id
        if uid and uid in seen:
            continue
        if uid:
            seen.add(uid)
        uniq.append(im)
    used_images = uniq[:MAX_ALBUM_IMAGES]
    order = [im.seq for im in used_images]
    logging.info(
        "album_finalize gid=%s images=%d order=%s",
        gid,
        len(used_images),
        order,
    )
    logging.info("html_mode=%s", state.html_mode)
    media = [(im.data, im.name) for im in used_images]
    poster_items, catbox_msg = await process_media(
        media, need_catbox=True, need_ocr=False
    )
    global LAST_CATBOX_MSG, LAST_HTML_MODE
    LAST_CATBOX_MSG = catbox_msg
    LAST_HTML_MODE = state.html_mode
    msg = state.message
    session_mode = None
    if msg and msg.from_user:
        session_mode = add_event_sessions.get(msg.from_user.id)
    if msg.forward_date or msg.forward_from_chat or getattr(msg, "forward_origin", None):
        await _process_forwarded(
            msg,
            db,
            bot,
            state.text,
            state.html,
            media,
            poster_media=poster_items,
            catbox_msg=catbox_msg,
        )
    else:
        mode_for_call: AddEventMode = session_mode or "event"
        await handle_add_event(
            msg,
            db,
            bot,
            session_mode=mode_for_call,
            force_festival=mode_for_call == "festival",
            media=media,
            poster_media=poster_items,
            catbox_msg=catbox_msg,
        )
        add_event_sessions.pop(msg.from_user.id, None)
    took = int((_time.monotonic() - start) * 1000)
    logging.info(
        "album_finalize_done gid=%s images_total=%d took_ms=%d used_images=%d catbox_result=%s",
        gid,
        images_total,
        took,
        len(used_images),
        LAST_CATBOX_MSG,
    )


async def _process_forwarded(
    message: types.Message,
    db: Database,
    bot: Bot,
    text: str,
    html: str | None,
    media: list[tuple[bytes, str]] | None,
    poster_media: Sequence[PosterMedia] | None = None,
    catbox_msg: str | None = None,
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            logging.info("user %s not registered or blocked", message.from_user.id)
            return
    link = None
    msg_id = None
    chat_id: int | None = None
    channel_name: str | None = None

    allowed: bool | None = None
    if message.forward_from_chat and message.forward_from_message_id:
        chat = message.forward_from_chat
        msg_id = message.forward_from_message_id
        chat_id = chat.id
        channel_name = chat.title or getattr(chat, "username", None)

        async with db.get_session() as session:
            ch = await session.get(Channel, chat_id)
            allowed = ch.is_registered if ch else False
        logging.info("forward from chat %s allowed=%s", chat_id, allowed)
        if allowed:
            if chat.username:
                link = f"https://t.me/{chat.username}/{msg_id}"
            else:
                cid = str(chat_id)
                if cid.startswith("-100"):
                    cid = cid[4:]
                else:
                    cid = cid.lstrip("-")
                link = f"https://t.me/c/{cid}/{msg_id}"
    else:
        fo = getattr(message, "forward_origin", None)
        if isinstance(fo, dict):
            fo_type = fo.get("type")
        else:
            fo_type = getattr(fo, "type", None)
        if fo_type in {"messageOriginChannel", "channel"}:
            chat_data = fo.get("chat") if isinstance(fo, dict) else getattr(fo, "chat", {})
            chat_id = chat_data.get("id") if isinstance(chat_data, dict) else getattr(chat_data, "id", None)
            msg_id = fo.get("message_id") if isinstance(fo, dict) else getattr(fo, "message_id", None)
            channel_name = (
                chat_data.get("title") if isinstance(chat_data, dict) else getattr(chat_data, "title", None)
            )
            if not channel_name:
                channel_name = (
                    chat_data.get("username") if isinstance(chat_data, dict) else getattr(chat_data, "username", None)
                )

            async with db.get_session() as session:
                ch = await session.get(Channel, chat_id)
                allowed = ch.is_registered if ch else False
            logging.info("forward from origin chat %s allowed=%s", chat_id, allowed)
            if allowed:
                username = chat_data.get("username") if isinstance(chat_data, dict) else getattr(chat_data, "username", None)
                if username:
                    link = f"https://t.me/{username}/{msg_id}"
                else:
                    cid = str(chat_id)
                    if cid.startswith("-100"):
                        cid = cid[4:]
                    else:
                        cid = cid.lstrip("-")
                    link = f"https://t.me/c/{cid}/{msg_id}"
    # determine where to update buttons: default to forwarded message itself
    target_chat_id = message.chat.id
    target_message_id = message.message_id
    if chat_id and msg_id:
        target_chat_id = chat_id
        target_message_id = msg_id
    logging.info(
        "FWD link=%s channel_id=%s name=%s allowed=%s",
        link,
        chat_id,
        channel_name,
        allowed,
    )
    if media is None:
        normalized_media: list[tuple[bytes, str]] = []
    elif isinstance(media, tuple):
        normalized_media = [media]
    else:
        normalized_media = list(media)
    poster_items: list[PosterMedia] = []
    local_catbox_msg = catbox_msg or ""
    if poster_media is not None:
        poster_items = list(poster_media)
    elif normalized_media:
        poster_items, local_catbox_msg = await process_media(
            normalized_media, need_catbox=True, need_ocr=False
        )
    global LAST_CATBOX_MSG
    LAST_CATBOX_MSG = local_catbox_msg
    logging.info(
        "FWD summary text_len=%d media_len=%d posters=%d",
        len(text or ""),
        len(normalized_media),
        len(poster_items),
    )
    logging.info("parsing forwarded text via LLM")
    try:
        results = await add_events_from_text(
            db,
            text,
            link,
            html,
            normalized_media,
            poster_media=poster_items,
            source_chat_id=target_chat_id,
            source_message_id=target_message_id,
            creator_id=user.user_id,
            source_channel=channel_name,
            bot=None,
        )
    except Exception as e:
        logging.exception("forward parse failed")
        snippet = (text or "")[:200]
        msg = f"Не удалось обработать сообщение: {type(e).__name__}: {e}"
        if snippet:
            msg += f"\n\n{snippet}"
        if link:
            msg += f"\n{link}"
        await notify_superadmin(db, bot, msg)
        return
    logging.info("forward parsed %d events", len(results))
    ocr_line = None
    if normalized_media and results.ocr_tokens_remaining is not None:
        base_line = (
            f"OCR: потрачено {results.ocr_tokens_spent}, осталось "
            f"{results.ocr_tokens_remaining}"
        )
        if results.ocr_limit_notice:
            ocr_line = f"{results.ocr_limit_notice}\n{base_line}"
        else:
            ocr_line = base_line
    if not results:
        logging.info("no events parsed from forwarded text")
        await bot.send_message(
            message.chat.id,
            "Я не смог найти события в пересланном посте. "
            "Попробуйте добавить событие вручную или свяжитесь с поддержкой.",
        )
        return
    for saved, added, lines, status in results:
        if status == "missing":
            buttons: list[list[types.InlineKeyboardButton]] = []
            if "time" in lines:
                buttons.append(
                    [types.InlineKeyboardButton(text="Добавить время", callback_data="asktime")]
                )
                buttons.append(
                    [types.InlineKeyboardButton(text="Изменить дату", callback_data="askdate")]
                )
            if "location_name" in lines:
                buttons.append(
                    [types.InlineKeyboardButton(text="Добавить локацию", callback_data="askloc")]
                )
            if "city" in lines:
                buttons.append(
                    [types.InlineKeyboardButton(text="Добавить город", callback_data="askcity")]
                )
            saved_id = getattr(saved, "id", None)
            if saved_id is not None:
                buttons.append(
                    [
                        types.InlineKeyboardButton(
                            text="Редактировать",
                            callback_data=f"edit:{saved_id}",
                        )
                    ]
                )
            keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
            await bot.send_message(
                message.chat.id,
                "Отсутствуют обязательные поля: " + ", ".join(lines),
                reply_markup=keyboard,
            )
            continue
        if isinstance(saved, Festival):
            async with db.get_session() as session:
                count = (
                    await session.scalar(
                        select(func.count()).where(Event.festival == saved.name)
                    )
                ) or 0
            logging.info(
                "festival_notify",
                extra={
                    "festival": saved.name,
                    "action": "created" if added else "updated",
                    "events_count_at_moment": count,
                },
            )
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        types.InlineKeyboardButton(
                            text="Создать события по дням",
                            callback_data=f"festdays:{saved.id}",
                        )
                    ]
                ]
            )
            text_out = "Festival added\n" + "\n".join(lines)
            if ocr_line:
                text_out = f"{text_out}\n{ocr_line}"
                ocr_line = None
            await bot.send_message(message.chat.id, text_out, reply_markup=markup)
            continue
        buttons: list[types.InlineKeyboardButton] = []
        if not saved.city:
            buttons.append(
                types.InlineKeyboardButton(
                    text="Добавить город", callback_data="askcity"
                )
            )
        if (
            not saved.is_free
            and saved.ticket_price_min is None
            and saved.ticket_price_max is None
        ):
            buttons.append(
                types.InlineKeyboardButton(
                    text="\u2753 Это бесплатное мероприятие",
                    callback_data=f"markfree:{saved.id}",
                )
            )
        buttons.append(
            types.InlineKeyboardButton(
                text="\U0001f6a9 Переключить на тихий режим",
                callback_data=f"togglesilent:{saved.id}",
            )
        )
        buttons.append(
            types.InlineKeyboardButton(
                text="Редактировать",
                callback_data=f"edit:{saved.id}",
            )
        )
        inline_keyboard: list[list[types.InlineKeyboardButton]]
        if len(buttons) > 1:
            inline_keyboard = [buttons[:-1], [buttons[-1]]]
        else:
            inline_keyboard = [[buttons[0]]]
        inline_keyboard = append_tourist_block(inline_keyboard, saved, "tg")
        markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
        extra_lines = [ocr_line] if ocr_line else None
        text_out = build_event_card_message(
            f"Event {status}", saved, lines, extra_lines=extra_lines
        )
        if ocr_line:
            ocr_line = None
        logging.info("sending response for event %s", saved.id)
        try:
            await bot.send_message(
                message.chat.id,
                text_out,
                reply_markup=markup,
            )
            await notify_event_added(db, bot, user, saved, added)
            await publish_event_progress(saved, db, bot, message.chat.id)
        except Exception as e:
            logging.error("failed to send event response: %s", e)


async def handle_forwarded(message: types.Message, db: Database, bot: Bot):
    logging.info(
        "received forwarded message %s from %s",
        message.message_id,
        message.from_user.id,
    )
    text = message.text or message.caption
    images = await extract_images(message, bot)
    logging.info(
        "forward text len=%d photos=%d",
        len(text or ""),
        len(images or []),
    )
    if message.media_group_id:
        gid = message.media_group_id
        if gid in processed_media_groups:
            logging.info("skip already processed album %s", gid)
            return
        state = pending_albums.get(gid)
        if not state:
            if len(pending_albums) >= MAX_PENDING_ALBUMS:
                old_gid, old_state = min(
                    pending_albums.items(), key=lambda kv: kv[1].created
                )
                if old_state.timer:
                    old_state.timer.cancel()
                age = int(_time.monotonic() - old_state.created)
                logging.info(
                    "album_drop_no_caption gid=%s buf_size=%d age_s=%d",
                    old_gid,
                    len(old_state.images),
                    age,
                )
                pending_albums.pop(old_gid, None)
            state = AlbumState(images=[])
            state.timer = asyncio.create_task(_drop_album_after_ttl(gid))
            pending_albums[gid] = state
        seq = message.forward_from_message_id
        if not seq:
            fo = getattr(message, "forward_origin", None)
            if isinstance(fo, dict):
                seq = fo.get("message_id")
            else:
                seq = getattr(fo, "message_id", None)
        if not seq:
            seq = message.message_id or int(message.date.timestamp())
        img_count = len(images or [])
        if images and len(state.images) < MAX_ALBUM_IMAGES:
            add = min(img_count, MAX_ALBUM_IMAGES - len(state.images))
            file_uid = None
            if message.photo:
                file_uid = message.photo[-1].file_unique_id
            elif (
                message.document
                and message.document.mime_type
                and message.document.mime_type.startswith("image/")
            ):
                file_uid = message.document.file_unique_id
            for data, name in images[:add]:
                state.images.append(AlbumImage(data=data, name=name, seq=seq, file_unique_id=file_uid))
        logging.info(
            "album_collect gid=%s seq=%s msg_id=%s has_text=%s images_in_msg=%d buf_size_after=%d",
            gid,
            seq,
            message.message_id,
            bool(text),
            len(images or []),
            len(state.images),
        )
        if text and not state.text:
            state.text = text
            state.html, state.html_mode = ensure_html_text(message)
            state.message = message
            if state.timer:
                state.timer.cancel()
            logging.info(
                "album_caption_seen gid=%s delay_ms=%d",
                gid,
                ALBUM_FINALIZE_DELAY_MS,
            )
            state.timer = asyncio.create_task(
                _finalize_album_after_delay(gid, db, bot)
            )
        return
    if not text:
        logging.info("forwarded message has no text")
        return
    media = images[:MAX_ALBUM_IMAGES] if images else None
    logging.info("IMG single message media_len=%d", len(media or []))
    html, _mode = ensure_html_text(message)
    await _process_forwarded(
        message,
        db,
        bot,
        text,
        html,
        media,
    )


async def handle_add_event_media_group(
    message: types.Message, db: Database, bot: Bot
) -> None:
    """Collect media group messages for /addevent sessions."""
    logging.info(
        "received add_event media group message %s from %s",
        message.message_id,
        message.from_user.id,
    )
    gid = message.media_group_id
    if not gid:
        return
    if gid in processed_media_groups:
        logging.info("skip already processed album %s", gid)
        return
    images = await extract_images(message, bot)
    state = pending_albums.get(gid)
    if not state:
        if len(pending_albums) >= MAX_PENDING_ALBUMS:
            old_gid, old_state = min(
                pending_albums.items(), key=lambda kv: kv[1].created
            )
            if old_state.timer:
                old_state.timer.cancel()
            age = int(_time.monotonic() - old_state.created)
            logging.info(
                "album_drop_no_caption gid=%s buf_size=%d age_s=%d",
                old_gid,
                len(old_state.images),
                age,
            )
            pending_albums.pop(old_gid, None)
        state = AlbumState(images=[])
        state.timer = asyncio.create_task(_drop_album_after_ttl(gid))
        pending_albums[gid] = state
    seq = message.message_id or int(message.date.timestamp())
    img_count = len(images or [])
    if images and len(state.images) < MAX_ALBUM_IMAGES:
        add = min(img_count, MAX_ALBUM_IMAGES - len(state.images))
        file_uid = None
        if message.photo:
            file_uid = message.photo[-1].file_unique_id
        elif (
            message.document
            and message.document.mime_type
            and message.document.mime_type.startswith("image/")
        ):
            file_uid = message.document.file_unique_id
        for data, name in images[:add]:
            state.images.append(
                AlbumImage(data=data, name=name, seq=seq, file_unique_id=file_uid)
            )
    text = message.text or message.caption
    logging.info(
        "album_collect gid=%s seq=%s msg_id=%s has_text=%s images_in_msg=%d buf_size_after=%d",
        gid,
        seq,
        message.message_id,
        bool(text),
        len(images or []),
        len(state.images),
    )
    if text and not state.text:
        state.text = text
        state.html, state.html_mode = ensure_html_text(message)
        state.message = message
        if state.timer:
            state.timer.cancel()
        logging.info(
            "album_caption_seen gid=%s delay_ms=%d",
            gid,
            ALBUM_FINALIZE_DELAY_MS,
        )
        state.timer = asyncio.create_task(
            _finalize_album_after_delay(gid, db, bot)
        )


async def telegraph_test():
    token = get_telegraph_token()
    if not token:
        print("Unable to obtain Telegraph token")
        return
    tg = Telegraph(access_token=token)
    page = await telegraph_create_page(
        tg, "Test Page", html_content="<p>test</p>", caller="event_pipeline"
    )
    logging.info("Created %s", page["url"])
    print("Created", page["url"])
    await telegraph_edit_page(
        tg,
        page["path"],
        title="Test Page",
        html_content="<p>updated</p>",
        caller="event_pipeline",
    )
    logging.info("Edited %s", page["url"])
    print("Edited", page["url"])


async def update_source_page(
    path: str,
    title: str,
    new_html: str,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    db: Database | None = None,
    *,
    catbox_urls: list[str] | None = None,
) -> tuple[str, int]:
    """Append text to an existing Telegraph page."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return "token missing"
    tg = Telegraph(access_token=token)
    try:
        logging.info("Fetching telegraph page %s", path)
        page = await telegraph_call(tg.get_page, path, return_html=True)
        html_content = page.get("content") or page.get("content_html") or ""
        catbox_msg = ""
        images: list[tuple[bytes, str]] = []
        if media:
            images = [media] if isinstance(media, tuple) else list(media)
        if catbox_urls is not None:
            urls = list(catbox_urls)
            catbox_msg = ""
        else:
            catbox_urls, catbox_msg = await upload_images(images)
            urls = catbox_urls
        has_cover = "<img" in html_content
        if has_cover:
            cover: list[str] = []
            tail = urls
        else:
            cover = urls[:1]
            tail = urls[1:]
        if cover:
            cover_html = f'<figure><img src="{html.escape(cover[0])}"/></figure>'
            html_content = cover_html + html_content
        new_html = normalize_hashtag_dates(new_html)
        cleaned = re.sub(r"</?tg-(?:emoji|spoiler)[^>]*>", "", new_html)
        cleaned = cleaned.replace(
            "\U0001f193\U0001f193\U0001f193\U0001f193", "Бесплатно"
        )
        new_block = (
            f"<p>{CONTENT_SEPARATOR}</p><p>" + cleaned.replace("\n", "<br/>") + "</p>"
        )
        hr_idx = html_content.lower().rfind("<hr")
        if hr_idx != -1:
            hr_end = html_content.find(">", hr_idx)
            if hr_end != -1:
                html_content = html_content[:hr_idx] + new_block + html_content[hr_idx:]
            else:
                html_content += new_block
        else:
            html_content += new_block
        nav_html = None
        if db:
            nav_html = await build_month_nav_html(db)
            html_content = apply_month_nav(html_content, nav_html)
        existing_imgs = html_content.count("<img")
        for url in tail:
            html_content += f'<img src="{html.escape(url)}"/>'
        total_imgs = existing_imgs + len(tail)
        if nav_html and total_imgs >= 2:
            html_content += nav_html
        html_content = apply_footer_link(html_content)
        html_content = lint_telegraph_html(html_content)
        logging.info(
            "Editing telegraph page %s", path,
        )
        await telegraph_edit_page(
            tg,
            path,
            title=title,
            html_content=html_content,
            caller="event_pipeline",
        )
        logging.info(
            "Updated telegraph page %s", path,
        )
        logging.info(
            "update_source_page: cover=%d tail=%d nav_dup=%s",
            len(cover),
            len(tail),
            bool(nav_html and total_imgs >= 2),
        )
        return catbox_msg, len(urls)
    except Exception as e:
        logging.error("Failed to update telegraph page: %s", e)
        return f"error: {e}", 0


async def update_source_page_ics(event_id: int, db: Database, url: str | None):
    """Insert or remove the ICS link in a Telegraph page."""
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev or not ev.telegraph_path:
        return
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    path = ev.telegraph_path
    title = ev.title or "Event"
    try:
        logging.info("Editing telegraph ICS for %s", path)
        page = await telegraph_call(tg.get_page, path, return_html=True)
        html_content = page.get("content") or page.get("content_html") or ""
        html_content = apply_ics_link(html_content, url)
        html_content = apply_footer_link(html_content)
        await telegraph_edit_page(
            tg,
            path,
            title=title,
            html_content=html_content,
            caller="event_pipeline",
            eid=ev.id,
        )
    except Exception as e:
        logging.error("Failed to update ICS link: %s", e)


async def get_source_page_text(path: str) -> str:
    """Return plain text from a Telegraph page."""
    raw_path = (path or "").strip()
    normalized_path = raw_path
    try:
        parsed = urlparse(raw_path)
        if parsed.scheme or parsed.netloc:
            normalized_path = parsed.path
    except Exception:
        normalized_path = raw_path
    normalized_path = (normalized_path or "").lstrip("/")

    try:
        token_info_fn = get_telegraph_token_info
    except NameError:  # Module imported without main namespace
        from main import get_telegraph_token_info as token_info_fn

    try:
        tg_call = telegraph_call
    except NameError:
        from main import telegraph_call as tg_call

    token_info = token_info_fn()

    logging.info(
        "telegraph_fetch start path=%s token_source=%s token_file=%s token_file_exists=%s token_file_readable=%s",
        normalized_path,
        token_info.source,
        token_info.token_file,
        token_info.token_file_exists,
        token_info.token_file_readable,
    )

    if not token_info.token:
        logging.warning(
            "telegraph_fetch no_token path=%s token_source=%s token_file_exists=%s",
            normalized_path,
            token_info.source,
            token_info.token_file_exists,
        )
        return ""

    fetch_path = normalized_path or path
    tg = Telegraph(access_token=token_info.token)
    try:
        page = await tg_call(tg.get_page, fetch_path, return_html=True)
    except Exception as e:
        logging.exception(
            "telegraph_fetch exception path=%s token_source=%s error=%s",
            normalized_path,
            token_info.source,
            type(e).__name__,
        )
        return ""
    resp_keys: list[str] | None = None
    len_html = 0
    len_text_raw = 0
    if isinstance(page, dict):
        resp_keys = list(page.keys())
        html_field = page.get("content_html") or page.get("content")
        if isinstance(html_field, str):
            len_html = len(html_field)
        raw_text_field = page.get("content")
        if isinstance(raw_text_field, str):
            len_text_raw = len(raw_text_field)
    logging.debug(
        "telegraph_fetch response path=%s resp_keys=%s len_html=%d len_text=%d",
        normalized_path,
        resp_keys,
        len_html,
        len_text_raw,
    )
    html_content = page.get("content") or page.get("content_html") or ""
    html_content = apply_ics_link(html_content, None)
    html_content = apply_month_nav(html_content, None)
    html_content = html_content.replace(FOOTER_LINK_HTML, "")
    html_content = html_content.replace(f"<p>{CONTENT_SEPARATOR}</p>", f"\n{CONTENT_SEPARATOR}\n")
    html_content = html_content.replace("<br/>", "\n").replace("<br>", "\n")
    html_content = re.sub(r"</p>\s*<p>", "\n", html_content)
    html_content = re.sub(r"<[^>]+>", "", html_content)
    text = html.unescape(html_content)
    text = text.replace(CONTENT_SEPARATOR, "").replace("\xa0", " ")
    cleaned = text.strip()
    if not cleaned:
        logging.warning(
            "telegraph_fetch empty_content path=%s token_source=%s resp_keys=%s len_html=%d len_text=%d",
            normalized_path,
            token_info.source,
            resp_keys,
            len_html,
            len(cleaned),
        )
    return cleaned


async def update_event_description(event: Event, db: Database) -> None:
    """Populate event.description from the Telegraph source page if missing."""
    if event.description:
        logging.info(
            "skip description update for event %s: already present", event.id
        )
        return
    if not event.telegraph_path:
        logging.info(
            "skip description update for event %s: no telegraph page", event.id
        )
        return
    logging.info(
        "updating description for event %s from %s",
        event.id,
        event.telegraph_path,
    )
    text = await get_source_page_text(event.telegraph_path)
    if not text:
        logging.info("no source text for event %s", event.id)
        return
    posters = await _fetch_event_posters(event.id, db)
    poster_texts = await get_event_poster_texts(event.id, db, posters=posters)
    poster_summary = _summarize_event_posters(posters)
    try:
        parse_kwargs: dict[str, Any] = {}
        if poster_texts:
            parse_kwargs["poster_texts"] = poster_texts
        if poster_summary:
            parse_kwargs["poster_summary"] = poster_summary
        parsed = await parse_event_via_4o(text, **parse_kwargs)
    except Exception as e:
        logging.error("Failed to parse source text for description: %s", e)
        return
    if not parsed:
        logging.info("4o returned no data for event %s", event.id)
        return
    desc = parsed[0].get("short_description", "").strip()
    if not desc:
        logging.info("no short description parsed for event %s", event.id)
        return
    async with db.get_session() as session:
        obj = await session.get(Event, event.id)
        if obj:
            obj.description = desc
            await session.commit()
            event.description = desc
            logging.info("stored description for event %s", event.id)


@dataclass(slots=True)
class SourcePageEventSummary:
    date: str | None = None
    end_date: str | None = None
    time: str | None = None
    event_type: str | None = None
    location_name: str | None = None
    location_address: str | None = None
    city: str | None = None
    ticket_price_min: int | None = None
    ticket_price_max: int | None = None
    ticket_link: str | None = None
    is_free: bool = False
    ticket_status: str | None = None  # 'available', 'sold_out', or None


def _format_summary_anchor_text(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = (parsed.netloc or "").strip()
    if not host:
        return url
    host = host.rstrip("/")
    path = (parsed.path or "").strip()
    path = path.rstrip("/")
    if path:
        display = f"{host}{path}"
    else:
        display = host
    if len(display) > 48:
        display = display[:47] + "…"
    return display or url


def _render_summary_anchor(url: str, text: str | None = None) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    href = cleaned
    try:
        parsed = urlparse(cleaned)
    except ValueError:
        parsed = None
    if parsed and not parsed.scheme:
        if cleaned.startswith("//"):
            href = "https:" + cleaned
        else:
            href = "https://" + cleaned.lstrip("/")
    display_text = text or _format_summary_anchor_text(href)
    return f'<a href="{html.escape(href)}">{html.escape(display_text)}</a>'


def _format_ticket_price(min_price: int | None, max_price: int | None) -> str:
    if (
        min_price is not None
        and max_price is not None
        and min_price != max_price
    ):
        return f"от {min_price} до {max_price} руб."
    if min_price is not None:
        return f"{min_price} руб."
    if max_price is not None:
        return f"{max_price} руб."
    return ""


async def _build_source_summary_block(
    event_summary: SourcePageEventSummary | None,
    *,
    ics_url: str | None = None,
) -> str:
    if not event_summary:
        return ""

    lines: list[str] = []

    start_date: date | None = None
    try:
        if event_summary.date:
            start_date = date.fromisoformat(event_summary.date)
    except ValueError:
        start_date = None

    if start_date:
        default_date_str = f"{start_date.day} {MONTHS[start_date.month - 1]}"
    elif event_summary.date:
        try:
            _, month, day = event_summary.date.split("-")
            default_date_str = f"{int(day)} {MONTHS[int(month) - 1]}"
        except Exception:
            default_date_str = event_summary.date
    else:
        default_date_str = ""

    end_date_obj: date | None = None
    if event_summary.end_date:
        try:
            end_date_obj = date.fromisoformat(event_summary.end_date)
        except ValueError:
            end_date_obj = None

    ongoing_exhibition = (
        (event_summary.event_type or "").strip().casefold() == "выставка"
        and start_date is not None
        and end_date_obj is not None
        and start_date <= date.today() <= end_date_obj
    )

    if ongoing_exhibition and end_date_obj is not None:
        end_month_name = MONTHS[end_date_obj.month - 1]
        year_suffix = ""
        if start_date.year != end_date_obj.year:
            year_suffix = f" {end_date_obj.year}"
        date_line = f"🗓 по {end_date_obj.day} {end_month_name}{year_suffix}"
    elif default_date_str:
        time_part = ""
        if event_summary.time and event_summary.time != "00:00":
            time_part = f" в {event_summary.time}"
        date_line = f"🗓 {default_date_str}{time_part}"
    else:
        date_line = ""

    date_line = date_line.strip()
    if date_line:
        escaped_date = html.escape(date_line)
        lines.append(escaped_date)
    
    # Add calendar link on separate line
    if ics_url:
        lines.append(
            f'📅 <a href="{html.escape(ics_url)}">{ICS_LABEL}</a>'
        )

    location_parts: list[str] = []
    existing_normalized: set[str] = set()
    for part in (event_summary.location_name, event_summary.location_address):
        if part and part.strip():
            location_parts.append(part)
            normalized = _normalize_location_part(part)
            if normalized:
                existing_normalized.add(normalized)
    if event_summary.city and event_summary.city.strip():
        city_normalized = _normalize_location_part(event_summary.city)
        if not city_normalized or city_normalized not in existing_normalized:
            location_parts.append(event_summary.city)
            if city_normalized:
                existing_normalized.add(city_normalized)

    location_line = ""
    if location_parts:
        location_text = ", ".join(part.strip() for part in location_parts if part.strip())
        if location_text.strip():
            location_line = f"📍 {location_text.strip()}"
    if location_line:
        lines.append(html.escape(location_line))

    ticket_segments: list[str] = []
    link_value = (event_summary.ticket_link or "").strip()
    price_text = _format_ticket_price(
        event_summary.ticket_price_min, event_summary.ticket_price_max
    )
    
    # Check ticket status for sold-out events
    if event_summary.ticket_status == "sold_out":
        ticket_segments.append(html.escape("❌ Билеты все проданы"))
    elif event_summary.is_free:
        if link_value:
            ticket_segments.append(html.escape("🆓 "))
            ticket_segments.append(
                _render_summary_anchor(link_value, "Бесплатно, по регистрации")
            )
        else:
            ticket_segments.append(html.escape("🆓 Бесплатно"))
    else:
        # Available tickets - add ✅ icon if ticket_status is explicitly 'available'
        status_icon = "✅ " if event_summary.ticket_status == "available" else ""
        if link_value:
            ticket_segments.append(html.escape(f"🎟 {status_icon}"))
            ticket_segments.append(_render_summary_anchor(link_value, "Билеты"))
            if price_text:
                ticket_segments.append(html.escape(f" {price_text}"))
        elif price_text:
            ticket_segments.append(html.escape(f"🎟 {status_icon}Билеты {price_text}"))

    if ticket_segments:
        lines.append("".join(ticket_segments).strip())

    if not lines:
        return ""

    return f"<p>{'<br/>'.join(lines)}</p>"


async def build_source_page_content(
    title: str,
    text: str,
    source_url: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    ics_url: str | None = None,
    db: Database | None = None,
    *,
    event_summary: SourcePageEventSummary | None = None,
    display_link: bool = True,
    catbox_urls: list[str] | None = None,
    image_mode: Literal["tail", "inline"] = "tail",
    page_mode: Literal["default", "history"] = "default",
    search_digest: str | None = None,
) -> tuple[str, str, int]:
    if image_mode not in {"tail", "inline"}:
        raise ValueError(f"unknown image_mode={image_mode}")
    if page_mode not in {"default", "history"}:
        raise ValueError(f"unknown page_mode={page_mode}")
    html_content = ""
    def strip_title(line_text: str) -> str:
        lines = line_text.splitlines()
        if lines and lines[0].strip() == title.strip():
            return "\n".join(lines[1:]).lstrip()
        return line_text
    images: list[tuple[bytes, str]] = []
    if media:
        images = [media] if isinstance(media, tuple) else list(media)
    if catbox_urls is not None:
        urls = list(catbox_urls)
        catbox_msg = ""
        input_count = len(catbox_urls)
    else:
        catbox_urls, catbox_msg = await upload_images(images)
        urls = catbox_urls
        input_count = 0
    # filter out video links and limit to first 12 images
    urls = [
        u for u in urls if not re.search(r"\.(?:mp4|webm|mkv|mov)(?:\?|$)", u, re.I)
    ][:12]
    cover = urls[:1]
    tail = urls[1:]
    if cover:
        html_content += f'<figure><img src="{html.escape(cover[0])}"/></figure>'
    summary_html = await _build_source_summary_block(
        event_summary, ics_url=ics_url
    )
    summary_added = bool(summary_html)
    if summary_html:
        html_content += summary_html
    elif ics_url:
        html_content += (
            f'<p>\U0001f4c5 <a href="{html.escape(ics_url)}">{ICS_LABEL}</a></p>'
        )
    
    # Add search_digest before long descriptions (> 500 chars)
    text_len = len((text or "").strip())
    if search_digest and search_digest.strip() and text_len > 500:
        digest_escaped = html.escape(search_digest.strip())
        html_content += f"<p>{digest_escaped}</p>"
        html_content += "<hr/>"
    emoji_pat = re.compile(r"<tg-emoji[^>]*>(.*?)</tg-emoji>", re.DOTALL)
    spoiler_pat = re.compile(r"<tg-spoiler[^>]*>(.*?)</tg-spoiler>", re.DOTALL)
    tg_emoji_cleaned = 0
    tg_spoiler_unwrapped = 0
    paragraphs: list[str] = []
    blank_paragraph_re = re.compile(
        r"<p>(?:&nbsp;|&#8203;|\s|<br\s*/?>)*</p>", re.IGNORECASE
    )
    def _wrap_plain_chunks(raw_chunk: str) -> list[str]:
        chunk = raw_chunk.strip()
        if not chunk:
            return []
        normalized = re.sub(r"<br\s*/?>", "<br/>", chunk, flags=re.IGNORECASE)
        normalized = normalized.replace("\r", "")
        parts = [
            part.strip()
            for part in re.split(r"(?:<br/>\s*){2,}|\n{2,}", normalized)
            if part.strip()
        ]
        wrapped: list[str] = []
        for part in parts:
            segment = part.replace("\n", "<br/>")
            segment = re.sub(r"^(?:<br/>\s*)+", "", segment, flags=re.IGNORECASE)
            segment = re.sub(r"(?:<br/>\s*)+$", "", segment, flags=re.IGNORECASE)
            wrapped.append(f"<p>{segment}</p>")
        return wrapped

    def _split_paragraph_block(block: str) -> list[str]:
        match = re.match(r"(<p[^>]*>)(.*?)(</p>)", block, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return [block]
        start_tag, body, end_tag = match.groups()
        pieces = re.split(r"(?:<br\s*/?>\s*){2,}", body, flags=re.IGNORECASE)
        result: list[str] = []
        for piece in pieces:
            cleaned = piece.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^(?:<br\s*/?>\s*)+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(?:<br\s*/?>\s*)+$", "", cleaned, flags=re.IGNORECASE)
            result.append(f"{start_tag}{cleaned}{end_tag}")
        return result or [block]

    def _fix_heading_paragraph_mismatches(raw: str) -> str:
        tag_re = re.compile(r"<(/?)(h[1-6]|p)([^>]*)>", re.IGNORECASE)
        block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
        result: list[str] = []
        pos = 0
        stack: list[str] = []

        def _flush_stack() -> None:
            while stack:
                result.append(f"</{stack.pop()}>")

        for match in tag_re.finditer(raw):
            start, end = match.span()
            result.append(raw[pos:start])
            closing = match.group(1) == "/"
            tag = match.group(2).lower()
            tail = match.group(3) or ""
            if not closing:
                if tag in block_tags and stack:
                    _flush_stack()
                stack.append(tag)
                result.append(f"<{tag}{tail}>")
            else:
                if not stack:
                    pos = end
                    continue
                if tag not in stack:
                    _flush_stack()
                else:
                    while stack and stack[-1] != tag:
                        result.append(f"</{stack.pop()}>")
                    if stack and stack[-1] == tag:
                        stack.pop()
                        result.append(f"</{tag}>")
            pos = end
        result.append(raw[pos:])
        while stack:
            result.append(f"</{stack.pop()}>")
        return "".join(result)

    def _editor_html_blocks(raw: str) -> list[str]:
        text_value = raw.strip()
        if not text_value:
            return []
        looks_like_html = bool(re.search(r"<\w+[^>]*>", text_value))
        if looks_like_html:
            sanitized = sanitize_telegram_html(text_value)
            sanitized = linkify_for_telegraph(sanitized)
        else:
            sanitized = md_to_html(text_value)
        sanitized = re.sub(r"<(\/?)h[12](\b)", r"<\1h3\2", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"<br\s*/?>", "<br/>", sanitized, flags=re.IGNORECASE)
        sanitized = sanitize_telegram_html(sanitized)
        block_re = re.compile(
            r"<(?P<tag>h[1-6]|p|ul|ol|blockquote|pre|table|figure)[^>]*>.*?</(?P=tag)>|<hr\b[^>]*>|<img\b[^>]*>",
            re.IGNORECASE | re.DOTALL,
        )
        blocks: list[str] = []
        pos = 0
        for match in block_re.finditer(sanitized):
            start, end = match.span()
            if start > pos:
                blocks.extend(_wrap_plain_chunks(sanitized[pos:start]))
            block_value = match.group(0)
            if block_value.lower().startswith("<p"):
                blocks.extend(_split_paragraph_block(block_value))
            else:
                blocks.append(block_value)
            pos = end
        if pos < len(sanitized):
            blocks.extend(_wrap_plain_chunks(sanitized[pos:]))
        return [block for block in blocks if block.strip()]

    if html_text:
        html_text = strip_title(html_text)
        html_text = normalize_hashtag_dates(html_text)
        html_text = html_text.replace("\r\n", "\n")
        html_text = sanitize_telegram_html(html_text)
        for k, v in CUSTOM_EMOJI_MAP.items():
            html_text = html_text.replace(k, v)
        html_text = _fix_heading_paragraph_mismatches(html_text)
        paragraphs = _editor_html_blocks(html_text)
    else:
        clean_text = strip_title(text)
        clean_text = normalize_hashtag_dates(clean_text)
        tg_emoji_cleaned = len(emoji_pat.findall(clean_text))
        tg_spoiler_unwrapped = len(spoiler_pat.findall(clean_text))
        clean_text = emoji_pat.sub(r"\1", clean_text)
        clean_text = spoiler_pat.sub(r"\1", clean_text)
        for k, v in CUSTOM_EMOJI_MAP.items():
            clean_text = clean_text.replace(k, v)
        for line in clean_text.splitlines():
            escaped = html.escape(line)
            linked = linkify_for_telegraph(escaped)
            if linked.strip():
                paragraphs.append(f"<p>{linked}</p>")
            else:
                paragraphs.append(BODY_SPACER_HTML)
    if paragraphs:
        if summary_added:
            html_content += BODY_SPACER_HTML
        normalized_paragraphs: list[str] = []
        for block in paragraphs:
            block_stripped = block.strip()
            if blank_paragraph_re.fullmatch(block_stripped):
                normalized_paragraphs.append(BODY_SPACER_HTML)
            else:
                normalized_paragraphs.append(block)
        paragraphs = normalized_paragraphs
    inline_used = 0
    if page_mode == "history" and paragraphs:
        anchor_re = re.compile(r"<a\b[^>]*href=(['\"])(.*?)\1[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)

        def _parse_href(raw: str | None) -> ParseResult | None:
            if not raw:
                return None
            candidate = html.unescape(raw).strip()
            if not candidate:
                return None
            parsed = urlparse(candidate)
            if parsed.scheme:
                final = parsed
            elif candidate.startswith("//"):
                final = urlparse("https:" + candidate)
            else:
                final = urlparse("https://" + candidate.lstrip("/"))
            return final

        def _normalized_parts(raw: str | None) -> tuple[str, str, str, str, str] | None:
            parsed = _parse_href(raw)
            if not parsed:
                return None
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            path = parsed.path or "/"
            if path != "/":
                path = path.rstrip("/")
            return host, path, parsed.params, parsed.query, parsed.fragment

        source_parts = _normalized_parts(source_url)

        def _is_vk_href(raw: str | None) -> bool:
            parsed = _parse_href(raw)
            if not parsed:
                return False
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host == "vk.com" or host.endswith(".vk.com")

        def _replace_anchor(match: re.Match[str]) -> str:
            href = match.group(2)
            parts = _normalized_parts(href)
            if source_parts and parts == source_parts:
                return match.group(0)
            if _is_vk_href(href):
                return match.group(3)
            return match.group(0)

        paragraphs = [anchor_re.sub(_replace_anchor, para) for para in paragraphs]
    if paragraphs:
        spacer = BODY_SPACER_HTML

        paragraph_tag_re = re.compile(r"<p\b", re.IGNORECASE)
        heading_tag_re = re.compile(r"<h[1-6]\b", re.IGNORECASE)
        media_tag_re = re.compile(r"<(?:figure|img)\b", re.IGNORECASE)

        def _should_add_spacer(previous_blocks: list[str], upcoming: str) -> bool:
            if not previous_blocks:
                return False
            last = previous_blocks[-1]
            if last == spacer:
                return False
            if page_mode != "history":
                return True
            if not paragraph_tag_re.match(upcoming.strip()):
                return True
            last_stripped = last.strip()
            if heading_tag_re.match(last_stripped):
                return False
            if media_tag_re.match(last_stripped):
                return False
            return True

        if image_mode == "inline" and tail:
            text_paragraphs = [p for p in paragraphs if p != BODY_SPACER_HTML]
            paragraph_count = len(text_paragraphs)
            image_count = len(tail)
            base_count = min(image_count, paragraph_count)
            positions: list[int] = []
            if base_count:
                step = paragraph_count / (base_count + 1)
                positions = [math.ceil((idx + 1) * step) for idx in range(base_count)]
            body_blocks: list[str] = []
            base_index = 0
            text_index = 0
            for block in paragraphs:
                if block == BODY_SPACER_HTML:
                    if _should_add_spacer(body_blocks, spacer):
                        body_blocks.append(spacer)
                    continue
                if _should_add_spacer(body_blocks, block):
                    body_blocks.append(spacer)
                body_blocks.append(block)
                text_index += 1
                inserted_for_para = False
                while base_index < base_count and positions[base_index] == text_index:
                    if not inserted_for_para and _should_add_spacer(body_blocks, spacer):
                        body_blocks.append(spacer)
                        inserted_for_para = True
                    body_blocks.append(
                        f'<img src="{html.escape(tail[base_index])}"/>'
                    )
                    base_index += 1
            for extra_url in tail[base_index:]:
                if _should_add_spacer(body_blocks, spacer):
                    body_blocks.append(spacer)
                body_blocks.append(f'<img src="{html.escape(extra_url)}"/>')
                base_index += 1
            inline_used = base_index
        else:
            body_blocks = []
            for block in paragraphs:
                if block == BODY_SPACER_HTML:
                    if _should_add_spacer(body_blocks, spacer):
                        body_blocks.append(spacer)
                    continue
                if _should_add_spacer(body_blocks, block):
                    body_blocks.append(spacer)
                body_blocks.append(block)
        html_content += "".join(body_blocks)
    elif image_mode == "inline" and tail:
        if summary_added:
            html_content += BODY_SPACER_HTML
        for extra_url in tail:
            html_content += f'<img src="{html.escape(extra_url)}"/>'
        inline_used = len(tail)
    if db and hasattr(db, "get_session") and text and text.strip():
        from models import Event, Festival
        from sqlalchemy import select

        async with db.get_session() as session:
            res = await session.execute(
                select(Event.festival, Festival.telegraph_path, Festival.telegraph_url)
                .join(Festival, Event.festival == Festival.name)
                .where(Event.source_text == text)
            )
            row = res.first()
            if row and row.telegraph_path:
                href = row.telegraph_url or f"https://telegra.ph/{row.telegraph_path.lstrip('/')}"
                html_content += (
                    BODY_SPACER_HTML
                    + f'<p>✨ <a href="{html.escape(href)}">{html.escape(row.festival)}</a></p>'
                    + BODY_SPACER_HTML
                )
    nav_html = None
    if db and page_mode != "history":
        nav_html = await build_month_nav_html(db)
        html_content = apply_month_nav(html_content, nav_html)
    if image_mode == "tail":
        for url in tail:
            html_content += f'<img src="{html.escape(url)}"/>'
    if nav_html and len(urls) >= 2:
        html_content += nav_html
    if page_mode == "history" and display_link and source_url:
        html_content += f'<p><a href="{html.escape(source_url)}">Источник</a></p>'
    if page_mode == "history":
        html_content = html_content.replace(FOOTER_LINK_HTML, "").rstrip()
        html_content = re.sub(
            r'(?:<p>(?:&nbsp;|&#8203;)</p>)?<p><a href="https://t\.me/kgdstories">[^<]+</a></p>',
            "",
            html_content,
        ).rstrip()
        html_content += HISTORY_FOOTER_HTML
    else:
        html_content = apply_footer_link(html_content)
    html_content = lint_telegraph_html(html_content)
    mode = "html" if html_text else "plain"
    logging.info("SRC build mode=%s urls_total=%d input_urls=%d", mode, len(urls), input_count)
    logging.info(
        "html_mode=%s tg_emoji_cleaned=%d tg_spoiler_unwrapped=%d",
        LAST_HTML_MODE,
        tg_emoji_cleaned,
        tg_spoiler_unwrapped,
    )
    logging.info(
        "build_source_page_content: cover=%d tail=%d nav_dup=%s catbox_msg=%s",
        len(cover),
        len(tail),
        bool(nav_html and len(urls) >= 2),
        catbox_msg,
    )
    return html_content, catbox_msg, len(urls)


async def create_source_page(
    title: str,
    text: str,
    source_url: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    ics_url: str | None = None,
    db: Database | None = None,
    *,
    display_link: bool = True,
    catbox_urls: list[str] | None = None,
    image_mode: Literal["tail", "inline"] = "tail",
    page_mode: Literal["default", "history"] = "default",
) -> tuple[str, str, str, int] | None:
    """Create a Telegraph page with the original event text."""
    if db and text and text.strip():
        from models import Event
        from sqlalchemy import select

        async with db.get_session() as session:
            res = await session.execute(
                select(Event.telegraph_url, Event.telegraph_path).where(
                    Event.source_text == text
                )
            )
            existing = res.first()
            if existing and existing.telegraph_path:
                return existing.telegraph_url, existing.telegraph_path, "", 0
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return None
    tg = Telegraph(access_token=token)
    html_content, catbox_msg, uploaded = await build_source_page_content(
        title,
        text,
        source_url,
        html_text,
        media,
        ics_url,
        db,
        display_link=display_link,
        catbox_urls=catbox_urls,
        image_mode=image_mode,
        page_mode=page_mode,
    )
    logging.info("SRC page compose uploaded=%d catbox_msg=%s", uploaded, catbox_msg)
    from telegraph.utils import html_to_nodes, InvalidHTML

    try:
        nodes = html_to_nodes(html_content)
    except InvalidHTML as exc:
        if not html_text:
            raise
        logging.warning(
            "Invalid HTML in source page, rebuilding without editor markup: %s", exc
        )
        html_content, catbox_msg, uploaded = await build_source_page_content(
            title,
            text,
            source_url,
            None,
            media,
            ics_url,
            db,
            display_link=display_link,
            catbox_urls=catbox_urls,
            image_mode=image_mode,
            page_mode=page_mode,
        )
        try:
            nodes = html_to_nodes(html_content)
        except InvalidHTML:
            logging.exception("Fallback source page content is still invalid")
            raise
    author_name = (
        "Полюбить Калининград Истории"
        if page_mode == "history"
        else "Полюбить Калининград Анонсы"
    )
    author_url: str | None = None
    if page_mode == "history":
        author_url = HISTORY_TELEGRAPH_AUTHOR_URL
    try:
        kwargs = dict(
            title=title,
            author_name=author_name,
            content=nodes,
            return_content=False,
            caller="event_pipeline",
        )
        if author_url:
            kwargs["author_url"] = author_url
        page = await telegraph_create_page(tg, **kwargs)
    except Exception as e:
        logging.error("Failed to create telegraph page: %s", e)
        return None
    url = normalize_telegraph_url(page.get("url"))
    logging.info(
        "SRC page created title=%s uploaded=%d url=%s",
        title,
        uploaded,
        url,
    )
    return url, page.get("path"), catbox_msg, uploaded


def create_app() -> web.Application:
    global db
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    webhook = os.getenv("WEBHOOK_URL")
    # Webhook is optional - only required for production mode
    # In dev mode, we'll skip webhook setup

    from main import IPv4AiohttpSession
    session = IPv4AiohttpSession(timeout=ClientTimeout(total=HTTP_TIMEOUT))
    bot = SafeBot(token, session=session)
    logging.info("DB_PATH=%s", DB_PATH)
    logging.info("FOUR_O_TOKEN found: %s", bool(os.getenv("FOUR_O_TOKEN")))
    dp = Dispatcher()
    dp.include_router(ik_poster_router)
    db = Database(DB_PATH)
    set_db(db)  # Set db in main.py's namespace for handlers
    dp.include_router(special_router)  # must be after db init
    import video_announce.handlers as video_handlers
    import preview_3d.handlers as preview_3d_handlers

    async def start_wrapper(message: types.Message):
        await handle_start(message, db, bot)

    async def register_wrapper(message: types.Message):
        await handle_register(message, db, bot)

    async def help_wrapper(message: types.Message):
        await handle_help(message, db, bot)

    async def ocrtest_wrapper(message: types.Message):
        await handle_ocrtest(message, db, bot)

    async def ocr_detail_wrapper(callback: types.CallbackQuery):
        await vision_test.select_detail(callback, bot)

    async def ocr_photo_wrapper(message: types.Message):
        images = await extract_images(message, bot)
        await vision_test.handle_photo(message, bot, images)

    async def requests_wrapper(message: types.Message):
        await handle_requests(message, db, bot)

    async def usage_test_wrapper(message: types.Message):
        await handle_usage_test(message, db, bot)

    async def tz_wrapper(message: types.Message):
        await handle_tz(message, db, bot)

    async def callback_wrapper(callback: types.CallbackQuery):
        await process_request(callback, db, bot)

    async def add_event_wrapper(message: types.Message):
        logging.info("add_event_wrapper start: user=%s", message.from_user.id)
        if message.from_user.id in add_event_sessions:
            return

        text = normalize_addevent_text(strip_leading_cmd(message.text or ""))
        if not text:
            text = normalize_addevent_text(strip_leading_cmd(message.caption or ""))
        if not text:
            logging.info("add_event_wrapper usage: empty input")
            await send_usage_fast(bot, message.chat.id)
            return
        message.text = f"/addevent {text}"
        await enqueue_add_event(message, db, bot)

    async def add_event_raw_wrapper(message: types.Message):
        logging.info("add_event_raw_wrapper start: user=%s", message.from_user.id)
        await enqueue_add_event_raw(message, db, bot)

    async def ask_4o_wrapper(message: types.Message):
        await handle_ask_4o(message, db, bot)

    async def list_events_wrapper(message: types.Message):
        await handle_events(message, db, bot)

    async def set_channel_wrapper(message: types.Message):
        await handle_set_channel(message, db, bot)

    async def channels_wrapper(message: types.Message):
        await handle_channels(message, db, bot)

    async def exhibitions_wrapper(message: types.Message):
        await handle_exhibitions(message, db, bot)

    async def digest_wrapper(message: types.Message):
        await show_digest_menu(message, db, bot)

    async def digest_select_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_lectures(callback, db, bot)

    async def digest_select_masterclasses_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_masterclasses(callback, db, bot)

    async def digest_select_exhibitions_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_exhibitions(callback, db, bot)

    async def digest_select_psychology_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_psychology(callback, db, bot)

    async def digest_select_science_pop_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_science_pop(callback, db, bot)

    async def digest_select_kraevedenie_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_kraevedenie(callback, db, bot)

    async def digest_select_networking_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_networking(callback, db, bot)

    async def digest_select_entertainment_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_entertainment(callback, db, bot)

    async def digest_select_markets_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_markets(callback, db, bot)

    async def digest_select_theatre_classic_wrapper(
        callback: types.CallbackQuery,
    ):
        await handle_digest_select_theatre_classic(callback, db, bot)

    async def digest_select_theatre_modern_wrapper(
        callback: types.CallbackQuery,
    ):
        await handle_digest_select_theatre_modern(callback, db, bot)

    async def digest_select_meetups_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_meetups(callback, db, bot)

    async def digest_select_movies_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_movies(callback, db, bot)

    async def digest_disabled_wrapper(callback: types.CallbackQuery):
        await callback.answer("Ещё не реализовано", show_alert=False)

    async def digest_toggle_wrapper(callback: types.CallbackQuery):
        await handle_digest_toggle(callback, bot)

    async def digest_refresh_wrapper(callback: types.CallbackQuery):
        await handle_digest_refresh(callback, bot)

    async def digest_send_wrapper(callback: types.CallbackQuery):
        await handle_digest_send(callback, db, bot)

    async def digest_notify_partners_wrapper(callback: types.CallbackQuery):
        await handle_digest_notify_partners(callback, db, bot)

    async def digest_hide_wrapper(callback: types.CallbackQuery):
        await handle_digest_hide(callback, bot)

    async def pages_wrapper(message: types.Message):
        await handle_pages(message, db, bot)

    async def pages_rebuild_wrapper(message: types.Message):
        await handle_pages_rebuild(message, db, bot)

    async def weekendimg_cmd_wrapper(message: types.Message):
        await handle_weekendimg_cmd(message, db, bot)

    async def weekendimg_cb_wrapper(callback: types.CallbackQuery):
        await handle_weekendimg_cb(callback, db, bot)

    async def weekendimg_photo_wrapper(message: types.Message):
        await handle_weekendimg_photo(message, db, bot)

    async def stats_wrapper(message: types.Message):
        await handle_stats(message, db, bot)

    async def fest_wrapper(message: types.Message):
        await handle_fest(message, db, bot)

    async def imp_groups_30d_wrapper(message: types.Message):
        await handle_imp_groups_30d(message, db, bot)

    async def imp_daily_14d_wrapper(message: types.Message):
        await handle_imp_daily_14d(message, db, bot)

    async def festival_edit_wrapper(message: types.Message):
        await handle_festival_edit_message(message, db, bot)

    async def users_wrapper(message: types.Message):
        await handle_users(message, db, bot)

    async def dumpdb_wrapper(message: types.Message):
        await handle_dumpdb(message, db, bot)

    async def tourist_export_wrapper(message: types.Message):
        await handle_tourist_export(message, db, bot)

    async def telegraph_fix_author_wrapper(message: types.Message):
        await handle_telegraph_fix_author(message, db, bot)

    async def restore_wrapper(message: types.Message):
        await handle_restore(message, db, bot)

    async def video_cmd_wrapper(message: types.Message):
        await video_handlers.handle_video_command(message, db, bot)

    async def kaggle_test_wrapper(message: types.Message):
        await video_handlers.handle_kaggle_test(message, db, bot)

    async def video_cb_wrapper(callback: types.CallbackQuery):
        await video_handlers.handle_video_callback(
            callback,
            db,
            bot,
            build_events_message=build_events_message,
            get_tz_offset=get_tz_offset,
            offset_to_timezone=offset_to_timezone,
        )

    async def video_instruction_wrapper(message: types.Message):
        await video_handlers.handle_instruction_message(message, db, bot)

    async def video_intro_wrapper(message: types.Message):
        await video_handlers.handle_intro_message(message, db, bot)

    async def video_payload_wrapper(message: types.Message):
        await video_handlers.handle_payload_import_message(message, db, bot)

    async def preview_3di_wrapper(message: types.Message):
        await preview_3d_handlers.handle_3di_command(message, db, bot)

    async def preview_3di_cb_wrapper(callback: types.CallbackQuery):
        await preview_3d_handlers.handle_3di_callback(callback, db, bot)

    async def edit_message_wrapper(message: types.Message):
        await handle_edit_message(message, db, bot)

    async def daily_time_wrapper(message: types.Message):
        await handle_daily_time_message(message, db, bot)

    async def vk_group_msg_wrapper(message: types.Message):
        await handle_vk_group_message(message, db, bot)

    async def vk_time_msg_wrapper(message: types.Message):
        await handle_vk_time_message(message, db, bot)

    async def partner_info_wrapper(message: types.Message):
        await handle_partner_info_message(message, db, bot)

    async def forward_wrapper(message: types.Message):
        await handle_forwarded(message, db, bot)

    async def reg_daily_wrapper(message: types.Message):
        await handle_regdailychannels(message, db, bot)

    async def daily_wrapper(message: types.Message):
        await handle_daily(message, db, bot)

    async def images_wrapper(message: types.Message):
        await handle_images(message, db, bot)

    async def vkgroup_wrapper(message: types.Message):
        await handle_vkgroup(message, db, bot)

    async def vktime_wrapper(message: types.Message):
        await handle_vktime(message, db, bot)

    async def vkphotos_wrapper(message: types.Message):
        await handle_vkphotos(message, db, bot)

    captcha_handler = partial(handle_vk_captcha, db=db, bot=bot)

    async def askloc_wrapper(callback: types.CallbackQuery):
        await handle_askloc(callback, db, bot)

    async def askcity_wrapper(callback: types.CallbackQuery):
        await handle_askcity(callback, db, bot)

    async def pages_rebuild_cb_wrapper(callback: types.CallbackQuery):
        await handle_pages_rebuild_cb(callback, db, bot)

    async def captcha_prompt_wrapper(callback: types.CallbackQuery):
        await handle_vk_captcha_prompt(callback, db, bot)

    async def captcha_delay_wrapper(callback: types.CallbackQuery):
        await handle_vk_captcha_delay(callback, db, bot)
    async def captcha_refresh_wrapper(callback: types.CallbackQuery):
        await handle_vk_captcha_refresh(callback, db, bot)

    async def menu_wrapper(message: types.Message):
        await handle_menu(message, db, bot)

    async def events_menu_wrapper(message: types.Message):
        await handle_events_menu(message, db, bot)

    async def events_date_wrapper(message: types.Message):
        await handle_events_date_message(message, db, bot)

    async def add_event_start_wrapper(message: types.Message):
        logging.info("add_event_start_wrapper start: user=%s", message.from_user.id)
        await handle_add_event_start(message, db, bot)

    async def add_festival_start_wrapper(message: types.Message):
        logging.info(
            "add_festival_start_wrapper start: user=%s", message.from_user.id
        )
        await handle_add_festival_start(message, db, bot)

    async def dom_iskusstv_start_wrapper(message: types.Message):
        await handle_dom_iskusstv_start(message, db, bot)

    async def add_event_session_wrapper(message: types.Message):
        logging.info("add_event_session_wrapper start: user=%s", message.from_user.id)
        session_mode = add_event_sessions.get(message.from_user.id)
        
        # Check for festival URL parsing
        if session_mode == "festival":
            text = (message.text or "").strip()
            from source_parsing.festival_parser import is_valid_url, process_festival_url
            
            if is_valid_url(text):
                # URL detected - use festival parser
                logging.info("festival_url_detected: user=%s url=%s", message.from_user.id, text)
                add_event_sessions.pop(message.from_user.id, None)
                
                await bot.send_message(message.chat.id, "🔄 Запускаю парсер фестиваля...")
                
                try:
                    async def status_callback(status: str):
                        await bot.send_message(message.chat.id, f"📊 {status}")
                    
                    festival, uds_url, llm_log_url = await process_festival_url(
                        db=db,
                        bot=bot,
                        chat_id=message.chat.id,
                        url=text,
                        status_callback=status_callback,
                    )
                    
                    # Build success message
                    lines = [
                        f"✅ Фестиваль добавлен: **{festival.name}**",
                    ]
                    if festival.telegraph_url:
                        lines.append(f"📄 [Открыть страницу фестиваля]({festival.telegraph_url})")
                    if uds_url:
                        lines.append(f"📊 [JSON отчёт парсинга]({uds_url})")
                    if llm_log_url:
                        lines.append(f"🔍 [LLM лог (отладка)]({llm_log_url})")
                    
                    await bot.send_message(
                        message.chat.id,
                        "\n".join(lines),
                        parse_mode="Markdown",
                    )
                    
                except Exception as e:
                    logging.exception("festival_parser_failed: %s", e)
                    await bot.send_message(
                        message.chat.id,
                        f"❌ Парсинг завершился ошибкой: {e}",
                    )
                return
        
        if message.media_group_id:
            await handle_add_event_media_group(message, db, bot)
        else:
            await enqueue_add_event(
                message, db, bot, session_mode=session_mode
            )

    async def vk_link_cmd_wrapper(message: types.Message):
        logging.info("vk_link_cmd_wrapper start: user=%s", message.from_user.id)
        await handle_vk_link_command(message, db, bot)

    async def vk_cmd_wrapper(message: types.Message):
        await handle_vk_command(message, db, bot)

    async def vk_add_start_wrapper(message: types.Message):
        await handle_vk_add_start(message, db, bot)

    async def vk_add_msg_wrapper(message: types.Message):
        await handle_vk_add_message(message, db, bot)

    async def pyramida_start_wrapper(message: types.Message):
        await handle_pyramida_start(message, db, bot)

    async def pyramida_input_wrapper(message: types.Message):
        await handle_pyramida_input(message, db, bot)

    async def dom_iskusstv_start_wrapper(message: types.Message):
        await handle_dom_iskusstv_start(message, db, bot)

    async def dom_iskusstv_input_wrapper(message: types.Message):
        await handle_dom_iskusstv_input(message, db, bot)

    async def vk_list_wrapper(message: types.Message):
        await handle_vk_list(message, db, bot)

    async def vk_check_wrapper(message: types.Message):
        await handle_vk_check(message, db, bot)

    async def vk_delete_wrapper(callback: types.CallbackQuery):
        await handle_vk_delete_callback(callback, db, bot)

    async def vk_list_page_wrapper(callback: types.CallbackQuery):
        await handle_vk_list_page_callback(callback, db, bot)

    async def vk_rejected_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_rejected_callback(callback, db, bot)

    async def vk_settings_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_settings_callback(callback, db, bot)

    async def vk_dtime_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_dtime_callback(callback, db, bot)

    async def vk_dtime_msg_wrapper(message: types.Message):
        await handle_vk_dtime_message(message, db, bot)

    async def vk_ticket_link_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_ticket_link_callback(callback, db, bot)

    async def vk_ticket_link_msg_wrapper(message: types.Message):
        await handle_vk_ticket_link_message(message, db, bot)

    async def vk_location_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_location_callback(callback, db, bot)

    async def vk_location_msg_wrapper(message: types.Message):
        await handle_vk_location_message(message, db, bot)

    async def vk_next_wrapper(callback: types.CallbackQuery):
        await handle_vk_next_callback(callback, db, bot)

    async def vk_review_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_review_cb(callback, db, bot)

    async def vk_miss_review_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_miss_review_callback(callback, db, bot)

    async def vk_extra_msg_wrapper(message: types.Message):
        await handle_vk_extra_message(message, db, bot)

    async def vk_story_instr_msg_wrapper(message: types.Message):
        await handle_vk_story_instruction_message(message, db, bot)

    async def tourist_note_wrapper(message: types.Message):
        await handle_tourist_note_message(message, db, bot)

    async def vk_shortpost_edit_msg_wrapper(message: types.Message):
        await handle_vk_shortpost_edit_message(message, db, bot)

    async def vk_crawl_now_wrapper(message: types.Message):
        await handle_vk_crawl_now(message, db, bot)

    async def vk_queue_wrapper(message: types.Message):
        await handle_vk_queue(message, db, bot)

    async def vk_requeue_imported_wrapper(message: types.Message):
        await handle_vk_requeue_imported(message, db, bot)

    async def vk_miss_review_wrapper(message: types.Message):
        await handle_vk_miss_review(message, db, bot)
    async def status_wrapper(message: types.Message):
        await handle_status(message, db, bot, app)

    async def trace_wrapper(message: types.Message):
        await handle_trace(message, db, bot)

    async def last_errors_wrapper(message: types.Message):
        await handle_last_errors(message, db, bot)

    async def debug_wrapper(message: types.Message):
        await handle_debug(message, db, bot)

    async def queue_reap_wrapper(message: types.Message):
        await handle_queue_reap(message, db, bot)

    async def mem_wrapper(message: types.Message):
        await handle_mem(message, db, bot)

    async def festmerge_do_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_do_callback(callback, db, bot)

    async def festmerge_to_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_to_callback(callback, db, bot)

    async def festmerge_page_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_page_callback(callback, db, bot)

    async def festmerge_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_callback(callback, db, bot)
 
    async def backfill_topics_wrapper(message: types.Message):
        await handle_backfill_topics(message, db, bot)

    async def festivals_fix_nav_wrapper(message: types.Message):
        await handle_festivals_fix_nav(message, db, bot)

    async def ics_fix_nav_wrapper(message: types.Message):
        await handle_ics_fix_nav(message, db, bot)

    dp.message.register(help_wrapper, Command("help"))
    dp.message.register(ocrtest_wrapper, Command("ocrtest"))
    dp.message.register(start_wrapper, Command("start"))
    dp.message.register(register_wrapper, Command("register"))
    dp.message.register(requests_wrapper, Command("requests"))
    dp.message.register(kaggle_test_wrapper, Command("kaggletest"))
    dp.message.register(video_cmd_wrapper, Command("v"))
    dp.message.register(preview_3di_wrapper, Command("3di"))
    dp.message.register(usage_test_wrapper, Command("usage_test"))
    dp.callback_query.register(
        callback_wrapper,
        lambda c: c.data.startswith("approve")
        or c.data.startswith("reject")
        or c.data.startswith("del:")
        or c.data.startswith("nav:")
        or c.data.startswith("edit:")
        or c.data.startswith("editfield:")
        or c.data.startswith("editdone:")
        or c.data.startswith("makefest:")
        or c.data.startswith("makefest_create:")
        or c.data.startswith("makefest_bind:")
        or c.data.startswith("unset:")
        or c.data.startswith("assetunset:")
        or c.data.startswith("set:")
        or c.data.startswith("assetset:")
        or c.data.startswith("dailyset:")
        or c.data.startswith("dailyunset:")
        or c.data.startswith("dailytime:")
        or c.data.startswith("dailysend:")
        or c.data.startswith("dailysendtom:")
        or c.data == "vkset"
        or c.data == "vkunset"
        or c.data.startswith("vktime:")
        or c.data.startswith("vkdailysend:")
        or c.data.startswith("menuevt:")
        or c.data.startswith("togglefree:")
        or c.data.startswith("markfree:")
        or c.data.startswith("togglesilent:")
        or c.data.startswith("createics:")
        or c.data.startswith("delics:")
        or c.data.startswith("partner:")
        or c.data.startswith("block:")
        or c.data.startswith("unblock:")
        or c.data.startswith("festedit:")
        or c.data.startswith("festeditfield:")
        or c.data == "festeditdone"
        or c.data.startswith("festpage:")
        or c.data.startswith("festdel:")
        or c.data.startswith("setfest:")
        or c.data.startswith("festdays:")
        or c.data.startswith("festimgs:")
        or c.data.startswith("festsetcover:")
        or c.data.startswith("festcover:")
        or c.data.startswith("festsyncevents:")
        or c.data.startswith("festreparse:")
        or c.data.startswith("requeue:")
        or c.data.startswith("tourist:")
    ,
    )
    dp.callback_query.register(
        video_cb_wrapper, lambda c: c.data and c.data.startswith("vid")
    )
    dp.callback_query.register(
        preview_3di_cb_wrapper, lambda c: c.data and c.data.startswith("3di:")
    )
    dp.callback_query.register(
        festmerge_do_wrapper, lambda c: c.data and c.data.startswith("festmerge_do:")
    )
    dp.callback_query.register(
        festmerge_to_wrapper, lambda c: c.data and c.data.startswith("festmerge_to:")
    )
    dp.callback_query.register(
        festmerge_page_wrapper, lambda c: c.data and c.data.startswith("festmergep:")
    )
    dp.callback_query.register(
        festmerge_wrapper, lambda c: c.data and c.data.startswith("festmerge:")
    )
    dp.callback_query.register(
        ocr_detail_wrapper,
        lambda c: c.data and c.data.startswith("ocr:detail:"),
    )
    dp.callback_query.register(askloc_wrapper, lambda c: c.data == "askloc")
    dp.callback_query.register(askcity_wrapper, lambda c: c.data == "askcity")
    dp.callback_query.register(
        pages_rebuild_cb_wrapper, lambda c: c.data and c.data.startswith("pages_rebuild:")
    )
    dp.callback_query.register(captcha_prompt_wrapper, lambda c: c.data == "captcha_input")
    dp.callback_query.register(captcha_delay_wrapper, lambda c: c.data == "captcha_delay")
    dp.callback_query.register(captcha_refresh_wrapper, lambda c: c.data == "captcha_refresh")
    dp.callback_query.register(
        vk_rejected_cb_wrapper, lambda c: c.data and c.data.startswith("vkrejected:")
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(
        video_instruction_wrapper,
        lambda m: video_handlers.is_waiting_instruction(m.from_user.id),
    )
    dp.message.register(
        video_intro_wrapper,
        lambda m: video_handlers.is_waiting_intro_text(m.from_user.id),
    )
    dp.message.register(
        video_payload_wrapper,
        lambda m: video_handlers.is_waiting_payload_import(m.from_user.id),
    )
    dp.message.register(
        add_event_session_wrapper, lambda m: m.from_user.id in add_event_sessions
    )
    dp.message.register(
        ocr_photo_wrapper,
        lambda m: vision_test.is_waiting(m.from_user.id),
        F.photo | F.document,
    )
    dp.message.register(add_event_wrapper, Command("addevent"))
    dp.message.register(add_event_raw_wrapper, Command("addevent_raw"))
    dp.message.register(ask_4o_wrapper, Command("ask4o"))
    dp.message.register(list_events_wrapper, Command("events"))
    dp.message.register(set_channel_wrapper, Command("setchannel"))
    dp.message.register(images_wrapper, Command("images"))
    dp.message.register(vkgroup_wrapper, Command("vkgroup"))
    dp.message.register(vktime_wrapper, Command("vktime"))
    dp.message.register(vkphotos_wrapper, Command("vkphotos"))
    dp.message.register(captcha_handler, Command("captcha"))
    dp.message.register(captcha_handler, F.reply_to_message)
    dp.message.register(menu_wrapper, Command("menu"))
    dp.message.register(events_menu_wrapper, lambda m: m.text == MENU_EVENTS)
    dp.message.register(events_date_wrapper, lambda m: m.from_user.id in events_date_sessions)
    dp.message.register(add_event_start_wrapper, lambda m: m.text == MENU_ADD_EVENT)
    dp.message.register(dom_iskusstv_start_wrapper, lambda m: m.text == MENU_DOM_ISKUSSTV)
    dp.message.register(
        add_festival_start_wrapper, lambda m: m.text == MENU_ADD_FESTIVAL
    )
    dp.message.register(vk_link_cmd_wrapper, Command("vklink"))
    dp.message.register(vk_cmd_wrapper, Command("vk"))
    dp.message.register(vk_crawl_now_wrapper, Command("vk_crawl_now"))
    dp.message.register(vk_queue_wrapper, Command("vk_queue"))
    dp.message.register(vk_requeue_imported_wrapper, Command("vk_requeue_imported"))
    dp.message.register(
        vk_miss_review_wrapper,
        Command(VK_MISS_REVIEW_COMMAND.lstrip("/")),
    )
    dp.message.register(vk_add_start_wrapper, lambda m: m.text == VK_BTN_ADD_SOURCE)
    dp.message.register(vk_list_wrapper, lambda m: m.text == VK_BTN_LIST_SOURCES)
    dp.message.register(vk_check_wrapper, lambda m: m.text == VK_BTN_CHECK_EVENTS)
    dp.message.register(vk_queue_wrapper, lambda m: m.text == VK_BTN_QUEUE_SUMMARY)
    dp.message.register(pyramida_start_wrapper, lambda m: m.text == VK_BTN_PYRAMIDA)
    dp.message.register(pyramida_input_wrapper, lambda m: m.from_user.id in pyramida_input_sessions)
    dp.message.register(dom_iskusstv_start_wrapper, lambda m: m.text == VK_BTN_DOM_ISKUSSTV)
    dp.message.register(dom_iskusstv_input_wrapper, lambda m: m.from_user.id in dom_iskusstv_input_sessions)
    dp.message.register(vk_add_msg_wrapper, lambda m: m.from_user.id in vk_add_source_sessions)

    dp.message.register(vk_extra_msg_wrapper, lambda m: m.from_user.id in vk_review_extra_sessions)
    dp.message.register(
        vk_story_instr_msg_wrapper,
        lambda m: (
            (state := vk_review_story_sessions.get(m.from_user.id)) is not None
            and state.awaiting_instructions
        ),
    )
    dp.message.register(
        tourist_note_wrapper, lambda m: m.from_user.id in tourist_note_sessions
    )
    dp.message.register(
        vk_shortpost_edit_msg_wrapper,
        lambda m: m.from_user.id in vk_shortpost_edit_sessions,
    )
    dp.message.register(partner_info_wrapper, lambda m: m.from_user.id in partner_info_sessions)
    dp.message.register(channels_wrapper, Command("channels"))
    dp.message.register(reg_daily_wrapper, Command("regdailychannels"))
    dp.message.register(daily_wrapper, Command("daily"))
    dp.message.register(exhibitions_wrapper, Command("exhibitions"))
    dp.message.register(digest_wrapper, Command("digest"))
    dp.callback_query.register(
        digest_select_wrapper, lambda c: c.data.startswith("digest:select:lectures:")
    )
    dp.callback_query.register(
        digest_select_masterclasses_wrapper,
        lambda c: c.data.startswith("digest:select:masterclasses:"),
    )
    dp.callback_query.register(
        digest_select_exhibitions_wrapper,
        lambda c: c.data.startswith("digest:select:exhibitions:"),
    )
    dp.callback_query.register(
        digest_select_psychology_wrapper,
        lambda c: c.data.startswith("digest:select:psychology:"),
    )
    dp.callback_query.register(
        digest_select_science_pop_wrapper,
        lambda c: c.data.startswith("digest:select:science_pop:"),
    )
    dp.callback_query.register(
        digest_select_kraevedenie_wrapper,
        lambda c: c.data.startswith("digest:select:kraevedenie:"),
    )
    dp.callback_query.register(
        digest_select_networking_wrapper,
        lambda c: c.data.startswith("digest:select:networking:"),
    )
    dp.callback_query.register(
        digest_select_entertainment_wrapper,
        lambda c: c.data.startswith("digest:select:entertainment:"),
    )
    dp.callback_query.register(
        digest_select_markets_wrapper,
        lambda c: c.data.startswith("digest:select:markets:"),
    )
    dp.callback_query.register(
        digest_select_theatre_classic_wrapper,
        lambda c: c.data.startswith("digest:select:theatre_classic:"),
    )
    dp.callback_query.register(
        digest_select_theatre_modern_wrapper,
        lambda c: c.data.startswith("digest:select:theatre_modern:"),
    )
    dp.callback_query.register(
        digest_select_meetups_wrapper,
        lambda c: c.data.startswith("digest:select:meetups:"),
    )
    dp.callback_query.register(
        digest_select_movies_wrapper,
        lambda c: c.data.startswith("digest:select:movies:"),
    )
    dp.callback_query.register(
        digest_disabled_wrapper, lambda c: c.data == "digest:disabled"
    )
    dp.callback_query.register(
        digest_toggle_wrapper, lambda c: c.data.startswith("dg:t:")
    )
    dp.callback_query.register(
        digest_refresh_wrapper, lambda c: c.data.startswith("dg:r:")
    )
    dp.callback_query.register(
        digest_send_wrapper, lambda c: c.data.startswith("dg:s:")
    )
    dp.callback_query.register(
        digest_notify_partners_wrapper, lambda c: c.data.startswith("dg:np:")
    )
    dp.callback_query.register(
        digest_hide_wrapper, lambda c: c.data.startswith("dg:x:")
    )
    dp.message.register(fest_wrapper, Command("fest"))

    dp.message.register(weekendimg_cmd_wrapper, Command("weekendimg"))
    dp.callback_query.register(
        weekendimg_cb_wrapper, lambda c: c.data and c.data.startswith("weekimg:")
    )
    dp.message.register(
        weekendimg_photo_wrapper, lambda m: m.from_user.id in weekend_img_wait
    )

    dp.message.register(pages_wrapper, Command("pages"))
    dp.message.register(pages_rebuild_wrapper, Command("pages_rebuild"))
    dp.message.register(stats_wrapper, Command("stats"))
    dp.message.register(imp_groups_30d_wrapper, Command("imp_groups_30d"))
    dp.message.register(imp_daily_14d_wrapper, Command("imp_daily_14d"))
    dp.message.register(status_wrapper, Command("status"))
    dp.message.register(trace_wrapper, Command("trace"))
    dp.message.register(last_errors_wrapper, Command("last_errors"))
    dp.message.register(debug_wrapper, Command("debug"))
    dp.message.register(queue_reap_wrapper, Command("queue_reap"))
    dp.message.register(mem_wrapper, Command("mem"))
    dp.message.register(backfill_topics_wrapper, Command("backfill_topics"))
    dp.message.register(festivals_fix_nav_wrapper, Command("festivals_fix_nav"))
    dp.message.register(festivals_fix_nav_wrapper, Command("festivals_nav_dedup"))
    dp.message.register(ics_fix_nav_wrapper, Command("ics_fix_nav"))
    dp.message.register(users_wrapper, Command("users"))
    dp.message.register(dumpdb_wrapper, Command("dumpdb"))
    dp.message.register(tourist_export_wrapper, Command("tourist_export"))
    dp.message.register(telegraph_fix_author_wrapper, Command("telegraph_fix_author"))
    dp.message.register(restore_wrapper, Command("restore"))
    # Register /parse command from service module
    from source_parsing.commands import register_parse_command
    register_parse_command(dp, db, bot)
    dp.message.register(
        edit_message_wrapper, lambda m: m.from_user.id in editing_sessions
    )
    dp.message.register(
        daily_time_wrapper, lambda m: m.from_user.id in daily_time_sessions
    )
    dp.message.register(
        vk_group_msg_wrapper, lambda m: m.from_user.id in vk_group_sessions
    )
    dp.message.register(
        vk_time_msg_wrapper, lambda m: m.from_user.id in vk_time_sessions
    )
    dp.message.register(
        vk_dtime_msg_wrapper, lambda m: m.from_user.id in vk_default_time_sessions
    )
    dp.message.register(
        vk_ticket_link_msg_wrapper,
        lambda m: m.from_user.id in vk_default_ticket_link_sessions,
    )
    dp.message.register(
        vk_location_msg_wrapper,
        lambda m: m.from_user.id in vk_default_location_sessions,
    )
    dp.callback_query.register(
        vk_list_page_wrapper, lambda c: c.data.startswith("vksrcpage:")
    )
    dp.callback_query.register(vk_delete_wrapper, lambda c: c.data.startswith("vkdel:"))
    dp.callback_query.register(
        vk_settings_cb_wrapper, lambda c: c.data.startswith("vkset:")
    )
    dp.callback_query.register(vk_dtime_cb_wrapper, lambda c: c.data.startswith("vkdt:"))
    dp.callback_query.register(
        vk_ticket_link_cb_wrapper, lambda c: c.data.startswith("vklink:")
    )
    dp.callback_query.register(
        vk_location_cb_wrapper, lambda c: c.data.startswith("vkloc:")
    )
    dp.callback_query.register(vk_next_wrapper, lambda c: c.data.startswith("vknext:"))
    dp.callback_query.register(vk_review_cb_wrapper, lambda c: c.data and c.data.startswith("vkrev:"))
    dp.callback_query.register(
        vk_miss_review_cb_wrapper,
        lambda c: c.data and c.data.startswith(VK_MISS_CALLBACK_PREFIX),
    )
    dp.message.register(

        festival_edit_wrapper, lambda m: m.from_user.id in festival_edit_sessions

    )
    dp.message.register(
        forward_wrapper,
        lambda m: bool(m.forward_date)
        or "forward_origin" in getattr(m, "model_extra", {}),
    )
    dp.my_chat_member.register(partial(handle_my_chat_member, db=db))

    app = web.Application()
    SimpleRequestHandler(dp, bot).register(app, path="/webhook")
    setup_application(app, dp, bot=bot)
    
    # Store bot and dispatcher in app context for dev mode
    app["bot"] = bot
    app["dispatcher"] = dp

    async def health_handler(request: web.Request) -> web.Response:
        async with span("healthz"):
            return web.Response(text="ok")

    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/metrics", metrics_handler)

    async def on_startup(app: web.Application):
        await init_db_and_scheduler(app, db, bot, webhook)

    async def on_shutdown(app: web.Application):
        await bot.session.close()
        if "add_event_watch" in app:
            app["add_event_watch"].cancel()
            with contextlib.suppress(Exception):
                await app["add_event_watch"]
        if "add_event_worker" in app:
            app["add_event_worker"].cancel()
            with contextlib.suppress(Exception):
                await app["add_event_worker"]
        if "daily_scheduler" in app:
            app["daily_scheduler"].cancel()
            with contextlib.suppress(Exception):
                await app["daily_scheduler"]
        scheduler_cleanup()
        await close_vk_session()
        close_supabase_client()

    global _startup_handler_registered
    if not _startup_handler_registered:
        app.on_startup.append(on_startup)
        _startup_handler_registered = True
    app.on_shutdown.append(on_shutdown)
    return app


def _normalize_event_description(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


async def _handle_digest_select(
    callback: types.CallbackQuery,
    db: Database,
    bot: Bot,
    *,
    digest_type: str,
    preview_builder: Callable[[str, Database, datetime], Awaitable[tuple[str, List[str], int, List[Event], List[str]]]],
    items_noun: str,
    panel_text: str,
) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4 or parts[2] != digest_type:
        return
    digest_id = parts[3]

    chat_id = callback.message.chat.id if callback.message else None
    if chat_id is None:
        await callback.answer()
        return

    logging.info(
        "digest.type.selected digest_id=%s type=%s chat_id=%s user_id=%s callback_id=%s",
        digest_id,
        digest_type,
        chat_id,
        callback.from_user.id,
        callback.id,
    )

    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)
    now = datetime.now(tz).replace(tzinfo=None)

    intro, lines, horizon, events, norm_titles = await preview_builder(
        digest_id, db, now
    )
    if not events:
        await bot.send_message(
            chat_id,
            f"Пока ничего нет в ближайшие {horizon} дней с учётом правила “+2 часа”.",
        )
        return

    items: List[dict] = []
    for idx, (ev, line, norm_title) in enumerate(
        zip(events, lines, norm_titles), start=1
    ):
        cover_url = None
        if ev.telegraph_url:
            try:
                covers = await extract_catbox_covers_from_telegraph(
                    ev.telegraph_url, event_id=ev.id
                )
                cover_url = covers[0] if covers else None
            except Exception:
                cover_url = None
        norm_topics = normalize_topics(getattr(ev, "topics", []))
        items.append(
            {
                "event_id": ev.id,
                "creator_id": ev.creator_id,
                "index": idx,
                "title": ev.title,
                "norm_title": norm_title,
                "event_type": ev.event_type,
                "norm_description": _normalize_event_description(ev.description),
                "norm_topics": norm_topics,
                "date": ev.date,
                "end_date": ev.end_date,
                "link": pick_display_link(ev),
                "cover_url": cover_url,
                "line_html": line,
            }
        )

    async with db.get_session() as session_db:
        result = await session_db.execute(
            select(Channel).where(Channel.daily_time.is_not(None))
        )
        channels = result.scalars().all()

    session_data = {
        "chat_id": chat_id,
        "preview_msg_ids": [],
        "panel_msg_id": None,
        "items": items,
        "intro_html": intro,
        "footer_html": '<a href="https://t.me/kenigevents">Полюбить Калининград | Анонсы</a>',
        "excluded": set(),
        "horizon_days": horizon,
        "channels": [
            {
                "channel_id": ch.channel_id,
                "name": ch.title or ch.username or str(ch.channel_id),
                "username": ch.username,
            }
            for ch in channels
        ],
        "items_noun": items_noun,
        "panel_text": panel_text,
        "digest_type": digest_type,
    }

    digest_preview_sessions[digest_id] = session_data

    caption, attach, vis_len, kept = await _send_preview(
        session_data, digest_id, bot
    )

    logging.info(
        "digest.panel.new digest_id=%s type=%s total=%s caption_len_visible=%s attached=%s",
        digest_id,
        digest_type,
        len(items),
        vis_len,
        int(attach),
    )

    await callback.answer()


async def handle_digest_select_lectures(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="lectures",
        preview_builder=build_lectures_digest_preview,
        items_noun="лекций",
        panel_text="Управление дайджестом лекций\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_masterclasses(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="masterclasses",
        preview_builder=build_masterclasses_digest_preview,
        items_noun="мастер-классов",
        panel_text="Управление дайджестом мастер-классов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_exhibitions(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="exhibitions",
        preview_builder=build_exhibitions_digest_preview,
        items_noun="выставок",
        panel_text="Управление дайджестом выставок\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_psychology(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="psychology",
        preview_builder=build_psychology_digest_preview,
        items_noun="психологических событий",
        panel_text="Управление дайджестом психологии\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_science_pop(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="science_pop",
        preview_builder=build_science_pop_digest_preview,
        items_noun="научно-популярных событий",
        panel_text="Управление дайджестом научпопа\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_kraevedenie(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="kraevedenie",
        preview_builder=build_kraevedenie_digest_preview,
        items_noun="краеведческих событий",
        panel_text="Управление дайджестом краеведения\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_networking(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="networking",
        preview_builder=build_networking_digest_preview,
        items_noun="нетворкингов",
        panel_text="Управление дайджестом нетворкингов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_entertainment(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="entertainment",
        preview_builder=build_entertainment_digest_preview,
        items_noun="развлечений",
        panel_text="Управление дайджестом развлечений\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_markets(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="markets",
        preview_builder=build_markets_digest_preview,
        items_noun="маркетов",
        panel_text="Управление дайджестом маркетов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_theatre_classic(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="theatre_classic",
        preview_builder=build_theatre_classic_digest_preview,
        items_noun="классических спектаклей",
        panel_text="Управление дайджестом классического театра\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_theatre_modern(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="theatre_modern",
        preview_builder=build_theatre_modern_digest_preview,
        items_noun="современных спектаклей",
        panel_text="Управление дайджестом современного театра\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_meetups(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="meetups",
        preview_builder=build_meetups_digest_preview,
        items_noun="встреч и клубов",
        panel_text="Управление дайджестом встреч и клубов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_movies(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="movies",
        preview_builder=build_movies_digest_preview,
        items_noun="кинопоказов",
        panel_text="Управление дайджестом кинопоказов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def run_dev_mode():
    """Run bot in development mode with polling (no webhooks)."""
    logging.info("="*60)
    logging.info("BOT STARTING IN DEVELOPMENT MODE")
    logging.info("Mode: DEV_MODE | Connection: POLLING | Webhook: DISABLED")
    logging.info("="*60)
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
    
    # Create the application (this sets up all handlers)
    # WEBHOOK_URL is optional now, so this will work
    app = create_app()
    
    # Extract bot and dispatcher from the application context
    # They were stored in app in lines 14092-14093 of create_app()
    bot = app.get("bot")
    dp = app.get("dispatcher")
    
    if not bot or not dp:
        logging.error("Could not extract bot/dp from app")
        raise RuntimeError("Failed to extract bot and dispatcher from app")
    
    global db
    # db was already initialized in create_app
    
    # Initialize database and start background tasks
    # This is normally called in on_startup for prod mode, but we need to call it manually for dev mode
    await init_db_and_scheduler(app, db, bot, None)  # None webhook for dev mode
    
    # Delete any existing webhook
    logging.info("Deleting webhook for dev mode")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logging.info("Webhook deleted successfully")
    except Exception as e:
        logging.warning("Failed to delete webhook: %s", e)
    
    logging.info("="*60)
    logging.info("✓ DEV MODE READY: Bot is now polling for updates")
    logging.info("="*60)
    
    try:
        # Start polling
        await dp.start_polling(bot, allowed_updates=["message", "callback_query", "my_chat_member"])
    finally:
        # Cleanup - use the on_shutdown from app
        await app["bot"].session.close()
        if "add_event_watch" in app:
            app["add_event_watch"].cancel()
            with contextlib.suppress(Exception):
                await app["add_event_watch"]
        if "add_event_worker" in app:
            app["add_event_worker"].cancel()
            with contextlib.suppress(Exception):
                await app["add_event_worker"]
        if "daily_scheduler" in app:
            app["daily_scheduler"].cancel()
            with contextlib.suppress(Exception):
                await app["daily_scheduler"]
        scheduler_cleanup()
        await close_vk_session()
        close_supabase_client()




def run_prod_mode():
    """Run bot in production mode with webhooks."""
    logging.info("="*60)
    logging.info("BOT STARTING IN PRODUCTION MODE")
    logging.info("Mode: PROD_MODE | Connection: WEBHOOK | Polling: DISABLED")
    logging.info("="*60)
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    
    logging.info("Starting aiohttp server on %s:%s", host, port)
    web.run_app(
        create_app(),
        host=host,
        port=port,
    )
    logging.info("="*60)
    logging.info("✓ PROD MODE READY: Bot is listening for webhooks")
    logging.info("="*60)


if __name__ == "__main__":
    import sys
    
    # Check for special test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        # Check DEV_MODE environment variable
        dev_mode = os.getenv("DEV_MODE") == "1"
        
        if dev_mode:
            # Run in development mode with polling
            asyncio.run(run_dev_mode())
        else:
            # Run in production mode with webhooks
            run_prod_mode()
