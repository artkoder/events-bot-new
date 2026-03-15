from __future__ import annotations

import html
from datetime import date, datetime
from typing import Iterable, Mapping

from .parser import audience_line, collapse_ws

RU_WEEKDAY_SHORT = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
RU_MONTH_GEN = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}

MAX_DIGEST_ITEMS_PER_MESSAGE = 8
MAX_MEDIA_ITEMS = 5
MAX_TEXT_LEN = 3900


def _parse_date(value: str | None) -> date | None:
    raw = collapse_ws(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def format_date_time(date_iso: str | None, time_text: str | None) -> str | None:
    d = _parse_date(date_iso)
    if not d:
        return time_text or None
    base = f"{RU_WEEKDAY_SHORT[d.weekday()]}, {d.day} {RU_MONTH_GEN.get(d.month, '')}".strip()
    if time_text:
        return f"{base}, {time_text}"
    return base


def _html_text(value: object | None) -> str:
    return html.escape(collapse_ws("" if value is None else str(value)), quote=True)


def _html_link(label: object | None, url: str | None) -> str:
    text = _html_text(label)
    href = collapse_ws(url)
    if not text:
        return ""
    if not href:
        return text
    return f'<a href="{html.escape(href, quote=True)}">{text}</a>'


def _channel_link(row: Mapping[str, object]) -> str | None:
    source_username = collapse_ws(str(row.get("source_username") or "")).lstrip("@")
    if source_username:
        return f"https://t.me/{source_username}"
    channel_url = collapse_ws(str(row.get("channel_url") or ""))
    if channel_url.startswith("https://t.me/"):
        parts = channel_url.split("/")
        if len(parts) >= 4 and parts[3]:
            return "/".join(parts[:4])
    return None


def _channel_label(row: Mapping[str, object]) -> str:
    source_title = collapse_ws(str(row.get("source_title") or ""))
    if source_title:
        return source_title
    guide_names = row.get("guide_names") or []
    if isinstance(guide_names, list):
        joined = collapse_ws(", ".join(str(x) for x in guide_names[:3] if collapse_ws(str(x))))
        if joined:
            return joined
    return "Канал"


def format_occurrence_card(row: Mapping[str, object], *, index: int) -> str:
    likes_mark = collapse_ws(str(row.get("popularity_mark") or ""))
    title = collapse_ws(str(row.get("canonical_title") or "Экскурсия"))
    source_post_url = collapse_ws(str(row.get("channel_url") or ""))
    header_title = _html_link(title, source_post_url) if source_post_url else _html_text(title)
    header = f"{index}. {likes_mark} {header_title}".replace("  ", " ").strip()
    lines = [header]
    channel_line = _html_link(_channel_label(row), _channel_link(row))
    if channel_line:
        lines.append(channel_line)
    date_line = format_date_time(str(row.get("date") or ""), str(row.get("time") or ""))
    if date_line:
        lines.append(f"🗓 {_html_text(date_line)}")
    audience = row.get("audience_fit") or []
    if isinstance(audience, list):
        audience_text = audience_line([str(x) for x in audience])
        if audience_text:
            lines.append(f"👥 Для кого: {_html_text(audience_text)}")
    blurb = collapse_ws(str(row.get("digest_blurb") or row.get("summary_one_liner") or ""))
    if blurb:
        lines.append(f"🧭 {_html_text(blurb)}")
    meeting_point = collapse_ws(str(row.get("meeting_point") or ""))
    if meeting_point:
        lines.append(f"📍 {_html_text(meeting_point)}")
    price_text = collapse_ws(str(row.get("price_text") or ""))
    if price_text:
        lines.append(f"💸 {_html_text(price_text)}")
    seats_text = collapse_ws(str(row.get("seats_text") or ""))
    if seats_text:
        lines.append(f"🎟 {_html_text(seats_text)}")
    booking_text = collapse_ws(str(row.get("booking_text") or ""))
    booking_url = collapse_ws(str(row.get("booking_url") or ""))
    if booking_text and booking_url:
        lines.append(f"✍️ Запись: {_html_link(booking_text, booking_url)}")
    elif booking_url:
        lines.append(f'✍️ Запись: <a href="{html.escape(booking_url, quote=True)}">ссылка</a>')
    elif booking_text:
        lines.append(f"✍️ Запись: {_html_text(booking_text)}")
    return "\n".join(lines)


def build_digest_messages(
    rows: Iterable[Mapping[str, object]],
    *,
    family: str,
) -> list[str]:
    items = list(rows)
    if not items:
        title = "Новых экскурсионных находок пока нет." if family == "new_occurrences" else "Сигналов last call пока нет."
        return [title]

    family_title = (
        f"Новые экскурсии гидов: {len(items)} находки на ближайшие дни"
        if family == "new_occurrences"
        else f"Last call по экскурсиям: {len(items)} сигналов"
    )
    texts: list[str] = []
    current = [family_title, ""]
    current_len = len(family_title) + 2
    count_in_current = 0
    for idx, row in enumerate(items, start=1):
        card = format_occurrence_card(row, index=idx)
        candidate_len = current_len + len(card) + 2
        if current and count_in_current >= MAX_DIGEST_ITEMS_PER_MESSAGE:
            texts.append("\n".join(current).strip())
            current = [f"Продолжение дайджеста экскурсий ({len(texts)+1}/?)", ""]
            current_len = sum(len(x) + 1 for x in current)
            count_in_current = 0
        if candidate_len > MAX_TEXT_LEN and count_in_current > 0:
            texts.append("\n".join(current).strip())
            current = [f"Продолжение дайджеста экскурсий ({len(texts)+1}/?)", "", card]
            current_len = sum(len(x) + 1 for x in current)
            count_in_current = 1
            continue
        current.append(card)
        current.append("")
        current_len = candidate_len
        count_in_current += 1
    if current:
        texts.append("\n".join(current).strip())
    total = len(texts)
    if total > 1:
        fixed: list[str] = []
        for idx, text in enumerate(texts, start=1):
            if idx == 1:
                fixed.append(text)
                continue
            lines = text.splitlines()
            if lines:
                lines[0] = f"Продолжение дайджеста экскурсий ({idx}/{total})"
            fixed.append("\n".join(lines).strip())
        texts = fixed
    return texts


def build_media_caption(*, family: str, item_count: int, media_count: int) -> str:
    family_title = "Новые экскурсии гидов" if family == "new_occurrences" else "Last call по экскурсиям"
    return f"{family_title}\nВ альбоме карточки 1-{min(item_count, media_count)}."
