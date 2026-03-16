from __future__ import annotations

import html
import re
from datetime import date, datetime
from typing import Iterable, Mapping

from .parser import audience_line, collapse_ws
from .place_aliases import normalize_public_place

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
MAX_MEDIA_ITEMS = 10
MAX_TEXT_LEN = 3900
CARD_SEPARATOR = "──────────"


def _meaningful_text(value: object | None) -> str | None:
    text = collapse_ws("" if value is None else str(value))
    if not text:
        return None
    if text.lower() in {
        "не определено",
        "не указано",
        "нет",
        "n/a",
        "none",
        "одна дата",
        "one date",
        "single date",
        "только одна дата",
    }:
        return None
    return text


def _looks_raw_price_copy(value: object | None) -> bool:
    text = _meaningful_text(value)
    if not text:
        return False
    if re.search(r"\d+\s*/\s*\d+", text):
        return True
    return any(token in text for token in (",пенсион", ", дети", "/с ", "₽/с"))


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


def _normalize_phone_link(raw: object | None) -> tuple[str, str] | None:
    text = _meaningful_text(raw)
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    if len(digits) != 11 or not digits.startswith("7"):
        return None
    display = f"+7 {digits[1:4]} {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
    return display, f"tel:+{digits}"


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


def _fact_pack_value(row: Mapping[str, object], key: str) -> object | None:
    fact_pack = row.get("fact_pack")
    if isinstance(fact_pack, Mapping):
        return fact_pack.get(key)
    return None


def _profile_fact_value(row: Mapping[str, object], key: str) -> object | None:
    fact_pack = row.get("guide_profile_facts")
    if isinstance(fact_pack, Mapping):
        return fact_pack.get(key)
    return None


def _guide_names_line(row: Mapping[str, object]) -> str | None:
    guide_names = row.get("guide_names") or _fact_pack_value(row, "guide_names") or []
    if not isinstance(guide_names, list):
        return None
    names = [collapse_ws(str(item)) for item in guide_names if collapse_ws(str(item))]
    names = [item for item in names if len(item.split()) >= 2]
    if not names:
        return None
    return ", ".join(names[:3])


def _looks_organizer_identity(value: str | None, row: Mapping[str, object]) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    source_kind = collapse_ws(str(row.get("source_kind") or ""))
    if source_kind not in {"organization_with_tours", "excursion_operator", "aggregator"}:
        return False
    source_title = collapse_ws(str(row.get("source_title") or ""))
    low = text.lower()
    source_low = source_title.lower()
    if source_low and (low == source_low or low.startswith(source_low)):
        return True
    return any(
        token in low
        for token in (
            "организац",
            "организатор",
            "туроператор",
            "турагент",
            "проект",
            "команда",
            "экскурс",
            "путешеств",
            "тур по ",
        )
    )


def _organizer_line(row: Mapping[str, object]) -> str | None:
    direct = _meaningful_text(row.get("organizer_line"))
    if direct:
        return direct
    source_kind = collapse_ws(str(row.get("source_kind") or ""))
    marketing_name = _meaningful_text(
        row.get("guide_profile_marketing_name") or _profile_fact_value(row, "marketing_name")
    )
    if marketing_name and source_kind in {"organization_with_tours", "excursion_operator", "aggregator"}:
        return marketing_name
    source_title = _meaningful_text(row.get("source_title"))
    if source_title and source_kind in {"organization_with_tours", "excursion_operator", "aggregator"}:
        return source_title
    organizer_names = row.get("organizer_names") or _fact_pack_value(row, "organizer_names") or []
    if isinstance(organizer_names, list):
        names = [collapse_ws(str(item)) for item in organizer_names if collapse_ws(str(item))]
        if names:
            return ", ".join(names[:3])
    return None


def _audience_region_line(row: Mapping[str, object]) -> str | None:
    direct = _meaningful_text(row.get("audience_region_line") or _fact_pack_value(row, "audience_region_line"))
    if direct:
        return direct
    return None


def _public_location_text(row: Mapping[str, object]) -> str | None:
    raw = _meaningful_text(row.get("city") or _fact_pack_value(row, "city"))
    if not raw:
        return None
    return normalize_public_place(raw)


def _trim_card_separator(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    if trimmed and trimmed[-1] == CARD_SEPARATOR:
        trimmed.pop()
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def format_occurrence_card(row: Mapping[str, object], *, index: int) -> str:
    likes_mark = collapse_ws(str(row.get("popularity_mark") or ""))
    llm_mark = _meaningful_text(row.get("lead_emoji") or _fact_pack_value(row, "lead_emoji"))
    mark = llm_mark or likes_mark
    title = collapse_ws(str(row.get("canonical_title") or "Экскурсия"))
    source_post_url = collapse_ws(str(row.get("source_post_url") or row.get("channel_url") or ""))
    header_title = _html_link(title, source_post_url) if source_post_url else _html_text(title)
    header = f"{index}. {mark} {header_title}".replace("  ", " ").strip()
    lines = [header]
    channel_line = _html_text(_channel_label(row))
    if channel_line:
        lines.append(channel_line)
    guide_names = row.get("guide_names") or []
    guide_count = len(guide_names) if isinstance(guide_names, list) else 0
    if guide_count >= 2:
        names_line = _guide_names_line(row)
        if names_line:
            lines.append(f"👥 Гиды: {_html_text(names_line)}")
    else:
        guide_line = _meaningful_text(row.get("guide_line") or _profile_fact_value(row, "guide_line"))
        if not guide_line:
            guide_line = _guide_names_line(row)
        if _looks_organizer_identity(guide_line, row):
            organizer_line = _organizer_line(row) or guide_line
            guide_line = None
        else:
            organizer_line = _organizer_line(row)
        if guide_line and collapse_ws(guide_line).lower() != collapse_ws(_channel_label(row)).lower():
            lines.append(f"👤 Гид: {_html_text(guide_line)}")
        else:
            if organizer_line and collapse_ws(organizer_line).lower() != collapse_ws(_channel_label(row)).lower():
                lines.append(f"🏢 Организатор: {_html_text(organizer_line)}")
    date_line = collapse_ws(str(row.get("schedule_line") or "")) or format_date_time(
        str(row.get("date") or ""),
        str(row.get("time") or ""),
    )
    if date_line:
        lines.append(f"🗓 {_html_text(date_line)}")
    city_text = _public_location_text(row)
    if city_text:
        lines.append(f"🏙 Локация: {_html_text(city_text)}")
    audience_region_text = _audience_region_line(row)
    if audience_region_text:
        lines.append(f"🏠 {_html_text(audience_region_text)}")
    audience_text = _meaningful_text(row.get("audience_line"))
    if not audience_text:
        audience = row.get("audience_fit") or []
        if isinstance(audience, list):
            audience_text = audience_line([str(x) for x in audience])
    if audience_text:
        lines.append(f"👥 Кому подойдёт: {_html_text(audience_text)}")
    group_format = _meaningful_text(row.get("group_format_line") or row.get("group_format") or _fact_pack_value(row, "group_format"))
    if group_format:
        lines.append(f"👥 Формат: {_html_text(group_format)}")
    blurb = _meaningful_text(row.get("digest_blurb") or row.get("summary_one_liner"))
    if blurb:
        lines.append(f"🧭 {_html_text(blurb)}")
    route_text = _meaningful_text(row.get("route_line") or row.get("route_summary") or _fact_pack_value(row, "route_summary"))
    if route_text:
        lines.append(f"🗺 Что в маршруте: {_html_text(route_text)}")
    duration_text = _meaningful_text(row.get("duration_line") or row.get("duration_text") or _fact_pack_value(row, "duration_text"))
    if duration_text:
        lines.append(f"⏱ Продолжительность: {_html_text(duration_text)}")
    meeting_point = _meaningful_text(row.get("meeting_point_line") or row.get("meeting_point"))
    if meeting_point:
        lines.append(f"📍 Место сбора: {_html_text(meeting_point)}")
    price_text = _meaningful_text(row.get("price_line"))
    if _looks_raw_price_copy(price_text):
        price_text = None
    if not price_text:
        raw_price = _meaningful_text(row.get("price_text"))
        if raw_price and not _looks_raw_price_copy(raw_price):
            price_text = raw_price
    if price_text:
        lines.append(f"💸 {_html_text(price_text)}")
    seats_text = _meaningful_text(row.get("seats_line") or row.get("seats_text"))
    if seats_text:
        lines.append(f"🎟 {_html_text(seats_text)}")
    booking_text = _meaningful_text(row.get("booking_line") or row.get("booking_text"))
    booking_url = _meaningful_text(row.get("booking_url"))
    if booking_text and not booking_url:
        phone_link = _normalize_phone_link(booking_text)
        if phone_link:
            booking_text, booking_url = phone_link
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
        trailer = ["", CARD_SEPARATOR, ""] if idx < len(items) else [""]
        candidate_len = current_len + len(card) + sum(len(part) + 1 for part in trailer)
        if current and count_in_current >= MAX_DIGEST_ITEMS_PER_MESSAGE:
            texts.append("\n".join(_trim_card_separator(current)).strip())
            current = [f"Продолжение дайджеста экскурсий ({len(texts)+1}/?)", ""]
            current_len = sum(len(x) + 1 for x in current)
            count_in_current = 0
        if candidate_len > MAX_TEXT_LEN and count_in_current > 0:
            texts.append("\n".join(_trim_card_separator(current)).strip())
            current = [f"Продолжение дайджеста экскурсий ({len(texts)+1}/?)", "", card]
            if idx < len(items):
                current.extend(["", CARD_SEPARATOR, ""])
            else:
                current.append("")
            current_len = sum(len(x) + 1 for x in current)
            count_in_current = 1
            continue
        current.append(card)
        current.extend(trailer)
        current_len = candidate_len
        count_in_current += 1
    if current:
        texts.append("\n".join(_trim_card_separator(current)).strip())
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
