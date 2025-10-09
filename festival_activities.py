"""Festival activities parsing and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import functools
import re
from pathlib import Path
from typing import Any, Iterable, Sequence
import textwrap

import yaml

from markup import telegraph_br

MONTHS = [
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


def _format_day_pretty(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]}"


class FestivalActivitiesError(ValueError):
    """Raised when the YAML payload cannot be parsed."""


@dataclass(slots=True)
class FestivalActivitiesResult:
    """Normalised payload returned by :func:`parse_festival_activities_yaml`."""

    activities: list[dict[str, Any]]
    website_url: str | None = None


_DEFAULT_LOCATIONS_PATH = Path(__file__).resolve().parent / "docs" / "LOCATIONS.md"


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


@functools.lru_cache(maxsize=1)
def load_canonical_venues(path: Path | None = None) -> dict[str, dict[str, str | None]]:
    """Return mapping of canonical venue names to their address/city."""

    actual = path or _DEFAULT_LOCATIONS_PATH
    mapping: dict[str, dict[str, str | None]] = {}
    if not actual.exists():
        return mapping
    text = actual.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        address = parts[1] if len(parts) >= 3 else None
        city = parts[2] if len(parts) >= 3 else parts[1]
        if city:
            city = city.lstrip("#") or None
        if address and city and len(parts) > 3:
            # Support lines with extra commas in the address.
            address = ", ".join(parts[1:-1])
            city = parts[-1].lstrip("#") or None
        mapping[_normalize_key(name)] = {
            "name": name,
            "address": address,
            "city": city,
        }
    return mapping


def _ensure_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FestivalActivitiesError(f"Field '{field}' must be a non-empty string")
    return value.strip()


_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _normalize_time(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise FestivalActivitiesError(f"Field '{field}' must be a string")
    value = value.strip()
    m = _TIME_RE.fullmatch(value)
    if not m:
        raise FestivalActivitiesError(f"Field '{field}' must be in HH:MM format")
    hours = int(m.group(1))
    minutes = int(m.group(2))
    if hours > 23 or minutes > 59:
        raise FestivalActivitiesError(f"Field '{field}' must be a valid time")
    return f"{hours:02d}:{minutes:02d}"


def _normalize_date(value: Any, *, field: str) -> str:
    if not isinstance(value, (str, date)):
        raise FestivalActivitiesError(f"Field '{field}' must be a date string")
    if isinstance(value, date):
        return value.isoformat()
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:  # pragma: no cover - defensive
        raise FestivalActivitiesError(f"Field '{field}' must be YYYY-MM-DD") from exc


def normalize_venues(
    venues: Iterable[Any],
    *,
    default_city: str | None = None,
    canonical: dict[str, dict[str, str | None]] | None = None,
) -> list[dict[str, str | None]]:
    """Return venues normalized against the canonical list."""

    canonical = canonical or load_canonical_venues()
    cleaned: list[dict[str, str | None]] = []
    for raw in venues:
        if isinstance(raw, str):
            name = raw.strip()
            address = None
            city = None
        elif isinstance(raw, dict):
            name = _ensure_str(raw.get("name"), field="venues[].name")
            address = (raw.get("address") or "").strip() or None
            city = (raw.get("city") or "").strip() or None
        else:
            raise FestivalActivitiesError("Each venue must be a string or dict")

        if not name:
            raise FestivalActivitiesError("Venue name cannot be empty")

        canonical_item = canonical.get(_normalize_key(name))
        if canonical_item:
            name = canonical_item["name"] or name
            address = address or canonical_item.get("address")
            city = city or canonical_item.get("city")
        city = city or default_city
        cleaned.append({"name": name, "address": address, "city": city})

    if not cleaned:
        raise FestivalActivitiesError("At least one venue is required")
    return cleaned


def _normalize_schedule(raw_slots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for idx, slot in enumerate(raw_slots):
        if not isinstance(slot, dict):
            raise FestivalActivitiesError("Schedule entries must be mappings")
        date_text = _normalize_date(slot.get("date"), field=f"schedule[{idx}].date")
        start_time = _normalize_time(
            slot.get("start") or slot.get("time"), field=f"schedule[{idx}].start"
        )
        end_time = _normalize_time(slot.get("end"), field=f"schedule[{idx}].end")
        label = (slot.get("label") or "").strip() or None
        notes = (slot.get("notes") or "").strip() or None
        price = (slot.get("price") or "").strip() or None
        cta = slot.get("cta") or {}
        if cta:
            if not isinstance(cta, dict):
                raise FestivalActivitiesError("Schedule CTA must be a mapping")
            cta_text = (cta.get("text") or "").strip() or None
            cta_url = (cta.get("url") or "").strip() or None
        else:
            cta_text = None
            cta_url = None
        slots.append(
            {
                "date": date_text,
                "start_time": start_time or None,
                "end_time": end_time or None,
                "label": label,
                "notes": notes,
                "price": price,
                "cta_text": cta_text,
                "cta_url": cta_url,
            }
        )
    return slots


def parse_festival_activities_yaml(text: str) -> FestivalActivitiesResult:
    """Parse YAML describing festival activities (schema version 2)."""

    try:
        stripped = textwrap.dedent(text).strip()
        payload = yaml.safe_load(stripped) if stripped else {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise FestivalActivitiesError("Invalid YAML") from exc

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise FestivalActivitiesError("Top-level YAML must be a mapping")

    version = payload.get("version", 2)
    if version != 2:
        raise FestivalActivitiesError("Only schema version 2 is supported")

    defaults = payload.get("defaults") or {}
    if defaults and not isinstance(defaults, dict):
        raise FestivalActivitiesError("'defaults' must be a mapping")
    default_city = (defaults.get("city") or "").strip() or None

    festival_info = payload.get("festival") or {}
    if festival_info and not isinstance(festival_info, dict):
        raise FestivalActivitiesError("'festival' must be a mapping")
    website_url = (festival_info.get("website") or "").strip() or None

    raw_activities = payload.get("activities")
    if raw_activities is None:
        return FestivalActivitiesResult(activities=[], website_url=website_url)
    if not isinstance(raw_activities, list):
        raise FestivalActivitiesError("'activities' must be a list")

    canonical = load_canonical_venues()
    normalised: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_activities):
        if not isinstance(item, dict):
            raise FestivalActivitiesError("Each activity must be a mapping")
        title = _ensure_str(item.get("title"), field=f"activities[{idx}].title")
        kind = _ensure_str(item.get("kind") or defaults.get("kind"), field=f"activities[{idx}].kind")
        summary = (item.get("summary") or "").strip() or None
        description = (item.get("description") or "").strip() or None
        anytime = bool(item.get("anytime"))
        on_request = bool(item.get("on_request"))
        tags = [str(tag).strip() for tag in item.get("tags", []) if str(tag).strip()]

        venue_value = item.get("venues")
        if venue_value is None:
            single = item.get("venue")
            if single is None:
                raise FestivalActivitiesError("Activity must provide venues")
            venue_value = [single]
        elif isinstance(venue_value, (str, dict)):
            venue_value = [venue_value]
        if not isinstance(venue_value, Iterable):
            raise FestivalActivitiesError("'venues' must be iterable")

        venues = normalize_venues(
            venue_value,
            default_city=default_city or (item.get("city") or "").strip() or None,
            canonical=canonical,
        )

        schedule_raw = item.get("schedule") or item.get("dates") or []
        if isinstance(schedule_raw, dict):
            schedule_raw = [schedule_raw]
        if schedule_raw and not isinstance(schedule_raw, list):
            raise FestivalActivitiesError("'schedule' must be a list or mapping")
        schedule = _normalize_schedule(schedule_raw)

        global_cta = item.get("cta") or {}
        if global_cta:
            if not isinstance(global_cta, dict):
                raise FestivalActivitiesError("Activity CTA must be a mapping")
            cta_text = (global_cta.get("text") or "").strip() or None
            cta_url = (global_cta.get("url") or "").strip() or None
        else:
            cta_text = None
            cta_url = None

        normalised.append(
            {
                "title": title,
                "kind": kind,
                "summary": summary,
                "description": description,
                "anytime": anytime,
                "on_request": on_request,
                "tags": tags,
                "venues": venues,
                "schedule": schedule,
                "cta_text": cta_text,
                "cta_url": cta_url,
            }
        )

    return FestivalActivitiesResult(activities=normalised, website_url=website_url)


def activity_primary_city(activity: dict[str, Any]) -> str | None:
    for venue in activity.get("venues", []):
        city = venue.get("city")
        if city:
            return city
    return None


def group_activities_by_city(
    activities: Sequence[dict[str, Any]]
) -> list[tuple[str | None, list[dict[str, Any]]]]:
    groups: list[tuple[str | None, list[dict[str, Any]]]] = []
    index: dict[str | None, list[dict[str, Any]]] = {}
    for activity in activities:
        city = activity_primary_city(activity)
        bucket = index.setdefault(city, [])
        bucket.append(activity)
    for city, items in index.items():
        groups.append((city, items))
    return groups


def format_activity_slot(slot: dict[str, Any]) -> str:
    slot_date = datetime.strptime(slot["date"], "%Y-%m-%d").date()
    parts: list[str] = [_format_day_pretty(slot_date)]
    time_parts: list[str] = []
    if slot.get("start_time") and slot.get("end_time"):
        time_parts.append(f"{slot['start_time']}–{slot['end_time']}")
    elif slot.get("start_time"):
        time_parts.append(slot["start_time"])
    if time_parts:
        parts.append(" ".join(time_parts))
    if slot.get("label"):
        parts.append(slot["label"])
    line = ", ".join(parts)
    extras: list[str] = []
    if slot.get("price"):
        extras.append(slot["price"])
    if slot.get("notes"):
        extras.append(slot["notes"])
    if extras:
        line = f"{line} · {' · '.join(extras)}"
    return line


def _render_activity(activity: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    nodes.append({"tag": "p", "children": [{"tag": "strong", "children": [activity["title"]]}]})
    if activity.get("summary"):
        nodes.append({"tag": "p", "children": [activity["summary"]]})
    if activity.get("description"):
        nodes.append({"tag": "p", "children": [activity["description"]]})
    for venue in activity.get("venues", []):
        parts = [venue.get("name") or ""]
        if venue.get("address"):
            parts.append(venue["address"])
        if venue.get("city"):
            parts.append(venue["city"])
        nodes.append({"tag": "p", "children": [", ".join(parts)]})
    schedule = activity.get("schedule") or []
    if schedule:
        items: list[dict[str, Any]] = []
        for slot in schedule:
            line_children: list[Any] = [format_activity_slot(slot)]
            if slot.get("cta_url"):
                link_text = slot.get("cta_text") or slot["cta_url"]
                line_children.extend(
                    [
                        " — ",
                        {"tag": "a", "attrs": {"href": slot["cta_url"]}, "children": [link_text]},
                    ]
                )
            items.append({"tag": "li", "children": line_children})
        nodes.append({"tag": "ul", "children": items})
    if activity.get("cta_url"):
        link_text = activity.get("cta_text") or activity["cta_url"]
        nodes.append(
            {
                "tag": "p",
                "children": [
                    {"tag": "a", "attrs": {"href": activity["cta_url"]}, "children": [link_text]}
                ],
            }
        )
    nodes.extend(telegraph_br())
    return nodes


def _render_city_group(city: str | None, activities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if city:
        nodes.append({"tag": "h4", "children": [city]})
    for activity in activities:
        nodes.extend(_render_activity(activity))
    return nodes


def activities_to_nodes(activities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not activities:
        return []

    nodes: list[dict[str, Any]] = [{"tag": "h2", "children": ["АКТИВНОСТИ"]}]
    remaining: list[dict[str, Any]] = []

    anytime_items = [item for item in activities if item.get("anytime")]
    on_request_items = [
        item for item in activities if item.get("on_request") and not item.get("anytime")
    ]
    for item in activities:
        if item in anytime_items or item in on_request_items:
            continue
        remaining.append(item)

    if anytime_items:
        nodes.append({"tag": "h3", "children": ["Можно в любой день"]})
        for city, items in group_activities_by_city(anytime_items):
            nodes.extend(_render_city_group(city, items))

    if on_request_items:
        nodes.append({"tag": "h3", "children": ["По запросу / по записи"]})
        for city, items in group_activities_by_city(on_request_items):
            nodes.extend(_render_city_group(city, items))

    if remaining:
        kinds: dict[str, list[dict[str, Any]]] = {}
        for item in remaining:
            kinds.setdefault(item["kind"], []).append(item)
        for kind, items in kinds.items():
            nodes.append({"tag": "h3", "children": [kind]})
            for city, city_items in group_activities_by_city(items):
                nodes.extend(_render_city_group(city, city_items))

    return nodes


def format_activities_preview(activities: Sequence[dict[str, Any]]) -> str:
    if not activities:
        return "(нет активностей)"
    lines: list[str] = []
    for activity in activities:
        lines.append(f"- {activity['title']} [{activity['kind']}]")
        schedule = activity.get("schedule") or []
        for slot in schedule:
            lines.append(f"  • {format_activity_slot(slot)}")
        if not schedule:
            lines.append("  • без расписания")
    return "\n".join(lines)


async def save_festival_activities(
    session, festival, result: FestivalActivitiesResult
) -> None:  # pragma: no cover - exercised in tests
    festival.activities_json = result.activities
    if result.website_url:
        festival.website_url = result.website_url
    session.add(festival)
    await session.commit()
    await session.refresh(festival)
