from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

MAX_YAML_SIZE = 256 * 1024
SCHEMA_VERSION = 2


class FestivalActivitiesError(ValueError):
    """Raised when the activities YAML payload cannot be parsed."""


@dataclass(slots=True)
class NormalizedLocation:
    name: str | None = None
    address: str | None = None
    city: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, str]:
        data: dict[str, str] = {}
        if self.name:
            data["name"] = self.name
        if self.address:
            data["address"] = self.address
        if self.city:
            data["city"] = self.city
        if self.note:
            data["note"] = self.note
        return data


@dataclass(slots=True)
class FestivalActivity:
    kind: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    time: str | None = None
    price: str | None = None
    age: str | None = None
    tags: tuple[str, ...] = ()
    note: str | None = None
    cta_label: str | None = None
    cta_url: str | None = None
    location: NormalizedLocation = field(default_factory=NormalizedLocation)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"title": self.title}
        if self.subtitle:
            payload["subtitle"] = self.subtitle
        if self.description:
            payload["description"] = self.description
        if self.time:
            payload["time"] = self.time
        if self.price:
            payload["price"] = self.price
        if self.age:
            payload["age"] = self.age
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.note:
            payload["note"] = self.note
        if self.cta_label or self.cta_url:
            payload["cta"] = {
                "label": self.cta_label,
                "url": self.cta_url,
            }
        loc_dict = self.location.as_dict()
        if loc_dict:
            payload["location"] = loc_dict
        payload["kind"] = self.kind
        return payload


@dataclass(slots=True)
class FestivalActivityGroup:
    kind: str
    title: str
    items: list[FestivalActivity]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "items": [item.to_json() for item in self.items],
        }


@dataclass(slots=True)
class FestivalActivitiesParseResult:
    groups: list[FestivalActivityGroup]
    warnings: list[str] = field(default_factory=list)
    festival_site: str | None = None

    def to_json_payload(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = [
            {"kind": "meta", "version": SCHEMA_VERSION, "festival_site": self.festival_site}
        ]
        for group in self.groups:
            data.append(group.to_json())
        return data


def parse_festival_activities_yaml(text: str) -> FestivalActivitiesParseResult:
    """Parse festival activities YAML (schema v2) with validation and warnings."""

    payload = text.encode("utf-8")
    if len(payload) > MAX_YAML_SIZE:
        raise FestivalActivitiesError("Activities YAML is larger than 256 KB")

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - PyYAML error text is unstable
        raise FestivalActivitiesError("Invalid YAML: unable to parse") from exc

    if not isinstance(data, Mapping):
        raise FestivalActivitiesError("Top level YAML structure must be a mapping")

    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise FestivalActivitiesError("Activities YAML must declare version: 2")

    warnings: list[str] = []
    site = data.get("festival_site")
    if site is not None and not _is_url(site):
        warnings.append("festival_site should be a valid URL")
        logging.warning("festival_activities invalid_site url=%r", site)

    groups: list[FestivalActivityGroup] = []
    for kind, title in (
        ("always_on", "Можно в любой день"),
        ("by_request", "По запросу / по записи"),
    ):
        raw_items = data.get(kind) or []
        if raw_items is None:
            raw_items = []
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            raise FestivalActivitiesError(f"Section '{kind}' must be a list")
        items: list[FestivalActivity] = []
        for idx, raw in enumerate(raw_items):
            ctx = f"{kind}[{idx}]"
            if not isinstance(raw, Mapping):
                raise FestivalActivitiesError(f"Activity {ctx} must be a mapping")
            try:
                items.append(_parse_activity(kind, raw, ctx, warnings))
            except FestivalActivitiesError:
                raise
        if items:
            groups.append(FestivalActivityGroup(kind=kind, title=title, items=items))

    return FestivalActivitiesParseResult(groups=groups, warnings=warnings, festival_site=site)


def _parse_activity(
    kind: str,
    raw: Mapping[str, Any],
    ctx: str,
    warnings: list[str],
) -> FestivalActivity:
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise FestivalActivitiesError(f"Activity {ctx} is missing title")

    subtitle = _optional_str(raw, "subtitle", ctx)
    description = _optional_str(raw, "description", ctx)
    time = _optional_str(raw, "time", ctx)
    price = _optional_str(raw, "price", ctx)
    age = _optional_str(raw, "age", ctx)
    note = _optional_str(raw, "note", ctx)

    tags_value = raw.get("tags", [])
    tags: tuple[str, ...]
    if tags_value in (None, ""):
        tags = ()
    elif isinstance(tags_value, Sequence) and not isinstance(tags_value, (str, bytes)):
        tags_list: list[str] = []
        for pos, tag in enumerate(tags_value):
            if not isinstance(tag, str) or not tag.strip():
                warnings.append(f"{ctx}: tag #{pos+1} is empty")
                logging.warning("festival_activities empty_tag ctx=%s", ctx)
                continue
            tags_list.append(tag.strip())
        tags = tuple(tags_list)
    else:
        raise FestivalActivitiesError(f"{ctx}: tags must be a list of strings")

    cta_label: str | None = None
    cta_url: str | None = None
    cta_raw = raw.get("cta")
    if cta_raw is not None:
        if not isinstance(cta_raw, Mapping):
            raise FestivalActivitiesError(f"{ctx}: cta must be a mapping")
        cta_label = _optional_str(cta_raw, "label", ctx)
        cta_url = _optional_str(cta_raw, "url", ctx)
        if cta_url and not _is_url(cta_url):
            warnings.append(f"{ctx}: CTA URL looks invalid")
            logging.warning("festival_activities invalid_cta_url ctx=%s url=%r", ctx, cta_url)
        if cta_url and not cta_label:
            cta_label = "Подробнее"
        if cta_label and not cta_url:
            warnings.append(f"{ctx}: CTA label provided without URL")
            logging.warning("festival_activities missing_cta_url ctx=%s", ctx)

    location = _parse_location(raw.get("location"), ctx, warnings)

    return FestivalActivity(
        kind=kind,
        title=title.strip(),
        subtitle=subtitle,
        description=description,
        time=time,
        price=price,
        age=age,
        tags=tags,
        note=note,
        cta_label=cta_label,
        cta_url=cta_url,
        location=location,
    )


def _optional_str(data: Mapping[str, Any], key: str, ctx: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)) and key == "time":
        seconds = int(value)
        minutes = seconds % 60
        hours = seconds // 60
        value = f"{hours:02d}:{minutes:02d}"
    elif not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _parse_location(value: Any, ctx: str, warnings: list[str]) -> NormalizedLocation:
    if value is None:
        return NormalizedLocation()
    if isinstance(value, str):
        raw_name = value.strip()
        return _normalize_location({"name": raw_name}, ctx, warnings)
    if not isinstance(value, Mapping):
        raise FestivalActivitiesError(f"{ctx}: location must be a mapping or string")
    raw_name = _optional_str(value, "name", ctx)
    address = _optional_str(value, "address", ctx)
    city = _optional_str(value, "city", ctx)
    note = _optional_str(value, "note", ctx)
    return _normalize_location(
        {"name": raw_name, "address": address, "city": city, "note": note},
        ctx,
        warnings,
    )


def _normalize_location(
    data: Mapping[str, str | None], ctx: str, warnings: list[str]
) -> NormalizedLocation:
    name = (data.get("name") or "").strip()
    address = (data.get("address") or "").strip() or None
    city = (data.get("city") or "").strip() or None
    note = (data.get("note") or "").strip() or None

    if not name and not address and not city:
        return NormalizedLocation(note=note)

    canonical = _standard_locations().get(name.casefold())
    if canonical:
        canon_name, canon_address, canon_city = canonical
        if address and address != canon_address:
            warnings.append(f"{ctx}: overriding canonical address for '{name}'")
            logging.warning(
                "festival_activities address_override ctx=%s name=%r addr=%r", ctx, name, address
            )
        if city and canon_city and city != canon_city:
            warnings.append(f"{ctx}: overriding canonical city for '{name}'")
            logging.warning(
                "festival_activities city_override ctx=%s name=%r city=%r", ctx, name, city
            )
        name = canon_name
        address = canon_address
        city = canon_city
    elif name:
        warnings.append(f"{ctx}: location '{name}' is not in docs/LOCATIONS.md")
        logging.warning("festival_activities unknown_location ctx=%s name=%r", ctx, name)

    return NormalizedLocation(name=name or None, address=address, city=city, note=note)


def _is_url(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("http://") or value.startswith("https://")


@lru_cache(maxsize=1)
def _standard_locations() -> dict[str, tuple[str, str | None, str | None]]:
    base = Path(__file__).resolve().parent
    doc_path = base / "docs" / "LOCATIONS.md"
    mapping: dict[str, tuple[str, str | None, str | None]] = {}
    if not doc_path.exists():  # pragma: no cover - repository invariant
        return mapping
    text = doc_path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        name = parts[0]
        address = parts[1] if len(parts) > 1 else None
        city = parts[2] if len(parts) > 2 else None
        mapping[name.casefold()] = (name, address, city)
    return mapping


def load_groups_from_json(data: Sequence[Mapping[str, Any]] | None) -> list[FestivalActivityGroup]:
    if not data:
        return []
    groups: list[FestivalActivityGroup] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        kind = entry.get("kind")
        if kind == "meta":
            continue
        if kind not in {"always_on", "by_request"}:
            continue
        title = entry.get("title") or ("Можно в любой день" if kind == "always_on" else "По запросу / по записи")
        raw_items = entry.get("items") or []
        items: list[FestivalActivity] = []
        if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
            for raw in raw_items:
                if not isinstance(raw, Mapping):
                    continue
                location = _normalize_location_for_json(raw.get("location"))
                tags = raw.get("tags")
                if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
                    tags_tuple = tuple(str(tag) for tag in tags)
                else:
                    tags_tuple = ()
                cta_raw = raw.get("cta") if isinstance(raw.get("cta"), Mapping) else {}
                cta_label = None
                cta_url = None
                if isinstance(cta_raw, Mapping):
                    lbl = cta_raw.get("label")
                    url = cta_raw.get("url")
                    cta_label = str(lbl) if isinstance(lbl, str) and lbl else None
                    cta_url = str(url) if isinstance(url, str) and url else None
                activity = FestivalActivity(
                    kind=kind,
                    title=str(raw.get("title") or "").strip() or "Без названия",
                    subtitle=_str_or_none(raw.get("subtitle")),
                    description=_str_or_none(raw.get("description")),
                    time=_str_or_none(raw.get("time")),
                    price=_str_or_none(raw.get("price")),
                    age=_str_or_none(raw.get("age")),
                    tags=tags_tuple,
                    note=_str_or_none(raw.get("note")),
                    cta_label=cta_label,
                    cta_url=cta_url,
                    location=location,
                )
                items.append(activity)
        groups.append(
            FestivalActivityGroup(kind=kind, title=str(title), items=items)
        )
    return groups


def _normalize_location_for_json(value: Any) -> NormalizedLocation:
    if isinstance(value, Mapping):
        return NormalizedLocation(
            name=_str_or_none(value.get("name")),
            address=_str_or_none(value.get("address")),
            city=_str_or_none(value.get("city")),
            note=_str_or_none(value.get("note")),
        )
    if isinstance(value, str):
        return NormalizedLocation(name=value)
    return NormalizedLocation()


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def build_activity_card_lines(activity: FestivalActivity) -> list[str]:
    lines: list[str] = []
    if activity.subtitle:
        lines.append(activity.subtitle)
    if activity.time:
        lines.append(f"🕒 {activity.time}")
    location_line = format_location_line(activity.location)
    if location_line:
        lines.append(f"📍 {location_line}")
    details: list[str] = []
    if activity.price:
        details.append(activity.price)
    if activity.age:
        details.append(activity.age)
    if activity.tags:
        details.append(", ".join(activity.tags))
    if details:
        lines.append(" • ".join(details))
    if activity.description:
        lines.append(activity.description)
    if activity.note:
        lines.append(activity.note)
    if activity.cta_url:
        label = activity.cta_label or "Подробнее"
        lines.append(f"➡️ {label} ({activity.cta_url})")
    return lines


def format_location_line(location: NormalizedLocation) -> str:
    parts = [p for p in (location.name, location.address) if p]
    if location.city:
        parts.append(location.city)
    return ", ".join(parts)


def activities_to_telegraph_nodes(groups: Iterable[FestivalActivityGroup]) -> list[dict]:
    from markup import telegraph_br

    nodes: list[dict] = []
    for group in groups:
        if not group.items:
            continue
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.append({"tag": "h3", "children": [group.title]})
        for activity in group.items:
            nodes.append({"tag": "h4", "children": [activity.title]})
            for line in build_activity_card_lines(activity):
                if line.startswith("➡️ ") and "(" in line and line.endswith(")"):
                    label, url = line[2:].strip().split(" (", 1)
                    url = url[:-1]
                    nodes.append(
                        {
                            "tag": "p",
                            "children": [
                                {
                                    "tag": "a",
                                    "attrs": {"href": url},
                                    "children": [label.strip()],
                                }
                            ],
                        }
                    )
                else:
                    nodes.append({"tag": "p", "children": [line]})
            nodes.extend(telegraph_br())
    return nodes


TEMPLATE_YAML = """# version must always be 2
version: 2
festival_site: https://example.com

# Activities available on any day of the festival
always_on:
  - title: Постоянная экспозиция
    subtitle: Каждый день фестиваля
    time: 10:00–20:00
    location:
      name: Музей Янтаря
    description: >-
      Короткое описание активности, одно-два предложения.
    price: Бесплатно
    cta:
      label: Подробнее
      url: https://example.com/expo

# Activities that require booking or personal request
by_request:
  - title: Закрытая экскурсия по музею
    note: Требуется предварительная запись
    location:
      name: Музей Янтаря
    cta:
      label: Оставить заявку
      url: https://example.com/request
"""

