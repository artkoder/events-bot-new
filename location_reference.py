from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


_LOCATION_NOISE_PREFIXES_RE = re.compile(
    r"^(?:"
    r"кинотеатр|"
    r"арт[- ]?пространство|"
    r"пространство|"
    r"арт[- ]?площадка|"
    r"культурн(?:ый|ое) центр|"
    r"центр|"
    r"площадка|"
    r"клуб"
    r")\s+",
    re.IGNORECASE,
)

_ADDRESS_ABBR_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(?:проспект|пр(?:\s*|-)?(?:кт|т)?|пр\.)\b"), "пр"),
    (re.compile(r"(?i)\b(?:улица|ул\.?)\b"), "ул"),
    (re.compile(r"(?i)\b(?:площадь|пл\.?)\b"), "пл"),
    (re.compile(r"(?i)\b(?:набережная|наб\.?)\b"), "наб"),
    (re.compile(r"(?i)\b(?:бульвар|бул\.?)\b"), "бульвар"),
    (re.compile(r"(?i)\b(?:переулок|пер\.?)\b"), "пер"),
)

_ADDRESS_NOISE_RE = re.compile(
    r"(?iu)\b(?:ул(?:ица)?|пр(?:оспект|осп)?|пр-?т|пр|пер(?:еулок)?|пер|б-р|бульвар|пл(?:ощадь)?|пл|наб(?:ережная)?|наб|д(?:ом)?)\b"
)


def normalize_venue_key(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = (
        text.replace("\u00ab", " ")
        .replace("\u00bb", " ")
        .replace("\u201c", " ")
        .replace("\u201d", " ")
        .replace("\u201e", " ")
        .replace("\u2019", " ")
        .replace('"', " ")
        .replace("'", " ")
        .replace("`", " ")
    )
    text = _LOCATION_NOISE_PREFIXES_RE.sub("", text).strip()
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address_key(value: str | None, *, city: str | None = None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    text = raw.casefold().replace("ё", "е")
    text = re.sub(r"[«»\"'`]", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    for patt, repl in _ADDRESS_ABBR_REPLACEMENTS:
        text = patt.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _ADDRESS_NOISE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    city_key = normalize_venue_key(city)
    if city_key:
        text = re.sub(rf"(?i)\b{re.escape(city_key)}\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?i)\b(?:г|город)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class KnownVenue:
    canonical_line: str
    name: str
    address: str
    city: str
    name_key: str
    line_key: str


@lru_cache(maxsize=1)
def read_known_venues_lines() -> tuple[str, ...]:
    loc_path = os.path.join("docs", "reference", "locations.md")
    if not os.path.exists(loc_path):
        return ()
    try:
        with open(loc_path, "r", encoding="utf-8") as f:
            locations = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except Exception:
        return ()
    return tuple(locations)


@lru_cache(maxsize=1)
def read_known_venues() -> tuple[KnownVenue, ...]:
    venues: list[KnownVenue] = []
    for line in read_known_venues_lines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        name = parts[0]
        city = parts[-1] if len(parts) >= 2 else ""
        address = ", ".join(parts[1:-1]).strip() if len(parts) >= 3 else ""
        city_clean = city.lstrip("#").strip()
        venues.append(
            KnownVenue(
                canonical_line=line,
                name=name,
                address=address,
                city=city_clean,
                name_key=normalize_venue_key(name),
                line_key=normalize_venue_key(line),
            )
        )
    return tuple(venues)


@lru_cache(maxsize=1)
def read_known_venue_alias_lines() -> tuple[str, ...]:
    alias_path = os.path.join("docs", "reference", "location-aliases.md")
    if not os.path.exists(alias_path):
        return ()
    try:
        with open(alias_path, "r", encoding="utf-8") as f:
            aliases = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except Exception:
        return ()
    return tuple(aliases)


@lru_cache(maxsize=1)
def read_known_venue_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for line in read_known_venue_alias_lines():
        if "=>" not in line:
            continue
        alias_raw, canonical_raw = line.split("=>", 1)
        alias_key = normalize_venue_key(alias_raw)
        canonical_key = normalize_venue_key(canonical_raw)
        if alias_key and canonical_key:
            aliases[alias_key] = canonical_key
    return aliases


def _filter_venues_by_city(
    venues: tuple[KnownVenue, ...],
    city: str | None,
) -> tuple[KnownVenue, ...]:
    city_key = normalize_venue_key(city)
    if not city_key:
        return venues
    by_city = [v for v in venues if normalize_venue_key(v.city) == city_key]
    if by_city:
        return tuple(by_city)
    return venues


def _match_known_venue_alias(
    key: str,
    venues: tuple[KnownVenue, ...],
) -> KnownVenue | None:
    canonical_key = read_known_venue_aliases().get(key)
    if not canonical_key:
        return None
    for venue in venues:
        if canonical_key in {venue.line_key, venue.name_key}:
            return venue
    return None


def match_known_venue_by_address(
    address: str | None, *, city: str | None = None
) -> KnownVenue | None:
    addr_key = normalize_address_key(address, city=city)
    if not addr_key:
        return None
    venues = read_known_venues()
    if not venues:
        return None

    venues = _filter_venues_by_city(venues, city)

    exact: list[KnownVenue] = []
    for venue in venues:
        if not venue.address:
            continue
        venue_key = normalize_address_key(venue.address, city=venue.city or city)
        if venue_key and venue_key == addr_key:
            exact.append(venue)
    if len(exact) == 1:
        return exact[0]

    fuzzy: list[KnownVenue] = []
    for venue in venues:
        if not venue.address:
            continue
        venue_key = normalize_address_key(venue.address, city=venue.city or city)
        if not venue_key:
            continue
        if addr_key in venue_key or venue_key in addr_key:
            fuzzy.append(venue)
    if len(fuzzy) == 1:
        return fuzzy[0]
    return None


def match_known_venue(value: str | None, *, city: str | None = None) -> KnownVenue | None:
    key = normalize_venue_key(value)
    if not key:
        return None
    venues = read_known_venues()
    if not venues:
        return None

    venues = _filter_venues_by_city(venues, city)

    alias_match = _match_known_venue_alias(key, venues)
    if alias_match is not None:
        return alias_match

    for venue in venues:
        if key == venue.line_key or key == venue.name_key:
            return venue

    matches = [
        venue
        for venue in venues
        if venue.name_key and (key == venue.name_key or key in venue.name_key or venue.name_key in key)
    ]
    if len(matches) == 1:
        return matches[0]

    try:
        from difflib import SequenceMatcher
    except Exception:
        return None

    scored: list[tuple[float, KnownVenue]] = []
    for venue in venues:
        if not venue.name_key:
            continue
        scored.append((SequenceMatcher(None, key, venue.name_key).ratio(), venue))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    best_score, best_venue = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.92 and (best_score - second) >= 0.05:
        return best_venue

    stop = {
        "г",
        "город",
        "им",
        "имени",
        "ул",
        "улица",
        "проспект",
        "пр",
        "пл",
        "дом",
        "д",
        "к",
        "корп",
        "офис",
        "зал",
        "сцена",
        "театр",
        "музей",
        "бар",
        "клуб",
        "центр",
        "пространство",
        "школа",
        "библиотека",
        "галерея",
        "арена",
        "дворец",
        "резиденция",
        "музыкальная",
        "областная",
        "городская",
        "детская",
        "молодежный",
        "молодежныйи",
        "молодежныи",
        "молодежный",
    }

    def _tokens(s: str) -> set[str]:
        parts = re.findall(r"[a-zа-яё0-9]{4,}", s, flags=re.IGNORECASE)
        out = {p.casefold().replace("ё", "е") for p in parts if p}
        return {t for t in out if t not in stop and len(t) >= 4}

    key_tokens = _tokens(key)
    if not key_tokens:
        return None

    from collections import Counter

    freq: Counter[str] = Counter()
    venue_tokens: list[tuple[KnownVenue, set[str]]] = []
    for venue in venues:
        venue_token_set = _tokens(venue.name_key)
        venue_tokens.append((venue, venue_token_set))
        for token in venue_token_set:
            freq[token] += 1

    best: tuple[int, KnownVenue, set[str]] | None = None
    second_score = 0
    for venue, venue_token_set in venue_tokens:
        score = len(key_tokens & venue_token_set)
        if best is None or score > best[0]:
            if best is not None:
                second_score = max(second_score, best[0])
            best = (score, venue, venue_token_set)
        elif score > second_score:
            second_score = score

    if not best or best[0] <= 0:
        return None

    overlap = key_tokens & best[2]
    if best[0] >= 2 and (best[0] - second_score) >= 1:
        return best[1]
    if best[0] == 1 and (best[0] - second_score) >= 1:
        only = next(iter(overlap)) if overlap else ""
        if only and freq.get(only, 0) == 1:
            return best[1]
    return None


def normalise_event_location_from_reference(
    event_obj: dict[str, Any],
) -> KnownVenue | None:
    if not isinstance(event_obj, dict):
        return None
    raw_city = event_obj.get("city")
    raw_location_name = event_obj.get("location_name")
    raw_location_address = event_obj.get("location_address")

    venue_by_name = match_known_venue(raw_location_name, city=raw_city)
    venue_by_addr = match_known_venue_by_address(raw_location_address, city=raw_city)

    venue = venue_by_name
    addr_raw = str(raw_location_address or "").strip()
    addr_conflicts_with_name_match = False
    if venue_by_name is not None and addr_raw:
        raw_addr_key = normalize_address_key(addr_raw, city=raw_city)
        venue_addr_key = normalize_address_key(
            venue_by_name.address,
            city=venue_by_name.city or raw_city,
        )
        if raw_addr_key and venue_addr_key and raw_addr_key != venue_addr_key:
            addr_conflicts_with_name_match = True

    if venue_by_addr is not None and venue_by_addr != venue_by_name:
        venue = venue_by_addr
    elif addr_conflicts_with_name_match:
        return None

    if venue is None:
        return None

    event_obj["location_name"] = venue.canonical_line
    if venue.address:
        if (not addr_raw) or (
            normalize_address_key(addr_raw, city=raw_city)
            == normalize_address_key(venue.address, city=venue.city or raw_city)
        ):
            event_obj["location_address"] = venue.address
    if venue.city:
        # Once a source is matched to a single canonical venue from the reference list,
        # that venue's city is the authoritative value for the event.
        event_obj["city"] = venue.city
    return venue
