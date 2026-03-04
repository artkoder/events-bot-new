from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher

from sqlmodel import select

from db import Database
from models import Event

logger = logging.getLogger(__name__)

_TITLE_CLEAN_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_TITLE_SPACE_RE = re.compile(r"\s+")
_TIME_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$")

# Noise tokens that often prefix otherwise-identical titles.
_TITLE_NOISE_WORDS: set[str] = {
    "спектакль",
    "концерт",
    "лекция",
    "мастер",
    "мастеркласс",
    "мастер-класс",
    "встреча",
    "показ",
    "премьера",
}


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _parse_iso_date(value: str | None) -> date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw.split("..", 1)[0].strip())
    except Exception:
        return None


def _normalize_title(text: str | None) -> str:
    if not text:
        return ""
    normalized = (
        str(text)
        .strip()
        .lower()
        .replace("ё", "е")
        .replace("\u00a0", " ")
    )
    normalized = _TITLE_CLEAN_RE.sub(" ", normalized)
    normalized = _TITLE_SPACE_RE.sub(" ", normalized).strip()
    return normalized


def _title_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in text.split() if len(t) >= 3]


def _strip_noise_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _TITLE_NOISE_WORDS]


def _titles_match_for_linking(a: str | None, b: str | None, *, threshold: float = 0.90) -> bool:
    """High-precision match for linking same-event occurrences across different dates.

    Important: This is intentionally stricter than Smart Update merge matching.
    Linking must avoid false positives because it is user-facing ("Другие даты").
    """
    na = _normalize_title(a)
    nb = _normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    toks_a = _strip_noise_tokens(_title_tokens(na))
    toks_b = _strip_noise_tokens(_title_tokens(nb))
    if toks_a and toks_b:
        if toks_a == toks_b:
            return True
        set_a = set(toks_a)
        set_b = set(toks_b)
        if set_a and set_b and (set_a.issubset(set_b) or set_b.issubset(set_a)):
            return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _time_sort_key(value: str | None) -> tuple[int, int, int]:
    raw = (value or "").strip()
    if not raw or raw == "00:00":
        return (1, 0, 0)  # unknown time goes last within the day
    m = _TIME_RE.match(raw)
    if not m:
        return (1, 0, 0)
    try:
        hh = int(m.group("h"))
        mm = int(m.group("m"))
    except Exception:
        return (1, 0, 0)
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return (1, 0, 0)
    return (0, hh, mm)


def _event_sort_key(date_value: str | None, time_value: str | None, event_id: int) -> tuple[date, tuple[int, int, int], int]:
    d = _parse_iso_date(date_value) or date.max
    return (d, _time_sort_key(time_value), int(event_id))


@dataclass(slots=True)
class LinkedEventsRecomputeResult:
    group_event_ids: list[int]
    changed_event_ids: list[int]
    capped: bool = False


async def recompute_linked_event_ids(
    db: Database,
    event_id: int,
    *,
    past_days: int | None = None,
    future_days: int | None = None,
    max_candidates: int | None = None,
    max_group_size: int | None = None,
) -> LinkedEventsRecomputeResult:
    """Recompute `Event.linked_event_ids` for an event and keep backlinks consistent.

    The linking rule is deliberately simple and deterministic:
    - same `location_name`
    - title matches by a strict fuzzy matcher (`_titles_match_for_linking`)
    - different dates/times are allowed (we link occurrences, not merge them)
    """
    eid = int(event_id)
    past = past_days if isinstance(past_days, int) else _env_int("LINKED_EVENTS_PAST_DAYS", 120, min_value=0, max_value=3650)
    future = future_days if isinstance(future_days, int) else _env_int("LINKED_EVENTS_FUTURE_DAYS", 365, min_value=0, max_value=3650)
    candidates_limit = (
        int(max_candidates)
        if isinstance(max_candidates, int)
        else _env_int("LINKED_EVENTS_MAX_CANDIDATES", 800, min_value=50, max_value=5000)
    )
    group_limit = (
        int(max_group_size)
        if isinstance(max_group_size, int)
        else _env_int("LINKED_EVENTS_MAX_GROUP_SIZE", 120, min_value=10, max_value=2000)
    )

    async with db.get_session() as session:
        base = await session.get(Event, eid)
        if not base:
            return LinkedEventsRecomputeResult(group_event_ids=[], changed_event_ids=[])
        loc = (getattr(base, "location_name", None) or "").strip()
        title = (getattr(base, "title", None) or "").strip()
        date_raw = (getattr(base, "date", None) or "").strip()
        end_date_raw = (getattr(base, "end_date", None) or "").strip()
        old_links = set(int(x) for x in (getattr(base, "linked_event_ids", None) or []) if str(x).lstrip("-").isdigit())

        if not loc or not title:
            return LinkedEventsRecomputeResult(group_event_ids=[], changed_event_ids=[])

        # Only link single-day events: multi-day entries already represent "many dates" as a range.
        if ".." in date_raw or end_date_raw:
            return LinkedEventsRecomputeResult(group_event_ids=[], changed_event_ids=[])

        base_day = _parse_iso_date(date_raw)
        if not base_day:
            return LinkedEventsRecomputeResult(group_event_ids=[], changed_event_ids=[])

        start_iso = (base_day - timedelta(days=past)).isoformat()
        end_iso = (base_day + timedelta(days=future)).isoformat()

        rows = (
            await session.execute(
                select(Event.id, Event.title, Event.date, Event.time)
                .where(
                    Event.location_name == loc,
                    Event.date >= start_iso,
                    Event.date <= end_iso,
                )
                .order_by(Event.date.asc(), Event.time.asc(), Event.id.asc())
                .limit(candidates_limit)
            )
        ).all()

        candidates: list[tuple[int, str, str | None, str | None]] = []
        seen_ids: set[int] = set()
        for rid, rtitle, rdate, rtime in rows:
            try:
                rid_int = int(rid)
            except Exception:
                continue
            if rid_int in seen_ids:
                continue
            seen_ids.add(rid_int)
            candidates.append(
                (
                    rid_int,
                    str(rtitle or ""),
                    str(rdate) if rdate is not None else None,
                    str(rtime) if rtime is not None else None,
                )
            )
        if eid not in seen_ids:
            # Defensive: ensure the base event participates even if its date is outside the window
            # due to malformed date strings.
            candidates.append((eid, title, date_raw, getattr(base, "time", None)))

        group_rows: list[tuple[int, str | None, str | None]] = []
        for rid_int, rtitle, rdate, rtime in candidates:
            if not rtitle.strip():
                continue
            if _titles_match_for_linking(title, rtitle, threshold=0.90):
                group_rows.append((rid_int, rdate, rtime))
        # Always include self.
        if eid not in {rid for rid, _d, _t in group_rows}:
            group_rows.append((eid, date_raw, getattr(base, "time", None)))

        group_rows.sort(key=lambda it: _event_sort_key(it[1], it[2], it[0]))
        ordered_group_ids = [rid for rid, _d, _t in group_rows]

        capped = False
        if len(ordered_group_ids) > group_limit:
            capped = True
            try:
                idx = ordered_group_ids.index(eid)
            except ValueError:
                idx = 0
            before = max(0, idx - (group_limit // 2))
            after = min(len(ordered_group_ids), before + group_limit)
            before = max(0, after - group_limit)
            ordered_group_ids = ordered_group_ids[before:after]

        group_set = set(ordered_group_ids)
        # Remove self from stored links, keep stable ordering by date/time.
        desired_by_id: dict[int, list[int]] = {
            rid: [x for x in ordered_group_ids if x != rid] for rid in ordered_group_ids
        }

        to_remove_from_others = old_links - group_set
        touched_ids = set(ordered_group_ids) | set(to_remove_from_others) | {eid}

        if not touched_ids:
            return LinkedEventsRecomputeResult(group_event_ids=[], changed_event_ids=[])

        res = await session.execute(select(Event).where(Event.id.in_(touched_ids)))
        touched_events = list(res.scalars().all())
        by_id = {int(getattr(e, "id") or 0): e for e in touched_events if getattr(e, "id", None) is not None}

        changed: list[int] = []

        def _clean_ids(raw: object, *, self_id: int) -> list[int]:
            out: list[int] = []
            seen: set[int] = set()
            if isinstance(raw, list):
                items = raw
            else:
                items = []
            for it in items:
                try:
                    v = int(it)
                except Exception:
                    continue
                if v == self_id:
                    continue
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
            return out

        # 1) Update the computed group: every event gets the full group list (minus itself).
        for rid in ordered_group_ids:
            ev = by_id.get(int(rid))
            if not ev:
                continue
            desired = desired_by_id.get(int(rid), [])
            cur = _clean_ids(getattr(ev, "linked_event_ids", None), self_id=int(rid))
            if cur != desired:
                ev.linked_event_ids = list(desired)
                changed.append(int(rid))

        # 2) Remove backlink from events that were previously linked but are no longer in group.
        for rid in to_remove_from_others:
            ev = by_id.get(int(rid))
            if not ev:
                continue
            cur = _clean_ids(getattr(ev, "linked_event_ids", None), self_id=int(rid))
            desired = [x for x in cur if x != eid]
            if cur != desired:
                ev.linked_event_ids = list(desired)
                changed.append(int(rid))

        if changed:
            for rid in sorted(set(changed)):
                session.add(by_id[rid])
            await session.commit()
        return LinkedEventsRecomputeResult(
            group_event_ids=ordered_group_ids,
            changed_event_ids=sorted(set(changed)),
            capped=capped,
        )

