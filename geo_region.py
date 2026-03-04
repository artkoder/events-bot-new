from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from sqlalchemy import text


KALININGRAD_OBLAST_WIKIDATA_QID_DEFAULT = "Q2085"
KALININGRAD_OBLAST_REGION_CODE = "RU-KGD"

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

_UA = "events-bot-new/geo-region-filter (contact: repo user)"

_wikidata_lock = asyncio.Lock()
_wikidata_last_ts = 0.0


@dataclass(frozen=True, slots=True)
class RegionDecision:
    allowed: bool | None
    city_norm: str
    region_code: str | None = None
    region_name: str | None = None
    source: str | None = None
    wikidata_qid: str | None = None
    details: dict[str, Any] | None = None

    @property
    def reason(self) -> str:
        if self.allowed is True:
            return f"in_region city={self.city_norm} region={self.region_code or ''}".strip()
        if self.allowed is False:
            return f"out_of_region city={self.city_norm} region={self.region_name or self.region_code or ''}".strip()
        return f"unknown_region city={self.city_norm}".strip()


def _truthy_env(name: str, default: str = "1") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_city(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    value = value.replace("#", " ")
    value = re.sub(r"[(),.]+", " ", value)
    value = re.sub(r"(?i)\\b(г|г\\s*\\.|город|пос|пос\\s*\\.|пгт|село|деревня|п\\s*\\.|п\\s*п\\.)\\b", " ", value)
    value = re.sub(r"\\s+", " ", value).strip()
    return value.casefold()


def _load_allowlist_norm() -> set[str]:
    path = Path("docs/reference/kaliningrad_oblast_places.md")
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    out: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        norm = _normalize_city(line)
        if norm:
            out.add(norm)
    return out


_ALLOWLIST_NORM: set[str] | None = None


def _allowlist_norm() -> set[str]:
    global _ALLOWLIST_NORM
    if _ALLOWLIST_NORM is None:
        _ALLOWLIST_NORM = _load_allowlist_norm()
    return _ALLOWLIST_NORM


async def _wikidata_throttle(min_interval_sec: float = 0.5) -> None:
    global _wikidata_last_ts
    async with _wikidata_lock:
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = (_wikidata_last_ts + min_interval_sec) - now
        if wait > 0:
            await asyncio.sleep(min(2.0, wait))
        _wikidata_last_ts = loop.time()


async def _http_get_json(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    await _wikidata_throttle()
    hdrs = {"User-Agent": _UA, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    async with httpx.AsyncClient(timeout=15.0, headers=hdrs, follow_redirects=True) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def _wikidata_search_entities(query: str, *, limit: int = 6) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    data = await _http_get_json(
        _WIKIDATA_API,
        params={
            "action": "wbsearchentities",
            "format": "json",
            "language": "ru",
            "uselang": "ru",
            "type": "item",
            "search": q,
            "limit": max(1, min(limit, 10)),
        },
    )
    items = data.get("search") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict) and isinstance(x.get("id"), str)]


async def _wikidata_sparql(query: str) -> Any:
    return await _http_get_json(
        _WIKIDATA_SPARQL,
        params={"format": "json", "query": query},
        headers={"Accept": "application/sparql-results+json"},
    )


async def _wikidata_ask_in_region(*, qid: str, region_qid: str) -> bool | None:
    if not qid or not region_qid:
        return None
    qid = qid.strip()
    region_qid = region_qid.strip()
    if not re.fullmatch(r"Q\\d+", qid) or not re.fullmatch(r"Q\\d+", region_qid):
        return None
    q = f"ASK {{ wd:{qid} wdt:P131* wd:{region_qid} . }}"
    try:
        data = await _wikidata_sparql(q)
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("boolean"), bool):
        return bool(data["boolean"])
    return None


async def _wikidata_federal_subject_label(*, qid: str) -> str | None:
    """Best-effort: get the Russian federal subject label for a place."""
    if not qid or not re.fullmatch(r"Q\\d+", qid.strip()):
        return None
    qid = qid.strip()
    # Q10864048 = "federal subject of Russia"
    q = (
        "SELECT ?subjLabel WHERE { "
        f"wd:{qid} wdt:P131* ?subj . "
        "?subj wdt:P31/wdt:P279* wd:Q10864048 . "
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". } '
        "} LIMIT 1"
    )
    try:
        data = await _wikidata_sparql(q)
    except Exception:
        return None
    try:
        bindings = data["results"]["bindings"]
        if not bindings:
            return None
        label = bindings[0]["subjLabel"]["value"]
        return str(label).strip() if label else None
    except Exception:
        return None


def _extract_json_object(text_value: str) -> dict[str, Any] | None:
    raw = (text_value or "").strip()
    if not raw:
        return None
    # Try direct JSON first.
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def _gemma_region_fallback(*, city: str, gemma_client: Any | None) -> RegionDecision:
    city_norm = _normalize_city(city)
    if not city_norm:
        return RegionDecision(allowed=None, city_norm="")
    if gemma_client is None:
        return RegionDecision(allowed=None, city_norm=city_norm, source="none")

    prompt = (
        "Ты географический ассистент. Определи, находится ли населённый пункт в Калининградской области (Россия).\n"
        "Верни строго JSON без markdown:\n"
        '{"is_kaliningrad_oblast": true|false, "region_name": "…", "confidence": 0.0}\n\n'
        f'Населённый пункт: "{city.strip()}"\n'
    )
    try:
        raw, _usage = await gemma_client.generate_content_async(
            model="gemma-3-27b",
            prompt=prompt,
            generation_config={"temperature": 0},
            max_output_tokens=256,
        )
    except Exception:
        return RegionDecision(allowed=None, city_norm=city_norm, source="gemma_error")
    data = _extract_json_object(str(raw or ""))
    if not isinstance(data, dict):
        return RegionDecision(allowed=None, city_norm=city_norm, source="gemma_parse_error")
    is_k = data.get("is_kaliningrad_oblast")
    region_name = data.get("region_name")
    if isinstance(is_k, bool):
        return RegionDecision(
            allowed=bool(is_k),
            city_norm=city_norm,
            region_code=KALININGRAD_OBLAST_REGION_CODE if is_k else None,
            region_name=str(region_name).strip() if isinstance(region_name, str) and region_name.strip() else None,
            source="gemma",
            details={"raw": str(raw or "")[:800]},
        )
    return RegionDecision(allowed=None, city_norm=city_norm, source="gemma_unknown")


async def _db_get_cached(db, *, city_norm: str) -> RegionDecision | None:
    if not city_norm:
        return None
    async with db.get_session() as session:
        row = (
            await session.execute(
                text(
                    "SELECT is_kaliningrad_oblast, region_code, region_name, source, wikidata_qid "
                    "FROM geo_city_region_cache WHERE city_norm = :city_norm LIMIT 1"
                ),
                {"city_norm": city_norm},
            )
        ).first()
    if not row:
        return None
    allowed = row[0]
    if allowed is None:
        allowed_val: bool | None = None
    else:
        allowed_val = bool(allowed)
    return RegionDecision(
        allowed=allowed_val,
        city_norm=city_norm,
        region_code=row[1],
        region_name=row[2],
        source=row[3],
        wikidata_qid=row[4],
    )


async def _db_put_cached(db, *, decision: RegionDecision) -> None:
    if not decision.city_norm:
        return
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    async with db.get_session() as session:
        await session.execute(
            text(
                "INSERT INTO geo_city_region_cache("
                "city_norm, is_kaliningrad_oblast, region_code, region_name, source, wikidata_qid, details, created_at, updated_at"
                ") VALUES("
                ":city_norm, :is_kaliningrad_oblast, :region_code, :region_name, :source, :wikidata_qid, :details, :created_at, :updated_at"
                ") ON CONFLICT(city_norm) DO UPDATE SET "
                "is_kaliningrad_oblast=excluded.is_kaliningrad_oblast, "
                "region_code=excluded.region_code, "
                "region_name=excluded.region_name, "
                "source=excluded.source, "
                "wikidata_qid=excluded.wikidata_qid, "
                "details=excluded.details, "
                "updated_at=excluded.updated_at"
            ),
            {
                "city_norm": decision.city_norm,
                "is_kaliningrad_oblast": None if decision.allowed is None else int(bool(decision.allowed)),
                "region_code": decision.region_code,
                "region_name": decision.region_name,
                "source": decision.source,
                "wikidata_qid": decision.wikidata_qid,
                "details": json.dumps(decision.details or {}, ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            },
        )
        await session.commit()


def _extract_city_candidates(city: str | None, location_address: str | None) -> Iterable[str]:
    if city and city.strip():
        yield city.strip()
    addr = (location_address or "").strip()
    if addr:
        # Try "..., <city>" heuristics.
        parts = [p.strip() for p in re.split(r"[,;/]", addr) if p.strip()]
        if parts:
            yield parts[-1]


async def decide_kaliningrad_oblast(
    db,
    *,
    city: str | None,
    location_address: str | None = None,
    gemma_client: Any | None = None,
) -> RegionDecision:
    """Return decision whether the event city belongs to Kaliningrad Oblast (RU-KGD).

    Strategy:
    1) Fast allowlist match (docs/reference/kaliningrad_oblast_places.md)
    2) Cache lookup (geo_city_region_cache)
    3) Wikidata search + region check (P131*)
    4) Gemma fallback for ambiguous/failed cases
    """
    if not _truthy_env("REGION_FILTER_ENABLED", "1"):
        return RegionDecision(allowed=True, city_norm=_normalize_city(city), source="disabled")

    region_qid = (os.getenv("REGION_FILTER_WIKIDATA_QID") or KALININGRAD_OBLAST_WIKIDATA_QID_DEFAULT).strip()
    region_qid = region_qid if re.fullmatch(r"Q\\d+", region_qid) else KALININGRAD_OBLAST_WIKIDATA_QID_DEFAULT

    for cand in _extract_city_candidates(city, location_address):
        city_norm = _normalize_city(cand)
        if not city_norm:
            continue

        if city_norm in _allowlist_norm():
            d = RegionDecision(
                allowed=True,
                city_norm=city_norm,
                region_code=KALININGRAD_OBLAST_REGION_CODE,
                region_name="Калининградская область",
                source="allowlist",
            )
            await _db_put_cached(db, decision=d)
            return d

        cached = await _db_get_cached(db, city_norm=city_norm)
        if cached is not None and cached.allowed is not None:
            return cached

        details: dict[str, Any] = {"query": cand}
        try:
            hits = await _wikidata_search_entities(cand)
        except Exception as e:
            details["wikidata_search_error"] = type(e).__name__
            hits = []

        details["wikidata_hits"] = [
            {"id": h.get("id"), "label": h.get("label"), "description": h.get("description")}
            for h in hits[:6]
        ]

        allowed_qid = None
        subject_label = None
        any_known = False
        for hit in hits[:6]:
            qid = str(hit.get("id") or "").strip()
            if not qid:
                continue
            ask = await _wikidata_ask_in_region(qid=qid, region_qid=region_qid)
            if ask is None:
                continue
            any_known = True
            if ask:
                allowed_qid = qid
                break
        if allowed_qid:
            subject_label = await _wikidata_federal_subject_label(qid=allowed_qid)
            d = RegionDecision(
                allowed=True,
                city_norm=city_norm,
                region_code=KALININGRAD_OBLAST_REGION_CODE,
                region_name=subject_label or "Калининградская область",
                source="wikidata",
                wikidata_qid=allowed_qid,
                details=details,
            )
            await _db_put_cached(db, decision=d)
            return d

        if any_known and hits:
            # We have confident negatives for the top candidates -> treat as out-of-region.
            qid0 = str(hits[0].get("id") or "").strip() if hits else None
            subject_label = await _wikidata_federal_subject_label(qid=qid0) if qid0 else None
            d = RegionDecision(
                allowed=False,
                city_norm=city_norm,
                region_code=None,
                region_name=subject_label,
                source="wikidata",
                wikidata_qid=qid0,
                details=details,
            )
            await _db_put_cached(db, decision=d)
            return d

        # Fallback: Gemma (only if Wikidata wasn't decisive).
        d = await _gemma_region_fallback(city=cand, gemma_client=gemma_client)
        if d.allowed is not None:
            await _db_put_cached(db, decision=d)
            return d

        d = RegionDecision(allowed=None, city_norm=city_norm, source="unknown", details=details)
        await _db_put_cached(db, decision=d)
        return d

    return RegionDecision(allowed=None, city_norm=_normalize_city(city), source="no_city")
