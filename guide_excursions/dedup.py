from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any, Mapping, Sequence

from .llm_support import GuideSecretsProviderAdapter, guide_account_name, resolve_candidate_key_ids
from .parser import collapse_ws

logger = logging.getLogger(__name__)

GUIDE_EXCURSIONS_DEDUP_ENABLED = (
    (os.getenv("GUIDE_EXCURSIONS_DEDUP_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED = (
    (os.getenv("GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_EXCURSIONS_DEDUP_MODEL = (os.getenv("GUIDE_EXCURSIONS_DEDUP_MODEL") or "gemma-3-27b").strip() or "gemma-3-27b"
GUIDE_EXCURSIONS_DEDUP_MAX_PAIRS = max(
    1,
    min(int((os.getenv("GUIDE_EXCURSIONS_DEDUP_MAX_PAIRS") or "12") or 12), 40),
)
GUIDE_EXCURSIONS_GOOGLE_KEY_ENV = (
    os.getenv("GUIDE_EXCURSIONS_GOOGLE_KEY_ENV") or "GOOGLE_API_KEY2"
).strip() or "GOOGLE_API_KEY2"
GUIDE_EXCURSIONS_GOOGLE_FALLBACK_KEY_ENV = (
    os.getenv("GUIDE_EXCURSIONS_GOOGLE_FALLBACK_KEY_ENV") or "GOOGLE_API_KEY"
).strip() or "GOOGLE_API_KEY"
GUIDE_EXCURSIONS_GOOGLE_ACCOUNT_ENV = (
    os.getenv("GUIDE_EXCURSIONS_GOOGLE_ACCOUNT_ENV") or "GOOGLE_API_LOCALNAME2"
).strip() or "GOOGLE_API_LOCALNAME2"
GUIDE_EXCURSIONS_GOOGLE_ACCOUNT_FALLBACK_ENV = (
    os.getenv("GUIDE_EXCURSIONS_GOOGLE_ACCOUNT_FALLBACK_ENV") or "GOOGLE_API_LOCALNAME"
).strip() or "GOOGLE_API_LOCALNAME"

_PAIR_DECISION_CACHE: dict[str, dict[str, Any]] = {}
_STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко", "у", "о", "об", "от", "за",
    "для", "из", "или", "а", "но", "что", "как", "это", "эта", "этот", "эти", "мы",
    "вы", "он", "она", "они", "его", "ее", "их", "при", "над", "под", "до", "после",
    "без", "через", "уже", "будет", "будут", "приглашаем", "присоединяйтесь", "марта",
    "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря",
    "января", "февраля", "субботу", "воскресенье", "пятницу", "четверг", "среду", "вторник",
    "понедельник", "суббота",
}


@dataclass(slots=True)
class PairCandidate:
    left_id: int
    right_id: int
    score: float
    features: dict[str, Any]


@dataclass(slots=True)
class DedupResult:
    display_rows: list[dict[str, Any]]
    covered_occurrence_ids: list[int]
    suppressed_occurrence_ids: list[int]
    pair_decisions: list[dict[str, Any]]

def _normalize_text(value: object | None) -> str:
    return collapse_ws("" if value is None else str(value)).lower().replace("ё", "е")


def _normalized_title_core(value: object | None) -> str:
    title = _normalize_text(value)
    title = re.sub(r"^[^a-zа-я0-9]+", "", title)
    title = re.sub(r"^(на|по|в|во|к|ко)\s+", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .,!?:;\"'()[]{}")
    return title


def _token_set(*parts: object | None) -> set[str]:
    payload = " ".join(_normalize_text(part) for part in parts if part is not None)
    tokens = set(re.findall(r"[a-zа-я0-9]{3,}", payload))
    return {token for token in tokens if token not in _STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _contains_focus_teaser_phrase(row: Mapping[str, Any]) -> bool:
    payload = _normalize_text(
        row.get("dedup_source_text")
        or row.get("summary_one_liner")
        or row.get("digest_blurb")
        or ""
    )
    return any(
        phrase in payload
        for phrase in (
            "в рамках нашего путешествия",
            "в рамках нашей поездки",
            "в рамках путешествия",
            "в рамках поездки",
            "в преддверии прогулки",
            "которая состоится завтра",
            "еще есть возможность на нее записаться",
            "у вас еще есть возможность на нее записаться",
        )
    )


def _same_channel_post(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_url = collapse_ws(str(left.get("channel_url") or ""))
    right_url = collapse_ws(str(right.get("channel_url") or ""))
    return bool(left_url and left_url == right_url)


def _looks_generic_title(row: Mapping[str, Any]) -> bool:
    title_tokens = _token_set(row.get("canonical_title"))
    source_tokens = _token_set(row.get("source_title"))
    title = _normalize_text(row.get("canonical_title"))
    if not title_tokens:
        return True
    if source_tokens and title_tokens <= source_tokens:
        return True
    if title in {
        "хранители руин",
        "прогулки хранителей",
        "экскурсии и путешествия на март-апрель.",
        "экскурсии и путешествия на март апрель.",
    }:
        return True
    if len(title_tokens) == 1 and len(title) <= 14:
        return True
    return False


def _row_quality_score(row: Mapping[str, Any]) -> float:
    score = float(row.get("_score") or 0.0)
    if str(row.get("source_kind") or "") != "aggregator":
        score += 0.8
    if _contains_focus_teaser_phrase(row):
        score -= 0.6
    if collapse_ws(str(row.get("booking_url") or "")):
        score += 0.45
    if collapse_ws(str(row.get("booking_text") or "")):
        score += 0.2
    if collapse_ws(str(row.get("price_text") or "")):
        score += 0.15
    if collapse_ws(str(row.get("meeting_point") or "")):
        score += 0.15
    if collapse_ws(str(row.get("time") or "")):
        score += 0.1
    score += min(len(collapse_ws(str(row.get("summary_one_liner") or ""))), 320) / 1000.0
    return score


def _has_logistics(row: Mapping[str, Any]) -> bool:
    return any(
        collapse_ws(str(row.get(key) or ""))
        for key in ("time", "booking_url", "booking_text", "price_text", "meeting_point")
    )


def _looks_schedule_or_update_rollup(row: Mapping[str, Any]) -> bool:
    payload = _normalize_text(
        " ".join(
            str(row.get(part) or "")
            for part in ("canonical_title", "summary_one_liner", "dedup_source_text")
        )
    )
    return any(
        phrase in payload
        for phrase in (
            "расписание экскурсий",
            "расписание на март",
            "уже набрана",
            "мест нет",
            "лист ожидания",
            "завтра в ",
            "отправление",
            "предположительное возвращение",
        )
    )


def _canonical_side(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    return "left" if _row_quality_score(left) >= _row_quality_score(right) else "right"


def _build_pair_features(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_tokens = _token_set(left.get("canonical_title"), left.get("summary_one_liner"))
    right_tokens = _token_set(right.get("canonical_title"), right.get("summary_one_liner"))
    shared_tokens = sorted(left_tokens & right_tokens)
    left_title_core = _normalized_title_core(left.get("canonical_title"))
    right_title_core = _normalized_title_core(right.get("canonical_title"))
    return {
        "same_source": str(left.get("source_username") or "") == str(right.get("source_username") or ""),
        "same_source_kind": str(left.get("source_kind") or "") == str(right.get("source_kind") or ""),
        "aggregator_involved": "aggregator" in {str(left.get("source_kind") or ""), str(right.get("source_kind") or "")},
        "same_date": str(left.get("date") or "") == str(right.get("date") or ""),
        "same_channel_post": _same_channel_post(left, right),
        "same_time": bool(collapse_ws(str(left.get("time") or "")) and str(left.get("time") or "") == str(right.get("time") or "")),
        "same_booking_url": bool(
            collapse_ws(str(left.get("booking_url") or "")) and str(left.get("booking_url") or "") == str(right.get("booking_url") or "")
        ),
        "same_price_text": bool(
            collapse_ws(str(left.get("price_text") or "")) and str(left.get("price_text") or "") == str(right.get("price_text") or "")
        ),
        "token_overlap_score": round(_jaccard(left_tokens, right_tokens), 3),
        "shared_route_tokens": shared_tokens[:16],
        "same_title_core": bool(left_title_core and left_title_core == right_title_core),
        "left_focus_teaser_phrase": _contains_focus_teaser_phrase(left),
        "right_focus_teaser_phrase": _contains_focus_teaser_phrase(right),
        "left_generic_title": _looks_generic_title(left),
        "right_generic_title": _looks_generic_title(right),
        "left_schedule_rollup": _looks_schedule_or_update_rollup(left),
        "right_schedule_rollup": _looks_schedule_or_update_rollup(right),
    }


def _candidate_prefilter(features: Mapping[str, Any]) -> tuple[bool, float]:
    same_source = bool(features.get("same_source"))
    same_date = bool(features.get("same_date"))
    same_channel_post = bool(features.get("same_channel_post"))
    token_score = float(features.get("token_overlap_score") or 0.0)
    shared_tokens = list(features.get("shared_route_tokens") or [])
    same_booking = bool(features.get("same_booking_url"))
    same_time = bool(features.get("same_time"))
    aggregator_involved = bool(features.get("aggregator_involved"))
    teaser_phrase = bool(features.get("left_focus_teaser_phrase") or features.get("right_focus_teaser_phrase"))
    same_title_core = bool(features.get("same_title_core"))
    schedule_rollup = bool(features.get("left_schedule_rollup") or features.get("right_schedule_rollup"))

    if not same_date:
        return False, 0.0
    if same_source and same_channel_post:
        return True, 0.82 + token_score
    if same_source and same_title_core and schedule_rollup:
        return True, 0.78 + token_score
    if same_source and (token_score >= 0.12 or len(shared_tokens) >= 3 or same_booking):
        return True, 0.65 + token_score
    if same_source and teaser_phrase:
        return True, 0.6 + token_score
    if same_source and teaser_phrase and len(shared_tokens) >= 2:
        return True, 0.74 + token_score
    if aggregator_involved and token_score >= 0.22:
        return True, 0.55 + token_score
    if aggregator_involved and same_booking and same_time:
        return True, 0.48
    return False, token_score


def _heuristic_pair_decision(left: Mapping[str, Any], right: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any] | None:
    same_source = bool(features.get("same_source"))
    same_date = bool(features.get("same_date"))
    same_channel_post = bool(features.get("same_channel_post"))
    token_score = float(features.get("token_overlap_score") or 0.0)
    shared_tokens = list(features.get("shared_route_tokens") or [])
    same_booking = bool(features.get("same_booking_url"))
    same_time = bool(features.get("same_time"))
    left_teaser = bool(features.get("left_focus_teaser_phrase"))
    right_teaser = bool(features.get("right_focus_teaser_phrase"))
    left_generic = bool(features.get("left_generic_title"))
    right_generic = bool(features.get("right_generic_title"))
    teaser_phrase = bool(left_teaser or right_teaser)
    same_title_core = bool(features.get("same_title_core"))
    left_schedule_rollup = bool(features.get("left_schedule_rollup"))
    right_schedule_rollup = bool(features.get("right_schedule_rollup"))

    if same_source and same_date and same_channel_post and (left_generic ^ right_generic):
        canonical_side = "right" if left_generic else "left"
        return {
            "decision": "same_occurrence",
            "relation": "same_post_generic_vs_specific",
            "canonical_side": canonical_side,
            "confidence": 0.86,
            "reason_short": "Both cards come from the same source post and one title is generic while the other is specific.",
            "decided_by": "heuristic",
        }
    if same_source and same_date and teaser_phrase and len(shared_tokens) >= 2:
        if left_teaser and not right_teaser:
            canonical_side = "right"
        elif right_teaser and not left_teaser:
            canonical_side = "left"
        else:
            canonical_side = _canonical_side(left, right)
        return {
            "decision": "same_occurrence",
            "relation": "master_announce_vs_focus_teaser",
            "canonical_side": canonical_side,
            "confidence": 0.74,
            "reason_short": "One post explicitly looks like a focused teaser within the broader same-date trip.",
            "decided_by": "heuristic",
        }
    if same_source and same_date and teaser_phrase and (_has_logistics(left) ^ _has_logistics(right)):
        if left_teaser and not right_teaser:
            canonical_side = "right"
        elif right_teaser and not left_teaser:
            canonical_side = "left"
        else:
            canonical_side = "left" if _has_logistics(left) else "right"
        return {
            "decision": "same_occurrence_update",
            "relation": "same_day_teaser_update_vs_logistics_announce",
            "canonical_side": canonical_side,
            "confidence": 0.79,
            "reason_short": "One same-day card looks like a teaser/update, the other carries the concrete outing logistics.",
            "decided_by": "heuristic",
        }
    if same_source and same_date and teaser_phrase and (left_generic ^ right_generic):
        canonical_side = "right" if left_generic else "left"
        return {
            "decision": "same_occurrence_update",
            "relation": "same_day_teaser_update_vs_specific_announce",
            "canonical_side": canonical_side,
            "confidence": 0.81,
            "reason_short": "One card looks like a same-day teaser/update with a generic title, the other like the concrete outing.",
            "decided_by": "heuristic",
        }
    if same_source and same_date and same_title_core and (left_schedule_rollup or right_schedule_rollup):
        if _has_logistics(left) and not _has_logistics(right):
            canonical_side = "left"
        elif _has_logistics(right) and not _has_logistics(left):
            canonical_side = "right"
        elif left_schedule_rollup and not right_schedule_rollup:
            canonical_side = "right"
        elif right_schedule_rollup and not left_schedule_rollup:
            canonical_side = "left"
        else:
            canonical_side = _canonical_side(left, right)
        return {
            "decision": "same_occurrence_update",
            "relation": "same_day_schedule_rollup_vs_departure_update",
            "canonical_side": canonical_side,
            "confidence": 0.84,
            "reason_short": "Both cards share the same title core and date; one is a schedule/reminder rollup while the other carries the concrete same-day outing details.",
            "decided_by": "heuristic",
        }
    if same_source and same_date and token_score >= 0.32 and len(shared_tokens) >= 4:
        canonical_side = _canonical_side(left, right)
        return {
            "decision": "same_occurrence",
            "relation": "same_source_same_date_overlap",
            "canonical_side": canonical_side,
            "confidence": 0.72,
            "reason_short": "Same source, same date and strong route overlap.",
            "decided_by": "heuristic",
        }
    if same_source and same_date and same_booking and same_time and token_score <= 0.05:
        return {
            "decision": "distinct",
            "relation": "same_contact_but_different_product",
            "canonical_side": "neither",
            "confidence": 0.78,
            "reason_short": "Same booking contact alone is not enough and content overlap is near-zero.",
            "decided_by": "heuristic",
        }
    return None


def _pair_cache_key(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    ids = sorted([int(left.get("id") or 0), int(right.get("id") or 0)])
    return f"{ids[0]}:{ids[1]}"


def _guide_account_name() -> str | None:
    return guide_account_name(
        primary_account_env=GUIDE_EXCURSIONS_GOOGLE_ACCOUNT_ENV,
        fallback_account_env=GUIDE_EXCURSIONS_GOOGLE_ACCOUNT_FALLBACK_ENV,
    )


@lru_cache(maxsize=1)
def _get_dedup_client():
    try:
        from google_ai import GoogleAIClient, SecretsProvider
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("guide_dedup: google_ai client unavailable: %s", exc)
        return None, None
    supabase = None
    incident_notifier = None
    try:
        from main import get_supabase_client, notify_llm_incident  # type: ignore

        supabase = get_supabase_client()
        incident_notifier = notify_llm_incident
    except Exception:
        pass
    client = GoogleAIClient(
        supabase_client=supabase,
        secrets_provider=GuideSecretsProviderAdapter(
            SecretsProvider(),
            primary_key_env=GUIDE_EXCURSIONS_GOOGLE_KEY_ENV,
            fallback_key_env=GUIDE_EXCURSIONS_GOOGLE_FALLBACK_KEY_ENV,
        ),
        consumer="guide_excursions_dedup",
        account_name=_guide_account_name(),
        default_env_var_name=GUIDE_EXCURSIONS_GOOGLE_KEY_ENV,
        incident_notifier=incident_notifier,
    )
    candidate_key_ids = resolve_candidate_key_ids(
        supabase=supabase,
        primary_key_env=GUIDE_EXCURSIONS_GOOGLE_KEY_ENV,
        fallback_key_env=GUIDE_EXCURSIONS_GOOGLE_FALLBACK_KEY_ENV,
        consumer="guide_excursions_dedup",
    )
    return client, candidate_key_ids


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


async def _ask_pair_judge_llm(left: Mapping[str, Any], right: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any] | None:
    client, candidate_key_ids = _get_dedup_client()
    if client is None:
        return None
    schema = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": [
                    "same_occurrence",
                    "same_occurrence_update",
                    "same_template_other_occurrence",
                    "distinct",
                    "uncertain",
                ],
            },
            "relation": {"type": "string"},
            "canonical_side": {"type": "string", "enum": ["left", "right", "neither"]},
            "confidence": {"type": "number"},
            "reason_short": {"type": "string"},
        },
        "required": ["decision", "relation", "canonical_side", "confidence", "reason_short"],
    }
    payload = {
        "task": "classify_pair",
        "left": {
            "occurrence_id": int(left.get("id") or 0),
            "source_kind": str(left.get("source_kind") or ""),
            "source_username": str(left.get("source_username") or ""),
            "source_title": str(left.get("source_title") or ""),
            "title": str(left.get("canonical_title") or ""),
            "date": str(left.get("date") or ""),
            "time": left.get("time"),
            "meeting_point": left.get("meeting_point"),
            "price_text": left.get("price_text"),
            "booking_text": left.get("booking_text"),
            "booking_url": left.get("booking_url"),
            "audience_fit": list(left.get("audience_fit") or []),
            "summary_one_liner": str(left.get("summary_one_liner") or ""),
            "post_excerpt": collapse_ws(str(left.get("dedup_source_text") or ""))[:1400],
        },
        "right": {
            "occurrence_id": int(right.get("id") or 0),
            "source_kind": str(right.get("source_kind") or ""),
            "source_username": str(right.get("source_username") or ""),
            "source_title": str(right.get("source_title") or ""),
            "title": str(right.get("canonical_title") or ""),
            "date": str(right.get("date") or ""),
            "time": right.get("time"),
            "meeting_point": right.get("meeting_point"),
            "price_text": right.get("price_text"),
            "booking_text": right.get("booking_text"),
            "booking_url": right.get("booking_url"),
            "audience_fit": list(right.get("audience_fit") or []),
            "summary_one_liner": str(right.get("summary_one_liner") or ""),
            "post_excerpt": collapse_ws(str(right.get("dedup_source_text") or ""))[:1400],
        },
        "candidate_features": dict(features),
    }
    prompt = (
        "Ты решаешь задачу дедупликации анонсов экскурсий.\n"
        "Определи, описывают ли две карточки один и тот же конкретный выход экскурсии.\n"
        "Будь консервативен: совпадение даты, телефона или канала само по себе недостаточно.\n"
        "Если один пост выглядит как тизер/фокус на часть маршрута внутри более общего анонса на ту же дату, это same_occurrence.\n"
        "Верни только JSON без markdown.\n"
        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        raw, _usage = await client.generate_content_async(
            model=GUIDE_EXCURSIONS_DEDUP_MODEL,
            prompt=prompt,
            generation_config={"temperature": 0},
            max_output_tokens=420,
            candidate_key_ids=list(candidate_key_ids) if candidate_key_ids else None,
        )
    except Exception as exc:
        logger.warning("guide_dedup: llm pair judge failed left=%s right=%s err=%s", left.get("id"), right.get("id"), exc)
        return None
    data = _extract_json(raw or "")
    if not isinstance(data, dict):
        return None
    return data


async def _decide_pair(left: Mapping[str, Any], right: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    heuristic = _heuristic_pair_decision(left, right, features)
    if heuristic is not None:
        return heuristic
    cache_key = _pair_cache_key(left, right)
    cached = _PAIR_DECISION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    if not GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED:
        out = {
            "decision": "uncertain",
            "relation": "no_llm",
            "canonical_side": "neither",
            "confidence": 0.0,
            "reason_short": "LLM dedup disabled; conservative no-merge fallback.",
            "decided_by": "fallback",
        }
        _PAIR_DECISION_CACHE[cache_key] = dict(out)
        return out
    llm = await _ask_pair_judge_llm(left, right, features)
    if isinstance(llm, dict):
        llm["decided_by"] = "llm"
        _PAIR_DECISION_CACHE[cache_key] = dict(llm)
        return llm
    out = {
        "decision": "uncertain",
        "relation": "llm_unavailable",
        "canonical_side": "neither",
        "confidence": 0.0,
        "reason_short": "LLM dedup unavailable; conservative no-merge fallback.",
        "decided_by": "fallback",
    }
    _PAIR_DECISION_CACHE[cache_key] = dict(out)
    return out


def _candidate_pairs(rows: Sequence[Mapping[str, Any]]) -> list[PairCandidate]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        date_iso = collapse_ws(str(row.get("date") or ""))
        if not date_iso:
            continue
        by_date.setdefault(date_iso, []).append(row)
    out: list[PairCandidate] = []
    for group in by_date.values():
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            features = _build_pair_features(left, right)
            allowed, score = _candidate_prefilter(features)
            if not allowed:
                continue
            out.append(
                PairCandidate(
                    left_id=int(left.get("id") or 0),
                    right_id=int(right.get("id") or 0),
                    score=float(score),
                    features=features,
                )
            )
    out.sort(key=lambda item: (-item.score, -float(item.features.get("token_overlap_score") or 0.0), item.left_id, item.right_id))
    return out[:GUIDE_EXCURSIONS_DEDUP_MAX_PAIRS]


class _UnionFind:
    def __init__(self, ids: Sequence[int]):
        self.parent = {int(i): int(i) for i in ids}

    def find(self, item: int) -> int:
        root = self.parent[item]
        while root != self.parent[root]:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        rl = self.find(left)
        rr = self.find(right)
        if rl != rr:
            self.parent[rr] = rl


async def deduplicate_occurrence_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    limit: int,
) -> DedupResult:
    items = [dict(row) for row in rows]
    if not GUIDE_EXCURSIONS_DEDUP_ENABLED or len(items) <= 1:
        display = [dict(row) for row in items[:limit]]
        ids = [int(row.get("id") or 0) for row in display]
        return DedupResult(display_rows=display, covered_occurrence_ids=ids, suppressed_occurrence_ids=[], pair_decisions=[])

    index_by_id = {int(row.get("id") or 0): idx for idx, row in enumerate(items)}
    row_by_id = {int(row.get("id") or 0): row for row in items}
    pairs = _candidate_pairs(items)
    pair_decisions: list[dict[str, Any]] = []
    uf = _UnionFind(list(row_by_id.keys()))
    canonical_votes: dict[int, float] = {}
    for pair in pairs:
        left = row_by_id.get(pair.left_id)
        right = row_by_id.get(pair.right_id)
        if not left or not right:
            continue
        decision = await _decide_pair(left, right, pair.features)
        entry = {
            "left_id": pair.left_id,
            "right_id": pair.right_id,
            "features": dict(pair.features),
            "decision": dict(decision),
        }
        pair_decisions.append(entry)
        if decision.get("decision") in {"same_occurrence", "same_occurrence_update"}:
            uf.union(pair.left_id, pair.right_id)
            side = str(decision.get("canonical_side") or "neither")
            confidence = float(decision.get("confidence") or 0.0)
            if side == "left":
                canonical_votes[pair.left_id] = canonical_votes.get(pair.left_id, 0.0) + max(0.2, confidence)
            elif side == "right":
                canonical_votes[pair.right_id] = canonical_votes.get(pair.right_id, 0.0) + max(0.2, confidence)

    clusters: dict[int, list[int]] = {}
    for row_id in row_by_id:
        root = uf.find(row_id)
        clusters.setdefault(root, []).append(row_id)

    cluster_meta: list[dict[str, Any]] = []
    for member_ids in clusters.values():
        member_rows = [row_by_id[mid] for mid in member_ids if mid in row_by_id]
        canonical = max(
            member_rows,
            key=lambda item: (
                canonical_votes.get(int(item.get("id") or 0), 0.0),
                _row_quality_score(item),
            ),
        )
        canonical_id = int(canonical.get("id") or 0)
        order_idx = min(index_by_id[mid] for mid in member_ids if mid in index_by_id)
        cluster_meta.append(
            {
                "canonical_id": canonical_id,
                "member_ids": list(member_ids),
                "order_idx": order_idx,
                "display_row": canonical,
            }
        )
    cluster_meta.sort(key=lambda item: int(item["order_idx"]))

    display_rows: list[dict[str, Any]] = []
    covered_occurrence_ids: list[int] = []
    suppressed_ids: list[int] = []
    for meta in cluster_meta:
        if len(display_rows) >= limit:
            break
        display_rows.append(dict(meta["display_row"]))
        covered_occurrence_ids.extend(int(mid) for mid in meta["member_ids"])
        suppressed_ids.extend(int(mid) for mid in meta["member_ids"] if int(mid) != int(meta["canonical_id"]))

    covered_occurrence_ids = list(dict.fromkeys(covered_occurrence_ids))
    suppressed_ids = list(dict.fromkeys(suppressed_ids))
    logger.info(
        "guide_dedup: family=%s raw=%s display=%s covered=%s suppressed=%s pairs=%s",
        family,
        len(items),
        len(display_rows),
        len(covered_occurrence_ids),
        len(suppressed_ids),
        len(pair_decisions),
    )
    return DedupResult(
        display_rows=display_rows,
        covered_occurrence_ids=covered_occurrence_ids,
        suppressed_occurrence_ids=suppressed_ids,
        pair_decisions=pair_decisions,
    )
