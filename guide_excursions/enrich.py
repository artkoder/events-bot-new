from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .llm_support import GuideSecretsProviderAdapter, guide_account_name, resolve_candidate_key_ids
from .parser import collapse_ws

logger = logging.getLogger(__name__)

GUIDE_OCCURRENCE_ENRICH_ENABLED = (
    (os.getenv("GUIDE_OCCURRENCE_ENRICH_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_OCCURRENCE_ENRICH_MODEL = (os.getenv("GUIDE_OCCURRENCE_ENRICH_MODEL") or "gemma-3-27b").strip() or "gemma-3-27b"
GUIDE_OCCURRENCE_ENRICH_BATCH_SIZE = max(
    1,
    min(int((os.getenv("GUIDE_OCCURRENCE_ENRICH_BATCH_SIZE") or "3") or 3), 10),
)
GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV = (
    os.getenv("GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV") or "GOOGLE_API_KEY2"
).strip() or "GOOGLE_API_KEY2"
GUIDE_OCCURRENCE_ENRICH_GOOGLE_FALLBACK_KEY_ENV = (
    os.getenv("GUIDE_OCCURRENCE_ENRICH_GOOGLE_FALLBACK_KEY_ENV") or "GOOGLE_API_KEY"
).strip() or "GOOGLE_API_KEY"
GUIDE_OCCURRENCE_ENRICH_GOOGLE_ACCOUNT_ENV = (
    os.getenv("GUIDE_OCCURRENCE_ENRICH_GOOGLE_ACCOUNT_ENV") or "GOOGLE_API_LOCALNAME2"
).strip() or "GOOGLE_API_LOCALNAME2"
GUIDE_OCCURRENCE_ENRICH_GOOGLE_ACCOUNT_FALLBACK_ENV = (
    os.getenv("GUIDE_OCCURRENCE_ENRICH_GOOGLE_ACCOUNT_FALLBACK_ENV") or "GOOGLE_API_LOCALNAME"
).strip() or "GOOGLE_API_LOCALNAME"

_URLISH_RE = re.compile(r"https?://|t\.me/|tel:", re.I)
_USERNAME_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,64}")

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


def _normalize_text(value: Any) -> str:
    return collapse_ws(value).replace("\xa0", " ")


def _safe_string_list(value: Any, *, limit: int = 6) -> list[str]:
    out: list[str] = []
    items: list[Any]
    if isinstance(value, str):
        items = re.split(r"\s*[;,]\s*", value) if collapse_ws(value) else []
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return out
    for item in items:
        text = _normalize_text(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _compact_fact_pack(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "canonical_title",
        "city",
        "meeting_point",
        "route_summary",
        "duration_text",
        "group_format",
        "summary_one_liner",
        "availability_mode",
        "guide_names",
        "audience_fit",
        "price_text",
        "status",
    ):
        raw = value.get(key)
        if raw in (None, "", [], {}):
            continue
        out[key] = raw
    return out


def _hook_payload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fact_pack = _compact_fact_pack(row.get("fact_pack"))
    return {
        "occurrence_id": int(row.get("occurrence_id") or row.get("id") or 0),
        "canonical_title": _normalize_text(row.get("canonical_title")),
        "summary_seed": _normalize_text(row.get("summary_one_liner") or row.get("digest_blurb")),
        "route_summary": _normalize_text(row.get("route_summary")),
        "guide_names": _safe_string_list(row.get("guide_names"), limit=4),
        "fact_pack": fact_pack,
        "post_excerpt": _normalize_text(row.get("post_excerpt"))[:900],
    }


def _audience_payload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fact_pack = _compact_fact_pack(row.get("fact_pack"))
    return {
        "occurrence_id": int(row.get("occurrence_id") or row.get("id") or 0),
        "canonical_title": _normalize_text(row.get("canonical_title")),
        "city": _normalize_text(row.get("city")),
        "summary_seed": _normalize_text(row.get("summary_one_liner") or row.get("digest_blurb")),
        "route_summary": _normalize_text(row.get("route_summary")),
        "audience_fit": _safe_string_list(row.get("audience_fit"), limit=6),
        "group_format": _normalize_text(row.get("group_format")),
        "fact_pack": fact_pack,
        "post_excerpt": _normalize_text(row.get("post_excerpt"))[:1200],
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text))


def _sanitize_evidence(items: Any, *, limit: int = 5) -> list[str]:
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        text = _normalize_text(item)
        if not text or text in out:
            continue
        text = re.sub(r"\s+", " ", text).strip(" .;:-")
        if not text or _URLISH_RE.search(text):
            continue
        out.append(text[:120])
        if len(out) >= limit:
            break
    return out


def _sanitize_main_hook(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    text = text.strip(" .,:;!?")
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return None
    if _word_count(text) < 3 or _word_count(text) > 16:
        return None
    if len(text) > 140:
        return None
    return text


def _normalize_fit_label(value: Any) -> str | None:
    raw = _normalize_text(value).lower()
    if raw in {"locals", "tourists", "mixed"}:
        return raw
    return None


def _normalize_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except Exception:
        return None
    return max(0, min(score, 100))


def _normalize_percent_score(value: Any) -> int | None:
    score = _normalize_score(value)
    if score is None:
        return None
    # Gemma sometimes returns compact 0..10 rubric values even when the prompt
    # explicitly asks for 0..100. Treat that as a percent-style shorthand so the
    # signal still materializes into facts/admin surfaces instead of collapsing
    # into near-zero confidence.
    if 0 < score <= 10:
        return score * 10
    return score


def _retry_after_seconds(message: str) -> float | None:
    match = re.search(r"retry after\s+(\d+)\s*ms", message or "", flags=re.I)
    if not match:
        return None
    try:
        delay_ms = int(match.group(1))
    except Exception:
        return None
    if delay_ms <= 0:
        return None
    return max(0.5, min(delay_ms / 1000.0, 65.0))


@lru_cache(maxsize=4)
def _get_enrich_runtime(*, consumer: str = "guide_occurrence_enrich"):
    try:
        from google_ai import GoogleAIClient, SecretsProvider
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("guide_enrich: google_ai client unavailable: %s", exc)
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
            primary_key_env=GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV,
            fallback_key_env=GUIDE_OCCURRENCE_ENRICH_GOOGLE_FALLBACK_KEY_ENV,
        ),
        consumer=consumer,
        account_name=guide_account_name(
            primary_account_env=GUIDE_OCCURRENCE_ENRICH_GOOGLE_ACCOUNT_ENV,
            fallback_account_env=GUIDE_OCCURRENCE_ENRICH_GOOGLE_ACCOUNT_FALLBACK_ENV,
        ),
        default_env_var_name=GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV,
        incident_notifier=incident_notifier,
    )
    candidate_key_ids = resolve_candidate_key_ids(
        supabase=supabase,
        primary_key_env=GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV,
        fallback_key_env=GUIDE_OCCURRENCE_ENRICH_GOOGLE_FALLBACK_KEY_ENV,
        consumer=consumer,
    )
    return client, candidate_key_ids


async def _ask_batch(client: Any, *, prompt: str, candidate_key_ids: Sequence[str] | None) -> dict[str, Any] | None:
    for attempt in range(3):
        try:
            raw, _usage = await client.generate_content_async(
                model=GUIDE_OCCURRENCE_ENRICH_MODEL,
                prompt=prompt,
                generation_config={"temperature": 0},
                max_output_tokens=2400,
                candidate_key_ids=list(candidate_key_ids) if candidate_key_ids else None,
            )
            return _extract_json(raw or "")
        except Exception as exc:
            retry_after = _retry_after_seconds(str(exc))
            if retry_after is not None and attempt < 2:
                logger.info("guide_enrich: batch retrying after %.1fs due to provider limit", retry_after)
                await asyncio.sleep(retry_after)
                continue
            logger.warning("guide_enrich: llm batch failed: %s", exc)
            return None
    return None


async def _ask_main_hook_batch(
    client: Any,
    payload_rows: Sequence[dict[str, Any]],
    *,
    candidate_key_ids: Sequence[str] | None,
) -> dict[int, dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "occurrence_id": {"type": "integer"},
                        "main_hook": {"type": "string"},
                        "confidence": {"type": "integer"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["occurrence_id", "main_hook", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    prompt = (
        "Ты выделяешь главный grounded hook экскурсии по структурированным фактам и короткому фрагменту исходного поста.\n"
        "Верни только JSON.\n"
        "Для каждой карточки верни:\n"
        "- occurrence_id\n"
        "- main_hook: короткая русская фраза 3-16 слов; это не логистика и не рекламный слоган, а самая цепляющая содержательная особенность маршрута;\n"
        "- confidence: 0..100\n"
        "- evidence: 2-4 коротких текстовых маркера из input.\n"
        "Правила:\n"
        "1. main_hook должен быть строго grounded в input.\n"
        "1a. Сохраняй доминирующий термин из title/summary: если маршрут назван прогулкой, не переименовывай его в экскурсию; если назван экскурсией, не переименовывай его в прогулку.\n"
        "2. Не используй URL, usernames, телефоны, даты, цены и призывы записываться.\n"
        "3. Не пиши hype и не преувеличивай.\n"
        "4. Предпочитай именную фразу, а не полное предложение с конструкциями вроде `экскурсия раскрывает...` или `прогулка расскажет...`.\n"
        "5. Если сильного hook нет, выбери самый конкретный содержательный фокус маршрута.\n"
        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Input:\n{json.dumps({'items': list(payload_rows)}, ensure_ascii=False)}"
    )
    data = await _ask_batch(client, prompt=prompt, candidate_key_ids=candidate_key_ids)
    items = data.get("items") if isinstance(data, dict) else None
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            occurrence_id = int(item.get("occurrence_id") or 0)
        except Exception:
            continue
        if occurrence_id <= 0:
            continue
        main_hook = _sanitize_main_hook(item.get("main_hook"))
        confidence = _normalize_score(item.get("confidence"))
        evidence = _sanitize_evidence(item.get("evidence"), limit=4)
        if not main_hook:
            continue
        out[occurrence_id] = {
            "main_hook": main_hook,
            "main_hook_confidence": confidence if confidence is not None else 0,
            "main_hook_evidence": evidence,
        }
    return out


async def _ask_audience_region_batch(
    client: Any,
    payload_rows: Sequence[dict[str, Any]],
    *,
    candidate_key_ids: Sequence[str] | None,
) -> dict[int, dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "occurrence_id": {"type": "integer"},
                        "label": {"type": "string"},
                        "local_score": {"type": "integer"},
                        "tourist_score": {"type": "integer"},
                        "confidence": {"type": "integer"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "ambiguity": {"type": "string"},
                    },
                    "required": [
                        "occurrence_id",
                        "label",
                        "local_score",
                        "tourist_score",
                        "confidence",
                        "evidence",
                        "ambiguity",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    prompt = (
        "Проанализируй экскурсионный анонс для Калининградской области и определи, кому он больше подходит:\n"
        "- locals\n"
        "- tourists\n"
        "- mixed\n"
        "Опирайся только на признаки из input.\n"
        "Критерии:\n"
        "- предполагаемый уровень знакомства аудитории с местом;\n"
        "- акцент на новизне vs переоткрытии знакомого;\n"
        "- обзорность маршрута vs локальная глубина;\n"
        "- язык локальной идентичности / принадлежности;\n"
        "- плотность логистики для незнакомой аудитории;\n"
        "- type of call to action: жители / гости / оба.\n"
        "Верни только JSON.\n"
        "label должен быть только locals, tourists или mixed.\n"
        "local_score, tourist_score и confidence должны быть в диапазоне 0..100, не 0..10.\n"
        "evidence: 3-5 коротких текстовых маркеров из input.\n"
        "ambiguity: коротко, что мешает однозначности.\n"
        "Не додумывай аудиторию без маркеров.\n"
        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Input:\n{json.dumps({'items': list(payload_rows)}, ensure_ascii=False)}"
    )
    data = await _ask_batch(client, prompt=prompt, candidate_key_ids=candidate_key_ids)
    items = data.get("items") if isinstance(data, dict) else None
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            occurrence_id = int(item.get("occurrence_id") or 0)
        except Exception:
            continue
        if occurrence_id <= 0:
            continue
        label = _normalize_fit_label(item.get("label"))
        local_score = _normalize_percent_score(item.get("local_score"))
        tourist_score = _normalize_percent_score(item.get("tourist_score"))
        confidence = _normalize_percent_score(item.get("confidence"))
        evidence = _sanitize_evidence(item.get("evidence"), limit=5)
        ambiguity = _normalize_text(item.get("ambiguity"))[:220]
        if not label or local_score is None or tourist_score is None or confidence is None or not evidence:
            continue
        out[occurrence_id] = {
            "audience_region_fit_label": label,
            "audience_region_local_score": local_score,
            "audience_region_tourist_score": tourist_score,
            "audience_region_confidence": confidence,
            "audience_region_evidence": evidence,
            "audience_region_ambiguity": ambiguity or None,
        }
    return out


async def apply_occurrence_enrichment(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    if not GUIDE_OCCURRENCE_ENRICH_ENABLED:
        return {}
    if not (
        (os.getenv(GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV) or "").strip()
        or (os.getenv(GUIDE_OCCURRENCE_ENRICH_GOOGLE_FALLBACK_KEY_ENV) or "").strip()
    ):
        return {}
    targets = [row for row in rows if int(row.get("occurrence_id") or row.get("id") or 0) > 0]
    if not targets:
        return {}
    client, candidate_key_ids = _get_enrich_runtime()
    if client is None:
        return {}

    merged: dict[int, dict[str, Any]] = {}
    for idx in range(0, len(targets), GUIDE_OCCURRENCE_ENRICH_BATCH_SIZE):
        batch = list(targets[idx : idx + GUIDE_OCCURRENCE_ENRICH_BATCH_SIZE])
        hook_rows = [_hook_payload_row(row) for row in batch]
        audience_rows = [_audience_payload_row(row) for row in batch]
        hook_out = await _ask_main_hook_batch(client, hook_rows, candidate_key_ids=candidate_key_ids)
        audience_out = await _ask_audience_region_batch(client, audience_rows, candidate_key_ids=candidate_key_ids)
        for row in batch:
            occurrence_id = int(row.get("occurrence_id") or row.get("id") or 0)
            if occurrence_id <= 0:
                continue
            item: dict[str, Any] = {}
            if occurrence_id in hook_out:
                item.update(hook_out[occurrence_id])
            if occurrence_id in audience_out:
                item.update(audience_out[occurrence_id])
            if item:
                merged[occurrence_id] = item
    return merged


def _profile_payload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": int(row.get("profile_id") or 0),
        "source_username": _normalize_text(row.get("source_username")),
        "source_title": _normalize_text(row.get("source_title")),
        "display_name": _normalize_text(row.get("display_name")),
        "marketing_name": _normalize_text(row.get("marketing_name")),
        "about_text": _normalize_text(row.get("about_text"))[:1600],
        "sample_titles": _safe_string_list(row.get("sample_titles"), limit=6),
        "sample_summaries": _safe_string_list(row.get("sample_summaries"), limit=4),
        "sample_hooks": _safe_string_list(row.get("sample_hooks"), limit=4),
        "guide_names": _safe_string_list(row.get("guide_names"), limit=6),
    }


def _sanitize_profile_name(value: Any) -> str | None:
    text = _normalize_text(value).strip(" .,:;!?")
    if not text:
        return None
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return None
    if len(text) > 90 or _word_count(text) < 2 or _word_count(text) > 6:
        return None
    return text


def _sanitize_profile_line(value: Any) -> str | None:
    text = _normalize_text(value).strip(" .,:;!?")
    if not text:
        return None
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return None
    if len(text) > 140 or _word_count(text) < 2 or _word_count(text) > 16:
        return None
    return text


def _sanitize_profile_summary(value: Any) -> str | None:
    text = _normalize_text(value).strip()
    if not text:
        return None
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return None
    if len(text) > 200 or _word_count(text) < 5 or _word_count(text) > 32:
        return None
    return text


def _sanitize_profile_tags(value: Any, *, limit: int) -> list[str]:
    out: list[str] = []
    for item in _safe_string_list(value, limit=limit):
        text = _normalize_text(item).strip(" .,:;!?")
        if not text or _URLISH_RE.search(text) or _USERNAME_RE.search(text):
            continue
        if text in out or _word_count(text) > 8:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


async def _ask_profile_batch(
    client: Any,
    payload_rows: Sequence[dict[str, Any]],
    *,
    candidate_key_ids: Sequence[str] | None,
) -> dict[int, dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "profile_id": {"type": "integer"},
                        "display_name": {"type": "string"},
                        "guide_line": {"type": "string"},
                        "summary_short": {"type": "string"},
                        "credentials": {"type": "array", "items": {"type": "string"}},
                        "expertise_tags": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "integer"},
                    },
                    "required": [
                        "profile_id",
                        "display_name",
                        "guide_line",
                        "summary_short",
                        "credentials",
                        "expertise_tags",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    prompt = (
        "Ты materialize-ишь публичный профиль гида по facts-first input.\n"
        "Верни только JSON.\n"
        "Для каждой карточки верни:\n"
        "- profile_id\n"
        "- display_name: реальное имя и фамилия, только если они явно grounded в about_text или других input hints; иначе пустая строка.\n"
        "- guide_line: короткая русская строка 3-14 слов для публичного поля «Гид», желательно имя плюс grounded специализация/регалия.\n"
        "- summary_short: одно grounded предложение о профиле без рекламы.\n"
        "- credentials: 0-4 коротких grounded регалии/квалификации.\n"
        "- expertise_tags: 0-5 коротких grounded тематик/сильных сторон.\n"
        "- confidence: 0..100.\n"
        "Правила:\n"
        "1. Используй только то, что явно есть в input.\n"
        "2. Не придумывай имя человека, если в about_text его нет.\n"
        "3. Не используй URL, usernames, телефоны и призывы записываться.\n"
        "4. Если источник больше похож на бренд/проект, а не на конкретного человека, не форсируй персональное ФИО.\n"
        "5. Стиль нейтральный, информативный, без hype.\n"
        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Input:\n{json.dumps({'items': list(payload_rows)}, ensure_ascii=False)}"
    )
    data = await _ask_batch(client, prompt=prompt, candidate_key_ids=candidate_key_ids)
    items = data.get("items") if isinstance(data, dict) else None
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            profile_id = int(item.get("profile_id") or 0)
        except Exception:
            continue
        if profile_id <= 0:
            continue
        display_name = _sanitize_profile_name(item.get("display_name"))
        guide_line = _sanitize_profile_line(item.get("guide_line"))
        summary_short = _sanitize_profile_summary(item.get("summary_short"))
        credentials = _sanitize_profile_tags(item.get("credentials"), limit=4)
        expertise_tags = _sanitize_profile_tags(item.get("expertise_tags"), limit=5)
        confidence = _normalize_percent_score(item.get("confidence"))
        if not any([display_name, guide_line, summary_short, credentials, expertise_tags]):
            continue
        out[profile_id] = {
            "display_name": display_name,
            "guide_line": guide_line,
            "summary_short": summary_short,
            "credentials": credentials,
            "expertise_tags": expertise_tags,
            "confidence": confidence if confidence is not None else 0,
        }
    return out


async def apply_profile_enrichment(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    if not GUIDE_OCCURRENCE_ENRICH_ENABLED:
        return {}
    if not (
        (os.getenv(GUIDE_OCCURRENCE_ENRICH_GOOGLE_KEY_ENV) or "").strip()
        or (os.getenv(GUIDE_OCCURRENCE_ENRICH_GOOGLE_FALLBACK_KEY_ENV) or "").strip()
    ):
        return {}
    targets = [row for row in rows if int(row.get("profile_id") or 0) > 0]
    if not targets:
        return {}
    client, candidate_key_ids = _get_enrich_runtime(consumer="guide_profile_enrich")
    if client is None:
        return {}

    merged: dict[int, dict[str, Any]] = {}
    for idx in range(0, len(targets), GUIDE_OCCURRENCE_ENRICH_BATCH_SIZE):
        batch = list(targets[idx : idx + GUIDE_OCCURRENCE_ENRICH_BATCH_SIZE])
        payload = [_profile_payload_row(row) for row in batch]
        batch_out = await _ask_profile_batch(client, payload, candidate_key_ids=candidate_key_ids)
        if batch_out:
            merged.update(batch_out)
    return merged
