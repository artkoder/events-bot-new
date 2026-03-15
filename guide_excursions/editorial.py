from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping, Sequence

from .parser import PHONE_RE, URL_RE, USERNAME_RE, collapse_ws, has_public_invite_signal, looks_context_only, looks_operational_only

logger = logging.getLogger(__name__)

GUIDE_EDITORIAL_ENABLED = (
    (os.getenv("GUIDE_EDITORIAL_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_EDITORIAL_LLM_ENABLED = (
    (os.getenv("GUIDE_EDITORIAL_LLM_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_EDITORIAL_MODEL = (os.getenv("GUIDE_EDITORIAL_MODEL") or "gemma-3-27b").strip() or "gemma-3-27b"
GUIDE_EDITORIAL_GOOGLE_KEY_ENV = (
    os.getenv("GUIDE_EDITORIAL_GOOGLE_KEY_ENV") or "GOOGLE_API_KEY2"
).strip() or "GOOGLE_API_KEY2"
GUIDE_EDITORIAL_GOOGLE_FALLBACK_KEY_ENV = (
    os.getenv("GUIDE_EDITORIAL_GOOGLE_FALLBACK_KEY_ENV") or "GOOGLE_API_KEY"
).strip() or "GOOGLE_API_KEY"


class _GuideSecretsProviderAdapter:
    def __init__(self, base: Any):
        self.base = base

    def get_secret(self, name: str) -> str | None:
        if name == "GOOGLE_API_KEY":
            return (
                self.base.get_secret(GUIDE_EDITORIAL_GOOGLE_KEY_ENV)
                or self.base.get_secret(GUIDE_EDITORIAL_GOOGLE_FALLBACK_KEY_ENV)
            )
        return self.base.get_secret(name)


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


def _normalize_phone(raw: str) -> tuple[str, str] | None:
    value = collapse_ws(raw)
    if not value:
        return None
    digits = re.sub(r"[^\d+]", "", value)
    if not digits:
        return None
    return value, f"tel:{digits}"


def build_booking_candidates(row: Mapping[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str | None, url: str | None, kind: str) -> None:
        text = collapse_ws(label)
        href = collapse_ws(url)
        if not text or not href:
            return
        key = (text, href)
        if key in seen:
            return
        seen.add(key)
        out.append({"text": text, "url": href, "kind": kind})

    explicit_text = collapse_ws(str(row.get("booking_text") or ""))
    explicit_url = collapse_ws(str(row.get("booking_url") or ""))
    add(explicit_text or "Запись", explicit_url, "explicit")

    about_text = collapse_ws(str(row.get("source_about_text") or ""))
    about_links = row.get("source_about_links") or []
    if not isinstance(about_links, list):
        about_links = []

    for link in about_links:
        href = collapse_ws(str(link))
        if not href:
            continue
        match = re.search(r"(?:https?://)?t\.me/([A-Za-z0-9_]{4,64})", href, re.I)
        if match:
            uname = match.group(1)
            add(f"@{uname}", f"https://t.me/{uname}", "source_about")
    for uname in USERNAME_RE.findall(about_text):
        add(f"@{uname}", f"https://t.me/{uname}", "source_about")
    for phone in PHONE_RE.findall(about_text):
        normalized = _normalize_phone(phone)
        if normalized:
            add(normalized[0], normalized[1], "source_about")
    for href in URL_RE.findall(about_text):
        if "t.me/" in href.lower():
            continue
        add("Сайт / запись", href, "source_about")
    return out


def neutralize_relative_blurb(blurb: str | None, *, date_label: str | None = None, time_text: str | None = None) -> str | None:
    text = collapse_ws(blurb)
    if not text:
        return None
    replacement = collapse_ws(date_label)
    if replacement and time_text and time_text not in replacement:
        replacement = f"{replacement}, {time_text}"
    elif time_text and not replacement:
        replacement = time_text
    if replacement:
        text = re.sub(r"(?i)\bзавтра\b", replacement, text)
        text = re.sub(r"(?i)\bсегодня\b", replacement, text)
        text = re.sub(r"(?i)\bвчера\b", replacement, text)
    return text


def repair_title_fallback(title: str | None, *, source_excerpt: str | None = None) -> str | None:
    value = collapse_ws(title)
    if not value:
        return None
    excerpt = collapse_ws(source_excerpt)
    low_excerpt = excerpt.lower()
    if value[:1].islower():
        if f"путешествие на {value.lower()}" in low_excerpt:
            return f"Путешествие на {value}"
        if f"экскурсия на {value.lower()}" in low_excerpt:
            return f"Экскурсия на {value}"
        return value[:1].upper() + value[1:]
    if value.lower().startswith("по зеленоградску") and "расширенная экскурсия по зеленоградску" in low_excerpt:
        return "Расширенная экскурсия по Зеленоградску"
    return value


def apply_editorial_fallback(row: Mapping[str, Any], *, date_label: str | None = None) -> tuple[dict[str, Any] | None, str]:
    item = dict(row)
    booking_candidates = build_booking_candidates(item)
    if not collapse_ws(str(item.get("booking_url") or "")) and booking_candidates:
        item["booking_text"] = booking_candidates[0]["text"]
        item["booking_url"] = booking_candidates[0]["url"]
    excerpt = collapse_ws(str(item.get("dedup_source_text") or ""))
    if looks_context_only(excerpt):
        return None, "context_only"
    if looks_operational_only(excerpt) and not collapse_ws(str(item.get("booking_url") or "")):
        return None, "operational_without_booking"
    if not has_public_invite_signal(excerpt) and not collapse_ws(str(item.get("booking_url") or "")):
        return None, "weak_public_signal_without_booking"
    item["canonical_title"] = repair_title_fallback(item.get("canonical_title"), source_excerpt=excerpt) or item.get("canonical_title")
    item["digest_blurb"] = neutralize_relative_blurb(
        str(item.get("digest_blurb") or item.get("summary_one_liner") or ""),
        date_label=date_label,
        time_text=collapse_ws(str(item.get("time") or "")) or None,
    )
    return item, "fallback"


def _get_editorial_client():
    try:
        from google_ai import GoogleAIClient, SecretsProvider
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("guide_editorial: google_ai client unavailable: %s", exc)
        return None
    supabase = None
    incident_notifier = None
    try:
        from main import get_supabase_client, notify_llm_incident  # type: ignore

        supabase = get_supabase_client()
        incident_notifier = notify_llm_incident
    except Exception:
        pass
    return GoogleAIClient(
        supabase_client=supabase,
        secrets_provider=_GuideSecretsProviderAdapter(SecretsProvider()),
        consumer="guide_excursions_editorial",
        incident_notifier=incident_notifier,
    )


async def _ask_editorial_llm(payload_rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]] | None:
    client = _get_editorial_client()
    if client is None:
        return None
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "occurrence_id": {"type": "integer"},
                        "decision": {"type": "string", "enum": ["publish", "suppress"]},
                        "title": {"type": ["string", "null"]},
                        "blurb": {"type": ["string", "null"]},
                        "booking_choice_idx": {"type": ["integer", "null"]},
                        "reason_short": {"type": "string"},
                    },
                    "required": ["occurrence_id", "decision", "title", "blurb", "booking_choice_idx", "reason_short"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    prompt = (
        "Ты редактор дайджеста экскурсий. Для каждой карточки реши, можно ли публиковать её в публичном дайджесте.\n"
        "Правила:\n"
        "1. suppress, если это размышление, отчёт, общий контекстный пост или operational update для уже собранной группы.\n"
        "2. suppress, если нет надёжного публичного способа связаться/записаться и текст выглядит рискованным для читателя.\n"
        "3. title должен описывать именно этот конкретный выход, а не будущую премьеру или другую секцию поста.\n"
        "4. blurb = одно короткое предложение по существу, без слов сегодня/завтра/вчера.\n"
        "5. booking_choice_idx выбирай только из переданных кандидатов; если подходящего нет, верни null.\n"
        "Верни только JSON без markdown.\n"
        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Input:\n{json.dumps({'items': list(payload_rows)}, ensure_ascii=False)}"
    )
    try:
        raw, _usage = await client.generate_content_async(
            model=GUIDE_EDITORIAL_MODEL,
            prompt=prompt,
            generation_config={"temperature": 0},
            max_output_tokens=2200,
        )
    except Exception as exc:
        logger.warning("guide_editorial: llm batch failed: %s", exc)
        return None
    data = _extract_json(raw or "")
    if not isinstance(data, dict):
        return None
    items = data.get("items")
    if not isinstance(items, list):
        return None
    out: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            occurrence_id = int(item.get("occurrence_id") or 0)
        except Exception:
            continue
        if occurrence_id > 0:
            out[occurrence_id] = dict(item)
    return out or None


async def refine_digest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    date_formatter: Any,
) -> tuple[list[dict[str, Any]], list[int], dict[int, str]]:
    if not GUIDE_EDITORIAL_ENABLED:
        kept = [dict(row) for row in rows]
        return kept, [], {}

    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        date_label = date_formatter(str(item.get("date") or ""), str(item.get("time") or ""))
        item["_date_label"] = date_label
        item["_booking_candidates"] = build_booking_candidates(item)
        prepared.append(item)

    llm_payload: list[dict[str, Any]] = []
    if GUIDE_EDITORIAL_LLM_ENABLED and prepared:
        for item in prepared:
            llm_payload.append(
                {
                    "occurrence_id": int(item.get("id") or 0),
                    "family": family,
                    "title": collapse_ws(str(item.get("canonical_title") or "")),
                    "date": collapse_ws(str(item.get("date") or "")),
                    "time": collapse_ws(str(item.get("time") or "")),
                    "status": collapse_ws(str(item.get("status") or "")),
                    "seats_text": collapse_ws(str(item.get("seats_text") or "")),
                    "summary_one_liner": collapse_ws(str(item.get("summary_one_liner") or "")),
                    "digest_blurb": collapse_ws(str(item.get("digest_blurb") or "")),
                    "post_excerpt": collapse_ws(str(item.get("dedup_source_text") or ""))[:1800],
                    "source_kind": collapse_ws(str(item.get("source_kind") or "")),
                    "source_title": collapse_ws(str(item.get("source_title") or "")),
                    "source_about_text": collapse_ws(str(item.get("source_about_text") or ""))[:1000],
                    "booking_candidates": item["_booking_candidates"],
                }
            )
    llm_decisions = await _ask_editorial_llm(llm_payload) if llm_payload else None

    kept: list[dict[str, Any]] = []
    suppressed_ids: list[int] = []
    reasons: dict[int, str] = {}
    for item in prepared:
        occurrence_id = int(item.get("id") or 0)
        date_label = item.get("_date_label")
        fallback_item, fallback_reason = apply_editorial_fallback(item, date_label=date_label)
        if fallback_item is None:
            suppressed_ids.append(occurrence_id)
            reasons[occurrence_id] = fallback_reason
            continue
        decision = llm_decisions.get(occurrence_id) if isinstance(llm_decisions, dict) else None
        if isinstance(decision, dict):
            if str(decision.get("decision") or "").strip().lower() == "suppress":
                suppressed_ids.append(occurrence_id)
                reasons[occurrence_id] = collapse_ws(str(decision.get("reason_short") or "")) or "llm_suppress"
                continue
            title = collapse_ws(str(decision.get("title") or ""))
            blurb = collapse_ws(str(decision.get("blurb") or ""))
            if title:
                fallback_item["canonical_title"] = repair_title_fallback(title, source_excerpt=fallback_item.get("dedup_source_text")) or title
            if blurb:
                fallback_item["digest_blurb"] = neutralize_relative_blurb(
                    blurb,
                    date_label=date_label,
                    time_text=collapse_ws(str(fallback_item.get("time") or "")) or None,
                )
            choice_idx = decision.get("booking_choice_idx")
            if isinstance(choice_idx, int):
                candidates = fallback_item.get("_booking_candidates") or []
                if 0 <= choice_idx < len(candidates):
                    choice = candidates[choice_idx]
                    fallback_item["booking_text"] = choice.get("text")
                    fallback_item["booking_url"] = choice.get("url")
        kept.append(fallback_item)
    return kept, suppressed_ids, reasons
