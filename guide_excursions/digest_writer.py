from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .editorial import neutralize_relative_blurb, repair_title_fallback
from .llm_support import GuideSecretsProviderAdapter, guide_account_name, resolve_candidate_key_ids
from .parser import collapse_ws

logger = logging.getLogger(__name__)

GUIDE_DIGEST_WRITER_ENABLED = (
    (os.getenv("GUIDE_DIGEST_WRITER_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_DIGEST_WRITER_MODEL = (os.getenv("GUIDE_DIGEST_WRITER_MODEL") or "gemma-3-27b").strip() or "gemma-3-27b"
GUIDE_DIGEST_WRITER_GOOGLE_KEY_ENV = (
    os.getenv("GUIDE_DIGEST_WRITER_GOOGLE_KEY_ENV") or "GOOGLE_API_KEY2"
).strip() or "GOOGLE_API_KEY2"
GUIDE_DIGEST_WRITER_GOOGLE_FALLBACK_KEY_ENV = (
    os.getenv("GUIDE_DIGEST_WRITER_GOOGLE_FALLBACK_KEY_ENV") or "GOOGLE_API_KEY"
).strip() or "GOOGLE_API_KEY"
GUIDE_DIGEST_WRITER_GOOGLE_ACCOUNT_ENV = (
    os.getenv("GUIDE_DIGEST_WRITER_GOOGLE_ACCOUNT_ENV") or "GOOGLE_API_LOCALNAME2"
).strip() or "GOOGLE_API_LOCALNAME2"
GUIDE_DIGEST_WRITER_GOOGLE_ACCOUNT_FALLBACK_ENV = (
    os.getenv("GUIDE_DIGEST_WRITER_GOOGLE_ACCOUNT_FALLBACK_ENV") or "GOOGLE_API_LOCALNAME"
).strip() or "GOOGLE_API_LOCALNAME"
GUIDE_DIGEST_WRITER_BATCH_SIZE = max(
    1,
    min(int((os.getenv("GUIDE_DIGEST_WRITER_BATCH_SIZE") or "2") or 2), 12),
)
GUIDE_DIGEST_WRITER_MAX_WAIT_SECONDS = max(
    30,
    min(int((os.getenv("GUIDE_DIGEST_WRITER_MAX_WAIT_SECONDS") or "240") or 240), 1800),
)
_BANNED_HYPE_STEMS = [
    "уникаль",
    "невероят",
    "потряса",
    "незабыва",
    "волшеб",
    "роскош",
    "идеаль",
    "лучший",
    "самый",
]
_USERNAME_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,64}")
_URLISH_RE = re.compile(r"https?://|t\.me/", re.I)

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


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        raw = collapse_ws(value)
        items = re.split(r"\s*[;,]\s*", raw) if raw else []
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return out
    for item in items:
        text = collapse_ws(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _fact_signal_count(row: Mapping[str, Any]) -> int:
    fact_pack = row.get("fact_pack") if isinstance(row.get("fact_pack"), Mapping) else {}
    count = 0
    for key in (
        "canonical_title",
        "date",
        "time",
        "duration_text",
        "city",
        "meeting_point",
        "route_summary",
        "price_text",
        "booking_text",
        "booking_url",
        "status",
        "seats_text",
        "summary_one_liner",
        "digest_blurb",
        "availability_mode",
        "post_kind",
        "group_format",
        "main_hook",
        "audience_region_fit_label",
    ):
        value = row.get(key)
        if value is None and isinstance(fact_pack, Mapping):
            value = fact_pack.get(key)
        if isinstance(value, list):
            if _string_list(value):
                count += 1
            continue
        if collapse_ws(value):
            count += 1
    for key in ("audience_fit", "guide_names", "organizer_names"):
        value = row.get(key)
        if value is None and isinstance(fact_pack, Mapping):
            value = fact_pack.get(key)
        if _string_list(value, limit=4):
            count += 1
    return count


def _fact_density_bucket(row: Mapping[str, Any]) -> str:
    count = _fact_signal_count(row)
    if count >= 9:
        return "high"
    if count >= 6:
        return "medium"
    return "low"


def _target_blurb_sentences(row: Mapping[str, Any]) -> int:
    bucket = _fact_density_bucket(row)
    if bucket == "high":
        return 3
    if bucket == "medium":
        return 2
    return 1


def _shell_fields_present(row: Mapping[str, Any]) -> list[str]:
    present: list[str] = []
    if collapse_ws(row.get("date")) or collapse_ws(row.get("time")):
        present.append("date_time")
    if collapse_ws(row.get("city")):
        present.append("city")
    if collapse_ws(row.get("meeting_point_line") or row.get("meeting_point")):
        present.append("meeting_point")
    if collapse_ws(row.get("duration_line") or row.get("duration_text")):
        present.append("duration")
    if collapse_ws(row.get("route_line") or row.get("route_summary")):
        present.append("route")
    if collapse_ws(row.get("price_line") or row.get("price_text")):
        present.append("price")
    if collapse_ws(row.get("booking_line") or row.get("booking_text") or row.get("booking_url")):
        present.append("booking")
    if collapse_ws(row.get("organizer_line")):
        present.append("organizer")
    if collapse_ws(row.get("seats_line") or row.get("seats_text")):
        present.append("availability")
    if collapse_ws(row.get("group_format_line") or row.get("group_format")):
        present.append("format")
    audience = row.get("audience_fit")
    if _string_list(audience, limit=4):
        present.append("audience")
    if collapse_ws(row.get("audience_region_fit_label") or (row.get("fact_pack") or {}).get("audience_region_fit_label")):
        present.append("audience_region_fit")
    return present


def _non_logistics_seed(row: Mapping[str, Any]) -> str | None:
    for key in ("digest_blurb", "summary_one_liner"):
        value = collapse_ws(row.get(key))
        if value:
            return value
    fact_pack = row.get("fact_pack")
    if isinstance(fact_pack, Mapping):
        for key in ("digest_blurb", "summary_one_liner"):
            value = collapse_ws(fact_pack.get(key))
            if value:
                return value
    return None


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
        "guide_names",
        "organizer_names",
        "audience_fit",
        "price_text",
        "main_hook",
        "main_hook_evidence",
        "audience_region_fit_label",
        "audience_region_evidence",
        "audience_region_ambiguity",
    ):
        raw = value.get(key)
        if raw in (None, "", [], {}):
            continue
        out[key] = raw
    return out


def _contains_banned_hype(text: str | None) -> bool:
    low = collapse_ws(text).lower()
    return any(stem in low for stem in _BANNED_HYPE_STEMS)


def _word_count(text: str | None) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", collapse_ws(text)))


def _duplicates_shell_logistics(text: str | None, row: Mapping[str, Any], *, date_label: str | None) -> bool:
    low = collapse_ws(text).lower()
    if not low:
        return False
    candidates = [
        date_label,
        row.get("time"),
        row.get("price_line"),
        row.get("price_text"),
        row.get("booking_text"),
        row.get("booking_url"),
        row.get("seats_line"),
        row.get("seats_text"),
    ]
    for candidate in candidates:
        normalized = collapse_ws(candidate).lower()
        if normalized and normalized in low:
            return True
    return False


def _dominant_term_family(row: Mapping[str, Any]) -> str | None:
    candidates = [
        row.get("canonical_title"),
        row.get("summary_one_liner"),
        row.get("digest_blurb"),
        row.get("route_summary"),
        (row.get("fact_pack") or {}).get("canonical_title") if isinstance(row.get("fact_pack"), Mapping) else None,
        (row.get("fact_pack") or {}).get("summary_one_liner") if isinstance(row.get("fact_pack"), Mapping) else None,
        (row.get("fact_pack") or {}).get("route_summary") if isinstance(row.get("fact_pack"), Mapping) else None,
    ]
    combined = " ".join(collapse_ws(value) for value in candidates if collapse_ws(value)).lower().replace("ё", "е")
    if "прогул" in combined:
        return "walk"
    if "экскурс" in combined:
        return "excursion"
    return None


def _term_family_conflict(text: str | None, row: Mapping[str, Any]) -> bool:
    low = collapse_ws(text).lower().replace("ё", "е")
    if not low:
        return False
    family = _dominant_term_family(row)
    if family == "walk" and "экскурс" in low and "прогул" not in low:
        return True
    if family == "excursion" and "прогул" in low and "экскурс" not in low:
        return True
    return False


def _duplicates_audience_region(text: str | None, row: Mapping[str, Any]) -> bool:
    low = collapse_ws(text).lower().replace("ё", "е")
    if not low:
        return False
    has_region_fit = bool(
        collapse_ws(row.get("audience_region_line"))
        or collapse_ws((row.get("fact_pack") or {}).get("audience_region_fit_label")) if isinstance(row.get("fact_pack"), Mapping) else False
    )
    if not has_region_fit:
        return False
    patterns = (
        ("местн", "гост"),
        ("жител", "гост"),
        ("турист", "местн"),
        ("впервые", "знаком"),
    )
    return any(all(token in low for token in group) for group in patterns)


def _accept_title_candidate(title: str | None) -> bool:
    text = collapse_ws(title)
    if not text:
        return False
    if _contains_banned_hype(text):
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    return 3 <= _word_count(text) <= 12


def _accept_lead_emoji_candidate(value: str | None) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if len(text) > 4:
        return False
    if any(char.isalnum() for char in text):
        return False
    return True


def _accept_blurb_candidate(blurb: str | None, row: Mapping[str, Any], *, date_label: str | None) -> bool:
    text = collapse_ws(blurb)
    if not text:
        return False
    if _contains_banned_hype(text):
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    if _duplicates_shell_logistics(text, row, date_label=date_label):
        return False
    if _duplicates_audience_region(text, row):
        return False
    if _term_family_conflict(text, row):
        return False
    return _word_count(text) >= 5


def _accept_guide_line_candidate(value: str | None) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    return 2 <= _word_count(text) <= 14 and len(text) <= 120


def _accept_organizer_line_candidate(value: str | None) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    return 2 <= _word_count(text) <= 16 and len(text) <= 140


def _accept_price_line_candidate(value: str | None, row: Mapping[str, Any]) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    raw_price = collapse_ws(row.get("price_text") or (row.get("fact_pack") or {}).get("price_text"))
    if not raw_price:
        return False
    if re.search(r"\d+\s*/\s*\d+", text):
        return False
    if any(token in text for token in (",пенсион", ", дети", "/с ", "₽/с")):
        return False
    explicit_categories = re.search(r"взрос|дет|пенсион|льгот|школь", raw_price, flags=re.I)
    generated_categories = re.search(r"взрос|дет|пенсион|льгот|школь", text, flags=re.I)
    if not explicit_categories and generated_categories:
        return False
    return 1 <= _word_count(text) <= 18 and len(text) <= 120


def _accept_route_line_candidate(value: str | None) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    if _contains_banned_hype(text):
        return False
    return 4 <= _word_count(text) <= 26 and len(text) <= 180


def _accept_audience_region_line_candidate(value: str | None) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if _URLISH_RE.search(text) or _USERNAME_RE.search(text):
        return False
    lowered = text.lower()
    if lowered.startswith("кому больше:") or lowered.startswith("для кого:"):
        return False
    if text.lower() in {
        "и для жителей региона, и для гостей",
        "и для местных, и для гостей",
        "подойдёт и местным, и гостям",
        "подойдет и местным, и гостям",
        "для местных и гостей",
    }:
        return False
    return 2 <= _word_count(text) <= 18 and len(text) <= 140


def _accept_booking_line_candidate(value: str | None, row: Mapping[str, Any]) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    booking_url = collapse_ws(row.get("booking_url"))
    if not booking_url:
        return False
    if _URLISH_RE.search(text):
        return False
    if booking_url.startswith("tel:"):
        digits = re.sub(r"[^\d]", "", text)
        return 6 <= len(digits) <= 15 and len(text) <= 40
    if booking_url.lower().startswith("https://t.me/"):
        return bool(re.fullmatch(r"@[A-Za-z0-9_]{4,64}", text))
    return 1 <= _word_count(text) <= 8 and len(text) <= 80


@lru_cache(maxsize=1)
def _get_digest_writer_runtime():
    try:
        from google_ai import GoogleAIClient, SecretsProvider
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("guide_digest_writer: google_ai client unavailable: %s", exc)
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
            primary_key_env=GUIDE_DIGEST_WRITER_GOOGLE_KEY_ENV,
            fallback_key_env=GUIDE_DIGEST_WRITER_GOOGLE_FALLBACK_KEY_ENV,
        ),
        consumer="guide_excursions_digest_batch",
        account_name=guide_account_name(
            primary_account_env=GUIDE_DIGEST_WRITER_GOOGLE_ACCOUNT_ENV,
            fallback_account_env=GUIDE_DIGEST_WRITER_GOOGLE_ACCOUNT_FALLBACK_ENV,
        ),
        default_env_var_name=GUIDE_DIGEST_WRITER_GOOGLE_KEY_ENV,
        incident_notifier=incident_notifier,
    )
    candidate_key_ids = resolve_candidate_key_ids(
        supabase=supabase,
        primary_key_env=GUIDE_DIGEST_WRITER_GOOGLE_KEY_ENV,
        fallback_key_env=GUIDE_DIGEST_WRITER_GOOGLE_FALLBACK_KEY_ENV,
        consumer="guide_excursions_digest_batch",
    )
    return client, candidate_key_ids


async def _ask_digest_batch_llm_batch(
    client: Any,
    payload_rows: Sequence[dict[str, Any]],
    *,
    candidate_key_ids: Sequence[str] | None,
) -> dict[int, dict[str, Any]] | None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "occurrence_id": {"type": "integer"},
                        "lead_emoji": {"type": "string"},
                        "title": {"type": "string"},
                        "digest_blurb": {"type": "string"},
                        "guide_line": {"type": "string"},
                        "organizer_line": {"type": "string"},
                        "price_line": {"type": "string"},
                        "route_line": {"type": "string"},
                        "audience_region_line": {"type": "string"},
                        "booking_line": {"type": "string"},
                    },
                    "required": ["occurrence_id", "title", "digest_blurb"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    prompt = (
        "Ты пишешь короткий публичный copy-block для digest-карточек экскурсий.\n"
        "Твоя задача: для каждой карточки вернуть title, digest_blurb и при возможности короткие public lines.\n"
        "Правила:\n"
        "1. Используй только grounded facts из fact_pack и input hints.\n"
        "2. В digest_blurb не повторяй date/time, city, meeting point, price, booking, seats, если они уже вынесены в shell_fields_present.\n"
        "2a. В digest_blurb не повторяй local-vs-tourist fit; это живёт в отдельной строке audience_region_line / `🏠 ...`.\n"
        "3. Если в input есть main_hook, первая фраза должна открываться именно им или его точным смыслом.\n"
        "4. Если main_hook нет, в первой фразе опирайся на самый конкретный не-логистический факт: тема маршрута, особенность прогулки, объект наблюдения, формат, фокус экскурсии.\n"
        "4b. Не смешивай термины `прогулка` и `экскурсия` без оснований.\n"
        "- если title_seed или grounded facts явно задают `прогулка`, сохраняй эту семью слов и не переименовывай её в `экскурсию`;\n"
        "- если title_seed или grounded facts явно задают `экскурсия`, не размягчай её в `прогулку`;\n"
        "- если тип неочевиден, предпочитай нейтральные слова `маршрут`, `выход`, `поездка`, а не неверный термин.\n"
        "5. Не используй hype и рекламные усилители. Запрещены слова/основы: "
        + ", ".join(_BANNED_HYPE_STEMS)
        + ".\n"
        "6. lead_emoji: один тематический эмодзи для заголовка, только если он действительно grounded в маршруте/теме (`🐦`, `🏛️`, `🌲`, `🧱`, `🌊`). Не ставь generic `⭐️`/`❤️`, если тема не требует эмодзи. Если уместного эмодзи нет, верни пустую строку.\n"
        "7. title: 4-10 слов, без даты, цены, контактов и эмодзи; лучше точное название выхода, чем креативное переименование.\n"
        "8. digest_blurb: законченное описание из 1-3 предложений; количество предложений должно строго следовать target_blurb_sentences.\n"
        "9. Делай текст живым и интересным, чтобы читателю хотелось пойти на экскурсию, но без преувеличений и маркетингового нажима.\n"
        "10. Плотность описания определяется только набором фактов: если фактов мало, ограничься самым важным; если есть несколько содержательных не-логистических фактов, используй их, но не выходи за target_blurb_sentences.\n"
        "11. Не используй today/tomorrow/yesterday, не вставляй URL, телефоны и usernames в digest_blurb.\n"
        "12. guide_line: короткая строка для поля «Гид». Предпочитай реальное имя и grounded регалии из profile/about. Если надёжного имени нет, верни пустую строку.\n"
        "12a. organizer_line: если надёжного имени гида нет, но ясно, что это организация/проект/оператор, верни короткую строку для поля «Организатор». Если есть надёжный гид, organizer_line оставь пустой.\n"
        "13. price_line: краткая естественная русская формулировка цены без URL и без сырого копипаста, строго по фактам. Нормализуй схемы вроде `500/300 руб взрослые/дети,пенсионеры` в форму `500 ₽ для взрослых, 300 ₽ для детей и пенсионеров`, но не добавляй категории посетителей, если их нет в input. Если у конкретной карточки нет price fact, верни пустую строку.\n"
        "14. route_line: короткая строка для поля «Что в маршруте». Пиши только если в input есть конкретное содержательное наполнение маршрута; если есть только общая тема, верни пустую строку.\n"
        "15. audience_region_line: отдельная самостоятельная строка для house-line на основе audience_region_fit_label и evidence. Эта строка будет показана как `🏠 <твой текст>`, поэтому не начинай её с `Кому больше:` или `Для кого:`.\n"
        "Она должна описывать именно знакомство аудитории с регионом/местом, а не возраст или состав группы: возрастные группы уже живут в отдельной строке `Кому подойдёт`.\n"
        "Избегай канцеляризма и пустых generic-фраз вроде `и для жителей региона, и для гостей`.\n"
        "Примеры хорошего тона:\n"
        "- locals -> `Больше подойдёт тем, кто уже знает город и хочет увидеть знакомый район под новым углом`.\n"
        "- tourists -> `Хороший вариант для тех, кто только знакомится с Калининградской областью и хочет начать с понятного маршрута`.\n"
        "- mixed -> `Подойдёт и тем, кто уже живёт в регионе, и тем, кто только его открывает`.\n"
        "Если естественно и grounded сформулировать нельзя, верни пустую строку.\n"
        "15a. booking_line: короткая строка для поля `Запись`. Не начинай с `Запись:` и не копируй служебные фразы вроде `звоните`, `пишите`, `по телефону с ...`.\n"
        "Если booking_url это `tel:`, верни только один лучший номер телефона в читабельной форме; при наличии нескольких телефонов предпочитай мобильный.\n"
        "Если booking_url ведёт на telegram username, верни только `@username`.\n"
        "Если booking_url это сайт, верни короткую нейтральную подпись вроде `сайт организатора` или `форма записи`.\n"
        "16. Возвращай JSON и ничего больше.\n"
        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Input:\n{json.dumps({'items': list(payload_rows)}, ensure_ascii=False)}"
    )
    raw = None
    waited_seconds = 0.0
    for attempt in range(6):
        try:
            raw, _usage = await client.generate_content_async(
                model=GUIDE_DIGEST_WRITER_MODEL,
                prompt=prompt,
                generation_config={"temperature": 0.2},
                max_output_tokens=2400,
                candidate_key_ids=list(candidate_key_ids) if candidate_key_ids else None,
            )
            break
        except Exception as exc:
            retry_after = _retry_after_seconds(str(exc))
            if (
                retry_after is not None
                and attempt < 5
                and (waited_seconds + retry_after) <= float(GUIDE_DIGEST_WRITER_MAX_WAIT_SECONDS)
            ):
                logger.info("guide_digest_writer: batch retrying after %.1fs due to provider limit", retry_after)
                await asyncio.sleep(retry_after)
                waited_seconds += retry_after
                continue
            logger.warning("guide_digest_writer: llm batch failed: %s", exc)
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


async def _ask_digest_batch_llm(payload_rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]] | None:
    client, candidate_key_ids = _get_digest_writer_runtime()
    if client is None:
        return None
    merged: dict[int, dict[str, Any]] = {}
    for idx in range(0, len(payload_rows), GUIDE_DIGEST_WRITER_BATCH_SIZE):
        batch = list(payload_rows[idx : idx + GUIDE_DIGEST_WRITER_BATCH_SIZE])
        batch_out = await _ask_digest_batch_llm_batch(client, batch, candidate_key_ids=candidate_key_ids)
        if isinstance(batch_out, dict):
            merged.update(batch_out)
    return merged or None


async def apply_digest_batch_copy(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    date_formatter: Any,
) -> list[dict[str, Any]]:
    if not GUIDE_DIGEST_WRITER_ENABLED:
        return [dict(row) for row in rows]
    prepared = [dict(row) for row in rows]
    payload_rows: list[dict[str, Any]] = []
    for item in prepared:
        payload_rows.append(
            {
                "occurrence_id": int(item.get("id") or 0),
                "family": family,
                "title_seed": collapse_ws(str(item.get("canonical_title") or "")),
                "summary_seed": _non_logistics_seed(item),
                "fact_density": _fact_density_bucket(item),
                "target_blurb_sentences": _target_blurb_sentences(item),
                "shell_fields_present": _shell_fields_present(item),
                "fact_pack": _compact_fact_pack(item.get("fact_pack") or {}),
                "source_kind": collapse_ws(str(item.get("source_kind") or "")),
                "source_title": collapse_ws(str(item.get("source_title") or "")),
                "source_username": collapse_ws(str(item.get("source_username") or "")),
                "source_about_excerpt": collapse_ws(str(item.get("source_about_text") or ""))[:360],
                "guide_names": _string_list(item.get("guide_names"), limit=4),
                "audience_fit": _string_list(item.get("audience_fit"), limit=6),
                "main_hook": collapse_ws((item.get("fact_pack") or {}).get("main_hook")),
                "audience_region_fit_label": collapse_ws((item.get("fact_pack") or {}).get("audience_region_fit_label")),
                "guide_profile_summary": collapse_ws(str(item.get("guide_profile_summary") or "")),
                "guide_profile_facts": item.get("guide_profile_facts") or {},
                "organizer_names": _string_list(item.get("organizer_names"), limit=4),
                "booking_url": collapse_ws(str(item.get("booking_url") or "")),
                "booking_text_seed": collapse_ws(str(item.get("booking_line") or item.get("booking_text") or "")),
                "post_excerpt": collapse_ws(str(item.get("dedup_source_text") or ""))[:520],
            }
        )
    llm_out = await _ask_digest_batch_llm(payload_rows) if payload_rows else None
    out: list[dict[str, Any]] = []
    for item in prepared:
        occurrence_id = int(item.get("id") or 0)
        date_label = date_formatter(str(item.get("date") or ""), str(item.get("time") or ""))
        llm_row = llm_out.get(occurrence_id) if isinstance(llm_out, dict) else None
        if isinstance(llm_row, dict):
            lead_emoji = collapse_ws(str(llm_row.get("lead_emoji") or ""))
            title = collapse_ws(str(llm_row.get("title") or ""))
            blurb = collapse_ws(str(llm_row.get("digest_blurb") or ""))
            guide_line = collapse_ws(str(llm_row.get("guide_line") or ""))
            organizer_line = collapse_ws(str(llm_row.get("organizer_line") or ""))
            price_line = collapse_ws(str(llm_row.get("price_line") or ""))
            route_line = collapse_ws(str(llm_row.get("route_line") or ""))
            audience_region_line = collapse_ws(str(llm_row.get("audience_region_line") or ""))
            booking_line = collapse_ws(str(llm_row.get("booking_line") or ""))
            if _accept_lead_emoji_candidate(lead_emoji):
                item["lead_emoji"] = lead_emoji
            if _accept_title_candidate(title):
                item["canonical_title"] = repair_title_fallback(title, source_excerpt=item.get("dedup_source_text")) or title
            if _accept_blurb_candidate(blurb, item, date_label=date_label):
                item["digest_blurb"] = neutralize_relative_blurb(
                    blurb,
                    date_label=date_label,
                    time_text=collapse_ws(str(item.get("time") or "")) or None,
                )
            if _accept_guide_line_candidate(guide_line):
                item["guide_line"] = guide_line
                item.pop("organizer_line", None)
            elif _accept_organizer_line_candidate(organizer_line):
                item["organizer_line"] = organizer_line
            if _accept_price_line_candidate(price_line, item):
                item["price_line"] = price_line
            if _accept_route_line_candidate(route_line):
                item["route_line"] = route_line
            if _accept_audience_region_line_candidate(audience_region_line):
                item["audience_region_line"] = audience_region_line
            if _accept_booking_line_candidate(booking_line, item):
                item["booking_line"] = booking_line
        out.append(item)
    return out
