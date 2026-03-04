from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

from aiogram import F, Router, types
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import Command, CommandObject

from runtime import require_main_attr

logger = logging.getLogger(__name__)

admin_assist_router = Router(name="admin_assist")

_ASSIST_TTL_SEC = 15 * 60
_ASSIST_MAX_ARGS_LEN = 800
_ASSIST_MAX_REQUEST_LEN = 1200
_ASSIST_MAX_PROPOSALS = 3


@dataclass
class _AssistSession:
    mode: Literal["awaiting_request", "awaiting_clarify_answer"]
    request_text: str | None = None
    question: str | None = None


@dataclass(frozen=True)
class _ActionProposal:
    action_id: str
    args: dict[str, Any]
    confidence: float = 0.0


@dataclass
class _PendingAction:
    token: str
    user_id: int
    chat_id: int
    request_text: str
    proposals: list[_ActionProposal]
    selected_index: int = 0
    created_at: float = 0.0
    clarify_question: str = ""
    clarify_options: list[dict[str, str]] | None = None


_assist_sessions: dict[int, _AssistSession] = {}
_pending_actions: dict[str, _PendingAction] = {}


def _now_ts() -> float:
    return time.time()


def _is_forwarded(message: types.Message) -> bool:
    return bool(message.forward_date) or "forward_origin" in getattr(message, "model_extra", {})


def _cleanup_pending() -> None:
    now = _now_ts()
    stale = [k for k, v in _pending_actions.items() if now - (v.created_at or 0.0) > _ASSIST_TTL_SEC]
    for key in stale:
        _pending_actions.pop(key, None)


def _trim_one_line(value: str, *, max_len: int) -> str:
    cleaned = (value or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _norm_intent_text(value: str) -> str:
    cleaned = (value or "").casefold().strip()
    cleaned = cleaned.replace("ё", "е")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _sanitize_args_text(value: str) -> str:
    raw = value or ""
    if "\n" in raw or "\r" in raw:
        raise ValueError("args_text must be single-line")
    cleaned = _trim_one_line(raw, max_len=_ASSIST_MAX_ARGS_LEN)
    if cleaned.startswith("/"):
        raise ValueError("args_text must not start with '/'")
    return cleaned


def _parse_iso_date(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    if "T" in cleaned or " " in cleaned:
        raise ValueError(f"invalid iso date: {cleaned}")
    try:
        # Raises ValueError on invalid dates
        datetime.strptime(cleaned, "%Y-%m-%d").date()
    except Exception as exc:
        raise ValueError(f"invalid iso date: {cleaned}") from exc
    return cleaned


_TZ_OFFSET_RE = re.compile(r"^[+-]\d{2}:\d{2}$")
_TIME_HHMM_RE = re.compile(r"^\d{2}:\d{2}$")


def _parse_tz_offset(value: str) -> str:
    cleaned = (value or "").strip()
    if not _TZ_OFFSET_RE.match(cleaned):
        raise ValueError("invalid tz offset, expected ±HH:MM")
    return cleaned


def _parse_time_hhmm(value: str) -> str:
    cleaned = (value or "").strip()
    if not _TIME_HHMM_RE.match(cleaned):
        raise ValueError("invalid time, expected HH:MM")
    hh, mm = cleaned.split(":", 1)
    if int(hh) > 23 or int(mm) > 59:
        raise ValueError("invalid time, expected HH:MM")
    return cleaned


def _allowed_actions() -> dict[str, dict[str, Any]]:
    vk_miss_cmd = str(require_main_attr("VK_MISS_REVIEW_COMMAND") or "/vk_misses").strip() or "/vk_misses"
    # Keep the allowlist small and explicit. Add new actions here.
    return {
        "help": {
            "command": "/help",
            "risk": "safe",
            "desc": "Показать список команд по роли.",
            "args_schema": {},
            "examples": ["покажи команды", "что умеет бот"],
        },
        "menu": {
            "command": "/menu",
            "risk": "safe",
            "desc": "Показать главное меню с кнопками.",
            "args_schema": {},
            "examples": ["покажи меню", "кнопки"],
        },
        "events": {
            "command": "/events",
            "risk": "safe",
            "desc": "Показать события на дату (по умолчанию сегодня).",
            "args_schema": {"date": {"type": "date_iso", "required": False}},
            "examples": ["покажи события на завтра", "события на 2026-02-24"],
        },
        "exhibitions": {
            "command": "/exhibitions",
            "risk": "safe",
            "desc": "Показать активные выставки.",
            "args_schema": {},
            "examples": ["покажи выставки", "активные выставки"],
        },
        "pages": {
            "command": "/pages",
            "risk": "safe",
            "desc": "Показать Telegraph страницы месяца/выходных.",
            "args_schema": {},
            "examples": ["покажи страницы", "страницы телеграф"],
        },
        "pages_rebuild": {
            "command": "/pages_rebuild",
            "risk": "mutating",
            "desc": "Пересобрать страницы месяца/выходных (ручной запуск).",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["пересобери страницы", "pages_rebuild 2026-02 --future=2"],
        },
        "parse": {
            "command": "/parse",
            "risk": "mutating",
            "desc": "Запуск парсинга внешних источников (театры и др.).",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["запусти /parse", "parse check"],
        },
        "vk": {
            "command": "/vk",
            "risk": "mutating",
            "desc": "Открыть UI очереди VK (источники/проверка/импорт).",
            "args_schema": {},
            "examples": ["открой VK", "vk меню"],
        },
        "vk_queue": {
            "command": "/vk_queue",
            "risk": "safe",
            "desc": "Показать сводку очереди VK и кнопку проверки.",
            "args_schema": {},
            "examples": ["сводка очереди VK", "очередь vk"],
        },
        "vk_auto_import": {
            "command": "/vk_auto_import",
            "risk": "mutating",
            "desc": "Авторазбор очереди VK через Smart Update.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["vk_auto_import 10", "импортни VK все"],
        },
        "vk_auto_import_stop": {
            "command": "/vk_auto_import_stop",
            "risk": "mutating",
            "desc": "Остановить текущий прогон vk_auto_import (после текущего поста).",
            "args_schema": {},
            "examples": ["останови vk_auto_import", "vk_auto_import stop"],
        },
        "vk_crawl_now": {
            "command": "/vk_crawl_now",
            "risk": "mutating",
            "desc": "Запустить VK crawler сейчас.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["vk_crawl_now", "vk_crawl_now --backfill-days=7"],
        },
        "vk_misses": {
            "command": vk_miss_cmd,
            "risk": "mutating",
            "desc": "Открыть поток разбора пропусков VK (miss-review).",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["покажи пропуски vk", "vk_misses 20"],
        },
        "fest_queue": {
            "command": "/fest_queue",
            "risk": "mutating",
            "desc": "Фестивальная очередь: статус или запуск обработки.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["fest_queue -i", "запусти fest_queue"],
        },
        "ticket_sites_queue": {
            "command": "/ticket_sites_queue",
            "risk": "mutating",
            "desc": "Очередь ticket-сайтов: статус или запуск обработки.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["ticket_sites_queue -i", "ticket_sites_queue --limit=10"],
        },
        "tg": {
            "command": "/tg",
            "risk": "mutating",
            "desc": "UI управления Telegram Monitoring.",
            "args_schema": {},
            "examples": ["открой telegram monitoring", "tg мониторинг"],
        },
        "stats": {
            "command": "/stats",
            "risk": "safe",
            "desc": "Статистика Telegraph и shortlinks (просмотры/клики).",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": [
                "статистика telegraph",
                "просмотры страниц",
                "клики по vk.cc",
                "stats events",
            ],
        },
        "telegraph_cache_stats": {
            "command": "/telegraph_cache_stats",
            "risk": "safe",
            "desc": "Статистика Telegram web preview для Telegraph страниц (cached_page/photo): сколько страниц «в кэше», сколько битых.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": [
                "статистика кэша телеграф",
                "сколько битых страниц телеграф",
                "telegraph cache stats",
                "telegraph_cache_stats event",
            ],
        },
        "telegraph_cache_sanitize": {
            "command": "/telegraph_cache_sanitize",
            "risk": "mutating",
            "desc": "Запустить санитайзер/прогрев Telegram кэша для Telegraph страниц (Kaggle/Telethon) и поставить битые в очередь на пересборку.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": [
                "прогрей телеграф страницы",
                "санитайзер телеграф кэша",
                "проверь cached_page для телеграф",
                "telegraph_cache_sanitize --limit=100",
                "telegraph_cache_sanitize --no-enqueue",
            ],
        },
        "general_stats": {
            "command": "/general_stats",
            "risk": "safe",
            "desc": "Общий суточный отчёт: авторазбор VK/TG, мониторинг и LLM (Gemma) метрики.",
            "args_schema": {},
            "examples": [
                "общая статистика по авторазбору вк и телеграм и гемма",
                "общая статистика за сутки",
                "суточный отчет по системе",
                "general_stats",
            ],
        },
        "popular_posts": {
            "command": "/popular_posts",
            "risk": "safe",
            "desc": "ТОП популярных постов (TG/VK), которые создали события: выше медианы внутри канала за 3 суток и за 24 часа.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": [
                "покажи топ популярных постов",
                "статистика популярных постов по медиане",
                "покажи посты выше медианы по лайкам и просмотрам",
                "popular_posts",
                "popular_posts 20",
            ],
        },
        "status": {
            "command": "/status",
            "risk": "safe",
            "desc": "Uptime и статус джобов.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["status", "status 123"],
        },
        "trace": {
            "command": "/trace",
            "risk": "safe",
            "desc": "Показать trace по run_id.",
            "args_schema": {"args_text": {"type": "args_text", "required": True}},
            "examples": ["trace 2026-02-23T12:00:00Z"],
        },
        "last_errors": {
            "command": "/last_errors",
            "risk": "safe",
            "desc": "Показать последние ошибки.",
            "args_schema": {"args_text": {"type": "args_text", "required": False}},
            "examples": ["last_errors", "last_errors 20"],
        },
        "tz": {
            "command": "/tz",
            "risk": "mutating",
            "desc": "Установить смещение таймзоны для форматирования дат/времени.",
            "args_schema": {"offset": {"type": "tz_offset", "required": True}},
            "examples": ["поставь таймзону +02:00", "tz +02:00"],
        },
        "vkgroup": {
            "command": "/vkgroup",
            "risk": "mutating",
            "desc": "Установить ID группы VK или выключить (off).",
            "args_schema": {"value": {"type": "vkgroup_value", "required": True}},
            "examples": ["выключи vkgroup", "vkgroup 12345"],
        },
        "vktime": {
            "command": "/vktime",
            "risk": "mutating",
            "desc": "Настроить время VK публикаций (today|added HH:MM).",
            "args_schema": {
                "kind": {"type": "enum(today|added)", "required": True},
                "time": {"type": "time_hhmm", "required": True},
            },
            "examples": ["поставь время vk today 08:00", "vktime added 20:00"],
        },
        "vkphotos": {
            "command": "/vkphotos",
            "risk": "mutating",
            "desc": "Вкл/выкл отправку фото в VK постах.",
            "args_schema": {},
            "examples": ["включи vkphotos", "vkphotos"],
        },
        "images": {
            "command": "/images",
            "risk": "mutating",
            "desc": "Вкл/выкл загрузку фото в Catbox.",
            "args_schema": {},
            "examples": ["переключи загрузку картинок", "images"],
        },
        "ik_poster": {
            "command": "/ik_poster",
            "risk": "mutating",
            "desc": "ImageKit обработка афиш (Smart crop / GenFill).",
            "args_schema": {},
            "examples": ["обработай афишу imagekit", "ik_poster"],
        },
    }


def _build_command_text(action_id: str, args: dict[str, Any]) -> str:
    actions = _allowed_actions()
    action = actions.get(action_id)
    if not action:
        raise ValueError(f"unknown action_id: {action_id}")
    cmd = str(action.get("command") or "").strip()
    if not cmd:
        raise ValueError(f"empty command for action_id: {action_id}")

    if action_id == "events":
        date = _parse_iso_date(str((args or {}).get("date") or "").strip())
        return f"{cmd} {date}".strip()

    if action_id == "tz":
        offset = _parse_tz_offset(str((args or {}).get("offset") or ""))
        return f"{cmd} {offset}".strip()

    if action_id == "vkgroup":
        value = str((args or {}).get("value") or "").strip().lower()
        if not value:
            raise ValueError("vkgroup value is required")
        if value != "off":
            if not value.isdigit():
                raise ValueError("vkgroup value must be digits or 'off'")
        return f"{cmd} {value}".strip()

    if action_id == "vktime":
        kind = str((args or {}).get("kind") or "").strip().lower()
        if kind not in {"today", "added"}:
            raise ValueError("vktime kind must be today|added")
        tm = _parse_time_hhmm(str((args or {}).get("time") or ""))
        return f"{cmd} {kind} {tm}".strip()

    args_text = str((args or {}).get("args_text") or "").strip()
    if action_id == "trace" and not args_text:
        raise ValueError("trace args_text is required")
    if args_text:
        args_text = _sanitize_args_text(args_text)
        return f"{cmd} {args_text}".strip()
    return cmd


def _risk_label(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "safe":
        return "safe"
    if value == "mutating":
        return "mutating"
    if value == "dangerous":
        return "dangerous"
    return "mutating"


def _risk_human(value: str) -> str:
    value = _risk_label(value)
    if value == "safe":
        return "🟢 безопасно"
    if value == "dangerous":
        return "🔴 опасно"
    return "🟡 изменяет состояние"


def _summarize_action(action_id: str, command_text: str) -> str:
    actions = _allowed_actions()
    action = actions.get(action_id) or {}
    desc = str(action.get("desc") or "").strip()
    if desc:
        return f"{desc} ({command_text})"
    return command_text


def _heuristic_proposals(request_text: str) -> list[_ActionProposal] | None:
    """Return proposals without calling LLM for very common intents."""
    t = _norm_intent_text(request_text)
    if not t:
        return None

    has_stats_word = any(word in t for word in ("статист", "отчет", "отчёт", "сводк", "репорт"))
    if not has_stats_word:
        return None

    is_popular_posts_stats = (
        any(word in t for word in ("популяр", "топ"))
        and any(word in t for word in ("пост", "публикац", "лайк", "просмотр"))
    )
    if is_popular_posts_stats:
        return [_ActionProposal(action_id="popular_posts", args={}, confidence=0.95)]

    # Disambiguation: /general_stats vs /stats
    if "general_stats" in t:
        return [_ActionProposal(action_id="general_stats", args={}, confidence=1.0)]

    is_telegraph_stats = any(word in t for word in ("telegraph", "телеграф", "vk.cc", "vkcc", "shortlink", "шортлинк", "просмотр", "клик"))
    if is_telegraph_stats:
        return [_ActionProposal(action_id="stats", args={"args_text": "events"}, confidence=0.95)]

    is_system_stats = (
        "общая статист" in t
        or "общий отчет" in t
        or "общий отчёт" in t
        or "суточ" in t
        or "за сутки" in t
        or "24 часа" in t
        or "за 24" in t
        or any(word in t for word in ("авторазбор", "auto import", "vk_auto_import", "vk auto import", "мониторинг", "telegram monitoring", "гемма", "gemma", "llm", "лимит", "токен"))
    )
    if is_system_stats:
        return [_ActionProposal(action_id="general_stats", args={}, confidence=0.9)]

    # Generic "статистика" is ambiguous: offer both.
    return [
        _ActionProposal(action_id="general_stats", args={}, confidence=0.55),
        _ActionProposal(action_id="stats", args={"args_text": "events"}, confidence=0.45),
    ]


@lru_cache(maxsize=1)
def _read_prompt_base() -> str:
    path = os.path.join("docs", "llm", "admin-actions.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return (
            "You route an admin request to an existing bot command.\n"
            "Return JSON only.\n"
        )


def _build_prompt(request_text: str) -> str:
    today = datetime.now(require_main_attr("LOCAL_TZ")).date().isoformat()
    actions = _allowed_actions()
    payload = []
    for action_id, meta in actions.items():
        payload.append(
            {
                "action_id": action_id,
                "command": meta.get("command"),
                "risk": _risk_label(str(meta.get("risk") or "")),
                "desc": meta.get("desc"),
                "args_schema": meta.get("args_schema") or {},
                "examples": meta.get("examples") or [],
            }
        )
    schema = {
        "kind": "proposals|clarify|unknown",
        "proposals": [
            {"action_id": "string", "args": {"...": "..."}, "confidence": 0.0}
        ],
        "question": "string (only for kind=clarify)",
        "options": [{"label": "string", "add_to_request": "string"}],
    }
    base = _read_prompt_base()
    cleaned_request = _trim_one_line(request_text, max_len=_ASSIST_MAX_REQUEST_LEN)
    return (
        f"{base}\n\n"
        "Output MUST be a single JSON object.\n"
        "Do NOT wrap in code fences.\n"
        "Pick ONLY from action_id in ALLOWED_ACTIONS.\n"
        "If the request is ambiguous or missing required info, use kind=clarify.\n"
        f"Today is {today}.\n\n"
        f"OUTPUT_SCHEMA={json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}\n"
        f"ALLOWED_ACTIONS={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"REQUEST={json.dumps(cleaned_request, ensure_ascii=False)}\n"
    )


@lru_cache(maxsize=1)
def _get_gemma_client() -> Any:
    from google_ai import GoogleAIClient, SecretsProvider

    supabase = require_main_attr("get_supabase_client")()
    incident_notifier = require_main_attr("notify_llm_incident")
    return GoogleAIClient(
        supabase_client=supabase,
        secrets_provider=SecretsProvider(),
        consumer="admin_assist",
        incident_notifier=incident_notifier,
    )


def _extract_json(text: str) -> Any | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            return json.loads(cleaned[obj_start : obj_end + 1])
        except Exception:
            return None
    return None


async def _llm_plan(request_text: str) -> dict[str, Any] | None:
    model = (os.getenv("ADMIN_ASSISTANT_MODEL") or "gemma-3-27b").strip() or "gemma-3-27b"
    client = _get_gemma_client()
    prompt = _build_prompt(request_text)
    raw, _usage = await client.generate_content_async(
        model=model,
        prompt=prompt,
        generation_config={"temperature": 0},
        max_output_tokens=1024,
    )
    return _extract_json(raw)


def _validate_plan(data: Any) -> tuple[str, list[_ActionProposal], str, list[dict[str, str]]]:
    if not isinstance(data, dict):
        return "unknown", [], "", []
    kind = str(data.get("kind") or "").strip().lower()
    if kind not in {"proposals", "clarify", "unknown"}:
        kind = "unknown"
    question = _trim_one_line(str(data.get("question") or ""), max_len=240)
    options_raw = data.get("options") or []
    options: list[dict[str, str]] = []
    if isinstance(options_raw, list):
        for opt in options_raw[:4]:
            if not isinstance(opt, dict):
                continue
            label = _trim_one_line(str(opt.get("label") or ""), max_len=60)
            add = _trim_one_line(str(opt.get("add_to_request") or ""), max_len=120)
            if label and add:
                options.append({"label": label, "add_to_request": add})

    proposals: list[_ActionProposal] = []
    props_raw = data.get("proposals") or []
    if isinstance(props_raw, list):
        allowed = _allowed_actions()
        for item in props_raw[: _ASSIST_MAX_PROPOSALS]:
            if not isinstance(item, dict):
                continue
            action_id = str(item.get("action_id") or "").strip()
            if action_id not in allowed:
                continue
            args = item.get("args")
            if not isinstance(args, dict):
                args = {}
            try:
                # Validate buildability.
                _build_command_text(action_id, args)
            except Exception:
                continue
            conf = 0.0
            try:
                conf = float(item.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            conf = max(0.0, min(conf, 1.0))
            proposals.append(_ActionProposal(action_id=action_id, args=args, confidence=conf))

    if proposals:
        return "proposals", proposals, "", []
    if kind == "clarify" and question:
        return "clarify", [], question, options
    return "unknown", [], "", []


async def _require_superadmin(message: types.Message) -> bool:
    get_db = require_main_attr("get_db")
    db = get_db()
    if db is None:
        await message.answer("❌ Бот ещё не инициализирован. Попробуйте позже.")
        return False
    from models import User

    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not require_main_attr("has_admin_access")(user):
        await message.answer("⛔ Команда доступна только администраторам.")
        return False
    return True


async def start_admin_assist_interactive(message: types.Message) -> None:
    if not await _require_superadmin(message):
        return
    _assist_sessions[message.from_user.id] = _AssistSession(mode="awaiting_request")
    await message.answer(
        "🧠 Опишите действие простыми словами.\n"
        "Примеры:\n"
        "• «Покажи события на завтра»\n"
        "• «Открой очередь VK»\n"
        "• «Запусти vk_auto_import на 10 постов»\n\n"
        "Я предложу команду и спрошу подтверждение.\n"
        "Отмена: /assist_cancel",
    )


@admin_assist_router.message(Command("assist", "assistant", "a"))
async def cmd_admin_assist(message: types.Message, command: CommandObject) -> None:
    if not await _require_superadmin(message):
        return
    args = (command.args or "").strip()
    if not args:
        await start_admin_assist_interactive(message)
        return
    await _process_assist_request(message, args)


@admin_assist_router.message(Command("assist_cancel"))
async def cmd_admin_assist_cancel(message: types.Message) -> None:
    if message.from_user:
        _assist_sessions.pop(message.from_user.id, None)
    await message.answer("Ок, отменил.")


def _is_assist_session_input(message: types.Message) -> bool:
    if not message.from_user:
        return False
    session = _assist_sessions.get(message.from_user.id)
    if session is None:
        return False
    # Forwarded posts must keep working (Smart Update path). If an admin forwards a post
    # while /assist is waiting, we drop the assist session and let the regular forward
    # handler process the message.
    if _is_forwarded(message):
        return True
    text = (message.text or "").strip()
    if not text:
        return False
    # Don't swallow other commands.
    if text.startswith("/"):
        return False
    return True


@admin_assist_router.message(_is_assist_session_input)
async def handle_assist_session_message(message: types.Message) -> None:
    if _is_forwarded(message):
        _assist_sessions.pop(message.from_user.id, None)
        raise SkipHandler

    session = _assist_sessions.get(message.from_user.id)
    if not session:
        return
    text = (message.text or "").strip()
    text = _trim_one_line(text, max_len=_ASSIST_MAX_REQUEST_LEN)
    if not text:
        return

    if session.mode == "awaiting_request":
        _assist_sessions.pop(message.from_user.id, None)
        await _process_assist_request(message, text)
        return

    if session.mode == "awaiting_clarify_answer":
        base = _trim_one_line(session.request_text or "", max_len=_ASSIST_MAX_REQUEST_LEN)
        q = _trim_one_line(session.question or "", max_len=240)
        _assist_sessions.pop(message.from_user.id, None)
        merged = f"{base}\n\nCLARIFICATION_QUESTION={q}\nCLARIFICATION_ANSWER={text}".strip()
        await _process_assist_request(message, merged)
        return


async def _process_assist_request(message: types.Message, request_text: str) -> None:
    _cleanup_pending()
    req = _trim_one_line(request_text, max_len=_ASSIST_MAX_REQUEST_LEN)
    if not req:
        await message.answer("❌ Пустой запрос. Напишите, что сделать.")
        return

    await message.answer("⏳ Думаю…")
    heuristic = _heuristic_proposals(req)
    if heuristic:
        token = uuid.uuid4().hex[:12]
        pending = _PendingAction(
            token=token,
            user_id=message.from_user.id,
            chat_id=message.chat.id,
            request_text=req,
            proposals=heuristic,
            selected_index=0,
            created_at=_now_ts(),
        )
        _pending_actions[token] = pending
        await _send_pending_preview(message, pending)
        return
    try:
        plan = await _llm_plan(req)
    except Exception as exc:
        logger.warning("admin_assist: llm failed err=%s", exc, exc_info=True)
        await message.answer("❌ LLM сейчас недоступна. Попробуйте позже или используйте /help.")
        return

    kind, proposals, question, options = _validate_plan(plan)
    if kind == "clarify":
        if options:
            token = uuid.uuid4().hex[:12]
            _pending_actions[token] = _PendingAction(
                token=token,
                user_id=message.from_user.id,
                chat_id=message.chat.id,
                request_text=req,
                proposals=[],
                selected_index=0,
                created_at=_now_ts(),
                clarify_question=question or "Уточните запрос:",
                clarify_options=options,
            )
            kb = [
                [types.InlineKeyboardButton(text=opt["label"], callback_data=f"aa:clar:{token}:{idx}")]
                for idx, opt in enumerate(options)
            ]
            kb.append([types.InlineKeyboardButton(text="❌ Отмена", callback_data=f"aa:reject:{token}")])
            await message.answer(question or "Уточните запрос:", reply_markup=types.InlineKeyboardMarkup(inline_keyboard=kb))
            return
        _assist_sessions[message.from_user.id] = _AssistSession(
            mode="awaiting_clarify_answer",
            request_text=req,
            question=question,
        )
        await message.answer((question or "Нужно уточнение.") + "\n\nОтветьте сообщением. Отмена: /assist_cancel")
        return

    if kind != "proposals" or not proposals:
        await message.answer("🤷 Не понял, какую команду выбрать. Попробуйте переформулировать или используйте /help.")
        return

    token = uuid.uuid4().hex[:12]
    pending = _PendingAction(
        token=token,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        request_text=req,
        proposals=proposals,
        selected_index=0,
        created_at=_now_ts(),
    )
    _pending_actions[token] = pending
    await _send_pending_preview(message, pending)


def _render_pending_preview(pending: _PendingAction) -> tuple[str, types.InlineKeyboardMarkup]:
    actions = _allowed_actions()
    if pending.proposals:
        proposal = pending.proposals[pending.selected_index]
        meta = actions.get(proposal.action_id) or {}
        risk = _risk_human(str(meta.get("risk") or "mutating"))
        command_text = _build_command_text(proposal.action_id, proposal.args)
        summary = _summarize_action(proposal.action_id, command_text)

        lines = [
            "🧠 Нашёл, как сделать:",
            f"Запрос: {pending.request_text}",
            f"Действие: {summary}",
            f"Риск: {risk}",
            f"Команда: {command_text}",
            "",
            "Подтвердить выполнение?",
        ]
        text = "\n".join(lines)

        buttons: list[list[types.InlineKeyboardButton]] = [
            [
                types.InlineKeyboardButton(
                    text="✅ Подтвердить", callback_data=f"aa:confirm:{pending.token}"
                ),
                types.InlineKeyboardButton(
                    text="❌ Отклонить", callback_data=f"aa:reject:{pending.token}"
                ),
            ]
        ]
        if len(pending.proposals) > 1:
            for idx, p in enumerate(pending.proposals):
                try:
                    cmd = _build_command_text(p.action_id, p.args)
                except Exception:
                    cmd = p.action_id
                prefix = "✅ " if idx == pending.selected_index else ""
                label = _trim_one_line(f"{prefix}Вариант {idx + 1}: {cmd}", max_len=60)
                buttons.append(
                    [
                        types.InlineKeyboardButton(
                            text=label, callback_data=f"aa:pick:{pending.token}:{idx}"
                        )
                    ]
                )
        return text, types.InlineKeyboardMarkup(inline_keyboard=buttons)

    # Clarify mode (options)
    options = pending.clarify_options or []
    question = pending.clarify_question or "Уточните запрос:"
    kb = [
        [
            types.InlineKeyboardButton(
                text=opt["label"], callback_data=f"aa:clar:{pending.token}:{idx}"
            )
        ]
        for idx, opt in enumerate(options)
    ]
    kb.append([types.InlineKeyboardButton(text="❌ Отмена", callback_data=f"aa:reject:{pending.token}")])
    return question, types.InlineKeyboardMarkup(inline_keyboard=kb)


async def _send_pending_preview(message: types.Message, pending: _PendingAction) -> None:
    text, markup = _render_pending_preview(pending)
    await message.answer(text, reply_markup=markup)


def _make_raw_update(*, user: types.User, chat: types.Chat, text: str) -> dict[str, Any]:
    now = int(time.time())
    update_id = now + int(uuid.uuid4().int % 1000)
    message_id = now + int(uuid.uuid4().int % 1000)
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": now,
            "chat": {"id": int(chat.id), "type": str(chat.type)},
            "from": {
                "id": int(user.id),
                "is_bot": False,
                "first_name": user.first_name or "Admin",
                "username": user.username,
            },
            "text": text,
        },
    }


@admin_assist_router.callback_query(F.data.startswith("aa:"))
async def handle_assist_callback(callback: types.CallbackQuery) -> None:
    _cleanup_pending()
    data = callback.data or ""
    parts = data.split(":", 3)
    if len(parts) < 3:
        await callback.answer()
        return
    action = parts[1]
    token = parts[2]
    pending = _pending_actions.get(token)
    if pending is None:
        await callback.answer("Сессия устарела", show_alert=False)
        return
    if int(callback.from_user.id) != int(pending.user_id):
        await callback.answer("Не ваша сессия", show_alert=True)
        return

    if action == "reject":
        _pending_actions.pop(token, None)
        if callback.message:
            try:
                await callback.message.edit_text("❌ Отклонено.")
            except Exception:
                pass
        await callback.answer("Ок")
        return

    if action == "pick":
        if len(parts) != 4:
            await callback.answer()
            return
        try:
            idx = int(parts[3])
        except Exception:
            await callback.answer()
            return
        if not (0 <= idx < len(pending.proposals)):
            await callback.answer("Нет такого варианта", show_alert=False)
            return
        pending.selected_index = idx
        if callback.message:
            try:
                text, markup = _render_pending_preview(pending)
                await callback.message.edit_text(text, reply_markup=markup)
            except Exception:
                pass
        await callback.answer("Выбрал вариант")
        return

    if action == "clar":
        if len(parts) != 4:
            await callback.answer()
            return
        try:
            idx = int(parts[3])
        except Exception:
            await callback.answer()
            return
        options = pending.clarify_options or []
        if not (0 <= idx < len(options)):
            await callback.answer("Список устарел", show_alert=False)
            return
        add = options[idx].get("add_to_request") or ""
        merged = f"{pending.request_text}\n\nCLARIFY_PICK={add}".strip()
        if callback.message:
            try:
                await callback.message.edit_text("⏳ Думаю…")
            except Exception:
                pass
        try:
            plan = await _llm_plan(merged)
        except Exception as exc:
            logger.warning("admin_assist: llm clarify failed err=%s", exc, exc_info=True)
            if callback.message:
                try:
                    text, markup = _render_pending_preview(pending)
                    await callback.message.edit_text(
                        text + "\n\n❌ LLM недоступна, попробуйте позже.",
                        reply_markup=markup,
                    )
                except Exception:
                    pass
            await callback.answer("Ошибка LLM", show_alert=False)
            return
        kind2, proposals2, question2, options2 = _validate_plan(plan)
        if kind2 == "proposals" and proposals2:
            pending.proposals = proposals2
            pending.selected_index = 0
            pending.clarify_options = None
            pending.clarify_question = ""
        elif kind2 == "clarify" and question2:
            pending.proposals = []
            pending.selected_index = 0
            pending.clarify_question = question2
            pending.clarify_options = options2 or None
        else:
            pending.proposals = []
            pending.selected_index = 0
            pending.clarify_question = "🤷 Не понял уточнение. Попробуйте иначе или используйте /help."
            pending.clarify_options = None
        if callback.message:
            try:
                text, markup = _render_pending_preview(pending)
                await callback.message.edit_text(text, reply_markup=markup)
            except Exception:
                pass
        await callback.answer("Ок")
        return

    if action == "confirm":
        if not pending.proposals:
            await callback.answer("Нечего выполнять", show_alert=True)
            return
        proposal = pending.proposals[pending.selected_index]
        command_text = _build_command_text(proposal.action_id, proposal.args)
        if not callback.message:
            await callback.answer("Нет сообщения для выполнения", show_alert=True)
            return
        if callback.message:
            try:
                await callback.message.edit_text(f"▶️ Выполняю: {command_text}")
            except Exception:
                pass
        await callback.answer("Запускаю…")
        try:
            dp = require_main_attr("get_dispatcher")()
            if dp is None:
                raise RuntimeError("dispatcher is None")
            update = _make_raw_update(user=callback.from_user, chat=callback.message.chat, text=command_text)
            await dp.feed_raw_update(callback.bot, update)
            _pending_actions.pop(token, None)
            if callback.message:
                try:
                    await callback.message.edit_text(f"✅ Выполнено: {command_text}")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("admin_assist: exec failed err=%s", exc, exc_info=True)
            if callback.message:
                try:
                    await callback.message.edit_text(f"❌ Ошибка при выполнении: {exc}")
                except Exception:
                    pass
        return

    await callback.answer()
