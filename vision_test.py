from __future__ import annotations

import asyncio
import base64
import html
import logging
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher, ndiff
from typing import Literal

from aiohttp import ClientError, ClientSession
from aiogram import Bot, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from cachetools import TTLCache


DetailLevel = Literal["auto", "low", "high"]


@dataclass
class VisionSession:
    detail: DetailLevel = "auto"
    waiting_for_photo: bool = True
    last_texts: dict[str, str] = field(default_factory=dict)


_SESSIONS: TTLCache[int, VisionSession] = TTLCache(maxsize=128, ttl=30 * 60)
_HTTP_SESSION: ClientSession | None = None
_HTTP_SEMAPHORE: asyncio.Semaphore | None = None

DETAIL_LABELS = {
    "auto": "Авто",
    "low": "Низкая",
    "high": "Высокая",
}

DETAIL_ORDER: tuple[DetailLevel, ...] = ("auto", "low", "high")
FOUR_O_TIMEOUT = float(os.getenv("FOUR_O_TIMEOUT", "60"))


def _ensure_session(user_id: int) -> VisionSession | None:
    session = _SESSIONS.get(user_id)
    if session:
        _SESSIONS[user_id] = session  # refresh TTL
    return session


def _main_keyboard(detail: DetailLevel) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Сменить детализацию ({DETAIL_LABELS.get(detail, detail)})",
                    callback_data="ocr:detail:menu",
                )
            ],
            [InlineKeyboardButton(text="Завершить", callback_data="ocr:detail:cancel")],
        ]
    )


def _detail_keyboard(current: DetailLevel) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=("✅ " if lvl == current else "▫️ ") + DETAIL_LABELS.get(lvl, lvl),
                callback_data=f"ocr:detail:{lvl}",
            )
        ]
        for lvl in DETAIL_ORDER
    ]
    rows.append([InlineKeyboardButton(text="← Назад", callback_data="ocr:detail:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def start(
    message: types.Message,
    bot: Bot,
    *,
    http_session: ClientSession,
    http_semaphore: asyncio.Semaphore,
) -> None:
    """Start OCR comparison session for the user."""

    global _HTTP_SESSION, _HTTP_SEMAPHORE
    _HTTP_SESSION = http_session
    _HTTP_SEMAPHORE = http_semaphore

    session = _ensure_session(message.from_user.id)
    if not session:
        session = VisionSession()
        _SESSIONS[message.from_user.id] = session
    session.waiting_for_photo = True
    session.last_texts.clear()

    text = (
        "Отправьте афишу или фото для распознавания.\n"
        f"Текущая детализация: {DETAIL_LABELS.get(session.detail, session.detail)}."
    )
    await bot.send_message(
        message.chat.id,
        text,
        reply_markup=_main_keyboard(session.detail),
    )


async def select_detail(callback: types.CallbackQuery, bot: Bot) -> None:
    """Handle detail selection callbacks."""

    session = _ensure_session(callback.from_user.id)
    if not session:
        await callback.answer("Сессия не найдена", show_alert=True)
        return

    data = callback.data or ""
    _, _, value = data.partition("ocr:detail:")
    if not value:
        await callback.answer("Неизвестная команда", show_alert=True)
        return

    if value == "menu":
        await callback.message.edit_reply_markup(_detail_keyboard(session.detail))
        await callback.answer("Выберите детализацию")
        return

    if value == "back":
        await callback.message.edit_reply_markup(_main_keyboard(session.detail))
        await callback.answer()
        return

    if value == "cancel":
        cancel(callback.from_user.id)
        await callback.message.edit_reply_markup(None)
        await callback.answer("Сессия завершена")
        return

    if value in DETAIL_ORDER:
        session.detail = value  # type: ignore[assignment]
        session.waiting_for_photo = True
        await callback.message.edit_reply_markup(_main_keyboard(session.detail))
        await callback.answer(f"Детализация: {DETAIL_LABELS.get(value, value)}")
        return

    await callback.answer("Неизвестная команда", show_alert=True)


def cancel(user_id: int) -> None:
    """Stop OCR session for the user."""

    if user_id in _SESSIONS:
        del _SESSIONS[user_id]


def is_waiting(user_id: int) -> bool:
    session = _ensure_session(user_id)
    return bool(session and session.waiting_for_photo)


async def run_ocr(image: bytes, *, model: str, detail: str) -> tuple[str, dict[str, int]]:
    """Send OCR request to OpenAI vision model."""

    if _HTTP_SESSION is None or _HTTP_SEMAPHORE is None:
        raise RuntimeError("HTTP resources are not configured for OCR")

    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")

    url = os.getenv("FOUR_O_URL", "https://api.openai.com/v1/chat/completions")
    encoded = base64.b64encode(image).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "верни только распознанный текст"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Распознай текст на изображении."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async def _call() -> dict:
        async with _HTTP_SEMAPHORE:
            async with _HTTP_SESSION.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json()

    try:
        data = await asyncio.wait_for(_call(), FOUR_O_TIMEOUT)
    except (asyncio.TimeoutError, ClientError) as e:  # pragma: no cover - network errors
        logging.error("OCR request failed: model=%s detail=%s error=%s", model, detail, e)
        raise RuntimeError(f"OCR request failed: {e}") from e

    try:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = (message.get("content") or "").strip()
        usage = data.get("usage", {}) or {}
    except (AttributeError, IndexError, TypeError) as e:  # pragma: no cover - unexpected
        logging.error("Invalid OCR response: data=%s", data)
        raise RuntimeError("Incomplete OCR response") from e

    if not text:
        raise RuntimeError("Empty OCR response")

    tokens = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    return text, tokens


async def handle_photo(
    message: types.Message,
    bot: Bot,
    images: list[tuple[bytes, str]],
) -> None:
    """Process incoming photo for OCR comparison."""

    session = _ensure_session(message.from_user.id)
    if not session:
        await bot.send_message(message.chat.id, "Сессия не найдена")
        return

    if not images:
        await bot.send_message(
            message.chat.id,
            "Не удалось получить изображение.",
            reply_markup=_main_keyboard(session.detail),
        )
        return

    session.waiting_for_photo = False
    image_bytes, name = images[0]
    models = ("gpt-4o-mini", "gpt-4o")
    results: list[dict[str, object]] = []
    for model in models:
        try:
            text, tokens = await run_ocr(image_bytes, model=model, detail=session.detail)
            session.last_texts[model] = text
            results.append({"model": model, "text": text, "tokens": tokens, "error": None})
        except Exception as exc:  # pragma: no cover - depends on network
            logging.warning("OCR model failed: model=%s error=%s", model, exc)
            results.append({"model": model, "text": "", "tokens": {}, "error": str(exc)})

    lines = [f"Файл: {name}", f"Детализация: {DETAIL_LABELS.get(session.detail, session.detail)}"]
    for item in results:
        model = item["model"]
        text = html.escape(str(item["text"] or ""))
        error = item.get("error")
        lines.append("")
        lines.append(f"<b>{model}</b>:")
        if error:
            lines.append(f"<i>Ошибка: {html.escape(str(error))}</i>")
        else:
            lines.append(f"<pre>{text}</pre>")

    lines.append("")
    lines.append("<b>Токены</b>:")
    header = f"{'Модель':<12} {'prompt':>7} {'completion':>10} {'total':>7}"
    lines.append(f"<pre>{html.escape(header)}")
    for item in results:
        tokens = item.get("tokens") or {}
        if tokens:
            row = (
                f"{item['model']:<12} "
                f"{tokens.get('prompt_tokens', 0):>7} "
                f"{tokens.get('completion_tokens', 0):>10} "
                f"{tokens.get('total_tokens', 0):>7}"
            )
        else:
            row = f"{item['model']:<12} {'-':>7} {'-':>10} {'-':>7}"
        lines.append(html.escape(row))
    lines.append("</pre>")

    success = [item for item in results if not item.get("error")]
    if len(success) == 2:
        text_a = success[0]["text"] or ""
        text_b = success[1]["text"] or ""
        ratio = SequenceMatcher(None, text_a, text_b).ratio()
        lines.append("")
        lines.append(f"Схожесть: {ratio:.3f}")
        diff_lines = [line for line in ndiff(str(text_a).splitlines(), str(text_b).splitlines()) if line[:1] in {"-", "+"}]
        if diff_lines:
            lines.append("Различия (первые 10 строк):")
            for line in diff_lines[:10]:
                lines.append(html.escape(line))
    else:
        errors = ", ".join(str(item.get("error")) for item in results if item.get("error"))
        lines.append("")
        lines.append(f"Сравнение недоступно: {html.escape(errors)}")

    session.waiting_for_photo = True
    text = "\n".join(lines)
    await bot.send_message(
        message.chat.id,
        text,
        parse_mode="HTML",
        reply_markup=_main_keyboard(session.detail),
    )


__all__ = [
    "start",
    "select_detail",
    "cancel",
    "is_waiting",
    "handle_photo",
    "run_ocr",
]
