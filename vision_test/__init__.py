from __future__ import annotations

import asyncio
import html
import logging
from difflib import SequenceMatcher, ndiff

from aiohttp import ClientSession
from aiogram import Bot, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from . import session as session_store
from .session import DetailLevel
from .ocr import OcrResult, configure_http, run_ocr

DETAIL_LABELS = {
    "auto": "Авто",
    "low": "Низкая",
    "high": "Высокая",
}

DETAIL_ORDER: tuple[DetailLevel, ...] = ("auto", "low", "high")

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

    configure_http(session=http_session, semaphore=http_semaphore)

    session = session_store.start_session(message.from_user.id)
    session.waiting_for_photo = True
    session.last_texts.clear()

    text = (
        "Отправьте изображение (афишу или фото) для распознавания.\n"
        f"Текущая детализация: {DETAIL_LABELS.get(session.detail, session.detail)}."
    )
    await bot.send_message(
        message.chat.id,
        text,
        reply_markup=_main_keyboard(session.detail),
    )


async def select_detail(callback: types.CallbackQuery, bot: Bot) -> None:
    """Handle detail selection callbacks."""

    session = session_store.get_session(callback.from_user.id)
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
        session = session_store.set_detail(callback.from_user.id, value)  # type: ignore[arg-type]
        await callback.message.edit_reply_markup(_main_keyboard(session.detail))
        await callback.answer(f"Детализация: {DETAIL_LABELS.get(value, value)}")
        return

    await callback.answer("Неизвестная команда", show_alert=True)


def cancel(user_id: int) -> None:
    """Stop OCR session for the user."""

    session_store.finish_session(user_id)


def is_waiting(user_id: int) -> bool:
    return session_store.is_waiting(user_id)




async def handle_photo(
    message: types.Message,
    bot: Bot,
    images: list[tuple[bytes, str]],
) -> None:
    """Process incoming photo for OCR comparison."""

    session = session_store.get_session(message.from_user.id)
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
    results: list[tuple[str, OcrResult | None, str | None]] = []
    for model in models:
        try:
            ocr_result = await run_ocr(image_bytes, model=model, detail=session.detail)
            session.last_texts[model] = ocr_result.text
            results.append((model, ocr_result, None))
        except Exception as exc:  # pragma: no cover - depends on network
            logging.warning("OCR model failed: model=%s error=%s", model, exc)
            results.append((model, None, str(exc)))

    detail_label = DETAIL_LABELS.get(session.detail, session.detail)
    lines = [f"Файл: {name}", f"Детализация: {detail_label}"]
    for model, result, error in results:
        lines.append("")
        lines.append(f"<b>{model}</b>:")
        if error:
            lines.append(f"<i>Ошибка: {html.escape(str(error))}</i>")
        elif result:
            lines.append(f"<pre>{html.escape(result.text)}</pre>")
        else:
            lines.append("<i>Нет данных</i>")

    lines.append("")
    lines.append("<b>Токены</b>:")
    header = f"{'Модель':<12} {'prompt':>7} {'completion':>10} {'total':>7}"
    lines.append(f"<pre>{html.escape(header)}")
    for model, result, _ in results:
        if result:
            usage = result.usage
            row = (
                f"{model:<12} "
                f"{usage.prompt_tokens:>7} "
                f"{usage.completion_tokens:>10} "
                f"{usage.total_tokens:>7}"
            )
        else:
            row = f"{model:<12} {'-':>7} {'-':>10} {'-':>7}"
        lines.append(html.escape(row))
    lines.append("</pre>")

    success = [result for _, result, error in results if result and not error]
    if len(success) == 2:
        text_a = success[0].text
        text_b = success[1].text
        ratio = SequenceMatcher(None, text_a, text_b).ratio()
        lines.append("")
        lines.append(f"Схожесть: {ratio:.3f}")
        diff_lines = [line for line in ndiff(text_a.splitlines(), text_b.splitlines()) if line[:1] in {"-", "+"}]
        if diff_lines:
            lines.append("Различия (первые 10 строк):")
            for line in diff_lines[:10]:
                lines.append(html.escape(line))
    else:
        errors = ", ".join(str(error) for _, _, error in results if error)
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
    "OcrResult",
]
