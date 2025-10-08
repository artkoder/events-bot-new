from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Final

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from imagekit_poster import PosterGravity, PosterResizeMode, PosterTransformation, process_poster

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for /ik_poster")
    return value


def _get_credentials() -> dict[str, str]:
    return {
        "public_key": _env("IMAGEKIT_PUBLIC_KEY"),
        "private_key": _env("IMAGEKIT_PRIVATE_KEY"),
        "url_endpoint": _env("IMAGEKIT_URL_ENDPOINT"),
    }


@dataclass(frozen=True)
class PosterMode:
    title: str
    transformations: tuple[PosterTransformation, ...]


POSTER_MODES: Final[dict[str, PosterMode]] = {
    "story": PosterMode(
        title="История 9:16",
        transformations=(
            PosterTransformation(
                name="story",
                width=1080,
                height=1920,
                mode=PosterResizeMode.CROP,
                gravity=PosterGravity.CENTER,
                quality=90,
            ),
        ),
    ),
    "square": PosterMode(
        title="Квадрат 1:1",
        transformations=(
            PosterTransformation(
                name="square",
                width=1080,
                height=1080,
                mode=PosterResizeMode.PAD,
                background="000000",
                quality=90,
            ),
        ),
    ),
    "landscape": PosterMode(
        title="Широкий 16:9",
        transformations=(
            PosterTransformation(
                name="landscape",
                width=1920,
                height=1080,
                mode=PosterResizeMode.CROP,
                gravity=PosterGravity.CENTER,
                quality=90,
            ),
        ),
    ),
}


class IkPosterStates(StatesGroup):
    waiting_photo = State()
    choosing_mode = State()


ik_poster_router = Router(name="ik_poster")


@ik_poster_router.message(Command("ik_poster"))
async def cmd_ik_poster(message: Message, state: FSMContext) -> None:
    await state.set_state(IkPosterStates.waiting_photo)
    await message.answer("Отправьте постер фотографией или файлом")


def _build_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, mode in POSTER_MODES.items():
        rows.append([InlineKeyboardButton(text=mode.title, callback_data=f"ik-poster:{key}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@ik_poster_router.message(IkPosterStates.waiting_photo)
async def handle_photo(message: Message, state: FSMContext) -> None:
    buffer = BytesIO()
    filename = "poster.jpg"

    if message.photo:
        photo = message.photo[-1]
        await message.bot.download(photo.file_id, destination=buffer)
        filename = f"{photo.file_unique_id}.jpg"
    elif (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        await message.bot.download(message.document.file_id, destination=buffer)
        filename = message.document.file_name or filename
    else:
        await message.answer("Пожалуйста, пришлите изображение постера.")
        return

    await state.update_data(image=buffer.getvalue(), filename=filename)
    await state.set_state(IkPosterStates.choosing_mode)
    await message.answer("Выберите режим обработки", reply_markup=_build_keyboard())


@ik_poster_router.callback_query(
    IkPosterStates.choosing_mode, F.data.startswith("ik-poster:")
)
async def handle_mode(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    mode_key = callback.data.split(":", 1)[1]
    mode = POSTER_MODES.get(mode_key)
    if not mode:
        await callback.message.answer("Неизвестный режим. Попробуйте ещё раз.")
        return

    data = await state.get_data()
    raw_image = data.get("image")
    image = bytes(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
    filename = data.get("filename", "poster.jpg")

    if not image:
        await callback.message.answer(
            "Не удалось найти изображение. Начните заново с команды /ik_poster.",
        )
        await state.clear()
        return

    operator_chat = os.getenv("OPERATOR_CHAT_ID")
    try:
        credentials = _get_credentials()
    except RuntimeError as exc:  # pragma: no cover - configuration error
        logger.exception("ImageKit credentials are not configured")
        await callback.message.answer(str(exc))
        await state.clear()
        return

    await callback.message.edit_reply_markup()
    await callback.message.answer(
        f"Обрабатываю постер в режиме «{mode.title}»..."
    )

    try:
        urls = process_poster(
            image,
            file_name=filename,
            transformations=mode.transformations,
            **credentials,
        )
    except Exception:  # pragma: no cover - network/SDK errors
        logger.exception("Failed to process poster via ImageKit")
        await callback.message.answer(
            "Не удалось обработать постер. Попробуйте позже или обратитесь к оператору.",
        )
        await state.clear()
        return

    lines = [f"Готово! Режим: {mode.title}"]
    for name, url in urls.items():
        lines.append(f"{name}: {url}")
    result_text = "\n".join(lines)

    await callback.message.answer(result_text)

    if operator_chat:
        try:
            await callback.message.bot.send_message(operator_chat, result_text)
        except Exception:  # pragma: no cover - network errors
            logger.exception("Failed to send poster result to operator chat")

    await state.clear()


@ik_poster_router.callback_query(IkPosterStates.choosing_mode)
async def handle_unknown_callback(callback: CallbackQuery) -> None:
    await callback.answer("Используйте кнопки ниже, чтобы выбрать режим.", show_alert=False)
