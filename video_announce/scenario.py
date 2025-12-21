from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timezone

from aiogram import types
from sqlalchemy import select

from db import Database
from models import (
    User,
    Event,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceSession,
    VideoAnnounceSessionStatus,
)
from .finalize import prepare_final_texts
from .kaggle_client import KaggleClient
from .selection import (
    build_payload,
    build_selection,
    fetch_profiles,
    payload_as_json,
    prepare_session_items,
)
from .types import SelectionContext, SessionOverview, VideoProfile

logger = logging.getLogger(__name__)


class VideoAnnounceScenario:
    def __init__(self, db: Database, bot, chat_id: int, user_id: int):
        self.db = db
        self.bot = bot
        self.chat_id = chat_id
        self.user_id = user_id

    async def _load_user(self) -> User | None:
        async with self.db.get_session() as session:
            return await session.get(User, self.user_id)

    async def ensure_access(self) -> bool:
        user = await self._load_user()
        if not user or not user.is_superadmin:
            await self.bot.send_message(self.chat_id, "Not authorized")
            return False
        return True

    async def has_rendering(self) -> VideoAnnounceSession | None:
        async with self.db.get_session() as session:
            res = await session.execute(
                select(VideoAnnounceSession).where(
                    VideoAnnounceSession.status == VideoAnnounceSessionStatus.RENDERING
                )
            )
            return res.scalars().first()

    async def _summaries(self) -> list[SessionOverview]:
        async with self.db.get_session() as session:
            res = await session.execute(
                select(VideoAnnounceSession)
                .order_by(VideoAnnounceSession.created_at.desc())
                .limit(5)
            )
            sessions = res.scalars().all()
            overviews: list[SessionOverview] = []
            for sess in sessions:
                items_res = await session.execute(
                    select(VideoAnnounceItem).where(
                        VideoAnnounceItem.session_id == sess.id
                    )
                )
                items = items_res.scalars().all()
                events: list[Event] = []
                if items:
                    event_ids = [item.event_id for item in items]
                    ev_res = await session.execute(
                        select(Event).where(Event.id.in_(event_ids))
                    )
                    events = ev_res.scalars().all()
                overviews.append(SessionOverview(session=sess, items=items, events=events))
            return overviews

    async def show_menu(self) -> None:
        if not await self.ensure_access():
            return
        rendering = await self.has_rendering()
        text_parts = ["Меню видео-анонсов"]
        if rendering:
            text_parts.append("\nРендеринг уже запущен, UI временно заблокирован.")
        keyboard: list[list[types.InlineKeyboardButton]] = []
        profiles = await fetch_profiles()
        if not rendering:
            for p in profiles:
                keyboard.append(
                    [
                        types.InlineKeyboardButton(
                            text=f"🎬 {p.title}", callback_data=f"vidprofile:{p.key}"
                        )
                    ]
                )
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="Обновить статусы", callback_data="vidstatus:refresh"
                )
            ]
        )

        overview_lines: list[str] = []
        for ov in await self._summaries():
            overview_lines.append(
                f"Сессия #{ov.session.id}: {ov.session.status.value} ({ov.count} событий)"
            )
            if ov.session.video_url:
                overview_lines.append(f" → {ov.session.video_url}")
        if overview_lines:
            text_parts.append("\n" + "\n".join(overview_lines))

        markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
        await self.bot.send_message(self.chat_id, "\n".join(text_parts), reply_markup=markup)

    async def start_session(self, profile_key: str) -> None:
        if not await self.ensure_access():
            return
        existing = await self.has_rendering()
        if existing:
            await self.bot.send_message(
                self.chat_id,
                f"Сессия #{existing.id} уже рендерится, дождитесь завершения",
            )
            return

        ctx = SelectionContext(tz=timezone.utc, profile=VideoProfile(profile_key, "", ""))
        ranked = await build_selection(self.db, ctx, client=KaggleClient())
        async with self.db.get_session() as session:
            obj = VideoAnnounceSession(
                status=VideoAnnounceSessionStatus.RENDERING,
                started_at=datetime.now(timezone.utc),
            )
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            await prepare_session_items(self.db, obj, ranked)
        await self.bot.send_message(
            self.chat_id, f"Сессия #{obj.id} запущена, собираем материалы"
        )
        asyncio.create_task(self._render_and_notify(obj, ranked))

    async def _render_and_notify(self, session_obj: VideoAnnounceSession, ranked) -> None:
        try:
            await prepare_final_texts(self.db, session_obj.id, ranked)
        except Exception:
            logger.exception("video_announce: failed to prepare final texts")
        payload = build_payload(session_obj, ranked, tz=timezone.utc)
        json_text = payload_as_json(payload, timezone.utc)
        preview_lines = []
        for r in ranked[:5]:
            dt = r.event.date.split("..", 1)[0]
            preview_lines.append(
                f"#{r.position} · {dt} · {r.event.emoji or ''} {r.event.title} ({r.score})"
            )
        preview = "\n".join(preview_lines)
        try:
            await self.bot.send_message(
                self.chat_id,
                "<b>Черновик JSON для видеоролика:</b>\n<pre>" + html.escape(json_text) + "</pre>",
                parse_mode="HTML",
            )
            await self.bot.send_message(self.chat_id, preview or "Нет событий")
        finally:
            async with self.db.get_session() as session:
                fresh = await session.get(VideoAnnounceSession, session_obj.id)
                if fresh:
                    fresh.status = VideoAnnounceSessionStatus.DONE
                    fresh.finished_at = datetime.now(timezone.utc)
                    fresh.video_url = fresh.video_url or "pending_delivery"
                    await session.commit()

    async def refresh_status(self) -> None:
        lines = ["Статусы сессий:"]
        for ov in await self._summaries():
            lines.append(
                f"#{ov.session.id}: {ov.session.status.value} ({ov.count} событий)"
            )
        await self.bot.send_message(self.chat_id, "\n".join(lines))


async def handle_prefix_action(prefix: str, callback: types.CallbackQuery, scenario: VideoAnnounceScenario) -> bool:
    if prefix == "vidprofile":
        _, profile = callback.data.split(":", 1)
        await scenario.start_session(profile)
        await callback.answer("Профиль выбран")
        return True
    if prefix == "vidstatus":
        await scenario.refresh_status()
        await callback.answer("Обновлено")
        return True
    return False
