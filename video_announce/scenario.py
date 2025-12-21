from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
from .kaggle_client import DEFAULT_KERNEL_PATH, KaggleClient
from .poller import run_kernel_poller
from .selection import (
    build_payload,
    build_selection,
    fetch_profiles,
    payload_as_json,
    prepare_session_items,
)
from .types import RankedEvent, SelectionContext, SessionOverview, VideoProfile

logger = logging.getLogger(__name__)
VIDEO_TEST_CHAT_ID = int(os.getenv("VIDEO_ANNOUNCE_TEST_CHAT_ID", "0") or 0)
VIDEO_MAIN_CHAT_ID = int(os.getenv("VIDEO_ANNOUNCE_MAIN_CHAT_ID", "0") or 0)


class VideoAnnounceScenario:
    def __init__(self, db: Database, bot, chat_id: int, user_id: int):
        self.db = db
        self.bot = bot
        self.chat_id = chat_id
        self.user_id = user_id

    async def _has_access(self) -> bool:
        user = await self._load_user()
        return bool(user and user.is_superadmin)

    async def _load_user(self) -> User | None:
        async with self.db.get_session() as session:
            return await session.get(User, self.user_id)

    async def ensure_access(self) -> bool:
        if not await self._has_access():
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

    async def _load_session(self, session_id: int) -> VideoAnnounceSession | None:
        async with self.db.get_session() as session:
            return await session.get(VideoAnnounceSession, session_id)

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

        summaries = await self._summaries()
        failed_sessions = [
            ov.session for ov in summaries if ov.session.status == VideoAnnounceSessionStatus.FAILED
        ]
        if failed_sessions and not rendering:
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="🔁 Перезапустить последнюю", callback_data=f"vidrestart:{failed_sessions[0].id}"
                    )
                ]
            )

        overview_lines: list[str] = []
        for ov in summaries:
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
                status=VideoAnnounceSessionStatus.SELECTED,
            )
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            await prepare_session_items(self.db, obj, ranked)
        await self._send_selection_ui(obj.id)

    async def _render_and_notify(self, session_obj: VideoAnnounceSession, ranked) -> None:
        client = KaggleClient()
        finalized = []
        try:
            finalized = await prepare_final_texts(self.db, session_obj.id, ranked)
        except Exception:
            logger.exception("video_announce: failed to prepare final texts")
        try:
            payload = build_payload(session_obj, ranked, tz=timezone.utc)
            json_text = payload_as_json(payload, timezone.utc)
            preview_lines = []
            for r in ranked[:5]:
                dt = r.event.date.split("..", 1)[0]
                preview_lines.append(
                    f"#{r.position} · {dt} · {r.event.emoji or ''} {r.event.title} ({r.score})"
                )
            preview = "\n".join(preview_lines)
            await self.bot.send_message(
                self.chat_id,
                "<b>Черновик JSON для видеоролика:</b>\n<pre>"
                + html.escape(json_text)
                + "</pre>",
                parse_mode="HTML",
            )
            await self.bot.send_message(self.chat_id, preview or "Нет событий")
            dataset_slug = await self._create_dataset(session_obj, json_text, finalized)
            kernel_ref = await self._push_kernel(client, dataset_slug)
            session_obj.kaggle_dataset = dataset_slug
            session_obj.kaggle_kernel_ref = kernel_ref
            await self._store_kaggle_meta(session_obj.id, dataset_slug, kernel_ref)
        except Exception:
            logger.exception("video_announce: failed to push kaggle job")
            await self._mark_failed(session_obj.id, "kaggle push failed")
            return
        asyncio.create_task(
            run_kernel_poller(
                self.db,
                client,
                session_obj,
                bot=self.bot,
                notify_chat_id=self.chat_id,
                test_chat_id=VIDEO_TEST_CHAT_ID or None,
                main_chat_id=VIDEO_MAIN_CHAT_ID or None,
                poll_interval=60,
                timeout_minutes=40,
                dataset_slug=dataset_slug,
            )
        )

    async def _load_ranked_events(
        self, session_id: int, *, ready_only: bool = False
    ) -> list[RankedEvent]:
        async with self.db.get_session() as session:
            query = (
                select(VideoAnnounceItem)
                .where(VideoAnnounceItem.session_id == session_id)
                .order_by(VideoAnnounceItem.position)
            )
            if ready_only:
                query = query.where(VideoAnnounceItem.status == VideoAnnounceItemStatus.READY)
            res_items = await session.execute(query)
            items = res_items.scalars().all()
            if not items:
                return []
            event_ids = [it.event_id for it in items]
            ev_res = await session.execute(select(Event).where(Event.id.in_(event_ids)))
            events = {ev.id: ev for ev in ev_res.scalars().all()}
        ranked: list[RankedEvent] = []
        for item in items:
            ev = events.get(item.event_id)
            if not ev:
                continue
            ranked.append(RankedEvent(event=ev, score=0.0, position=item.position))
        return sorted(ranked, key=lambda r: r.position)

    async def _load_items_with_events(
        self, session_id: int
    ) -> list[tuple[VideoAnnounceItem, Event]]:
        async with self.db.get_session() as session:
            res = await session.execute(
                select(VideoAnnounceItem, Event)
                .join(Event, VideoAnnounceItem.event_id == Event.id)
                .where(VideoAnnounceItem.session_id == session_id)
                .order_by(VideoAnnounceItem.position)
            )
            return list(res.all())

    async def _selection_view(
        self, session_id: int
    ) -> tuple[str, types.InlineKeyboardMarkup]:
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return ("Сессия не найдена", types.InlineKeyboardMarkup(inline_keyboard=[]))
        pairs = await self._load_items_with_events(session_id)
        lines = [
            f"Сессия #{session_id}: {session_obj.status.value}",
            "Выберите события для рендера:",
        ]
        keyboard: list[list[types.InlineKeyboardButton]] = []
        allow_edit = session_obj.status == VideoAnnounceSessionStatus.SELECTED
        for item, ev in pairs:
            marker = "✅" if item.status == VideoAnnounceItemStatus.READY else "⬜"
            title = html.escape(ev.title[:40])
            lines.append(
                f"{marker} #{item.position} · {ev.date.split('..', 1)[0]} · {ev.emoji or ''} {title}"
            )
            if allow_edit:
                keyboard.append(
                    [
                        types.InlineKeyboardButton(
                            text=f"{marker} #{item.position}",
                            callback_data=f"vidtoggle:{session_id}:{ev.id}",
                        )
                    ]
                )
        if allow_edit:
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="📄 Сформировать JSON", callback_data=f"vidjson:{session_id}"
                    ),
                    types.InlineKeyboardButton(
                        text="🚀 Запустить рендер", callback_data=f"vidrender:{session_id}"
                    ),
                ]
            )
        markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
        return ("\n".join(lines), markup)

    async def _send_selection_ui(self, session_id: int) -> None:
        text, markup = await self._selection_view(session_id)
        await self.bot.send_message(self.chat_id, text, reply_markup=markup)

    async def _update_selection_message(
        self, message: types.Message, session_id: int
    ) -> None:
        text, markup = await self._selection_view(session_id)
        await message.edit_text(text, reply_markup=markup)

    async def toggle_item(self, session_id: int, event_id: int, message: types.Message) -> str:
        if not await self._has_access():
            return "Not authorized"
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"
            res = await session.execute(
                select(VideoAnnounceItem)
                .where(VideoAnnounceItem.session_id == session_id)
                .where(VideoAnnounceItem.event_id == event_id)
            )
            item = res.scalars().first()
            if not item:
                return "Событие не найдено"
            item.status = (
                VideoAnnounceItemStatus.SKIPPED
                if item.status == VideoAnnounceItemStatus.READY
                else VideoAnnounceItemStatus.READY
            )
            session.add(item)
            await session.commit()
        await self._update_selection_message(message, session_id)
        return "Обновлено"

    async def preview_json(self, session_id: int) -> str:
        if not await self.ensure_access():
            return ""
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return "Сессия уже запущена"
        ranked = await self._load_ranked_events(session_id, ready_only=True)
        if not ranked:
            return "Нет выбранных событий"
        payload = build_payload(session_obj, ranked, tz=timezone.utc)
        json_text = payload_as_json(payload, timezone.utc)
        preview_lines = []
        for r in ranked:
            dt = r.event.date.split("..", 1)[0]
            preview_lines.append(
                f"#{r.position} · {dt} · {r.event.emoji or ''} {r.event.title} ({r.score})"
            )
        await self.bot.send_message(
            self.chat_id,
            "<b>Текущий JSON:</b>\n<pre>" + html.escape(json_text) + "</pre>",
            parse_mode="HTML",
        )
        await self.bot.send_message(self.chat_id, "\n".join(preview_lines) or "Нет событий")
        return "Сформировано"

    async def start_render(self, session_id: int, message: types.Message | None = None) -> str:
        if not await self._has_access():
            return "Not authorized"
        if await self.has_rendering():
            return "Уже есть активный рендер"
        ranked = await self._load_ranked_events(session_id, ready_only=True)
        if not ranked:
            return "Нет выбранных событий"
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"
            sess.status = VideoAnnounceSessionStatus.RENDERING
            sess.started_at = datetime.now(timezone.utc)
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
        if message:
            await self._update_selection_message(message, session_id)
        await self.bot.send_message(
            self.chat_id, f"Сессия #{session_id} запущена, собираем материалы"
        )
        asyncio.create_task(self._render_and_notify(sess, ranked))
        return "Рендеринг запущен"

    async def restart_session(self, session_id: int) -> None:
        if not await self.ensure_access():
            return
        ranked = await self._load_ranked_events(session_id, ready_only=True)
        if not ranked:
            await self.bot.send_message(self.chat_id, "Не удалось собрать события для рестарта")
            return
        async with self.db.get_session() as session:
            obj = await session.get(VideoAnnounceSession, session_id)
            if not obj:
                await self.bot.send_message(self.chat_id, "Сессия не найдена")
                return
            if obj.status != VideoAnnounceSessionStatus.FAILED:
                await self.bot.send_message(self.chat_id, "Сессию можно рестартовать только после ошибки")
                return
            obj.status = VideoAnnounceSessionStatus.RENDERING
            obj.started_at = datetime.now(timezone.utc)
            obj.finished_at = None
            obj.error = None
            obj.video_url = None
            obj.kaggle_dataset = None
            obj.kaggle_kernel_ref = None
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
        await self.bot.send_message(
            self.chat_id, f"Сессия #{session_id} перезапущена, готовим материалы"
        )
        asyncio.create_task(self._render_and_notify(obj, ranked))

    async def _store_kaggle_meta(
        self, session_id: int, dataset_slug: str, kernel_ref: str | None
    ) -> None:
        async with self.db.get_session() as session:
            obj = await session.get(VideoAnnounceSession, session_id)
            if not obj:
                return
            obj.kaggle_dataset = dataset_slug
            obj.kaggle_kernel_ref = kernel_ref
            await session.commit()

    async def _mark_failed(self, session_id: int, error: str) -> None:
        async with self.db.get_session() as session:
            obj = await session.get(VideoAnnounceSession, session_id)
            if not obj:
                return
            obj.status = VideoAnnounceSessionStatus.FAILED
            obj.finished_at = datetime.now(timezone.utc)
            obj.error = error
            await session.commit()

    async def _create_dataset(
        self, session_obj: VideoAnnounceSession, json_text: str, finalized
    ) -> str:
        username = os.getenv("KAGGLE_USERNAME", "video-afisha")
        slug = f"video-afisha-session-{session_obj.id}"
        dataset_id = f"{username}/{slug}"
        meta = {
            "title": f"Video Afisha Session {session_obj.id}",
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }
        final_payload = [
            {
                "event_id": item.event_id,
                "title": item.title,
                "description": item.description,
                "use_ocr": item.use_ocr,
                "poster_source": item.poster_source,
            }
            for item in finalized
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "dataset-metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (tmp_path / "payload.json").write_text(json_text, encoding="utf-8")
            (tmp_path / "final_texts.json").write_text(
                json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_size = sum(f.stat().st_size for f in tmp_path.iterdir())
            if total_size > 50 * 1024 * 1024:
                raise RuntimeError("dataset payload exceeds 50MB")
            client = KaggleClient()
            try:
                await asyncio.to_thread(client.create_dataset, tmp_path)
            except Exception:
                logger.exception("video_announce: failed to create dataset, retry after delete")
                await asyncio.to_thread(client.delete_dataset, dataset_id, no_confirm=True)
                await asyncio.to_thread(client.create_dataset, tmp_path)
        return dataset_id

    async def _push_kernel(self, client: KaggleClient, dataset_slug: str) -> str:
        await asyncio.to_thread(
            client.push_kernel, dataset_sources=[dataset_slug], timeout="300"
        )
        meta_path = DEFAULT_KERNEL_PATH / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        kernel_ref = str(meta.get("id") or meta.get("slug") or "")
        if not kernel_ref:
            raise RuntimeError("kernel reference missing after push")
        return kernel_ref

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
    if prefix == "vidrestart":
        try:
            _, session_id = callback.data.split(":", 1)
            await scenario.restart_session(int(session_id))
        except Exception:
            logger.exception("video_announce: restart failed")
        await callback.answer("Рестарт")
        return True
    return False
