from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Sequence

from cachetools import TTLCache
from aiogram import types
from sqlalchemy import select
from PIL import Image

from db import Database
from models import (
    Channel,
    User,
    Event,
    EventPoster,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceSession,
    VideoAnnounceSessionStatus,
)
from main import (
    HTTP_SEMAPHORE,
    LOCAL_TZ,
    format_day_pretty,
    get_http_session,
    get_setting_value,
    set_setting_value,
)
from .finalize import prepare_final_texts
from .kaggle_client import DEFAULT_KERNEL_PATH, KaggleClient
from .poller import VIDEO_MAX_MB, run_kernel_poller
from .selection import (
    build_payload,
    build_selection,
    fetch_profiles,
    payload_as_json,
    prepare_session_items,
)
from .types import (
    RankedEvent,
    RenderPayload,
    SelectionBuildResult,
    SelectionContext,
    SessionOverview,
    VideoProfile,
)

logger = logging.getLogger(__name__)
CHANNEL_SETTING_KEY = "videoannounce_channels"
DEFAULT_PRIMARY_WINDOW_DAYS = 3
DEFAULT_FALLBACK_WINDOW_DAYS = 10
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_SELECTED_MIN = 6
DEFAULT_SELECTED_MAX = 8
PENDING_INSTRUCTION_TTL = 15 * 60


@dataclass
class PendingInstruction:
    session_id: int
    reuse_candidates: bool = False


_pending_instructions: TTLCache[int, PendingInstruction] = TTLCache(
    maxsize=64, ttl=PENDING_INSTRUCTION_TTL
)


def set_pending_instruction(user_id: int, pending: PendingInstruction) -> None:
    _pending_instructions[user_id] = pending


def take_pending_instruction(
    user_id: int, session_id: int | None = None
) -> PendingInstruction | None:
    pending = _pending_instructions.get(user_id)
    if pending and (session_id is None or pending.session_id == session_id):
        return _pending_instructions.pop(user_id, None)
    return None


def is_waiting_instruction(user_id: int) -> bool:
    return user_id in _pending_instructions


def read_positive_int_env(env_key: str, default: int) -> int:
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning(
            "video_announce: invalid %s=%r, falling back to default %s",
            env_key,
            raw_value,
            default,
        )
        return default


DATASET_PAYLOAD_MAX_MB = read_positive_int_env("VIDEO_ANNOUNCE_DATASET_MAX_MB", 50)

logger.info(
    "video_announce: limits configured dataset_max_mb=%s video_max_mb=%s",
    DATASET_PAYLOAD_MAX_MB,
    VIDEO_MAX_MB,
)


class VideoAnnounceScenario:
    def __init__(self, db: Database, bot, chat_id: int, user_id: int):
        self.db = db
        self.bot = bot
        self.chat_id = chat_id
        self.user_id = user_id

    async def _load_admin_channels(self) -> list[Channel]:
        async with self.db.get_session() as session:
            result = await session.execute(
                select(Channel)
                .where(Channel.is_admin.is_(True))
                .order_by(Channel.title, Channel.username, Channel.channel_id)
            )
            return result.scalars().all()

    def _format_channel_label(self, channel: Channel) -> str:
        if channel.username:
            return f"@{channel.username}"
        if channel.title:
            return channel.title
        return str(channel.channel_id)

    async def _load_channel_config(self) -> dict[str, dict[str, int]]:
        raw = await get_setting_value(self.db, CHANNEL_SETTING_KEY)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("video_announce: failed to parse channel config")
            return {}
        parsed: dict[str, dict[str, int]] = {}
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            profile_cfg: dict[str, int] = {}
            for kind in ("test", "main"):
                raw_val = value.get(kind)
                try:
                    int_val = int(raw_val)
                except Exception:
                    continue
                profile_cfg[kind] = int_val
            if profile_cfg:
                parsed[str(key)] = profile_cfg
        return parsed

    async def _save_channel_config(self, data: dict[str, dict[str, int]]) -> None:
        await set_setting_value(self.db, CHANNEL_SETTING_KEY, json.dumps(data))

    async def _get_profile_channels(self, profile_key: str) -> tuple[int | None, int | None]:
        config = await self._load_channel_config()
        profile_cfg = config.get(profile_key, {})
        return (profile_cfg.get("test"), profile_cfg.get("main"))

    async def _set_profile_channel(
        self, profile_key: str, chat_id: int, kind: str
    ) -> None:
        if kind not in {"test", "main"}:
            return
        channels = await self._load_admin_channels()
        allowed_ids = {ch.channel_id for ch in channels}
        if chat_id not in allowed_ids:
            logger.warning("video_announce: unknown channel %s for profile %s", chat_id, profile_key)
            return
        config = await self._load_channel_config()
        profile_cfg = dict(config.get(profile_key, {}))
        current = profile_cfg.get(kind)
        if current == chat_id:
            profile_cfg.pop(kind, None)
        else:
            profile_cfg[kind] = chat_id
        if profile_cfg:
            config[profile_key] = profile_cfg
        else:
            config.pop(profile_key, None)
        await self._save_channel_config(config)

    async def _resolve_session_channels(
        self, session_obj: VideoAnnounceSession
    ) -> tuple[int | None, int | None]:
        if session_obj.test_chat_id or session_obj.main_chat_id:
            return session_obj.test_chat_id, session_obj.main_chat_id
        if not session_obj.profile_key:
            return None, None
        test_chat_id, main_chat_id = await self._get_profile_channels(
            session_obj.profile_key
        )
        if test_chat_id or main_chat_id:
            async with self.db.get_session() as session:
                fresh = await session.get(VideoAnnounceSession, session_obj.id)
                if fresh:
                    fresh.test_chat_id = test_chat_id
                    fresh.main_chat_id = main_chat_id
                    await session.commit()
                    await session.refresh(fresh)
                    session_obj.test_chat_id = fresh.test_chat_id
                    session_obj.main_chat_id = fresh.main_chat_id
                    return fresh.test_chat_id, fresh.main_chat_id
        session_obj.test_chat_id = test_chat_id
        session_obj.main_chat_id = main_chat_id
        return test_chat_id, main_chat_id

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

    def _default_selection_params(self) -> dict[str, int | str]:
        now_local = datetime.now(LOCAL_TZ)
        target = (now_local + timedelta(days=1)).date()
        return {
            "target_date": target.isoformat(),
            "primary_window_days": DEFAULT_PRIMARY_WINDOW_DAYS,
            "fallback_window_days": DEFAULT_FALLBACK_WINDOW_DAYS,
            "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
            "default_selected_min": DEFAULT_SELECTED_MIN,
            "default_selected_max": DEFAULT_SELECTED_MAX,
        }

    def _normalize_required_periods(
        self, params: dict[str, int | str]
    ) -> list[dict[str, int | str]]:
        raw_periods = params.get("required_periods")
        if not isinstance(raw_periods, list):
            return []
        normalized: list[dict[str, int | str]] = []
        fallback_default = int(
            params.get("fallback_window_days", DEFAULT_FALLBACK_WINDOW_DAYS)
            or DEFAULT_FALLBACK_WINDOW_DAYS
        )
        for item in raw_periods:
            preset: dict[str, int | str] | None = None
            if isinstance(item, dict):
                preset = {
                    k: v
                    for k, v in item.items()
                    if k
                    in {
                        "target_date",
                        "primary_window_days",
                        "fallback_window_days",
                        "candidate_limit",
                        "default_selected_min",
                        "default_selected_max",
                    }
                }
            elif isinstance(item, str):
                start_raw, end_raw = (item.split("..", 1) + [item])[:2]
                start_date = self._parse_target_date(start_raw)
                end_date = self._parse_target_date(end_raw) or start_date
                if start_date:
                    delta_days = max((end_date - start_date).days, 0) if end_date else 0
                    preset = {
                        "target_date": start_date.isoformat(),
                        "fallback_window_days": max(fallback_default, delta_days),
                    }
            if preset:
                normalized.append(preset)
        return normalized

    def _parse_target_date(self, raw: str | None) -> date | None:
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None

    def _format_event_date(self, raw_date: str) -> str:
        try:
            return date.fromisoformat(raw_date.split("..", 1)[0]).strftime("%d.%m")
        except ValueError:
            return raw_date.split("..", 1)[0]

    def _format_event_datetime(self, ev: Event) -> str:
        date_label = self._format_event_date(ev.date)
        time_text = (ev.time or "").strip()
        if time_text:
            short_time = time_text[:5] if ":" in time_text else time_text
            return f"{date_label} {short_time}"
        return date_label

    def _parse_event_datetime(self, ev: Event) -> datetime | None:
        try:
            day = date.fromisoformat(ev.date.split("..", 1)[0])
        except ValueError:
            return None
        time_text = (ev.time or "").strip()
        time_part = time_text
        for sep in ("-", "–", "—"):
            time_part = time_part.split(sep, 1)[0]
        time_part = time_part.split()[0] if time_part else ""
        try:
            if time_part and ":" in time_part:
                hours, minutes = time_part.split(":", 1)
                parsed_time = datetime.strptime(f"{hours}:{minutes}", "%H:%M").time()
            else:
                parsed_time = datetime.min.time()
            return datetime.combine(day, parsed_time, tzinfo=LOCAL_TZ)
        except ValueError:
            return datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_TZ)

    def _event_sort_key(self, ranked: RankedEvent) -> tuple[datetime, int]:
        parsed_dt = self._parse_event_datetime(ranked.event)
        if parsed_dt is None:
            parsed_dt = datetime.max.replace(tzinfo=timezone.utc)
        return (parsed_dt, ranked.position or 0)

    def _normalize_emoji(self, emoji: str | None) -> str:
        if not emoji:
            return ""
        tokens = [part for part in emoji.strip().split() if part]
        seen = set()
        unique: list[str] = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            unique.append(token)
        return unique[0] if unique else ""

    def _date_range_label(self, params: dict[str, int | str]) -> str:
        primary = int(params.get("primary_window_days", DEFAULT_PRIMARY_WINDOW_DAYS) or 0)
        fallback = int(params.get("fallback_window_days", DEFAULT_FALLBACK_WINDOW_DAYS) or 0)
        base = self._parse_target_date(str(params.get("target_date"))) or datetime.now(LOCAL_TZ).date()
        end = base + timedelta(days=fallback)
        pretty_start = format_day_pretty(base)
        pretty_end = format_day_pretty(end)
        if base == end:
            return f"{pretty_start} (окно +{primary}/+{fallback})"
        return f"{pretty_start} – {pretty_end} (окно +{primary}/+{fallback})"

    def _visible_pairs(
        self,
        pairs: Sequence[tuple[VideoAnnounceItem, Event]],
        *,
        visible_limit: int,
    ) -> list[tuple[VideoAnnounceItem, Event]]:
        if visible_limit <= 0:
            return list(pairs)
        ordered: list[tuple[VideoAnnounceItem, Event]] = []
        seen: set[int] = set()
        for item, ev in pairs:
            if len(ordered) >= visible_limit:
                break
            ordered.append((item, ev))
            seen.add(item.event_id)
        for item, ev in pairs:
            include_count = item.include_count or getattr(ev, "video_include_count", 0) or 0
            if (item.is_mandatory or include_count > 0) and item.event_id not in seen:
                ordered.append((item, ev))
                seen.add(item.event_id)
        return ordered

    def _format_title(self, ev: Event) -> str:
        url = ev.telegraph_url or ev.source_post_url
        title = html.escape(ev.title[:80])
        if url:
            safe_url = html.escape(url)
            return f'<a href="{safe_url}">{title}</a>'
        return title

    def _chunk_buttons(self, buttons: list[types.InlineKeyboardButton], size: int = 3) -> list[list[types.InlineKeyboardButton]]:
        return [buttons[i : i + size] for i in range(0, len(buttons), size)]

    def _get_selection_params(self, session_obj: VideoAnnounceSession) -> dict[str, int | str]:
        params = self._default_selection_params()
        stored = session_obj.selection_params if isinstance(session_obj.selection_params, dict) else {}
        params.update({k: v for k, v in (stored or {}).items() if v is not None})
        return params

    async def _resolve_profile(self, profile_key: str | None) -> VideoProfile:
        profiles = await fetch_profiles()
        if profile_key:
            for profile in profiles:
                if profile.key == profile_key:
                    return profile
        return VideoProfile(profile_key or "default", profile_key or "", "")

    def _selection_ctx_from_params(
        self, profile: VideoProfile | None, params: dict[str, int | str]
    ) -> SelectionContext:
        primary = int(params.get("primary_window_days", DEFAULT_PRIMARY_WINDOW_DAYS) or 0)
        fallback = int(params.get("fallback_window_days", DEFAULT_FALLBACK_WINDOW_DAYS) or 0)
        candidate_limit = int(params.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT) or 0)
        default_selected_min = int(
            params.get("default_selected_min", DEFAULT_SELECTED_MIN) or 0
        )
        default_selected_max = int(
            params.get("default_selected_max", DEFAULT_SELECTED_MAX) or 0
        )
        target_date = self._parse_target_date(str(params.get("target_date")))
        instruction = (str(params.get("instruction") or "").strip()) or None
        return SelectionContext(
            tz=LOCAL_TZ,
            target_date=target_date,
            profile=profile,
            primary_window_days=primary,
            fallback_window_days=fallback,
            candidate_limit=max(candidate_limit, DEFAULT_SELECTED_MAX),
            default_selected_min=max(default_selected_min, 1),
            default_selected_max=max(default_selected_max, default_selected_min),
            instruction=instruction,
        )

    async def _build_selection_context(
        self, session_obj: VideoAnnounceSession
    ) -> SelectionContext:
        params = self._get_selection_params(session_obj)
        profile = await self._resolve_profile(session_obj.profile_key)
        return self._selection_ctx_from_params(profile, params)

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

    async def show_profile_channels(
        self, profile_key: str, message: types.Message | None = None
    ) -> None:
        if not await self.ensure_access():
            return
        profiles = await fetch_profiles()
        profile = next((p for p in profiles if p.key == profile_key), None)
        if not profile:
            await self.bot.send_message(self.chat_id, "Профиль не найден")
            return
        channels = await self._load_admin_channels()
        test_chat_id, main_chat_id = await self._get_profile_channels(profile_key)
        channel_names = {
            ch.channel_id: self._format_channel_label(ch)
            for ch in channels
            if ch.channel_id is not None
        }
        lines = [
            f"🎬 {profile.title}",
            "Настройте каналы публикации для этой рубрики.",
        ]
        if not channels:
            lines.append("Бот не найден в админках каналов — отправим в операторский чат.")
        test_label = channel_names.get(test_chat_id) if test_chat_id else None
        main_label = channel_names.get(main_chat_id) if main_chat_id else None
        lines.append(
            f"Тестовый: {test_label or 'не выбран (отправим в операторский чат)'}"
        )
        lines.append(
            f"Основной: {main_label or 'не выбран (только тестовая публикация)'}"
        )
        keyboard: list[list[types.InlineKeyboardButton]] = []
        for ch in channels:
            label = self._format_channel_label(ch)
            test_marker = "✅" if ch.channel_id == test_chat_id else "➕"
            main_marker = "✅" if ch.channel_id == main_chat_id else "➕"
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text=f"Тест {test_marker} · {label}",
                        callback_data=f"vidchan:{profile_key}:{ch.channel_id}:test",
                    ),
                    types.InlineKeyboardButton(
                        text=f"Осн. {main_marker}",
                        callback_data=f"vidchan:{profile_key}:{ch.channel_id}:main",
                    ),
                ]
            )
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="🚀 Запустить подбор", callback_data=f"vidstart:{profile_key}"
                )
            ]
        )
        markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
        text = "\n".join(lines)
        if message:
            await message.edit_text(text, reply_markup=markup)
        else:
            await self.bot.send_message(self.chat_id, text, reply_markup=markup)

    async def show_menu(self) -> None:
        if not await self.ensure_access():
            return
        rendering = await self.has_rendering()
        text_parts = [
            "Меню видео-анонсов",
            "Выберите профиль, отметьте каналы и запустите подбор.",
        ]
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

        params = self._default_selection_params()
        profile = await self._resolve_profile(profile_key)
        ctx = self._selection_ctx_from_params(profile, params)
        test_chat_id, main_chat_id = await self._get_profile_channels(profile_key)
        async with self.db.get_session() as session:
            obj = VideoAnnounceSession(
                status=VideoAnnounceSessionStatus.CREATED,
                profile_key=profile_key,
                selection_params=params,
                test_chat_id=test_chat_id,
                main_chat_id=main_chat_id,
            )
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
        set_pending_instruction(
            self.user_id, PendingInstruction(session_id=obj.id, reuse_candidates=False)
        )
        await self._prompt_instruction(obj, ctx)

    async def _prompt_instruction(
        self,
        session_obj: VideoAnnounceSession,
        ctx: SelectionContext | None = None,
        *,
        reuse: bool = False,
    ) -> None:
        if ctx is None:
            ctx = await self._build_selection_context(session_obj)
        params = self._get_selection_params(session_obj)
        required_periods = self._normalize_required_periods(params)
        action_hint = (
            "новую инструкцию для пересчёта текущего списка"
            if reuse
            else "инструкцию для подбора афиши"
        )
        lines = [
            f"Сессия #{session_obj.id}: отправьте {action_hint}.",
            "Можно прислать текстом или нажмите пропустить.",
        ]
        if ctx.profile:
            lines.append(f"Профиль: {ctx.profile.title}")
        period_buttons: list[types.InlineKeyboardButton] = []
        for idx, preset in enumerate(required_periods):
            merged_params = dict(params)
            merged_params.update(preset)
            label = self._date_range_label(merged_params)
            period_buttons.append(
                types.InlineKeyboardButton(
                    text=label,
                    callback_data=f"vidinstr:{session_obj.id}:preset:{idx}",
                )
            )
        if period_buttons:
            lines.append("Или выберите один из обязательных периодов:")
        action_buttons = [
            types.InlineKeyboardButton(
                text="Пропустить", callback_data=f"vidinstr:{session_obj.id}:skip"
            ),
            types.InlineKeyboardButton(
                text="Отмена", callback_data=f"vidinstr:{session_obj.id}:cancel"
            ),
        ]
        inline_keyboard: list[list[types.InlineKeyboardButton]] = []
        if period_buttons:
            inline_keyboard.extend(self._chunk_buttons(period_buttons, size=2))
        inline_keyboard.append(action_buttons)
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
        await self.bot.send_message(self.chat_id, "\n".join(lines), reply_markup=keyboard)

    async def apply_period_preset(self, session_id: int, preset_idx: int) -> str:
        if not await self._has_access():
            return "Not authorized"
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        params = self._get_selection_params(session_obj)
        presets = self._normalize_required_periods(params)
        if not presets or preset_idx < 0 or preset_idx >= len(presets):
            return "Период не найден"
        params.update(presets[preset_idx])
        async with self.db.get_session() as session:
            fresh = await session.get(VideoAnnounceSession, session_id)
            if not fresh:
                return "Сессия не найдена"
            fresh.selection_params = params
            session.add(fresh)
            await session.commit()
            await session.refresh(fresh)
            session_obj = fresh
        reuse = session_obj.status == VideoAnnounceSessionStatus.SELECTED
        if reuse:
            ranked = await self._recalculate_selection(session_obj)
            await self._send_selection_posts(session_obj, ranked)
        await self._prompt_instruction(session_obj, reuse=reuse)
        return "Период применён"

    async def _build_and_store_selection(
        self,
        session_obj: VideoAnnounceSession,
        *,
        candidates: Sequence[Event] | None = None,
        preserve_existing: bool = False,
    ) -> list[RankedEvent]:
        ctx = await self._build_selection_context(session_obj)
        result = await build_selection(
            self.db,
            ctx,
            client=KaggleClient(),
            session_id=session_obj.id,
            candidates=candidates,
        )
        if preserve_existing:
            await self._refresh_selection_items(session_obj, result)
        else:
            await prepare_session_items(
                self.db,
                session_obj,
                result.ranked,
                default_ready_ids=result.default_ready_ids,
            )
        return result.ranked

    def _build_input_message(
        self, session_obj: VideoAnnounceSession, ranked: Sequence[RankedEvent]
    ) -> str:
        params = self._get_selection_params(session_obj)
        instruction = (str(params.get("instruction") or "").strip())
        lines = [
            f"Сессия #{session_obj.id}: INPUT",
            f"Диапазон: {self._date_range_label(params)}",
            f"Инструкция: {html.escape(instruction[:300]) if instruction else '—'}",
            f"Всего кандидатов: {len(ranked)}",
            "📥 Кандидаты:",
            "<blockquote>",
        ]
        sorted_ranked = sorted(ranked, key=self._event_sort_key)
        for r in sorted_ranked:
            ev = r.event
            emoji = self._normalize_emoji(ev.emoji)
            date_label = self._format_event_datetime(ev)
            include_count = getattr(ev, "video_include_count", 0) or 0
            promo_marker = " · 🔥PROMO" if r.mandatory or include_count > 0 else ""
            score = f" · {r.score:.1f}" if r.score is not None else ""
            reason = (
                f" · {html.escape(r.reason[:140])}" if r.reason else ""
            )
            lines.append(
                f"{r.position}. {date_label} · {emoji} {self._format_title(ev)}{promo_marker}{score}{reason}"
            )
        lines.append("</blockquote>")
        return "\n".join(lines)

    async def _send_input_overview(
        self, session_obj: VideoAnnounceSession, ranked: Sequence[RankedEvent]
    ) -> None:
        text = self._build_input_message(session_obj, ranked)
        await self.bot.send_message(self.chat_id, text, parse_mode="HTML")

    async def _send_selection_posts(
        self,
        session_obj: VideoAnnounceSession,
        ranked: Sequence[RankedEvent],
        *,
        selection_message: types.Message | None = None,
    ) -> None:
        await self._send_input_overview(session_obj, ranked)
        if selection_message:
            await self._update_selection_message(selection_message, session_obj.id)
        else:
            await self._send_selection_ui(session_obj.id)

    async def apply_instruction(
        self,
        session_id: int,
        instruction: str | None,
        *,
        reuse_candidates: bool,
        pending: PendingInstruction | None = None,
    ) -> str:
        if not await self._has_access():
            return "Not authorized"
        pending = pending or take_pending_instruction(self.user_id, session_id)
        reuse_candidates = reuse_candidates or bool(
            pending and pending.reuse_candidates
        )
        sess: VideoAnnounceSession | None = None
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status not in {
                VideoAnnounceSessionStatus.CREATED,
                VideoAnnounceSessionStatus.SELECTED,
            }:
                return "Сессия уже запущена"
            params = self._get_selection_params(sess)
            if instruction:
                params["instruction"] = instruction
            else:
                params.pop("instruction", None)
            sess.selection_params = params
            if sess.status == VideoAnnounceSessionStatus.CREATED:
                sess.status = VideoAnnounceSessionStatus.SELECTED
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
        preserve_existing = False
        if not sess:
            return "Сессия не найдена"
        candidates: Sequence[Event] | None = None
        if reuse_candidates:
            pairs = await self._load_items_with_events(session_id)
            candidates = [ev for _, ev in pairs]
            preserve_existing = bool(candidates)
        ranked = await self._build_and_store_selection(
            sess,
            candidates=candidates,
            preserve_existing=preserve_existing,
        )
        await self._send_selection_posts(sess, ranked)
        if pending and reuse_candidates:
            return "Инструкция обновлена"
        if pending:
            return "Инструкция сохранена"
        return "Готово"

    async def request_new_instruction(self, session_id: int) -> str:
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return "Сессия уже запущена"
        set_pending_instruction(
            self.user_id, PendingInstruction(session_id=session_id, reuse_candidates=True)
        )
        await self._prompt_instruction(session_obj, reuse=True)
        return "Запрос обновлён"

    async def cancel_instruction(self, session_id: int) -> str:
        pending = take_pending_instruction(self.user_id, session_id)
        if not pending:
            return "Запрос инструкций устарел"
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if sess and sess.status == VideoAnnounceSessionStatus.CREATED:
                await session.delete(sess)
                await session.commit()
                return "Сессия отменена"
        return "Обновление отменено"

    async def _render_and_notify(self, session_obj: VideoAnnounceSession, ranked) -> None:
        client = KaggleClient()
        finalized = []
        try:
            finalized = await prepare_final_texts(self.db, session_obj.id, ranked)
        except Exception:
            logger.exception("video_announce: failed to prepare final texts")
        try:
            payload = await self._build_render_payload(session_obj, ranked)
            json_text = payload_as_json(payload, timezone.utc)
            preview_lines = []
            event_map = {ev.id: ev for ev in payload.events}
            item_map = {it.event_id: it for it in payload.items}
            for r in ranked[:5]:
                ev = event_map.get(r.event.id)
                item = item_map.get(r.event.id)
                if not ev or not item:
                    continue
                dt = ev.date.split("..", 1)[0]
                title = item.final_title or ev.title
                preview_lines.append(
                    f"#{r.position} · {dt} · {ev.emoji or ''} {title} ({r.score})"
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
        test_chat_id, main_chat_id = await self._resolve_session_channels(session_obj)
        asyncio.create_task(
            run_kernel_poller(
                self.db,
                client,
                session_obj,
                bot=self.bot,
                notify_chat_id=self.chat_id,
                test_chat_id=test_chat_id,
                main_chat_id=main_chat_id,
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
            ranked.append(
                RankedEvent(
                    event=ev,
                    score=item.llm_score or 0.0,
                    position=item.position,
                    reason=item.llm_reason,
                    mandatory=bool(item.is_mandatory),
                )
            )
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

    async def _build_render_payload(
        self, session_obj: VideoAnnounceSession, ranked: Sequence[RankedEvent]
    ) -> RenderPayload:
        ranked_ids = {r.event.id for r in ranked}
        pairs = await self._load_items_with_events(session_obj.id)
        ready_items = [
            item
            for item, _ in pairs
            if item.status == VideoAnnounceItemStatus.READY
            and item.event_id in ranked_ids
        ]
        return build_payload(
            session_obj, ranked, tz=timezone.utc, items=ready_items
        )

    async def _refresh_selection_items(
        self, session_obj: VideoAnnounceSession, result: SelectionBuildResult
    ) -> None:
        async with self.db.get_session() as session:
            res = await session.execute(
                select(VideoAnnounceItem).where(
                    VideoAnnounceItem.session_id == session_obj.id
                )
            )
            existing = res.scalars().all()
            existing_map = {item.event_id: item for item in existing}
            preserved_status = {
                item.event_id: item.status
                for item in existing
                if item.status in {VideoAnnounceItemStatus.READY, VideoAnnounceItemStatus.SKIPPED}
            }
            new_ids = {r.event.id for r in result.ranked}
            for item in existing:
                if item.event_id not in new_ids:
                    await session.delete(item)

            for idx, r in enumerate(result.ranked, start=1):
                item = existing_map.get(r.event.id) or VideoAnnounceItem(
                    session_id=session_obj.id, event_id=r.event.id
                )
                item.position = idx
                saved_status = preserved_status.get(r.event.id)
                if saved_status:
                    item.status = saved_status
                elif item.status not in {
                    VideoAnnounceItemStatus.READY,
                    VideoAnnounceItemStatus.SKIPPED,
                }:
                    item.status = (
                        VideoAnnounceItemStatus.READY
                        if r.event.id in result.default_ready_ids
                        else VideoAnnounceItemStatus.SKIPPED
                    )
                item.llm_score = r.score
                item.llm_reason = r.reason
                item.is_mandatory = r.mandatory
                item.include_count = getattr(r.event, "video_include_count", 0) or 0
                session.add(item)
            await session.commit()

    async def _recalculate_selection(
        self, session_obj: VideoAnnounceSession
    ) -> list[RankedEvent]:
        return await self._build_and_store_selection(
            session_obj, preserve_existing=True
        )

    async def _selection_view(
        self, session_id: int
    ) -> tuple[str, types.InlineKeyboardMarkup]:
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return ("Сессия не найдена", types.InlineKeyboardMarkup(inline_keyboard=[]))
        pairs = await self._load_items_with_events(session_id)
        params = self._get_selection_params(session_obj)
        default_selected_max = int(
            params.get("default_selected_max", DEFAULT_SELECTED_MAX) or DEFAULT_SELECTED_MAX
        )
        instruction = (str(params.get("instruction") or "").strip())
        lines = [
            f"Сессия #{session_id}: SELECTED",
            f"Диапазон: {self._date_range_label(params)}",
            "Выберите события для рендера:",
            f"Показываем топ-{default_selected_max} + промо (всего {len(pairs)})",
        ]
        if instruction:
            lines.append(f"Инструкция: {html.escape(instruction[:300])}")
        else:
            lines.append("Инструкция: —")
        keyboard: list[list[types.InlineKeyboardButton]] = []
        toggle_buttons: list[types.InlineKeyboardButton] = []
        allow_edit = session_obj.status == VideoAnnounceSessionStatus.SELECTED
        if allow_edit:
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="📝 Новая инструкция", callback_data=f"vidinstr:{session_id}:new"
                    )
                ]
            )
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="+1 день", callback_data=f"vidsel:{session_id}:plus1"
                    ),
                    types.InlineKeyboardButton(
                        text="+3 дня", callback_data=f"vidsel:{session_id}:plus3"
                    ),
                ]
            )
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="Сброс к завтра", callback_data=f"vidsel:{session_id}:reset"
                    ),
                    types.InlineKeyboardButton(
                        text="Пересчитать", callback_data=f"vidsel:{session_id}:recalc"
                    ),
                ]
            )
        visible_pairs = self._visible_pairs(pairs, visible_limit=default_selected_max)
        for item, ev in visible_pairs:
            marker = "✅" if item.status == VideoAnnounceItemStatus.READY else "⬜"
            emoji = self._normalize_emoji(ev.emoji)
            date_label = self._format_event_datetime(ev)
            pin = ""
            include_count = item.include_count or getattr(ev, "video_include_count", 0) or 0
            if include_count > 0:
                pin = f" 📌{include_count}"
            promo_marker = " 🔥PROMO" if item.is_mandatory or include_count > 0 else ""
            score = f" · {item.llm_score:.1f}" if item.llm_score is not None else ""
            reason = (
                f" · {html.escape(item.llm_reason[:140])}"
                if item.llm_reason
                else ""
            )
            title = self._format_title(ev)
            lines.append(
                f"{marker} #{item.position} · {date_label} · {emoji} {title}{pin}{promo_marker}{score}{reason}"
            )
            if allow_edit:
                toggle_buttons.append(
                    types.InlineKeyboardButton(
                        text=f"{marker} #{item.position}",
                        callback_data=f"vidtoggle:{session_id}:{ev.id}",
                    )
                )
        ready_count = sum(
            1 for item, _ in visible_pairs if item.status == VideoAnnounceItemStatus.READY
        )
        if ready_count:
            lines.insert(3, f"По умолчанию выбрано: {ready_count} из {len(visible_pairs)}")
        if allow_edit and toggle_buttons:
            keyboard.extend(self._chunk_buttons(toggle_buttons, size=3))
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
        await self.bot.send_message(
            self.chat_id, text, reply_markup=markup, parse_mode="HTML"
        )

    async def _update_selection_message(
        self, message: types.Message, session_id: int
    ) -> None:
        text, markup = await self._selection_view(session_id)
        await message.edit_text(text, reply_markup=markup, parse_mode="HTML")

    async def adjust_selection_params(
        self, session_id: int, action: str, message: types.Message
    ) -> str:
        if not await self._has_access():
            return "Not authorized"
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"
            params = self._get_selection_params(sess)
            base_date = self._parse_target_date(str(params.get("target_date"))) or (
                datetime.now(LOCAL_TZ).date() + timedelta(days=1)
            )
            if action == "plus1":
                params["target_date"] = (base_date + timedelta(days=1)).isoformat()
            elif action == "plus3":
                params["target_date"] = (base_date + timedelta(days=3)).isoformat()
            elif action == "reset":
                params = self._default_selection_params()
            elif action != "recalc":
                return "Неизвестное действие"
            sess.selection_params = params
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
        ranked = await self._recalculate_selection(sess)
        await self._send_selection_posts(sess, ranked, selection_message=message)
        return "Обновлено"

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
        payload = await self._build_render_payload(session_obj, ranked)
        json_text = payload_as_json(payload, timezone.utc)
        event_map = {ev.id: ev for ev in payload.events}
        item_map = {it.event_id: it for it in payload.items}
        preview_lines = []
        for r in ranked:
            ev = event_map.get(r.event.id)
            item = item_map.get(r.event.id)
            if not ev or not item:
                continue
            dt = ev.date.split("..", 1)[0]
            title = item.final_title or ev.title
            preview_lines.append(
                f"#{r.position} · {dt} · {ev.emoji or ''} {title} ({r.score})"
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

    async def _download_poster_bytes(self, poster: EventPoster) -> bytes | None:
        if not poster.catbox_url:
            return None
        session = get_http_session()
        try:
            async with HTTP_SEMAPHORE:
                resp = await session.get(poster.catbox_url)
                resp.raise_for_status()
                return await resp.read()
        except Exception:
            logger.warning(
                "video_announce: failed to download poster url=%s", poster.catbox_url
            )
            return None

    async def _export_posters(
        self,
        tmp_path: Path,
        items: Sequence[VideoAnnounceItem],
        poster_map: dict[int, EventPoster],
    ) -> None:
        if not items:
            return
        for item in items:
            poster = poster_map.get(item.event_id)
            if not poster:
                continue
            data = await self._download_poster_bytes(poster)
            if not data:
                continue
            try:
                with Image.open(BytesIO(data)) as img:
                    img.convert("RGB").save(tmp_path / f"{item.position}.png", format="PNG")
            except Exception:
                logger.exception(
                    "video_announce: failed to convert poster event_id=%s", poster.event_id
                )

    def _copy_assets(self, tmp_path: Path) -> None:
        assets_dir = Path(__file__).resolve().parent / "assets"
        assets = [
            (assets_dir / "Oswald-VariableFont_wght.ttf", tmp_path / "font.ttf"),
            (assets_dir / "Pulsarium.mp3", tmp_path / "Pulsarium.mp3"),
        ]
        for src, dest in assets:
            if not src.exists():
                logger.warning("video_announce: asset not found %s", src)
                continue
            shutil.copy2(src, dest)

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
            ready_items: list[VideoAnnounceItem] = []
            poster_map: dict[int, EventPoster] = {}
            async with self.db.get_session() as session:
                res_items = await session.execute(
                    select(VideoAnnounceItem)
                    .where(VideoAnnounceItem.session_id == session_obj.id)
                    .where(VideoAnnounceItem.status == VideoAnnounceItemStatus.READY)
                    .order_by(VideoAnnounceItem.position)
                )
                ready_items = list(res_items.scalars().all())
                if ready_items:
                    res_posters = await session.execute(
                        select(EventPoster)
                        .where(
                            EventPoster.event_id.in_(
                                [item.event_id for item in ready_items]
                            )
                        )
                        .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
                    )
                    for poster in res_posters.scalars().all():
                        if poster.event_id not in poster_map and poster.catbox_url:
                            poster_map[poster.event_id] = poster
            (tmp_path / "dataset-metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (tmp_path / "payload.json").write_text(json_text, encoding="utf-8")
            (tmp_path / "final_texts.json").write_text(
                json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await self._export_posters(tmp_path, ready_items, poster_map)
            self._copy_assets(tmp_path)
            total_size = sum(
                f.stat().st_size for f in tmp_path.glob("**/*") if f.is_file()
            )
            if total_size > DATASET_PAYLOAD_MAX_MB * 1024 * 1024:
                raise RuntimeError(
                    f"dataset payload exceeds {DATASET_PAYLOAD_MAX_MB}MB"
                )
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
        await scenario.show_profile_channels(profile, message=callback.message)
        await callback.answer("Профиль выбран")
        return True
    if prefix == "vidchan":
        try:
            _, profile, chat_id, kind = callback.data.split(":", 3)
            await scenario._set_profile_channel(profile, int(chat_id), kind)
            await scenario.show_profile_channels(profile, message=callback.message)
            await callback.answer("Настройки сохранены")
        except Exception:
            logger.exception("video_announce: update channels failed")
            await callback.answer("Не удалось сохранить", show_alert=True)
        return True
    if prefix == "vidstart":
        _, profile = callback.data.split(":", 1)
        await scenario.start_session(profile)
        await callback.answer("Сбор профиля")
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
    if prefix == "vidinstr":
        try:
            _, session_id, action = callback.data.split(":", 2)
            session_id_int = int(session_id)
        except Exception:
            return False
        if action == "skip":
            pending = take_pending_instruction(callback.from_user.id, session_id_int)
            msg = await scenario.apply_instruction(
                session_id_int,
                None,
                reuse_candidates=bool(pending and pending.reuse_candidates),
                pending=pending,
            )
        elif action == "cancel":
            msg = await scenario.cancel_instruction(session_id_int)
        elif action == "new":
            msg = await scenario.request_new_instruction(session_id_int)
        elif action.startswith("preset:"):
            try:
                preset_idx = int(action.split(":", 1)[1])
            except Exception:
                msg = "Период не найден"
            else:
                msg = await scenario.apply_period_preset(session_id_int, preset_idx)
        else:
            msg = "Неизвестное действие"
        await callback.answer(
            msg or "Готово",
            show_alert=msg
            not in {
                "Готово",
                "Инструкция сохранена",
                "Инструкция обновлена",
                "Запрос обновлён",
                "Период применён",
            },
        )
        return True
    return False
