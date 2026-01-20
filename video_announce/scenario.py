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
from typing import Any, Sequence

from cachetools import TTLCache
from aiogram import types
from sqlalchemy import select

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
from .about import normalize_about_with_fallback
from .finalize import prepare_final_texts
from .kaggle_client import DEFAULT_KERNEL_PATH, KaggleClient, list_local_kernels
from .poller import (
    VIDEO_MAX_MB,
    remember_status_message,
    run_kernel_poller,
    update_status_message,
)
from .selection import (
    build_payload,
    build_selection,
    fetch_candidates,
    fetch_profiles,
    fill_missing_about,
    payload_as_json,
    prepare_session_items,
)
from .custom_types import (
    RankedEvent,
    RenderPayload,
    SelectionBuildResult,
    SelectionContext,
    SessionOverview,
    VideoProfile,
)
from .pattern_preview import (
    generate_intro_preview,
    get_next_pattern,
    get_prev_pattern,
    ALL_PATTERNS,
    PATTERN_STICKER,
)

logger = logging.getLogger(__name__)
CHANNEL_SETTING_KEY = "videoannounce_channels"
DEFAULT_PRIMARY_WINDOW_DAYS = 3
DEFAULT_FALLBACK_WINDOW_DAYS = 10
DEFAULT_CANDIDATE_LIMIT = 80
MAX_CANDIDATE_LIMIT = 80
DEFAULT_SELECTED_MIN = 8
DEFAULT_SELECTED_MAX = 12
PENDING_INSTRUCTION_TTL = 15 * 60
IMPORT_PAYLOAD_FLAG_KEY = "imported_payload"
IMPORT_PAYLOAD_JSON_KEY = "imported_payload_json"


@dataclass
class PendingInstruction:
    session_id: int
    reuse_candidates: bool = False


_pending_instructions: TTLCache[int, PendingInstruction] = TTLCache(
    maxsize=64, ttl=PENDING_INSTRUCTION_TTL
)


@dataclass
class PendingIntroText:
    session_id: int


_pending_intro_texts: TTLCache[int, PendingIntroText] = TTLCache(
    maxsize=64, ttl=PENDING_INSTRUCTION_TTL
)


@dataclass
class PendingPayloadImport:
    profile_key: str


_pending_payload_imports: TTLCache[int, PendingPayloadImport] = TTLCache(
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


def set_pending_intro_text(user_id: int, pending: PendingIntroText) -> None:
    _pending_intro_texts[user_id] = pending


def take_pending_intro_text(
    user_id: int, session_id: int | None = None
) -> PendingIntroText | None:
    pending = _pending_intro_texts.get(user_id)
    if pending and (session_id is None or pending.session_id == session_id):
        return _pending_intro_texts.pop(user_id, None)
    return None


def is_waiting_intro_text(user_id: int) -> bool:
    return user_id in _pending_intro_texts


def set_pending_payload_import(user_id: int, pending: PendingPayloadImport) -> None:
    _pending_payload_imports[user_id] = pending


def take_pending_payload_import(user_id: int) -> PendingPayloadImport | None:
    return _pending_payload_imports.pop(user_id, None)


def is_waiting_payload_import(user_id: int) -> bool:
    return user_id in _pending_payload_imports


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

    def _default_required_periods(self, base_day: date | None = None) -> list[dict[str, int | str]]:
        today_local = base_day or datetime.now(LOCAL_TZ).date()
        tomorrow = today_local + timedelta(days=1)
        
        # Calculate next weekend (Saturday)
        weekend_offset = (5 - tomorrow.weekday()) % 7
        next_saturday = tomorrow + timedelta(days=weekend_offset)
        
        # Check if today is Saturday or Sunday
        is_weekend_now = today_local.weekday() >= 5  # 5=Saturday, 6=Sunday
        
        periods: list[dict[str, int | str]] = [
            {
                "title": "Завтра",
                "target_date": tomorrow.isoformat(),
                "fallback_window_days": 0,
                "primary_window_days": 0,
            },
            {
                "title": "3 дня",
                "target_date": tomorrow.isoformat(),
                "fallback_window_days": 2,
                "primary_window_days": 2,
            },
        ]
        
        # If today is Saturday or Sunday, add current weekend button
        if is_weekend_now:
            # Current weekend: this Saturday
            current_saturday = today_local - timedelta(days=today_local.weekday() - 5) if today_local.weekday() == 6 else today_local
            periods.append({
                "title": f"Эти выходные ({current_saturday.day}.{current_saturday.month:02d})",
                "target_date": current_saturday.isoformat(),
                "fallback_window_days": 1,
                "primary_window_days": 1,
            })
        
        # Next weekend
        periods.append({
            "title": f"Выходные ({next_saturday.day}.{next_saturday.month:02d})" if is_weekend_now else "Выходные",
            "target_date": next_saturday.isoformat(),
            "fallback_window_days": 1,
            "primary_window_days": 1,
        })
        
        periods.extend([
            {
                "title": "Неделя",
                "target_date": tomorrow.isoformat(),
                "fallback_window_days": 6,
            },
            {
                "title": "10 дней",
                "target_date": tomorrow.isoformat(),
                "fallback_window_days": 9,
            },
        ])
        
        return periods

    def _default_selection_params(self) -> dict[str, Any]:
        now_local = datetime.now(LOCAL_TZ)
        target = (now_local + timedelta(days=1)).date()
        return {
            "target_date": target.isoformat(),
            "primary_window_days": DEFAULT_PRIMARY_WINDOW_DAYS,
            "fallback_window_days": DEFAULT_FALLBACK_WINDOW_DAYS,
            "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
            "default_selected_min": DEFAULT_SELECTED_MIN,
            "default_selected_max": DEFAULT_SELECTED_MAX,
            "required_periods": self._default_required_periods(now_local.date()),
            "selected_required_period": None,
            "random_order": False,
        }

    def _normalize_required_periods(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        raw_periods = params.get("required_periods")
        if not isinstance(raw_periods, list):
            return []
        normalized: list[dict[str, Any]] = []
        fallback_default = int(
            params.get("fallback_window_days", DEFAULT_FALLBACK_WINDOW_DAYS)
            or DEFAULT_FALLBACK_WINDOW_DAYS
        )
        for raw_idx, item in enumerate(raw_periods):
            title: str | None = None
            explicit_label: str | None = None
            preset: dict[str, int | str] | None = None
            if isinstance(item, dict):
                raw_title = item.get("title")
                raw_label = item.get("label")
                title = raw_title.strip() if isinstance(raw_title, str) else None
                explicit_label = (
                    raw_label.strip() if isinstance(raw_label, str) else None
                )
                params_candidate = item.get("params") if isinstance(item.get("params"), dict) else item
                preset = {
                    k: v
                    for k, v in params_candidate.items()
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
                label = (
                    explicit_label
                    or self._format_required_preset_label(title, params, preset)
                    or self._date_range_label(preset)
                )
                normalized.append(
                    {
                        "params": preset,
                        "label": label,
                        "raw_index": raw_idx,
                        "title": title,
                        "explicit_label": explicit_label,
                    }
                )
        return normalized

    def _find_weekend_period_index(self, periods: list[dict[str, Any]]) -> int | None:
        for idx, raw in enumerate(periods):
            if isinstance(raw, dict):
                title = str(raw.get("title") or "").lower()
                if title.startswith("выходн"):
                    return idx
        return None

    def _infer_selected_required_period(
        self, params: dict[str, Any], presets: list[dict[str, Any]]
    ) -> int:
        comparison_keys = {
            "target_date",
            "primary_window_days",
            "fallback_window_days",
            "candidate_limit",
            "default_selected_min",
            "default_selected_max",
        }
        for idx, preset in enumerate(presets):
            preset_params = preset.get("params", {})
            if all(params.get(k) == preset_params.get(k) for k in comparison_keys):
                return idx
        return -1

    async def _candidate_count_for_period(
        self, profile: VideoProfile | None, base_params: dict[str, Any], preset: dict[str, Any]
    ) -> int:
        merged = dict(base_params)
        merged.update(preset)
        ctx = self._selection_ctx_from_params(profile, merged)
        try:
            events, _, _ = await fetch_candidates(self.db, ctx)
        except Exception:
            logger.exception("video_announce: failed to fetch candidates for preset")
            return 0
        return len(events)

    async def _pick_default_required_period(
        self, profile: VideoProfile | None, params: dict[str, Any]
    ) -> dict[str, Any]:
        periods = self._normalize_required_periods(params)
        if not periods:
            return params

        selected_idx = params.get("selected_required_period")
        if isinstance(selected_idx, int) and 0 <= selected_idx < len(periods):
            return params

        tomorrow = self._parse_target_date(str(params.get("target_date"))) or datetime.now(
            LOCAL_TZ
        ).date()
        weekend_idx = self._find_weekend_period_index(params.get("required_periods") or [])
        if tomorrow.weekday() >= 5 and weekend_idx is not None:
            params.update(periods[weekend_idx]["params"])
            params["selected_required_period"] = weekend_idx
            return params

        # Strict behavior: always pick the first period (usually "Tomorrow") by default
        # without auto-extending to 3 days even if event count is low.
        target_idx = 0
        params.update(periods[target_idx]["params"])
        params["selected_required_period"] = target_idx
        return params

    def _format_required_preset_label(
        self, title: str | None, base_params: dict[str, Any], preset_params: dict[str, Any]
    ) -> str:
        merged_params = dict(base_params)
        merged_params.update(preset_params)
        range_label = self._date_range_label(merged_params)
        if title:
            return f"{title} · {range_label}"
        return range_label

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

    def _extract_schedule_map(self, params: dict[str, Any]) -> dict[int, str]:
        raw = params.get("dedup_schedule")
        if not isinstance(raw, dict):
            return {}
        result: dict[int, str] = {}
        for key, value in raw.items():
            try:
                event_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip():
                result[event_id] = value.strip()
        return result

    def _extract_occurrences_map(
        self, params: dict[str, Any]
    ) -> dict[int, list[dict[str, list[str]]]]:
        raw = params.get("dedup_occurrences")
        if not isinstance(raw, dict):
            return {}
        result: dict[int, list[dict[str, list[str]]]] = {}
        for key, value in raw.items():
            try:
                event_id = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, list):
                continue
            cleaned: list[dict[str, list[str]]] = []
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                date_value = entry.get("date")
                times_value = entry.get("times")
                if not isinstance(date_value, str):
                    continue
                times: list[str] = []
                if isinstance(times_value, list):
                    times = [t for t in times_value if isinstance(t, str)]
                cleaned.append({"date": date_value, "times": times})
            if cleaned:
                result[event_id] = cleaned
        return result

    def _format_event_schedule(self, ev: Event, params: dict[str, Any]) -> str:
        schedule_map = params.get("dedup_schedule")
        raw = None
        if isinstance(schedule_map, dict) and ev.id is not None:
            raw = schedule_map.get(str(ev.id))
            if raw is None:
                raw = schedule_map.get(ev.id)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().replace("\n", " · ")
        return self._format_event_datetime(ev)

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

    def _intro_texts(self, params: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        intro_override = (str(params.get("intro_text_override") or "").strip()) or None
        intro_llm = (str(params.get("intro_text") or "").strip()) or None
        intro = intro_override or intro_llm
        return intro, intro_override, intro_llm

    def _get_selection_params(self, session_obj: VideoAnnounceSession) -> dict[str, Any]:
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
        random_order = bool(params.get("random_order"))
        return SelectionContext(
            tz=LOCAL_TZ,
            target_date=target_date,
            profile=profile,
            primary_window_days=primary,
            fallback_window_days=fallback,
            candidate_limit=min(
                max(candidate_limit, DEFAULT_SELECTED_MAX), MAX_CANDIDATE_LIMIT
            ),
            default_selected_min=max(default_selected_min, 1),
            default_selected_max=max(default_selected_max, default_selected_min),
            instruction=instruction,
            random_order=random_order,
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
                ),
                types.InlineKeyboardButton(
                    text="📥 Импортировать payload",
                    callback_data=f"vidimport:{profile_key}",
                ),
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
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="🚀 Запуск Завтра (Auto)", callback_data="vidauto:tomorrow"
                    )
                ]
            )
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="🧪 Тест Завтра (1 сц)", callback_data="vidauto:test_tomorrow"
                    )
                ]
            )
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
        
        # Show force-reset button if there's a stuck rendering session
        if rendering:
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="⚠️ Сбросить застрявшую", callback_data=f"vidforce_reset:{rendering.id}"
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
        params = await self._pick_default_required_period(profile, params)
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

    async def run_tomorrow_pipeline(
        self,
        *,
        profile_key: str = "default",
        selected_max: int = DEFAULT_SELECTED_MAX,
        test_mode: bool = False,
    ) -> None:
        if not await self.ensure_access():
            return
        existing = await self.has_rendering()
        if existing:
            await self.bot.send_message(
                self.chat_id,
                f"Сессия #{existing.id} уже рендерится, дождитесь завершения",
            )
            return

        now_local = datetime.now(LOCAL_TZ)
        tomorrow = now_local.date() + timedelta(days=1)

        params = self._default_selection_params()
        params.update(
            {
                "target_date": tomorrow.isoformat(),
                "primary_window_days": 0,
                "fallback_window_days": 0,
                "random_order": True,
                "default_selected_max": selected_max,
                "selected_required_period": None,
            }
        )
        params.pop("instruction", None)

        kernel_ref = self._pick_crumple_kernel_ref() or self._pick_default_kernel_ref()
        if not kernel_ref:
            await self.bot.send_message(self.chat_id, "Не удалось подобрать kernel для Kaggle")
            return

        test_chat_id, main_chat_id = await self._get_profile_channels(profile_key)
        async with self.db.get_session() as session:
            obj = VideoAnnounceSession(
                status=VideoAnnounceSessionStatus.SELECTED,
                profile_key=profile_key,
                selection_params=params,
                test_chat_id=test_chat_id,
                main_chat_id=main_chat_id,
                kaggle_kernel_ref=kernel_ref,
            )
            session.add(obj)
            await session.commit()
            await session.refresh(obj)

        await self.bot.send_message(
            self.chat_id,
            (
                f"Сессия #{obj.id} запущена: завтра ({tomorrow.isoformat()}), "
                f"случайный порядок, до {selected_max} событий"
                f"{' (🧪 тест: 1 сц)' if test_mode else ''}. Kernel: {kernel_ref}"
            ),
        )

        result = await self._build_and_store_selection(obj)
        if not result.default_ready_ids:
            await self.bot.send_message(
                self.chat_id,
                (
                    f"Сессия #{obj.id}: не найдено подходящих событий "
                    f"(нужны постеры с OCR) для {tomorrow.isoformat()}"
                ),
            )
            return

        await self.bot.send_message(
            self.chat_id,
            f"Сессия #{obj.id}: выбрано {len(result.default_ready_ids)} событий, запускаю Kaggle…",
        )
        msg = await self.start_render(
            obj.id,
            message=None,
            limit_scenes=1 if test_mode else None,
        )
        if msg and msg != "Рендеринг запущен":
            await self.bot.send_message(self.chat_id, f"Сессия #{obj.id}: {msg}")

    async def prompt_payload_import(self, profile_key: str) -> None:
        if not await self.ensure_access():
            return
        existing = await self.has_rendering()
        if existing:
            await self.bot.send_message(
                self.chat_id,
                f"Сессия #{existing.id} уже рендерится, дождитесь завершения",
            )
            return
        profiles = await fetch_profiles()
        profile = next((p for p in profiles if p.key == profile_key), None)
        if not profile:
            await self.bot.send_message(self.chat_id, "Профиль не найден")
            return
        set_pending_payload_import(
            self.user_id, PendingPayloadImport(profile_key=profile_key)
        )
        await self.bot.send_message(
            self.chat_id,
            "Пришлите JSON-файл payload.json для перезапуска рендера.",
        )

    def _parse_import_payload(self, raw_text: str) -> tuple[str, int]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Некорректный JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Payload должен быть JSON-объектом")
        scenes = data.get("scenes")
        intro = data.get("intro")
        if not isinstance(scenes, list) or not isinstance(intro, dict):
            raise ValueError("Payload должен содержать поля intro и scenes")
        if not scenes:
            raise ValueError("Payload не содержит сцен")
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
        return json_text, len(scenes)

    def _pick_default_kernel_ref(self) -> str | None:
        local_kernels = sorted(
            list_local_kernels(),
            key=lambda k: (k.get("title") or k.get("ref") or ""),
        )
        if local_kernels:
            ref = local_kernels[0].get("ref")
            if isinstance(ref, str) and ref:
                return ref
        username = os.getenv("KAGGLE_USERNAME", "")
        if username:
            return f"{username}/video-announce-renderer"
        return None

    def _pick_crumple_kernel_ref(self) -> str | None:
        local_kernels = list_local_kernels()
        for kernel in local_kernels:
            ref = kernel.get("ref")
            if ref == "local:CrumpleVideo":
                return ref
        for kernel in local_kernels:
            ref = kernel.get("ref")
            title = str(kernel.get("title") or "")
            if isinstance(ref, str) and ref and "crumple" in title.casefold():
                return ref
        return None

    def _extract_import_payload_json(
        self, session_obj: VideoAnnounceSession
    ) -> str | None:
        params = (
            session_obj.selection_params
            if isinstance(session_obj.selection_params, dict)
            else {}
        )
        payload_json = params.get(IMPORT_PAYLOAD_JSON_KEY)
        if isinstance(payload_json, str) and payload_json.strip():
            return payload_json
        return None

    def _is_import_session(self, session_obj: VideoAnnounceSession) -> bool:
        params = (
            session_obj.selection_params
            if isinstance(session_obj.selection_params, dict)
            else {}
        )
        return bool(params.get(IMPORT_PAYLOAD_FLAG_KEY))

    async def import_payload_and_render(
        self, profile_key: str, payload_json: str, *, scene_count: int | None = None
    ) -> str | None:
        if not await self._has_access():
            return "Not authorized"
        existing = await self.has_rendering()
        if existing:
            return f"Сессия #{existing.id} уже рендерится, дождитесь завершения"
        profiles = await fetch_profiles()
        profile = next((p for p in profiles if p.key == profile_key), None)
        if not profile:
            return "Профиль не найден"
        test_chat_id, main_chat_id = await self._get_profile_channels(profile_key)
        async with self.db.get_session() as session:
            obj = VideoAnnounceSession(
                status=VideoAnnounceSessionStatus.CREATED,
                profile_key=profile_key,
                selection_params={
                    IMPORT_PAYLOAD_FLAG_KEY: True,
                    IMPORT_PAYLOAD_JSON_KEY: payload_json,
                },
                test_chat_id=test_chat_id,
                main_chat_id=main_chat_id,
            )
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
        scene_note = f" ({scene_count} сцен)" if scene_count else ""
        await self.bot.send_message(
            self.chat_id,
            f"Импортирован payload{scene_note}. Сессия #{obj.id} готова к запуску.",
        )
        kernel_msg = await self.show_kernel_selection(obj.id, message=None)
        if kernel_msg != "Выбор kernel":
            return kernel_msg
        return None

    async def _prompt_instruction(
        self,
        session_obj: VideoAnnounceSession,
        ctx: SelectionContext | None = None,
        *,
        reuse: bool = False,
        message: types.Message | None = None,
    ) -> None:
        if ctx is None:
            ctx = await self._build_selection_context(session_obj)
        params = self._get_selection_params(session_obj)
        required_periods = self._normalize_required_periods(params)
        selected_idx = (
            params.get("selected_required_period")
            if isinstance(params.get("selected_required_period"), int)
            else -1
        )
        if (selected_idx < 0 or selected_idx >= len(required_periods)) and required_periods:
            selected_idx = self._infer_selected_required_period(params, required_periods)
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
            merged_params.update(preset["params"])
            label = preset.get("label")
            if not label:
                title_hint = preset.get("explicit_label") or preset.get("title")
                label = self._format_required_preset_label(title_hint, params, preset["params"])
            if not label:
                label = self._date_range_label(merged_params)
            checkbox = "☑️ " if idx == selected_idx else "⬜ "
            period_buttons.append(
                types.InlineKeyboardButton(
                    text=f"{checkbox}{label}",
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
        text = "\n".join(lines)
        if message:
            try:
                await self.bot.edit_message_text(
                    text=text,
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    reply_markup=keyboard,
                )
                return
            except Exception:
                logger.exception("video_announce: failed to edit instruction prompt")
        await self.bot.send_message(self.chat_id, text, reply_markup=keyboard)

    async def apply_period_preset(
        self, session_id: int, preset_idx: int, message: types.Message | None = None
    ) -> str:
        if not await self._has_access():
            return "Not authorized"
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        params = self._get_selection_params(session_obj)
        presets = self._normalize_required_periods(params)
        if not presets or preset_idx < 0 or preset_idx >= len(presets):
            return "Период не найден"
        params.update(presets[preset_idx]["params"])
        params["selected_required_period"] = preset_idx
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
            result = await self._recalculate_selection(session_obj)
            await self._send_selection_posts(session_obj, result)
        await self._prompt_instruction(session_obj, reuse=reuse, message=message)
        return "Период применён"

    async def _build_and_store_selection(
        self,
        session_obj: VideoAnnounceSession,
        *,
        candidates: Sequence[Event] | None = None,
        preserve_existing: bool = False,
    ) -> SelectionBuildResult:
        ctx = await self._build_selection_context(session_obj)
        params = self._get_selection_params(session_obj)
        schedule_map = (
            self._extract_schedule_map(params) if candidates is not None else None
        )
        occurrences_map = (
            self._extract_occurrences_map(params) if candidates is not None else None
        )
        result = await build_selection(
            self.db,
            ctx,
            client=KaggleClient(),
            session_id=session_obj.id,
            candidates=candidates,
            schedule_map=schedule_map,
            occurrences_map=occurrences_map,
            bot=self.bot,
            notify_chat_id=self.chat_id,
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
        await self._persist_schedule_map(
            session_obj, result.schedule_map, result.occurrences_map
        )
        await self._persist_intro_text(session_obj, result.intro_text, valid=result.intro_text_valid)
        return result

    async def _persist_schedule_map(
        self,
        session_obj: VideoAnnounceSession,
        schedule_map: dict[int, str],
        occurrences_map: dict[int, list[dict[str, list[str]]]],
    ) -> None:
        async with self.db.get_session() as session:
            fresh = await session.get(VideoAnnounceSession, session_obj.id)
            if not fresh:
                return
            params = self._get_selection_params(fresh)
            if schedule_map:
                params["dedup_schedule"] = {
                    str(event_id): text for event_id, text in schedule_map.items()
                }
            else:
                params.pop("dedup_schedule", None)
            if occurrences_map:
                params["dedup_occurrences"] = {
                    str(event_id): value
                    for event_id, value in occurrences_map.items()
                }
            else:
                params.pop("dedup_occurrences", None)
            fresh.selection_params = params
            session.add(fresh)
            await session.commit()
            await session.refresh(fresh)
            session_obj.selection_params = params

    async def _persist_intro_text(
        self, session_obj: VideoAnnounceSession, intro_text: str | None, valid: bool = True
    ) -> None:
        intro = (intro_text or "").strip()
        async with self.db.get_session() as session:
            fresh = await session.get(VideoAnnounceSession, session_obj.id)
            if not fresh:
                return
            params = self._get_selection_params(fresh)
            if intro:
                 params["intro_text"] = intro
            if not valid:
                 params["intro_text_valid"] = False
            else:
                 # If valid (or defaulting to True), remove the invalid flag if present?
                 # Or explicitly set True?
                 params["intro_text_valid"] = True

            fresh.selection_params = params
            session.add(fresh)
            await session.commit()
            await session.refresh(fresh)
            session_obj.selection_params = params

    async def _send_intro_controls(self, session_obj: VideoAnnounceSession) -> None:
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return
        params = self._get_selection_params(session_obj)
        intro, intro_override, intro_llm = self._intro_texts(params)

        # Check validity flag
        is_valid = params.get("intro_text_valid", True)

        if not intro and not intro_llm:
            intro_text = "⚠️ LLM не предложило интро — задайте его вручную."
        elif intro_override:
            intro_text = f"Текущее интро: {html.escape(intro_override)}"
            if intro_llm and intro_llm != intro_override:
                intro_text += "\nПредложение LLM: " + html.escape(intro_llm)
        else:
            intro_text = f"Предложение LLM: {html.escape(intro or intro_llm or '')}"
            if not is_valid:
                 intro_text = "⚠️ " + intro_text + "\n(Формат не соблюден, исправьте вручную)"

        lines = [
            f"Сессия #{session_obj.id}: интро для ролика",
            intro_text,
            "Нажмите изменить, чтобы отправить свой текст.",
        ]
        keyboard = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text="✏️ Изменить интро",
                        callback_data=f"vidintro:{session_obj.id}:edit",
                    )
                ],
                [
                    types.InlineKeyboardButton(
                        text="📄 Показать JSON", callback_data=f"vidjson:{session_obj.id}"
                    ),
                    types.InlineKeyboardButton(
                        text="🎨 Превью паттерна",
                        callback_data=f"vidpatshow:{session_obj.id}",
                    ),
                ],
            ]
        )
        await self.bot.send_message(
            self.chat_id, "\n".join(lines), reply_markup=keyboard, parse_mode="HTML"
        )

    async def prompt_intro_override(self, session_id: int) -> str:
        if not await self.ensure_access():
            return "Not authorized"
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return "Сессия уже запущена"
        set_pending_intro_text(self.user_id, PendingIntroText(session_id=session_id))
        await self.bot.send_message(
            self.chat_id,
            "Пришлите текст интро для ролика. Пустое сообщение вернёт вариант LLM.",
        )
        return "Ожидаю интро"

    async def save_intro_override(
        self, session_id: int, intro_text: str | None
    ) -> str:
        intro = (intro_text or "").strip()
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"
            params = self._get_selection_params(sess)
            if intro:
                params["intro_text_override"] = intro
            else:
                params.pop("intro_text_override", None)

            # Reset validity flag if manual override or clearing override (assuming user fixes it or reverts to LLM which might be flagged invalid but that's ok)
            # Actually if user types, we assume it's valid for now, or we could re-validate?
            # Requirements say "don't retry LLM", "operator fixes manually".
            # So if manual override is present, we consider it "valid" (or at least don't show the warning derived from LLM output).

            sess.selection_params = params
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
        await self._send_intro_controls(sess)
        return "Интро обновлено"

    # --- Pattern Selection Methods ---

    def _get_current_pattern(self, session_obj: VideoAnnounceSession) -> str:
        """Get current pattern from session params, default to STICKER."""
        params = self._get_selection_params(session_obj)
        return params.get("intro_pattern") or PATTERN_STICKER

    def _build_pattern_keyboard(
        self, session_id: int, current_pattern: str
    ) -> types.InlineKeyboardMarkup:
        """Build inline keyboard for pattern selection."""
        buttons = []
        for pattern in ALL_PATTERNS:
            label = f"✓ {pattern}" if pattern == current_pattern else pattern
            buttons.append(
                types.InlineKeyboardButton(
                    text=label, callback_data=f"vidpat:{session_id}:{pattern}"
                )
            )
        return types.InlineKeyboardMarkup(
            inline_keyboard=[
                buttons,
                [
                    types.InlineKeyboardButton(
                        text="✏️ Изменить текст", callback_data=f"vidintro:{session_id}:edit"
                    ),
                    types.InlineKeyboardButton(
                        text="✓ Подтвердить", callback_data=f"vidpatconfirm:{session_id}"
                    ),
                ],
            ]
        )

    async def show_pattern_selection(
        self, session_id: int, message: types.Message | None = None
    ) -> str:
        """Display pattern preview with selection buttons."""
        if not await self.ensure_access():
            return "Not authorized"
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return "Сессия уже запущена"

        params = self._get_selection_params(session_obj)
        intro, intro_override, _ = self._intro_texts(params)
        intro_text = intro_override or intro
        if not intro_text:
            return "Нет текста интро"

        current_pattern = self._get_current_pattern(session_obj)

        # Fetch cities from session items' events
        cities_list = []
        try:
            async with self.db.get_session() as session:
                from sqlalchemy import select
                from models import VideoAnnounceItem, Event
                items = (await session.execute(
                    select(VideoAnnounceItem).where(
                        VideoAnnounceItem.session_id == session_id,
                        VideoAnnounceItem.status == VideoAnnounceItemStatus.READY,
                    )
                )).scalars().all()
                event_ids = [it.event_id for it in items]
                if event_ids:
                    events = (await session.execute(
                        select(Event).where(Event.id.in_(event_ids))
                    )).scalars().all()
                    for ev in events:
                        if ev.city and ev.city not in cities_list:
                            cities_list.append(ev.city)
        except Exception:
            logger.exception("video_announce: failed to fetch cities for preview")

        cities = ", ".join(cities_list[:4]) if cities_list else None

        # Generate preview image
        try:
            preview_bytes = await asyncio.to_thread(
                generate_intro_preview, current_pattern, intro_text, cities
            )
        except Exception:
            logger.exception("video_announce: failed to generate pattern preview")
            return "Ошибка генерации превью"

        keyboard = self._build_pattern_keyboard(session_id, current_pattern)
        caption = f"Паттерн: <b>{current_pattern}</b>\nТекст: {html.escape(intro_text[:100])}"

        photo = types.BufferedInputFile(preview_bytes, filename="pattern_preview.png")

        if message:
            # Try to edit existing message with media
            try:
                media = types.InputMediaPhoto(media=photo, caption=caption, parse_mode="HTML")
                await self.bot.edit_message_media(
                    chat_id=self.chat_id,
                    message_id=message.message_id,
                    media=media,
                    reply_markup=keyboard,
                )
                return "Паттерн обновлён"
            except Exception:
                logger.exception("video_announce: failed to edit pattern message")
                # Fall through to send new message

        # Send new message
        await self.bot.send_photo(
            self.chat_id, photo=photo, caption=caption, parse_mode="HTML", reply_markup=keyboard
        )
        return "Выбор паттерна"

    async def switch_pattern(
        self, session_id: int, pattern: str, message: types.Message
    ) -> str:
        """Switch to a different pattern and update preview."""
        if pattern not in ALL_PATTERNS:
            return "Неизвестный паттерн"
        if not await self.ensure_access():
            return "Not authorized"

        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"

            params = self._get_selection_params(sess)
            params["intro_pattern"] = pattern
            sess.selection_params = params
            session.add(sess)
            await session.commit()

        # Re-render with new pattern
        return await self.show_pattern_selection(session_id, message)

    async def confirm_pattern(self, session_id: int, message: types.Message) -> str:
        """Confirm pattern selection and proceed to render."""
        if not await self.ensure_access():
            return "Not authorized"
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return "Сессия уже запущена"

        # Delete the pattern selection message
        try:
            await self.bot.delete_message(self.chat_id, message.message_id)
        except Exception:
            pass  # Ignore if delete fails

        # Proceed to kernel selection (existing flow)
        return await self.show_kernel_selection(session_id, message=None)

    # --- Sorting Screen Methods ---

    def _get_render_order(self, session_obj: VideoAnnounceSession) -> list[int]:
        """Get custom render order from session params, or empty list for default."""
        params = self._get_selection_params(session_obj)
        return list(params.get("render_order", []))

    async def _sort_view(
        self, session_id: int
    ) -> tuple[str, types.InlineKeyboardMarkup]:
        """Build sorting screen with up/down buttons for ready events."""
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return ("Сессия не найдена", types.InlineKeyboardMarkup(inline_keyboard=[]))
        
        pairs = await self._load_items_with_events(session_id)
        ready_pairs = [(item, ev) for item, ev in pairs if item.status == VideoAnnounceItemStatus.READY]
        
        if not ready_pairs:
            return ("Нет выбранных событий", types.InlineKeyboardMarkup(inline_keyboard=[]))

        # Get custom render order or use default position order
        render_order = self._get_render_order(session_obj)
        params = self._get_selection_params(session_obj)
        
        # Sort by render_order if exists, otherwise by position
        if render_order:
            order_map = {eid: idx for idx, eid in enumerate(render_order)}
            ready_pairs = sorted(
                ready_pairs, 
                key=lambda p: order_map.get(p[1].id, 999999)
            )
        
        lines = [
            f"Сессия #{session_id}: СОРТИРОВКА",
            "Расставьте очерёдность для видео:",
            "",
        ]
        
        keyboard: list[list[types.InlineKeyboardButton]] = []
        
        for render_pos, (item, ev) in enumerate(ready_pairs, start=1):
            emoji = self._normalize_emoji(ev.emoji)
            date_label = self._format_event_schedule(ev, params)
            title = self._format_title(ev)
            # Show render position vs original LLM position
            lines.append(f"{render_pos}. #{item.position} · {date_label} · {emoji} {title}")
            
            # Up/Down buttons per row
            row = []
            if render_pos > 1:
                row.append(types.InlineKeyboardButton(
                    text=f"⬆️ {render_pos}",
                    callback_data=f"vidsort:{session_id}:up:{ev.id}"
                ))
            else:
                row.append(types.InlineKeyboardButton(
                    text="  ",
                    callback_data=f"vidsort:{session_id}:noop"
                ))
            if render_pos < len(ready_pairs):
                row.append(types.InlineKeyboardButton(
                    text=f"⬇️ {render_pos}",
                    callback_data=f"vidsort:{session_id}:down:{ev.id}"
                ))
            else:
                row.append(types.InlineKeyboardButton(
                    text="  ",
                    callback_data=f"vidsort:{session_id}:noop"
                ))
            keyboard.append(row)
        
        # Back button
        keyboard.append([
            types.InlineKeyboardButton(
                text="✓ Готово", callback_data=f"vidsort:{session_id}:done"
            )
        ])
        
        markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
        return ("\n".join(lines), markup)

    async def show_sort_screen(
        self, session_id: int, message: types.Message | None = None
    ) -> str:
        """Display the sorting screen."""
        if not await self.ensure_access():
            return "Not authorized"
        session_obj = await self._load_session(session_id)
        if not session_obj:
            return "Сессия не найдена"
        if session_obj.status != VideoAnnounceSessionStatus.SELECTED:
            return "Сессия уже запущена"
        
        # Initialize render_order if not set
        params = self._get_selection_params(session_obj)
        if "render_order" not in params:
            pairs = await self._load_items_with_events(session_id)
            ready_pairs = [(item, ev) for item, ev in pairs if item.status == VideoAnnounceItemStatus.READY]
            params["render_order"] = [ev.id for _, ev in ready_pairs]
            async with self.db.get_session() as session:
                sess = await session.get(VideoAnnounceSession, session_id)
                if sess:
                    sess.selection_params = params
                    session.add(sess)
                    await session.commit()
        
        text, markup = await self._sort_view(session_id)
        
        if message:
            try:
                await message.edit_text(text, reply_markup=markup, parse_mode="HTML")
                return "Сортировка"
            except Exception:
                pass
        
        await self.bot.send_message(self.chat_id, text, reply_markup=markup, parse_mode="HTML")
        return "Сортировка"

    async def move_item(
        self, session_id: int, event_id: int, direction: str, message: types.Message
    ) -> str:
        """Move an item up or down in render order."""
        if not await self.ensure_access():
            return "Not authorized"
        
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"
            
            params = self._get_selection_params(sess)
            render_order = list(params.get("render_order", []))
            
            if event_id not in render_order:
                return "Событие не в списке"
            
            idx = render_order.index(event_id)
            
            if direction == "up" and idx > 0:
                render_order[idx], render_order[idx - 1] = render_order[idx - 1], render_order[idx]
            elif direction == "down" and idx < len(render_order) - 1:
                render_order[idx], render_order[idx + 1] = render_order[idx + 1], render_order[idx]
            else:
                return "Нельзя переместить"
            
            params["render_order"] = render_order
            sess.selection_params = params
            session.add(sess)
            await session.commit()
        
        # Update the sort screen
        text, markup = await self._sort_view(session_id)
        try:
            await message.edit_text(text, reply_markup=markup, parse_mode="HTML")
        except Exception:
            pass
        return "Перемещено"

    async def finish_sorting(self, session_id: int, message: types.Message) -> str:
        """Close sort screen and return to selection view."""
        if not await self.ensure_access():
            return "Not authorized"
        
        try:
            await self.bot.delete_message(self.chat_id, message.message_id)
        except Exception:
            pass
        
        await self._send_selection_ui(session_id)
        return "Готово"

    def _build_input_message(
        self, session_obj: VideoAnnounceSession, result: SelectionBuildResult
    ) -> str:
        params = self._get_selection_params(session_obj)
        instruction = (str(params.get("instruction") or "").strip())
        ranked_map = {r.event.id: r for r in result.ranked}
        lines = [
            f"Сессия #{session_obj.id}: INPUT",
            f"Диапазон: {self._date_range_label(params)}",
            f"Инструкция: {html.escape(instruction[:300]) if instruction else '—'}",
            f"Всего кандидатов: {len(result.candidates)}",
            "📥 Кандидаты:",
            "<blockquote expandable>",
        ]

        def candidate_sort_key(ev: Event) -> tuple[datetime, int]:
            parsed_dt = self._parse_event_datetime(ev)
            if parsed_dt is None:
                parsed_dt = datetime.max.replace(tzinfo=timezone.utc)
            r = ranked_map.get(ev.id)
            # Use large number for unranked to put them at the end of the day list
            # or 0 if we want them first. Assuming ranked have positions 1..N.
            # Using 999999 to put unranked last.
            pos = r.position if r and r.position else 999999
            return (parsed_dt, pos)

        sorted_candidates = sorted(result.candidates, key=candidate_sort_key)
        for ev in sorted_candidates:
            r = ranked_map.get(ev.id)
            marker = "✅" if ev.id in result.default_ready_ids else "⬜"
            emoji = self._normalize_emoji(ev.emoji)
            date_label = self._format_event_schedule(ev, params)
            include_count = getattr(ev, "video_include_count", 0) or 0
            promo_marker = " · 🔥PROMO" if (r and r.mandatory) or include_count > 0 else ""
            score = f" · {r.score:.1f}" if r and r.score is not None else ""
            reason = f" · {html.escape(r.reason[:140])}" if r and r.reason else ""
            ranking_label = f"#{r.position}" if r else "—"
            lines.append(
                f"{marker} {ranking_label} · {date_label} · {emoji} {self._format_title(ev)}{promo_marker}{score}{reason}"
            )
        lines.append("</blockquote>")
        return "\n".join(lines)

    async def _send_input_overview(
        self, session_obj: VideoAnnounceSession, result: SelectionBuildResult
    ) -> None:
        text = self._build_input_message(session_obj, result)
        await self.bot.send_message(self.chat_id, text, parse_mode="HTML")

    async def _send_selection_posts(
        self,
        session_obj: VideoAnnounceSession,
        result: SelectionBuildResult,
        *,
        selection_message: types.Message | None = None,
    ) -> None:
        await self._send_input_overview(session_obj, result)
        if selection_message:
            await self._update_selection_message(selection_message, session_obj.id)
        else:
            await self._send_selection_ui(session_obj.id)
        await self._send_intro_controls(session_obj)

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
        result = await self._build_and_store_selection(
            sess,
            candidates=candidates,
            preserve_existing=preserve_existing,
        )
        await self._send_selection_posts(sess, result)
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

    async def _render_and_notify(
        self,
        session_obj: VideoAnnounceSession,
        ranked,
        *,
        status_message: tuple[int, int] | None = None,
        payload: RenderPayload | None = None,
        payload_json: str | None = None,
    ) -> None:
        client = KaggleClient()
        status_chat_id = status_message[0] if status_message else self.chat_id
        status_message_id = status_message[1] if status_message else None
        if not status_message:
            status_message = await update_status_message(
                self.bot,
                session_obj,
                {},
                chat_id=status_chat_id,
                allow_send=True,
                note="Готовим Kaggle",
            )
            if status_message:
                status_chat_id, status_message_id = status_message
                remember_status_message(
                    session_obj.id, status_chat_id, status_message_id
                )
        finalized = []
        if ranked:
            try:
                # We still might want finalized texts for debugging, but not for dataset export as per requirement 3
                finalized = await prepare_final_texts(self.db, session_obj.id, ranked)
            except Exception:
                logger.exception("video_announce: failed to prepare final texts")
        try:
            json_text = payload_json
            if not json_text:
                payload = payload or await self._build_render_payload(session_obj, ranked)
                json_text = payload_as_json(payload, timezone.utc)

            if payload:
                # Validation Step: Check for photo_urls presence
                missing_photos = []
                for item in payload.items:
                    ev = next((e for e in payload.events if e.id == item.event_id), None)
                    if ev and item.status == VideoAnnounceItemStatus.READY:
                        urls = getattr(ev, "photo_urls", []) or []
                        if not any(urls):
                            missing_photos.append(ev.id)

                if missing_photos:
                    error_msg = f"Ошибка: у событий {missing_photos} отсутствуют photo_urls, запуск Kaggle отменён."
                    await self.bot.send_message(self.chat_id, error_msg)
                    await self._mark_failed(session_obj.id, error_msg)
                    failed = await self._load_session(session_obj.id)
                    if failed:
                        await update_status_message(
                            self.bot,
                            failed,
                            {},
                            chat_id=status_chat_id,
                            message_id=status_message_id,
                            allow_send=True,
                            note="Ошибка: нет фото",
                        )
                    return

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
                # JSON is already sent as file attachment, no need for text duplicate
                await self.bot.send_message(self.chat_id, preview or "Нет событий")
            dataset_slug = await self._create_dataset(session_obj, json_text, finalized)
            
            # Wait for Kaggle to fully process the dataset before attaching to kernel
            logger.info("video_announce: waiting 15s for dataset to be processed...")
            await asyncio.sleep(15)

            kernel_ref = session_obj.kaggle_kernel_ref
            if not kernel_ref:
                # Fallback: use default local kernel with dynamic username
                username = os.getenv("KAGGLE_USERNAME", "")
                kernel_ref = f"{username}/video-announce-renderer" if username else "video-announce-renderer"

            actual_ref = await self._push_kernel(client, dataset_slug, kernel_ref)
            if actual_ref != kernel_ref:
                logger.info("Kernel ref changed from %s to %s", kernel_ref, actual_ref)
                kernel_ref = actual_ref

            session_obj.kaggle_dataset = dataset_slug
            session_obj.kaggle_kernel_ref = kernel_ref
            await self._store_kaggle_meta(session_obj.id, dataset_slug, kernel_ref)
            try:
                kaggle_status = await asyncio.to_thread(
                    client.get_kernel_status, kernel_ref
                )
            except Exception:
                logger.exception("video_announce: failed to fetch initial status")
                kaggle_status = {}
            status_message = await update_status_message(
                self.bot,
                session_obj,
                kaggle_status,
                chat_id=status_chat_id,
                message_id=status_message_id,
                allow_send=True,
                note="Kernel запущен",
            )
            if status_message:
                status_chat_id, status_message_id = status_message
                remember_status_message(
                    session_obj.id, status_chat_id, status_message_id
                )
        except Exception:
            logger.exception("video_announce: failed to push kaggle job")
            await self._mark_failed(session_obj.id, "kaggle push failed")
            failed = await self._load_session(session_obj.id)
            if failed:
                await update_status_message(
                    self.bot,
                    failed,
                    {},
                    chat_id=status_chat_id,
                    message_id=status_message_id,
                    allow_send=True,
                    note="Ошибка запуска Kaggle",
                )
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
                status_chat_id=status_chat_id,
                status_message_id=status_message_id,
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

    async def _send_payload_file(
        self, session_obj: VideoAnnounceSession, json_text: str, *, caption: str
    ) -> None:
        filename = f"video_payload_{session_obj.id}.json"
        document = types.BufferedInputFile(json_text.encode("utf-8"), filename=filename)
        await self.bot.send_document(
            self.chat_id, document, caption=caption, disable_notification=True
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
                about_text = normalize_about_with_fallback(
                    r.about,
                    ocr_text=r.poster_ocr_text,
                )
                description_text = (r.description or "").strip()
                if item.status == VideoAnnounceItemStatus.READY:
                    item.final_about = about_text
                    if description_text:
                        item.final_description = description_text
                session.add(item)
            await session.commit()

    async def _recalculate_selection(
        self, session_obj: VideoAnnounceSession
    ) -> SelectionBuildResult:
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
        show_all = bool(params.get("show_all_candidates", False))
        instruction = (str(params.get("instruction") or "").strip())
        lines = [
            f"Сессия #{session_id}: SELECTED",
            f"Диапазон: {self._date_range_label(params)}",
            "Выберите события для рендера:",
        ]
        if show_all:
            lines.append(f"Показываем все кандидаты: {len(pairs)}")
        else:
            lines.append(f"Показываем топ-{default_selected_max} + промо (всего {len(pairs)})")
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
        # Show all candidates or limited view
        if show_all:
            visible_pairs = list(pairs)  # Show all
        else:
            visible_pairs = self._visible_pairs(pairs, visible_limit=default_selected_max)
        for item, ev in visible_pairs:
            marker = "✅" if item.status == VideoAnnounceItemStatus.READY else "⬜"
            emoji = self._normalize_emoji(ev.emoji)
            date_label = self._format_event_schedule(ev, params)
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
            # Use 5-column layout for compact display
            keyboard.extend(self._chunk_buttons(toggle_buttons, size=5))
        if allow_edit:
            # Expand/Collapse button
            if show_all:
                expand_btn = types.InlineKeyboardButton(
                    text="− Свернуть", callback_data=f"vidsel:{session_id}:collapse"
                )
            else:
                expand_btn = types.InlineKeyboardButton(
                    text="+ Все кандидаты", callback_data=f"vidsel:{session_id}:expand"
                )
            keyboard.append([expand_btn])
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="🔀 Сортировка", callback_data=f"vidsort:{session_id}:show"
                    ),
                ]
            )
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
            
            # Handle expand/collapse - just toggle param and update UI, no recalculation
            if action == "expand":
                params["show_all_candidates"] = True
                sess.selection_params = params
                session.add(sess)
                await session.commit()
                await self._update_selection_message(message, session_id)
                return "Развёрнуто"
            elif action == "collapse":
                params["show_all_candidates"] = False
                sess.selection_params = params
                session.add(sess)
                await session.commit()
                await self._update_selection_message(message, session_id)
                return "Свёрнуто"
            
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
        result = await self._recalculate_selection(sess)
        await self._send_selection_posts(sess, result, selection_message=message)
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
        await self._send_payload_file(
            session_obj,
            json_text,
            caption="Файл с текущим Kaggle JSON", 
        )
        await self.bot.send_message(self.chat_id, "\n".join(preview_lines) or "Нет событий")
        return "Сформировано"

    async def show_kernel_selection(self, session_id: int, message: types.Message | None = None) -> str:
        if not await self._has_access():
            return "Not authorized"

        keyboard: list[list[types.InlineKeyboardButton]] = []
        
        # 1. Local kernels from repository (with 📦 icon)
        local_kernels = list_local_kernels()
        if local_kernels:
            for k in local_kernels:
                ref = k.get("ref") or ""
                title = k.get("title") or ref
                if not ref:
                    continue
                keyboard.append(
                    [
                        types.InlineKeyboardButton(
                            text=f"📦 {title} (репозиторий)",
                            callback_data=f"vidkernel:{session_id}:{ref}",
                        )
                    ]
                )
        
        # 2. Kaggle kernels (with 📓 icon)
        username = os.getenv("KAGGLE_USERNAME")
        client = KaggleClient()
        try:
            kaggle_kernels = await asyncio.to_thread(client.kernels_list, user=username, page_size=10)
        except Exception:
            logger.exception("video_announce: failed to list Kaggle kernels")
            kaggle_kernels = []

        if kaggle_kernels:
            for k in kaggle_kernels:
                ref = k.get("ref") or ""
                title = k.get("title") or ref
                if not ref:
                    continue
                keyboard.append(
                    [
                        types.InlineKeyboardButton(
                            text=f"📓 {title} (Kaggle)",
                            callback_data=f"vidkernel:{session_id}:{ref}",
                        )
                    ]
                )
        
        # Refresh button
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="🔄 Обновить список",
                    callback_data=f"vidrender:{session_id}",
                )
            ]
        )

        if not local_kernels and not kaggle_kernels:
            return "Нет доступных kernels"

        text = f"Выберите Notebook для запуска:\n📦 — из репозитория\n📓 — с Kaggle"
        if message:
            try:
                await message.edit_text(text, reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard))
            except Exception:
                await self.bot.send_message(self.chat_id, text, reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard))
        else:
            await self.bot.send_message(self.chat_id, text, reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard))

        return "Выбор kernel"


    async def save_kernel_and_start(self, session_id: int, kernel_ref: str, message: types.Message | None = None) -> str:
        if not await self._has_access():
             return "Not authorized"

        logger.info(
            "video_announce: kernel selected session_id=%s kernel_ref=%s",
            session_id,
            kernel_ref,
        )
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            sess.kaggle_kernel_ref = kernel_ref
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

        payload_json = self._extract_import_payload_json(sess)
        if payload_json:
            return await self.start_import_render(session_id, payload_json, message=message)
        if self._is_import_session(sess):
            return "Payload не найден, импортируйте заново"
        return await self.start_render(session_id, message=message)

    async def start_import_render(
        self,
        session_id: int,
        payload_json: str,
        message: types.Message | None = None,
    ) -> str:
        if not await self._has_access():
            return "Not authorized"
        if await self.has_rendering():
            return "Уже есть активный рендер"
        if not payload_json.strip():
            return "Payload не найден"
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"
            if not sess.kaggle_kernel_ref:
                return "Kernel не выбран"
            if sess.status == VideoAnnounceSessionStatus.RENDERING:
                return "Сессия уже запущена"
            sess.status = VideoAnnounceSessionStatus.RENDERING
            sess.started_at = datetime.now(timezone.utc)
            sess.finished_at = None
            sess.error = None
            sess.video_url = None
            sess.kaggle_dataset = None
            params = (
                sess.selection_params
                if isinstance(sess.selection_params, dict)
                else {}
            )
            if params:
                params.pop(IMPORT_PAYLOAD_JSON_KEY, None)
                params.pop(IMPORT_PAYLOAD_FLAG_KEY, None)
                sess.selection_params = params
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
        await self.bot.send_message(
            self.chat_id,
            (
                f"Сессия #{session_id} запущена, готовим материалы. "
                f"Kernel: {sess.kaggle_kernel_ref}"
            ),
        )
        status_message = await update_status_message(
            self.bot,
            sess,
            {},
            chat_id=self.chat_id,
            allow_send=True,
            note="Готовим Kaggle",
        )
        asyncio.create_task(
            self._render_and_notify(
                sess,
                [],
                status_message=status_message,
                payload=None,
                payload_json=payload_json,
            )
        )
        return "Рендеринг запущен"

    async def start_render(
        self,
        session_id: int,
        message: types.Message | None = None,
        *,
        limit_scenes: int | None = None,
    ) -> str:
        if not await self._has_access():
            return "Not authorized"
        if await self.has_rendering():
            return "Уже есть активный рендер"
        ranked = await self._load_ranked_events(session_id, ready_only=True)
        if limit_scenes is not None:
            limit = int(limit_scenes)
            if limit > 0:
                ranked = ranked[:limit]
        if not ranked:
            return "Нет выбранных событий"
        payload: RenderPayload | None = None
        payload_json: str | None = None
        async with self.db.get_session() as session:
            sess = await session.get(VideoAnnounceSession, session_id)
            if not sess:
                return "Сессия не найдена"

            if not sess.kaggle_kernel_ref:
                # If for some reason we got here without a kernel ref (should be caught by UI flow)
                return "Kernel не выбран"

            if sess.status != VideoAnnounceSessionStatus.SELECTED:
                return "Сессия уже запущена"
            
            # Load items and events for fill_missing_about
            ranked_ids = [r.event.id for r in ranked if r.event.id is not None]
            res_items = await session.execute(
                select(VideoAnnounceItem)
                .where(VideoAnnounceItem.session_id == session_id)
                .where(VideoAnnounceItem.status == VideoAnnounceItemStatus.READY)
                .where(VideoAnnounceItem.event_id.in_(ranked_ids))
            )
            ready_items = list(res_items.scalars().all())
            event_ids = [item.event_id for item in ready_items]
            ev_res = await session.execute(select(Event).where(Event.id.in_(event_ids)))
            events_map = {ev.id: ev for ev in ev_res.scalars().all()}
            
            # Fill missing about via LLM
            missing_about = await fill_missing_about(
                self.db,
                session_id,
                ready_items,
                events_map,
                bot=self.bot,
                notify_chat_id=self.chat_id,
            )
            
            # Save generated about to items
            if missing_about:
                for item in ready_items:
                    if item.event_id in missing_about:
                        item.final_about = missing_about[item.event_id]
                        session.add(item)
                await session.commit()
            
            payload = await self._build_render_payload(sess, ranked)
            payload_json = payload_as_json(payload, timezone.utc)
            sess.status = VideoAnnounceSessionStatus.RENDERING
            sess.started_at = datetime.now(timezone.utc)
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
        if message:
            try:
                await self._update_selection_message(message, session_id)
            except Exception:
                pass
        await self.bot.send_message(
            self.chat_id, f"Сессия #{session_id} запущена, собираем материалы. Kernel: {sess.kaggle_kernel_ref}"
        )
        if payload_json:
            await self._send_payload_file(
                sess, payload_json, caption="Payload JSON перед запуском Kaggle"
            )
        status_message = await update_status_message(
            self.bot,
            sess,
            {},
            chat_id=self.chat_id,
            allow_send=True,
            note="Готовим Kaggle",
        )
        asyncio.create_task(
            self._render_and_notify(
                sess,
                ranked,
                status_message=status_message,
                payload=payload,
                payload_json=payload_json,
            )
        )
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
            if obj.status == VideoAnnounceSessionStatus.RENDERING:
                await self.bot.send_message(self.chat_id, "Сессия сейчас рендерится, дождитесь завершения")
                return
            obj.status = VideoAnnounceSessionStatus.SELECTED
            obj.started_at = None
            obj.finished_at = None
            obj.error = None
            obj.video_url = None
            obj.kaggle_dataset = None
            obj.kaggle_kernel_ref = None
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
        await self.bot.send_message(
            self.chat_id,
            f"Сессия #{session_id} подготовлена к перезапуску. Выберите kernel.",
        )
        kernel_msg = await self.show_kernel_selection(session_id, message=None)
        if kernel_msg != "Выбор kernel":
            await self.bot.send_message(self.chat_id, kernel_msg)

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

    async def force_reset_session(self, session_id: int) -> str:
        """Force-reset a stuck RENDERING session to FAILED status."""
        async with self.db.get_session() as session:
            obj = await session.get(VideoAnnounceSession, session_id)
            if not obj:
                return "Сессия не найдена"
            if obj.status != VideoAnnounceSessionStatus.RENDERING:
                return f"Сессия не в статусе RENDERING (текущий: {obj.status.value})"
            obj.status = VideoAnnounceSessionStatus.FAILED
            obj.finished_at = datetime.now(timezone.utc)
            obj.error = "manual force reset"
            await session.commit()
            logger.warning(
                "video_announce: session %s force-reset by user to FAILED",
                session_id,
            )
        await self.bot.send_message(
            self.chat_id,
            f"⚠️ Сессия #{session_id} принудительно сброшена в FAILED.\n"
            "UI разблокирован, можно создавать новые сессии.",
        )
        return f"Сессия #{session_id} сброшена"

    def _copy_assets(self, tmp_path: Path) -> None:
        assets_dir = Path(__file__).resolve().parent / "assets"
        # Requirement: "Kaggle dataset contains only: payload.json, original *.ttf, Pulsarium.mp3, dataset-metadata.json"
        # We need to find the font. The example says "Oswald-VariableFont_wght.ttf"
        font_name = "BebasNeue-Bold.ttf"
        assets = [
            (assets_dir / font_name, tmp_path / font_name),
            (assets_dir / "Pulsarium.mp3", tmp_path / "Pulsarium.mp3"),
        ]
        logger.info(
            "video_announce: copying assets from %s, looking for %s files",
            assets_dir,
            len(assets),
        )
        missing = []
        for src, dest in assets:
            if not src.exists():
                logger.error("video_announce: MISSING required asset %s", src)
                missing.append(src.name)
                continue
            shutil.copy2(src, dest)
            logger.info("video_announce: copied asset %s (%s bytes)", dest.name, dest.stat().st_size)
        if missing:
            raise RuntimeError(f"Missing required assets: {missing}. Check that assets folder is deployed.")

    async def _create_dataset(
        self, session_obj: VideoAnnounceSession, json_text: str, finalized
    ) -> str:
        username = os.getenv("KAGGLE_USERNAME", "")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME not set")
        slug = f"video-session-{session_obj.id}"
        dataset_id = f"{username}/{slug}"
        meta = {
            "title": f"Video Afisha Session {session_obj.id}",
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "dataset-metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            payload_path = tmp_path / "payload.json"
            payload_path.write_text(json_text, encoding="utf-8")
            logger.info(
                "video_announce: created payload.json (%s bytes)",
                payload_path.stat().st_size,
            )
            # Removed final_texts.json and images as per Requirement 3

            self._copy_assets(tmp_path)
            
            # Log all files in dataset before upload
            all_files = list(tmp_path.glob("*"))
            logger.info(
                "video_announce: dataset files before upload: %s",
                [(f.name, f.stat().st_size) for f in all_files],
            )
            
            total_size = sum(
                f.stat().st_size for f in tmp_path.glob("**/*") if f.is_file()
            )
            logger.info(
                "video_announce: dataset total size=%s bytes (limit=%sMB)",
                total_size,
                DATASET_PAYLOAD_MAX_MB,
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
        logger.info("video_announce: dataset created successfully id=%s", dataset_id)
        return dataset_id

    async def _push_kernel(
        self, client: KaggleClient, dataset_slug: str, kernel_ref: str | None = None
    ) -> str:
        if not kernel_ref:
            # Fallback (old behavior) should be avoided if we enforce selection
            raise RuntimeError("Kernel reference not provided")

        return await asyncio.to_thread(
            client.deploy_kernel_update, kernel_ref, dataset_slug
        )

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
    if prefix == "vidimport":
        _, profile = callback.data.split(":", 1)
        await scenario.prompt_payload_import(profile)
        await callback.answer("Ожидаю payload")
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
    if prefix == "vidforce_reset":
        try:
            _, session_id = callback.data.split(":", 1)
            msg = await scenario.force_reset_session(int(session_id))
            await callback.answer(msg, show_alert=True)
        except Exception:
            logger.exception("video_announce: force reset failed")
            await callback.answer("Ошибка сброса", show_alert=True)
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
                msg = await scenario.apply_period_preset(
                    session_id_int, preset_idx, message=callback.message
                )
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
