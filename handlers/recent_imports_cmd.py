from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from runtime import require_main_attr

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)

recent_imports_router = Router(name="recent_imports")

_MAX_TG_MESSAGE_LEN = 3600
_DEFAULT_HOURS = 24
_MAX_HOURS = 168
_SOURCE_LABELS: tuple[tuple[str, str], ...] = (
    ("telegram", "Telegram"),
    ("vk", "VK"),
    ("parse", "/parse"),
)
_STATUS_ICONS = {
    "created": "✅",
    "updated": "🔄",
}


@dataclass(slots=True, frozen=True)
class _RecentImportRow:
    source_group: str
    event_id: int
    title: str
    telegraph_url: str | None
    status: str
    last_imported_at: datetime | None


def _chunk_lines(lines: list[str], *, max_len: int = _MAX_TG_MESSAGE_LEN) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        text = str(line or "")
        line_len = len(text) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current).strip())
            current = [text]
            current_len = line_len
        else:
            current.append(text)
            current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _parse_hours_arg(text: str | None) -> int:
    raw = str(text or "").strip()
    if not raw:
        return _DEFAULT_HOURS
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        return _DEFAULT_HOURS
    arg = parts[1].strip()
    if not arg:
        return _DEFAULT_HOURS
    if not arg.isdigit():
        raise ValueError("hours must be an integer")
    hours = int(arg)
    if hours < 1 or hours > _MAX_HOURS:
        raise ValueError(f"hours must be between 1 and {_MAX_HOURS}")
    return hours


def _parse_sqlite_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _utc_sql(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _telegraph_url(telegraph_url: str | None, telegraph_path: str | None) -> str | None:
    url = str(telegraph_url or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    path = str(telegraph_path or "").strip().lstrip("/")
    if path:
        return f"https://telegra.ph/{path}"
    return None


async def _require_superadmin(db: Database, user_id: int) -> bool:
    from models import User

    async with db.get_session() as session:
        user = await session.get(User, int(user_id))
        return bool(user and not user.blocked and user.is_superadmin)


async def _load_recent_import_rows(
    db: Database,
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, list[_RecentImportRow]]:
    start_raw = _utc_sql(start_utc)
    end_raw = _utc_sql(end_utc)
    out: dict[str, list[_RecentImportRow]] = {key: [] for key, _label in _SOURCE_LABELS}

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            WITH recent AS (
                SELECT
                    CASE
                        WHEN es.source_type LIKE 'telegram%' OR es.source_type = 'tg' THEN 'telegram'
                        WHEN es.source_type LIKE 'vk%' THEN 'vk'
                        WHEN es.source_type LIKE 'parser:%' THEN 'parse'
                        ELSE ''
                    END AS source_group,
                    e.id AS event_id,
                    COALESCE(NULLIF(TRIM(e.title), ''), 'событие') AS title,
                    NULLIF(TRIM(COALESCE(e.telegraph_url, '')), '') AS telegraph_url,
                    NULLIF(TRIM(COALESCE(e.telegraph_path, '')), '') AS telegraph_path,
                    MAX(es.imported_at) AS last_imported_at,
                    CASE
                        WHEN datetime(e.added_at) >= datetime(?) AND datetime(e.added_at) < datetime(?) THEN 1
                        ELSE 0
                    END AS is_created
                FROM event_source es
                JOIN event e ON e.id = es.event_id
                WHERE datetime(es.imported_at) >= datetime(?) AND datetime(es.imported_at) < datetime(?)
                  AND (
                    es.source_type LIKE 'telegram%'
                    OR es.source_type = 'tg'
                    OR es.source_type LIKE 'vk%'
                    OR es.source_type LIKE 'parser:%'
                  )
                GROUP BY 1, 2, 3, 4, 5, 7
            )
            SELECT source_group, event_id, title, telegraph_url, telegraph_path, last_imported_at, is_created
            FROM recent
            WHERE source_group != ''
            ORDER BY
                CASE source_group
                    WHEN 'telegram' THEN 1
                    WHEN 'vk' THEN 2
                    WHEN 'parse' THEN 3
                    ELSE 9
                END,
                datetime(last_imported_at) DESC,
                event_id DESC
            """,
            (start_raw, end_raw, start_raw, end_raw),
        )
        rows = await cur.fetchall()

    for source_group, event_id, title, telegraph_url, telegraph_path, last_imported_at, is_created in rows or []:
        key = str(source_group or "").strip()
        if key not in out:
            continue
        try:
            ev_id = int(event_id)
        except Exception:
            continue
        out[key].append(
            _RecentImportRow(
                source_group=key,
                event_id=ev_id,
                title=str(title or "").strip() or "событие",
                telegraph_url=_telegraph_url(telegraph_url, telegraph_path),
                status="created" if int(is_created or 0) > 0 else "updated",
                last_imported_at=_parse_sqlite_ts(last_imported_at),
            )
        )
    return out


def _render_pages(
    *,
    hours: int,
    start_local: datetime,
    end_local: datetime,
    tz_label: str,
    grouped_rows: dict[str, list[_RecentImportRow]],
) -> list[str]:
    body_lines: list[str] = [
        f"Окно: {start_local.strftime('%Y-%m-%d %H:%M')} → {end_local.strftime('%Y-%m-%d %H:%M')} ({html.escape(tz_label)})",
        "",
    ]
    for source_key, source_label in _SOURCE_LABELS:
        rows = list(grouped_rows.get(source_key) or [])
        created_count = sum(1 for row in rows if row.status == "created")
        updated_count = sum(1 for row in rows if row.status == "updated")
        body_lines.append(
            f"<b>{html.escape(source_label)}</b> — {len(rows)} "
            f"(создано {created_count}, обновлено {updated_count})"
        )
        if not rows:
            body_lines.append("—")
            body_lines.append("")
            continue
        for row in rows:
            status_icon = _STATUS_ICONS.get(row.status, "•")
            title = html.escape(row.title)
            if row.telegraph_url:
                title_part = f'<a href="{html.escape(row.telegraph_url)}">{title}</a>'
            else:
                title_part = title
            body_lines.append(f"id={row.event_id} {status_icon} {title_part}")
        body_lines.append("")

    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    header = f"🧾 <b>События из Telegram / VK / /parse за последние {hours} ч.</b>"
    body_max_len = max(120, int(_MAX_TG_MESSAGE_LEN) - 220)
    body_chunks = _chunk_lines(body_lines, max_len=body_max_len)
    if not body_chunks:
        body_chunks = ["Нет данных."]
    if len(body_chunks) == 1:
        return [f"{header}\n{body_chunks[0]}".strip()]
    total = len(body_chunks)
    return [
        f"{header}\nСтраница {idx}/{total}\n\n{chunk}".strip()
        for idx, chunk in enumerate(body_chunks, start=1)
    ]


async def _send_recent_imports_report(message: Message, db: Database, *, hours: int) -> None:
    tz = require_main_attr("LOCAL_TZ")
    end_utc = datetime.now(timezone.utc).replace(microsecond=0)
    start_utc = end_utc - timedelta(hours=max(1, int(hours)))
    grouped_rows = await _load_recent_import_rows(db, start_utc=start_utc, end_utc=end_utc)
    pages = _render_pages(
        hours=hours,
        start_local=start_utc.astimezone(tz),
        end_local=end_utc.astimezone(tz),
        tz_label=tz.tzname(None) or "UTC",
        grouped_rows=grouped_rows,
    )
    for page in pages:
        await message.answer(page, parse_mode="HTML", disable_web_page_preview=True)


@recent_imports_router.message(Command("recent_imports"))
async def cmd_recent_imports(message: Message) -> None:
    get_db = require_main_attr("get_db")
    db = get_db()
    if db is None:
        await message.answer("❌ База данных ещё не инициализирована. Попробуйте позже.")
        return
    try:
        user_id = int(message.from_user.id)
    except Exception:
        await message.answer("❌ Не удалось определить пользователя.")
        return
    if not await _require_superadmin(db, user_id):
        await message.answer("❌ Команда доступна только администраторам.")
        return

    try:
        hours = _parse_hours_arg(message.text)
    except ValueError as exc:
        await message.answer(f"❌ {exc}. Использование: /recent_imports [hours]")
        return

    status_message = None
    try:
        status_message = await message.answer("⏳ Готовлю recent imports…")
    except Exception:
        status_message = None

    try:
        await _send_recent_imports_report(message, db, hours=hours)
    except Exception:
        logger.exception("recent_imports: failed")
        await message.answer("❌ Не удалось построить отчёт. Проверьте логи.")
    finally:
        if status_message is not None:
            try:
                await status_message.delete()
            except Exception:
                pass
