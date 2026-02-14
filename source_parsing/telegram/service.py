import asyncio
import base64
import html
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqlalchemy import select

from db import Database
from models import TelegramSource, TelegramSourceForceMessage
from source_parsing.telegram.deduplication import get_month_context_urls
from source_parsing.telegram.handlers import (
    TelegramMonitorEventInfo,
    TelegramMonitorReport,
    process_telegram_results,
)
from video_announce.kaggle_client import KaggleClient

from .split_secrets import encrypt_secret

logger = logging.getLogger(__name__)

_BOT_USERNAME_CACHE: str | None = None


async def _resolve_bot_username(bot) -> str | None:
    global _BOT_USERNAME_CACHE
    if _BOT_USERNAME_CACHE:
        return _BOT_USERNAME_CACHE
    try:
        me = await bot.get_me()
        username = (getattr(me, "username", None) or "").strip().lstrip("@")
        if username:
            _BOT_USERNAME_CACHE = username
            return username
    except Exception:
        return None
    return None


def _log_deeplink(bot_username: str, event_id: int) -> str:
    safe = (bot_username or "").strip().lstrip("@")
    return f"https://t.me/{safe}?start=log_{int(event_id)}"

CONFIG_DATASET_CIPHER = os.getenv("TG_MONITORING_CONFIG_CIPHER", "telegram-monitor-cipher")
CONFIG_DATASET_KEY = os.getenv("TG_MONITORING_CONFIG_KEY", "telegram-monitor-key")
KEEP_DATASETS = os.getenv("TG_MONITORING_KEEP_DATASETS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

# Prevent overlapping runs (manual UI vs scheduler) in a single bot process.
# Overlapping Kaggle kernels can reuse the same Telegram session concurrently and trigger
# Telegram-side throttling / auth-key issues.
_RUN_LOCK = asyncio.Lock()

KERNEL_REF = os.getenv("TG_MONITORING_KERNEL_REF", "artkoder/telegram-monitor-bot")
KERNEL_PATH = Path(os.getenv("TG_MONITORING_KERNEL_PATH", "kaggle/TelegramMonitor"))

DATASET_PROPAGATION_WAIT_SECONDS = int(os.getenv("TG_MONITORING_DATASET_WAIT", "30"))
POLL_INTERVAL_SECONDS = int(os.getenv("TG_MONITORING_POLL_INTERVAL", "30"))
TIMEOUT_MINUTES = int(os.getenv("TG_MONITORING_TIMEOUT_MINUTES", "90"))
KAGGLE_STARTUP_WAIT_SECONDS = int(os.getenv("TG_MONITORING_STARTUP_WAIT", "10"))
MAX_TG_MESSAGE_LEN = int(os.getenv("TG_MONITORING_MAX_MESSAGE_LEN", "3800"))
KEEP_FORCE_MESSAGE_IDS = (os.getenv("TG_MONITORING_KEEP_FORCE_MESSAGE_IDS") or "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _read_env_file_value(key: str) -> str | None:
    path = Path(".env")
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip()
    except Exception:
        return None
    return None


def _get_env_value(key: str) -> str:
    value = (os.getenv(key) or "").strip()
    if value:
        return value
    fallback = _read_env_file_value(key)
    return (fallback or "").strip()


def _require_env(key: str) -> str:
    value = _get_env_value(key)
    if not value:
        raise RuntimeError(f"{key} is missing")
    return value


def _parse_auth_bundle(env_key: str) -> dict[str, Any] | None:
    bundle_b64 = _get_env_value(env_key)
    if not bundle_b64:
        return None
    try:
        raw = base64.urlsafe_b64decode(bundle_b64.encode("ascii")).decode("utf-8")
        bundle = json.loads(raw)
    except Exception as exc:  # pragma: no cover - validation only
        raise RuntimeError(f"Invalid {env_key}: {exc}") from exc
    required_keys = [
        "session",
        "device_model",
        "system_version",
        "app_version",
        "lang_code",
        "system_lang_code",
    ]
    missing = [key for key in required_keys if not bundle.get(key)]
    if missing:
        raise RuntimeError(f"{env_key} missing keys: {', '.join(missing)}")
    return bundle


def _require_kaggle_username() -> str:
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not set")
    return username


async def _build_config_payload(
    db: Database,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    async with db.get_session() as session:
        res = await session.execute(
            select(TelegramSource).where(TelegramSource.enabled.is_(True))
        )
        sources = res.scalars().all()
        force_map: dict[int, list[int]] = {}
        source_ids = [src.id for src in sources if src.id is not None]
        if source_ids:
            res_force = await session.execute(
                select(TelegramSourceForceMessage).where(
                    TelegramSourceForceMessage.source_id.in_(source_ids)
                )
            )
            for row in res_force.scalars().all():
                force_map.setdefault(row.source_id, []).append(row.message_id)
            if force_map and not KEEP_FORCE_MESSAGE_IDS:
                # Treat force-message requests as one-shot by default to avoid re-processing
                # the same old posts on every monitoring run.
                await session.execute(
                    TelegramSourceForceMessage.__table__.delete().where(
                        TelegramSourceForceMessage.source_id.in_(list(force_map.keys()))
                    )
                )
                await session.commit()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "username": src.username,
                "last_scanned_message_id": src.last_scanned_message_id,
                "default_location": src.default_location,
                "default_ticket_link": src.default_ticket_link,
                "trust_level": src.trust_level,
                "force_message_ids": sorted(set(force_map.get(src.id or -1, []))),
            }
            for src in sources
        ],
        "channels": [src.username for src in sources],
        "telegraph_urls": await get_month_context_urls(db),
    }
    if run_id:
        payload["run_id"] = run_id
    return payload


def _build_secrets_payload() -> str:
    bundle_raw = _get_env_value("TELEGRAM_AUTH_BUNDLE_S22")
    bundle = None
    bundle_ok = False
    if bundle_raw:
        try:
            bundle = _parse_auth_bundle("TELEGRAM_AUTH_BUNDLE_S22")
            bundle_ok = True
        except Exception as exc:  # pragma: no cover - validation only
            logger.warning("tg_monitor.secrets_payload invalid bundle: %s", exc)
    payload = {
        "TG_API_ID": _require_env("TG_API_ID"),
        "TG_API_HASH": _require_env("TG_API_HASH"),
        "GOOGLE_API_KEY": _require_env("GOOGLE_API_KEY"),
    }
    logger.info(
        "tg_monitor.secrets_payload bundle_len=%s bundle_ok=%s tg_session=%s days_back=%s limit=%s",
        len(bundle_raw) if bundle_raw else 0,
        bundle_ok,
        bool(_get_env_value("TG_SESSION")),
        _get_env_value("TG_MONITORING_DAYS_BACK"),
        _get_env_value("TG_MONITORING_LIMIT"),
    )
    if bundle_raw:
        payload["TELEGRAM_AUTH_BUNDLE_S22"] = _require_env("TELEGRAM_AUTH_BUNDLE_S22")
        if bundle and bundle.get("session"):
            payload["TG_SESSION"] = bundle["session"]
            payload["TG_MONITORING_ALLOW_TG_SESSION"] = "1"
    else:
        payload["TG_SESSION"] = _require_env("TG_SESSION")
        payload["TG_MONITORING_ALLOW_TG_SESSION"] = "1"
    # Include any additional Google API keys for pooled rate limiting.
    for key, value in os.environ.items():
        if key.startswith("GOOGLE_API_KEY") and key not in payload and value:
            payload[key] = value
    # Pass Supabase credentials to Kaggle for global rate limiting.
    for key in (
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_KEY",
        "SUPABASE_SCHEMA",
        "SUPABASE_DISABLED",
    ):
        value = (os.getenv(key) or "").strip()
        if value:
            payload[key] = value
    # Pass through monitoring config flags for Kaggle runtime (non-secret).
    for key, value in os.environ.items():
        if not value or key in payload:
            continue
        if key.startswith(("TG_MONITORING_", "TG_GEMMA_")):
            payload[key] = value
        elif key == "GOOGLE_API_LOCALNAME":
            payload[key] = value
    return json.dumps(payload, ensure_ascii=False)


def _slugify(value: str, *, max_len: int = 60) -> str:
    raw = (value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = raw.strip("-")
    if not raw:
        raw = uuid.uuid4().hex[:8]
    if len(raw) > max_len:
        raw = raw[:max_len].rstrip("-")
    return raw


def _build_dataset_slug(prefix: str, run_id: str) -> str:
    safe_prefix = _slugify(prefix, max_len=40)
    safe_run = _slugify(run_id, max_len=16)
    if safe_run:
        slug = f"{safe_prefix}-{safe_run}"
    else:
        slug = safe_prefix
    return slug[:60].rstrip("-")


def _create_dataset(
    client: KaggleClient,
    username: str,
    slug_suffix: str,
    title: str,
    writer,
) -> str:
    slug = f"{username}/{slug_suffix}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        writer(tmp_path)
        metadata = {
            "title": title,
            "id": slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp_path / "dataset-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            client.create_dataset(tmp_path)
        except Exception:
            logger.exception(
                "tg_monitor.dataset_create_failed retry=delete dataset=%s", slug
            )
            # Common case on frequent E2E/prod reruns: dataset id already exists.
            # Prefer creating a new dataset version instead of hard failing.
            try:
                client.create_dataset_version(
                    tmp_path,
                    version_notes=f"refresh {slug_suffix}",
                    quiet=True,
                    convert_to_csv=False,
                    dir_mode="zip",
                )
                logger.info(
                    "tg_monitor.dataset_version_created dataset=%s", slug
                )
                return slug
            except Exception:
                logger.exception(
                    "tg_monitor.dataset_version_failed dataset=%s", slug
                )
            try:
                client.delete_dataset(slug, no_confirm=True)
            except Exception:
                logger.exception(
                    "tg_monitor.dataset_delete_failed dataset=%s", slug
                )
            client.create_dataset(tmp_path)
    return slug


async def _prepare_kaggle_datasets(
    *,
    client: KaggleClient,
    config_payload: dict[str, Any],
    secrets_payload: str,
    run_id: str,
) -> tuple[str, str]:
    encrypted, fernet_key = encrypt_secret(secrets_payload)
    username = _require_kaggle_username()
    slug_suffix = _slugify(run_id, max_len=16)

    def write_cipher(path: Path) -> None:
        (path / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (path / "secrets.enc").write_bytes(encrypted)

    def write_key(path: Path) -> None:
        (path / "fernet.key").write_bytes(fernet_key)

    slug_cipher = _create_dataset(
        client,
        username,
        _build_dataset_slug(CONFIG_DATASET_CIPHER, run_id),
        f"Telegram Monitor Cipher {slug_suffix}",
        write_cipher,
    )
    slug_key = _create_dataset(
        client,
        username,
        _build_dataset_slug(CONFIG_DATASET_KEY, run_id),
        f"Telegram Monitor Key {slug_suffix}",
        write_key,
    )
    return slug_cipher, slug_key


def _kernel_has_code(kernel_path: Path) -> bool:
    if not kernel_path.exists():
        return False
    meta_path = kernel_path / "kernel-metadata.json"
    code_file = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            code_file = meta.get("code_file")
        except Exception:
            code_file = None
    if code_file and (kernel_path / code_file).exists():
        return True
    return bool(list(kernel_path.glob("*.ipynb")) or list(kernel_path.glob("*.py")))


def _kernel_ref_from_meta(kernel_path: Path) -> str:
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        return KERNEL_REF
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_id = meta.get("id") or meta.get("slug") or KERNEL_REF
        username = (os.getenv("KAGGLE_USERNAME") or "").strip()
        if username and isinstance(kernel_id, str):
            if "/" in kernel_id:
                owner, slug = kernel_id.split("/", 1)
                if slug and owner != username:
                    return f"{username}/{slug}"
            else:
                return f"{username}/{kernel_id}"
        return kernel_id
    except Exception:
        return KERNEL_REF


async def _push_kernel(
    client: KaggleClient,
    dataset_sources: list[str],
) -> str:
    kernel_ref = _kernel_ref_from_meta(KERNEL_PATH)
    if _kernel_has_code(KERNEL_PATH):
        logger.info("tg_monitor: pushing local kernel %s", KERNEL_PATH)
        client.push_kernel(kernel_path=KERNEL_PATH, dataset_sources=dataset_sources)
        return kernel_ref
    logger.info("tg_monitor: local kernel code missing, deploying remote kernel")
    for slug in dataset_sources:
        kernel_ref = client.deploy_kernel_update(kernel_ref, slug)
    return kernel_ref


async def _poll_kaggle_kernel(
    client: KaggleClient,
    kernel_ref: str,
    *,
    run_id: str | None = None,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> tuple[str, dict | None, float]:
    started = time.monotonic()
    deadline = started + TIMEOUT_MINUTES * 60
    last_status: dict | None = None
    attempt = 0

    async def _notify(phase: str, status: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, status)
        except Exception:
            logger.exception("tg_monitor: status callback failed phase=%s", phase)

    await _notify("poll", None)
    while time.monotonic() < deadline:
        attempt += 1
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status
        state = (status.get("status") or "").upper()
        await _notify("poll", status)
        logger.info(
            "tg_monitor.kernel_poll run_id=%s kernel_ref=%s attempt=%s status=%s elapsed=%.1fs",
            run_id,
            kernel_ref,
            attempt,
            state or "UNKNOWN",
            time.monotonic() - started,
        )
        if not state:
            logger.info(
                "tg_monitor.kernel_poll_details run_id=%s kernel_ref=%s status_payload=%s",
                run_id,
                kernel_ref,
                status,
            )
        if state == "COMPLETE":
            await _notify("complete", last_status)
            return "complete", last_status, time.monotonic() - started
        if state in ("ERROR", "FAILED", "CANCELLED"):
            await _notify("failed", last_status)
            return "failed", last_status, time.monotonic() - started
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    await _notify("timeout", last_status)
    return "timeout", last_status, time.monotonic() - started


async def _cleanup_datasets(dataset_slugs: list[str]) -> None:
    if KEEP_DATASETS:
        logger.info("tg_monitor.datasets_kept slugs=%s", dataset_slugs)
        return
    client = KaggleClient()
    for slug in dataset_slugs:
        if not slug:
            continue
        try:
            logger.info("tg_monitor.dataset_delete slug=%s", slug)
            await asyncio.to_thread(client.delete_dataset, slug)
        except Exception:
            logger.exception("tg_monitor.dataset_delete_failed slug=%s", slug)


async def _download_results(
    client: KaggleClient, kernel_ref: str, run_id: str
) -> Path:
    output_dir = Path(tempfile.gettempdir()) / f"tg-monitor-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        files = await asyncio.to_thread(
            client.download_kernel_output, kernel_ref, path=str(output_dir), force=True
        )
        for name in files:
            if Path(name).name == "telegram_results.json":
                return output_dir / name
        if attempt < max_attempts:
            await asyncio.sleep(5)
    raise RuntimeError("telegram_results.json not found in Kaggle output")


def _chunk_lines(lines: list[str], max_len: int = MAX_TG_MESSAGE_LEN) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def _format_kaggle_phase(phase: str) -> str:
    labels = {
        "prepare": "подготовка",
        "pushed": "запуск в Kaggle",
        "poll": "выполнение",
        "complete": "завершено",
        "failed": "ошибка",
        "timeout": "таймаут",
    }
    return labels.get(phase, phase)


def _format_kaggle_status(status: dict | None) -> str:
    if not status:
        return "неизвестен"
    state = status.get("status")
    failure_msg = status.get("failureMessage") or status.get("failure_message")
    if not state:
        return "неизвестен"
    result = str(state)
    if failure_msg:
        result += f" ({failure_msg})"
    return result


def _format_kaggle_status_message(
    phase: str,
    kernel_ref: str,
    status: dict | None,
) -> str:
    lines = [
        "🛰️ Kaggle: Telegram Monitor",
        f"Kernel: {kernel_ref or '—'}",
        f"Этап: {_format_kaggle_phase(phase)}",
    ]
    if status is not None:
        lines.append(f"Статус Kaggle: {_format_kaggle_status(status)}")
    return "\n".join(lines)


def _format_event_block(
    label: str,
    events: list[TelegramMonitorEventInfo],
    *,
    icon: str,
    bot_username: str | None = None,
) -> list[str]:
    lines = [f"{icon} <b>{html.escape(label)}</b>: {len(events)}", ""]
    for item in events:
        title = html.escape(item.title or "Без названия")
        if item.telegraph_url:
            safe_url = html.escape(item.telegraph_url, quote=True)
            line = f"• <a href=\"{safe_url}\">{title}</a>"
        else:
            line = f"• {title}"
        line += f" (id={item.event_id})"
        meta: list[str] = []
        if item.date:
            meta.append(item.date)
        if item.time:
            meta.append(item.time)
        if meta:
            line += f" — {' '.join(meta)}"
        lines.append(line)
        if item.source_link:
            lines.append(f"Источник: {html.escape(item.source_link)}")
        if item.telegraph_url:
            lines.append(f"Telegraph: {html.escape(item.telegraph_url)}")
        else:
            lines.append("Telegraph: ⏳ в очереди")
        if item.log_cmd:
            if bot_username:
                href = html.escape(_log_deeplink(bot_username, int(item.event_id)), quote=True)
                text = html.escape(item.log_cmd)
                lines.append(f"Лог: <a href=\"{href}\">{text}</a>")
            else:
                lines.append(f"Лог: {html.escape(item.log_cmd)}")
        ics_url = (item.ics_url or "").strip()
        if ics_url:
            lines.append(f"ICS: {html.escape(ics_url)}")
        else:
            # If there's a time, ICS is expected; otherwise it's optional.
            if (item.time or "").strip():
                lines.append("ICS: ⏳")
            else:
                lines.append("ICS: —")
        stats = item.fact_stats or {}
        try:
            photos = int(getattr(item, "photo_count", None) or 0)
        except Exception:
            photos = 0
        added_posters_raw = getattr(item, "added_posters", None)
        try:
            added_posters = int(added_posters_raw) if added_posters_raw is not None else None
        except Exception:
            added_posters = None
        if added_posters is None:
            photos_label = f"Иллюстрации: {'⚠️0' if photos == 0 else photos}"
        else:
            photos_label = f"Иллюстрации: +{added_posters}, всего {'⚠️0' if photos == 0 else photos}"
        if stats:
            added = int(stats.get("added") or 0)
            dup = int(stats.get("duplicate") or 0)
            conf = int(stats.get("conflict") or 0)
            note = int(stats.get("note") or 0)
            lines.append(f"Факты: ✅{added} ↩️{dup} ⚠️{conf} ℹ️{note} | {photos_label}")
        else:
            lines.append(f"Факты: — | {photos_label}")
        metrics = item.metrics or {}
        if isinstance(metrics, dict) and metrics:
            parts: list[str] = []
            views = metrics.get("views")
            likes = metrics.get("likes")
            if isinstance(views, int) and views >= 0:
                parts.append(f"views={views}")
            if isinstance(likes, int) and likes >= 0:
                parts.append(f"likes={likes}")
            reactions = metrics.get("reactions")
            if isinstance(reactions, dict):
                for k in ("👍", "❤", "❤️", "🔥"):
                    v = reactions.get(k)
                    if isinstance(v, int) and v > 0:
                        parts.append(f"{k}={v}")
            if parts:
                lines.append(f"Метрики: {html.escape(' '.join(parts))}")
        if item.source_excerpt:
            lines.append(f"Текст: {html.escape(item.source_excerpt)}")
        lines.append("")
    return lines


async def _send_event_details(
    bot,
    chat_id: int,
    report: TelegramMonitorReport,
) -> None:
    bot_username = await _resolve_bot_username(bot) if bot else None
    sections: list[list[str]] = []
    if report.created_events:
        sections.append(
            _format_event_block(
                "Созданные события",
                report.created_events,
                icon="✅",
                bot_username=bot_username,
            )
        )
    if report.merged_events:
        sections.append(
            _format_event_block(
                "Обновлённые события",
                report.merged_events,
                icon="🔄",
                bot_username=bot_username,
            )
        )
    if not sections:
        await bot.send_message(
            chat_id,
            (
                "ℹ️ <b>Smart Update (детали событий)</b>\n"
                "✅ Созданные события: 0\n"
                "🔄 Обновлённые события: 0\n"
                "Изменений по событиям в этом прогоне нет."
            ),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return
    for lines in sections:
        for chunk in _chunk_lines(lines):
            await bot.send_message(
                chat_id,
                chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )


def format_report(report: TelegramMonitorReport) -> str:
    lines = [
        "🕵️ <b>Telegram Monitor</b>",
        f"run_id: {report.run_id or '—'}",
    ]
    if report.generated_at:
        lines.append(f"generated_at: {report.generated_at}")
    lines.extend(
        [
            f"Источников: {report.sources_total}",
            f"Сообщений (Kaggle): {report.messages_scanned}",
            f"Сообщений с событиями: {report.messages_with_events}",
            f"Сообщений пропущено: {report.messages_skipped}",
            f"Событий извлечено: {report.events_extracted}",
            f"Создано: {report.events_created}",
            f"Смёрджено: {report.events_merged}",
            f"Пропущено: {report.events_skipped}",
        ]
    )
    if report.errors:
        lines.append("")
        lines.append("Ошибки:")
        for err in report.errors[:5]:
            lines.append(f"- {err}")
        if len(report.errors) > 5:
            lines.append(f"... ещё {len(report.errors) - 5}")
    return "\n".join(lines)


async def run_telegram_monitor(
    db: Database,
    *,
    bot=None,
    chat_id: int | None = None,
    run_id: str | None = None,
    send_progress: bool = False,
) -> TelegramMonitorReport:
    run_id = run_id or uuid.uuid4().hex
    logger.info("tg_monitor.start run_id=%s", run_id)
    if _RUN_LOCK.locked():
        logger.warning("tg_monitor.skip reason=already_running run_id=%s", run_id)
        if bot and chat_id:
            try:
                await bot.send_message(
                    chat_id,
                    "⏳ Мониторинг уже запущен, ждём завершения.",
                    parse_mode="HTML",
                )
            except Exception:
                logger.exception("tg_monitor: failed to notify already-running")
        return TelegramMonitorReport(run_id=run_id, errors=["already_running"])

    async with _RUN_LOCK:
        return await _run_telegram_monitor_locked(
            db,
            bot=bot,
            chat_id=chat_id,
            run_id=run_id,
            send_progress=send_progress,
        )


async def _run_telegram_monitor_locked(
    db: Database,
    *,
    bot=None,
    chat_id: int | None = None,
    run_id: str,
    send_progress: bool = False,
) -> TelegramMonitorReport:
    logger.info("tg_monitor.lock_acquired run_id=%s", run_id)
    kaggle_status_message_id: int | None = None
    kaggle_kernel_ref = KERNEL_REF
    config_payload = await _build_config_payload(db, run_id=run_id)
    sources = config_payload.get("sources") or []
    logger.info(
        "tg_monitor.config run_id=%s sources=%d telegraph_urls=%d",
        run_id,
        len(sources),
        len(config_payload.get("telegraph_urls") or []),
    )
    if sources:
        logger.info(
            "tg_monitor.sources sample=%s",
            [src.get("username") for src in sources[:5]],
        )
    secrets_payload = _build_secrets_payload()
    try:
        payload_keys = sorted((json.loads(secrets_payload) or {}).keys())
        logger.info("tg_monitor.secrets_payload_keys=%s", payload_keys)
    except Exception as exc:
        logger.warning("tg_monitor.secrets_payload_keys failed: %s", exc)

    async def _notify(text: str) -> None:
        if not (send_progress and bot and chat_id):
            return
        try:
            await bot.send_message(chat_id, text, parse_mode="HTML")
        except Exception:
            logger.exception("tg_monitor: failed to send progress update")

    async def _update_kaggle_status(
        phase: str,
        kernel_ref: str,
        status: dict | None,
    ) -> None:
        nonlocal kaggle_status_message_id, kaggle_kernel_ref
        if not (send_progress and bot and chat_id):
            return
        if kernel_ref:
            kaggle_kernel_ref = kernel_ref
        text = _format_kaggle_status_message(phase, kaggle_kernel_ref, status)
        try:
            if kaggle_status_message_id is None:
                sent = await bot.send_message(chat_id, text)
                kaggle_status_message_id = sent.message_id
            else:
                await bot.edit_message_text(
                    text=text,
                    chat_id=chat_id,
                    message_id=kaggle_status_message_id,
                )
        except Exception:
            logger.exception("tg_monitor: failed to update kaggle status")
    await _notify("🔧 Подготовка конфигов и секретов для Kaggle…")
    await _update_kaggle_status("prepare", kaggle_kernel_ref, None)
    client = KaggleClient()
    dataset_cipher = ""
    dataset_key = ""
    try:
        dataset_cipher, dataset_key = await _prepare_kaggle_datasets(
            client=client,
            config_payload=config_payload,
            secrets_payload=secrets_payload,
            run_id=run_id,
        )
        await _notify("🗄️ Kaggle datasets готовы, запускаю kernel…")
        logger.info(
            "tg_monitor.datasets created run_id=%s cipher=%s key=%s",
            run_id,
            dataset_cipher,
            dataset_key,
        )
        if DATASET_PROPAGATION_WAIT_SECONDS > 0:
            logger.info(
                "tg_monitor.dataset_wait seconds=%d",
                DATASET_PROPAGATION_WAIT_SECONDS,
            )
            await asyncio.sleep(DATASET_PROPAGATION_WAIT_SECONDS)

        kernel_ref = await _push_kernel(client, [dataset_cipher, dataset_key])
        await _update_kaggle_status("pushed", kernel_ref, None)
        await _notify(f"🛰️ Kaggle kernel запущен: {kernel_ref}")
        logger.info(
            "tg_monitor.kernel pushed run_id=%s kernel_ref=%s datasets=%s",
            run_id,
            kernel_ref,
            [dataset_cipher, dataset_key],
        )
        await asyncio.sleep(KAGGLE_STARTUP_WAIT_SECONDS)

        status, status_data, duration = await _poll_kaggle_kernel(
            client,
            kernel_ref,
            run_id=run_id,
            status_callback=_update_kaggle_status,
        )
        logger.info(
            "tg_monitor.kernel status run_id=%s kernel_ref=%s status=%s duration=%.1fs",
            run_id,
            kernel_ref,
            status,
            duration,
        )
        if status != "complete":
            failure = status_data.get("failureMessage") if status_data else ""
            await _notify(
                f"❌ Kaggle kernel завершился с ошибкой ({status}). {failure}".strip()
            )
            raise RuntimeError(f"Kaggle kernel failed ({status}) {failure}".strip())

        results_path = await _download_results(client, kernel_ref, run_id)
        await _notify("⬇️ Результаты Kaggle скачаны, запускаю импорт…")
        logger.info(
            "tg_monitor.results_downloaded run_id=%s kernel_ref=%s path=%s",
            run_id,
            kernel_ref,
            results_path,
        )
        report = await process_telegram_results(results_path, db)
        logger.info("tg_monitor: completed in %.1fs", duration)

        # Optional: drain only the jobs for the affected events. This makes local/E2E runs
        # deterministic without enabling a global outbox worker that would try to process
        # the entire prod snapshot backlog.
        drain_raw = (os.getenv("TG_MONITORING_DRAIN_EVENT_JOBS") or "").strip().lower()
        if drain_raw:
            drain = drain_raw in {"1", "true", "yes"}
        else:
            # Default for operator-triggered runs: build Telegraph/ICS for the touched events
            # so the report contains actionable links (Telegraph + /log + ICS).
            if bot and chat_id:
                drain = True
            else:
                # Non-interactive runs (no chat context): follow worker presence.
                worker_raw = (os.getenv("ENABLE_JOB_OUTBOX_WORKER") or "1").strip().lower()
                drain = worker_raw not in {"1", "true", "yes"}

        event_ids = sorted(
            {
                int(info.event_id)
                for info in (report.created_events + report.merged_events)
                if getattr(info, "event_id", None)
            }
        )

        drain_max = int(os.getenv("TG_MONITORING_DRAIN_MAX_EVENTS", "12") or 12)
        should_drain = bool(drain and bot and event_ids and len(event_ids) <= max(1, drain_max))
        if drain and bot and event_ids and not should_drain and chat_id:
            await bot.send_message(
                chat_id,
                f"ℹ️ Пропускаю синхронный drain (events={len(event_ids)} > max={drain_max}). "
                "Telegraph/ICS появятся позже через JobOutbox.",
                disable_web_page_preview=True,
            )
        if should_drain and bot and event_ids:
            try:
                from main import JobTask, run_event_update_jobs

                allowed = {JobTask.ics_publish, JobTask.telegraph_build}
                if chat_id:
                    await bot.send_message(
                        chat_id,
                        "⏳ Обновляю Telegraph/ICS для созданных/обновлённых событий…",
                        disable_web_page_preview=True,
                    )
                for eid in event_ids:
                    await run_event_update_jobs(
                        db,
                        bot,
                        event_id=eid,
                        allowed_tasks=allowed,
                    )
                logger.info(
                    "tg_monitor.drain_jobs completed events=%d allowed=%s",
                    len(event_ids),
                    [t.value for t in sorted(allowed, key=lambda x: x.value)],
                )
            except Exception:
                logger.exception("tg_monitor.drain_jobs failed")

        # Refresh per-event URLs/stats after optional draining so operator report is actionable.
        try:
            from source_parsing.telegram.handlers import refresh_telegram_monitor_event_info

            for info in (report.created_events + report.merged_events):
                await refresh_telegram_monitor_event_info(db, info)
        except Exception:
            logger.exception("tg_monitor.refresh_event_info failed")

        if bot and chat_id:
            try:
                await bot.send_message(chat_id, format_report(report), parse_mode="HTML")
                await _send_event_details(bot, chat_id, report)
            except Exception:
                logger.exception("tg_monitor: failed to send report")

        return report
    finally:
        if dataset_cipher or dataset_key:
            await _cleanup_datasets([dataset_cipher, dataset_key])


async def telegram_monitor_scheduler(
    db: Database,
    bot,
    *,
    run_id: str | None = None,
) -> None:
    admin_chat_id = os.getenv("ADMIN_CHAT_ID")
    chat_id = int(admin_chat_id) if admin_chat_id and admin_chat_id.isdigit() else None
    try:
        await run_telegram_monitor(db, bot=bot, chat_id=chat_id, run_id=run_id)
    except Exception:
        logger.exception("tg_monitor: scheduler failed run_id=%s", run_id)
