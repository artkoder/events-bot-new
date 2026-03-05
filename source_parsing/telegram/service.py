import asyncio
import base64
import html
import json
import logging
import math
import os
import re
import tempfile
import time
import uuid
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlsplit, urlunsplit
import ssl

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from db import Database
from models import TelegramSource, TelegramSourceForceMessage
from ops_run import finish_ops_run, start_ops_run
from source_parsing.telegram.deduplication import get_month_context_urls
from telegram_sources import normalize_tg_username
from source_parsing.telegram.handlers import (
    TelegramMonitorEventInfo,
    TelegramMonitorImportProgress,
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


def _preview_friendly_tg_post_url(url: str | None) -> str | None:
    """Return Telegram post URL with `?single` for better in-app preview behavior.

    Keep canonical DB/source URL unchanged; this is strictly a render-time href tweak.
    """
    raw = str(url or "").strip()
    if not raw:
        return None
    if not raw.startswith(("http://", "https://")):
        return raw
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw
    host = (parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "t.me":
        return raw

    path_parts = [p for p in str(parsed.path or "").split("/") if p]
    if len(path_parts) != 2:
        return raw
    username, message_id = path_parts
    if not username or not message_id.isdigit():
        return raw

    pairs = list(parse_qsl(parsed.query or "", keep_blank_values=True))
    if any(str(k) == "single" for k, _ in pairs):
        return raw
    pairs.append(("single", ""))
    query = "&".join(f"{k}={v}" if v else str(k) for k, v in pairs)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))

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


class TgMonitoringAlreadyRunningError(RuntimeError):
    pass


def _tg_monitor_lock_path() -> Path:
    # Keep lock scoped to environment/bot code, so prod/test bots on same host don't block each other.
    base_bot_code = (os.getenv("BOT_CODE") or "announcements").strip() or "announcements"
    bot_code = f"{base_bot_code}_test" if (os.getenv("DEV_MODE") or "").strip() == "1" else base_bot_code
    raw = (os.getenv("TG_MONITORING_GLOBAL_LOCK_PATH") or "").strip()
    if raw:
        return Path(raw)
    return Path(tempfile.gettempdir()) / f"eventsbot-{bot_code}-tg-monitoring.lock"


@contextlib.asynccontextmanager
async def _tg_monitor_global_lock(
    *,
    bot: Any | None,
    chat_id: int | None,
    send_progress: bool,
    run_id: str,
    purpose: str,
):
    """
    Cross-process lock to avoid concurrent TG monitoring/import runs.
    This prevents:
    - duplicated progress UI/messages
    - SQLite `database is locked` from multiple bot instances writing at once
    """
    lock_path = _tg_monitor_lock_path()
    fd = None
    try:
        try:
            import fcntl  # type: ignore
        except Exception:
            # Non-POSIX env: fall back to in-process lock only.
            yield
            return

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if send_progress and bot and chat_id:
                try:
                    await bot.send_message(
                        int(chat_id),
                        (
                            "⏳ Telegram мониторинг/импорт уже запущен в другом процессе.\n"
                            f"purpose={html.escape(purpose)}\n"
                            f"lock=<code>{html.escape(str(lock_path))}</code>"
                        ),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except Exception:
                    logger.exception("tg_monitor: failed to notify global lock held")
            raise TgMonitoringAlreadyRunningError(f"tg_monitor already running (global lock): {lock_path}")

        # Best-effort: record run metadata for debugging (does not affect flock semantics).
        try:
            fd.seek(0)
            fd.truncate(0)
            fd.write(f"run_id={run_id}\npurpose={purpose}\nstarted_at={datetime.now(timezone.utc).isoformat()}\n")
            fd.flush()
        except Exception:
            pass

        yield
    finally:
        if fd is not None:
            try:
                import fcntl  # type: ignore

                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                fd.close()
            except Exception:
                pass

KERNEL_REF = os.getenv("TG_MONITORING_KERNEL_REF", "artkoder/telegram-monitor-bot")
KERNEL_PATH = Path(os.getenv("TG_MONITORING_KERNEL_PATH", "kaggle/TelegramMonitor"))

DATASET_PROPAGATION_WAIT_SECONDS = int(os.getenv("TG_MONITORING_DATASET_WAIT", "30"))
POLL_INTERVAL_SECONDS = int(os.getenv("TG_MONITORING_POLL_INTERVAL", "30"))
TIMEOUT_MINUTES = int(os.getenv("TG_MONITORING_TIMEOUT_MINUTES", "90"))
TIMEOUT_MODE = (os.getenv("TG_MONITORING_TIMEOUT_MODE") or "dynamic").strip().lower()
TIMEOUT_BASE_MINUTES = int(os.getenv("TG_MONITORING_TIMEOUT_BASE_MINUTES", "15"))
# Production baseline: ~3.64 minutes per source; safety multiplier accounts for Kaggle/TG variance.
TIMEOUT_PER_SOURCE_MINUTES = float(os.getenv("TG_MONITORING_TIMEOUT_PER_SOURCE_MINUTES", "3.64"))
TIMEOUT_SAFETY_MULTIPLIER = float(os.getenv("TG_MONITORING_TIMEOUT_SAFETY_MULTIPLIER", "1.3"))
TIMEOUT_MAX_MINUTES = int(os.getenv("TG_MONITORING_TIMEOUT_MAX_MINUTES", "360"))
KAGGLE_STARTUP_WAIT_SECONDS = int(os.getenv("TG_MONITORING_STARTUP_WAIT", "10"))
MAX_TG_MESSAGE_LEN = int(os.getenv("TG_MONITORING_MAX_MESSAGE_LEN", "3800"))
KEEP_FORCE_MESSAGE_IDS = (os.getenv("TG_MONITORING_KEEP_FORCE_MESSAGE_IDS") or "").strip().lower() in {
    "1",
    "true",
    "yes",
}
IMPORT_RETRY_ATTEMPTS = max(1, min(int(os.getenv("TG_MONITORING_IMPORT_RETRY_ATTEMPTS", "4")), 12))
IMPORT_RETRY_BASE_DELAY_SEC = max(
    0.2,
    min(float(os.getenv("TG_MONITORING_IMPORT_RETRY_BASE_DELAY_SEC", "2.0")), 30.0),
)
LOCAL_RESULTS_GLOB = (
    os.getenv("TG_MONITORING_LOCAL_RESULTS_GLOB", "tg-monitor-*/telegram_results.json")
    or "tg-monitor-*/telegram_results.json"
).strip()
RECREATE_SQLITE_CHUNK_SIZE = max(
    200,
    min(int((os.getenv("TG_MONITORING_RECREATE_CHUNK_SIZE") or "500") or 500), 900),
)


@dataclass(slots=True)
class TelegramRecreateImportStats:
    source_links_total: int = 0
    source_usernames_total: int = 0
    message_pairs_total: int = 0
    event_ids_found: int = 0
    scanned_matches_found: int = 0
    joboutbox_deleted: int = 0
    events_deleted: int = 0
    scanned_deleted: int = 0


@dataclass(slots=True)
class _TelegramRecreateScope:
    source_links: list[str]
    source_usernames: list[str]
    message_pairs_total: int
    message_ids_by_username: dict[str, list[int]]


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


def _resolve_auth_bundle_env_key() -> str | None:
    """Pick auth bundle source env for Telegram monitoring."""
    override = (_get_env_value("TG_MONITORING_AUTH_BUNDLE_ENV") or "").strip()
    if override:
        return override

    # Default (safe separation): Telegram Monitoring/Kaggle uses S22 only.
    if _get_env_value("TELEGRAM_AUTH_BUNDLE_S22"):
        return "TELEGRAM_AUTH_BUNDLE_S22"

    return None


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
    bundle_env_key = _resolve_auth_bundle_env_key()
    bundle_raw = _get_env_value(bundle_env_key) if bundle_env_key else ""
    bundle = None
    bundle_ok = False
    if bundle_raw:
        try:
            bundle = _parse_auth_bundle(bundle_env_key or "TELEGRAM_AUTH_BUNDLE_S22")
            bundle_ok = True
        except Exception as exc:  # pragma: no cover - validation only
            logger.warning("tg_monitor.secrets_payload invalid bundle: %s", exc)
    payload = {
        "TG_API_ID": _require_env("TG_API_ID"),
        "TG_API_HASH": _require_env("TG_API_HASH"),
        "GOOGLE_API_KEY": _require_env("GOOGLE_API_KEY"),
    }
    logger.info(
        "tg_monitor.secrets_payload bundle_env=%s bundle_len=%s bundle_ok=%s tg_session=%s days_back=%s limit=%s",
        bundle_env_key or "-",
        len(bundle_raw) if bundle_raw else 0,
        bundle_ok,
        bool(_get_env_value("TG_SESSION")),
        _get_env_value("TG_MONITORING_DAYS_BACK"),
        _get_env_value("TG_MONITORING_LIMIT"),
    )
    if bundle_raw:
        # Kaggle notebook currently reads TELEGRAM_AUTH_BUNDLE_S22, so we always
        # map the selected local env source into this canonical payload key.
        payload["TELEGRAM_AUTH_BUNDLE_S22"] = bundle_raw
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


def _compute_kaggle_poll_timeout_minutes(*, sources_count: int) -> int:
    if TIMEOUT_MODE in {"fixed", "static"}:
        return max(1, TIMEOUT_MINUTES)
    try:
        base = max(0, int(TIMEOUT_BASE_MINUTES))
    except Exception:
        base = 15
    try:
        per_source = max(0.0, float(TIMEOUT_PER_SOURCE_MINUTES))
    except Exception:
        per_source = 3.64
    try:
        safety = max(0.1, float(TIMEOUT_SAFETY_MULTIPLIER))
    except Exception:
        safety = 1.3
    try:
        max_minutes = max(1, int(TIMEOUT_MAX_MINUTES))
    except Exception:
        max_minutes = 360

    effective_per_source = per_source * safety
    est = base + int(math.ceil(max(0, int(sources_count)) * effective_per_source))
    # Backward-compatible floor: older config expected TIMEOUT_MINUTES to be "safe default".
    timeout = max(TIMEOUT_MINUTES, est)
    return min(max_minutes, max(1, timeout))


async def _poll_kaggle_kernel(
    client: KaggleClient,
    kernel_ref: str,
    *,
    run_id: str | None = None,
    timeout_minutes: int,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> tuple[str, dict | None, float]:
    started = time.monotonic()
    timeout_minutes = max(1, int(timeout_minutes))
    deadline = started + timeout_minutes * 60
    last_status: dict | None = None
    consecutive_poll_errors = 0
    attempt = 0

    async def _notify(phase: str, status: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, status)
        except Exception:
            logger.exception("tg_monitor: status callback failed phase=%s", phase)

    await _notify("poll", {"_poll_timeout_minutes": timeout_minutes, "_elapsed_seconds": 0.0})
    while time.monotonic() < deadline:
        attempt += 1
        try:
            status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            last_status = status
            consecutive_poll_errors = 0
        except Exception as exc:
            consecutive_poll_errors += 1
            is_ssl = isinstance(exc, ssl.SSLError) or exc.__class__.__name__.endswith("SSLError")
            is_conn = isinstance(exc, ConnectionError) or exc.__class__.__name__.endswith("ConnectionError")
            is_timeout = exc.__class__.__name__.endswith("Timeout")
            is_transient = is_ssl or is_conn or is_timeout
            if not is_transient:
                raise
            msg = str(exc) or repr(exc)
            if len(msg) > 280:
                msg = msg[:277] + "..."
            logger.warning(
                "tg_monitor.kernel_poll_transient_error run_id=%s kernel_ref=%s attempt=%s consecutive=%s err=%s",
                run_id,
                kernel_ref,
                attempt,
                consecutive_poll_errors,
                msg,
            )
            shown_state = ""
            if isinstance(last_status, dict):
                shown_state = str(last_status.get("status") or "").upper()
            await _notify(
                "poll_error",
                {
                    "status": shown_state or "RUNNING",
                    "failureMessage": f"⚠️ временная ошибка Kaggle API (SSL/сеть), продолжаю опрос: {msg}",
                    "_poll_timeout_minutes": timeout_minutes,
                    "_elapsed_seconds": time.monotonic() - started,
                },
            )
            await asyncio.sleep(min(60, 2 ** min(consecutive_poll_errors, 5)))
            continue

        state = (status.get("status") or "").upper()
        payload = dict(status or {})
        payload["_poll_timeout_minutes"] = timeout_minutes
        payload["_elapsed_seconds"] = time.monotonic() - started
        await _notify("poll", payload)
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
            done = dict(last_status or {})
            done["_poll_timeout_minutes"] = timeout_minutes
            done["_elapsed_seconds"] = time.monotonic() - started
            await _notify("complete", done)
            return "complete", last_status, time.monotonic() - started
        if state in ("ERROR", "FAILED", "CANCELLED"):
            failed = dict(last_status or {})
            failed["_poll_timeout_minutes"] = timeout_minutes
            failed["_elapsed_seconds"] = time.monotonic() - started
            await _notify("failed", failed)
            return "failed", last_status, time.monotonic() - started
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    # Deadline hit: do a final status fetch once to avoid false timeouts near completion.
    try:
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status or last_status
        state = str((status or {}).get("status") or "").upper()
        if state == "COMPLETE":
            done = dict(last_status or {})
            done["_poll_timeout_minutes"] = timeout_minutes
            done["_elapsed_seconds"] = time.monotonic() - started
            await _notify("complete", done)
            return "complete", last_status, time.monotonic() - started
    except Exception:
        pass

    timeout_payload = dict(last_status or {})
    timeout_payload.setdefault(
        "failureMessage",
        "локальный таймаут ожидания: Kaggle kernel мог продолжить выполнение",
    )
    timeout_payload["_poll_timeout_minutes"] = timeout_minutes
    timeout_payload["_elapsed_seconds"] = time.monotonic() - started
    await _notify("timeout", timeout_payload)
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


def find_latest_telegram_results_json(search_root: str | Path | None = None) -> Path:
    root = Path(search_root) if search_root is not None else Path(tempfile.gettempdir())
    pattern = LOCAL_RESULTS_GLOB or "tg-monitor-*/telegram_results.json"
    candidates: list[Path] = []
    for path in root.glob(pattern):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No local telegram_results.json found under {root} (pattern={pattern})"
        )

    def _sort_key(path: Path) -> tuple[float, float, str]:
        try:
            stat = path.stat()
            return (float(stat.st_mtime), float(stat.st_ctime), str(path))
        except Exception:
            return (0.0, 0.0, str(path))

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def find_recent_telegram_results_json(
    limit: int = 4,
    search_root: str | Path | None = None,
) -> list[Path]:
    """Return N most recent local telegram_results.json files (newest first)."""
    root = Path(search_root) if search_root is not None else Path(tempfile.gettempdir())
    pattern = LOCAL_RESULTS_GLOB or "tg-monitor-*/telegram_results.json"
    candidates: list[Path] = []
    for path in root.glob(pattern):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        return []

    def _sort_key(path: Path) -> tuple[float, float, str]:
        try:
            stat = path.stat()
            return (float(stat.st_mtime), float(stat.st_ctime), str(path))
        except Exception:
            return (0.0, 0.0, str(path))

    candidates.sort(key=_sort_key, reverse=True)
    lim = max(1, min(int(limit or 0), 20))
    return candidates[:lim]


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _chunked(items: list[Any], *, size: int = RECREATE_SQLITE_CHUNK_SIZE):
    step = max(1, int(size or RECREATE_SQLITE_CHUNK_SIZE))
    for i in range(0, len(items), step):
        yield items[i : i + step]


def _build_recreate_scope_from_results(results_path: str | Path) -> _TelegramRecreateScope:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"telegram_results.json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []

    links_seen: set[str] = set()
    pairs_seen: set[tuple[str, int]] = set()
    message_ids_by_username: dict[str, set[int]] = {}

    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        username = normalize_tg_username(item.get("source_username"))
        message_id = _safe_int(item.get("message_id"))

        source_link = str(item.get("source_link") or "").strip()
        if not source_link and username and message_id is not None:
            source_link = f"https://t.me/{username}/{int(message_id)}"
        if source_link:
            links_seen.add(source_link)

        if not username or message_id is None:
            continue
        pairs_seen.add((username, int(message_id)))
        message_ids_by_username.setdefault(username, set()).add(int(message_id))

    usernames_sorted = sorted(message_ids_by_username.keys())
    message_ids_sorted = {
        username: sorted(message_ids_by_username.get(username) or set())
        for username in usernames_sorted
    }
    return _TelegramRecreateScope(
        source_links=sorted(links_seen),
        source_usernames=usernames_sorted,
        message_pairs_total=len(pairs_seen),
        message_ids_by_username=message_ids_sorted,
    )


async def _select_event_ids_for_links(conn, source_links: list[str]) -> list[int]:
    if not source_links:
        return []
    event_ids: set[int] = set()
    for chunk in _chunked(source_links):
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            "SELECT DISTINCT event_id "
            "FROM event_source "
            f"WHERE source_type='telegram' AND source_url IN ({placeholders})"
        )
        cursor = await conn.execute(sql, tuple(chunk))
        rows = await cursor.fetchall()
        await cursor.close()
        for row in rows:
            event_id = _safe_int(row[0] if row else None)
            if event_id is not None:
                event_ids.add(int(event_id))
    return sorted(event_ids)


async def _select_source_ids_by_username(conn, usernames: list[str]) -> dict[str, int]:
    if not usernames:
        return {}
    source_ids: dict[str, int] = {}
    for chunk in _chunked(usernames):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"SELECT id, username FROM telegram_source WHERE username IN ({placeholders})"
        cursor = await conn.execute(sql, tuple(chunk))
        rows = await cursor.fetchall()
        await cursor.close()
        for row in rows:
            if not row:
                continue
            source_id = _safe_int(row[0])
            username = normalize_tg_username(row[1])
            if source_id is None or not username:
                continue
            source_ids[username] = int(source_id)
    return source_ids


async def _count_scanned_matches(
    conn,
    *,
    source_ids_by_username: dict[str, int],
    message_ids_by_username: dict[str, list[int]],
) -> int:
    total = 0
    for username in sorted(message_ids_by_username.keys()):
        source_id = source_ids_by_username.get(username)
        if source_id is None:
            continue
        message_ids = message_ids_by_username.get(username) or []
        for chunk in _chunked(message_ids):
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT COUNT(*) "
                "FROM telegram_scanned_message "
                f"WHERE source_id=? AND message_id IN ({placeholders})"
            )
            params = (int(source_id), *tuple(chunk))
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            await cursor.close()
            total += int((row or [0])[0] or 0)
    return int(total)


async def _delete_joboutbox_by_event_ids(conn, event_ids: list[int]) -> int:
    if not event_ids:
        return 0
    deleted = 0
    for chunk in _chunked(event_ids):
        placeholders = ",".join("?" for _ in chunk)
        before = int(getattr(conn, "total_changes", 0) or 0)
        cursor = await conn.execute(
            f"DELETE FROM joboutbox WHERE event_id IN ({placeholders})",
            tuple(chunk),
        )
        rowcount = int(getattr(cursor, "rowcount", -1) or -1)
        await cursor.close()
        if rowcount >= 0:
            deleted += rowcount
        else:
            after = int(getattr(conn, "total_changes", 0) or 0)
            deleted += max(0, after - before)
    return int(deleted)


async def _delete_events_by_ids(conn, event_ids: list[int]) -> int:
    if not event_ids:
        return 0
    deleted = 0
    for chunk in _chunked(event_ids):
        placeholders = ",".join("?" for _ in chunk)
        before = int(getattr(conn, "total_changes", 0) or 0)
        cursor = await conn.execute(
            f"DELETE FROM event WHERE id IN ({placeholders})",
            tuple(chunk),
        )
        rowcount = int(getattr(cursor, "rowcount", -1) or -1)
        await cursor.close()
        if rowcount >= 0:
            deleted += rowcount
        else:
            after = int(getattr(conn, "total_changes", 0) or 0)
            deleted += max(0, after - before)
    return int(deleted)


async def _delete_scanned_marks(
    conn,
    *,
    source_ids_by_username: dict[str, int],
    message_ids_by_username: dict[str, list[int]],
) -> int:
    deleted = 0
    for username in sorted(message_ids_by_username.keys()):
        source_id = source_ids_by_username.get(username)
        if source_id is None:
            continue
        message_ids = message_ids_by_username.get(username) or []
        for chunk in _chunked(message_ids):
            placeholders = ",".join("?" for _ in chunk)
            params = (int(source_id), *tuple(chunk))
            before = int(getattr(conn, "total_changes", 0) or 0)
            cursor = await conn.execute(
                (
                    "DELETE FROM telegram_scanned_message "
                    f"WHERE source_id=? AND message_id IN ({placeholders})"
                ),
                params,
            )
            rowcount = int(getattr(cursor, "rowcount", -1) or -1)
            await cursor.close()
            if rowcount >= 0:
                deleted += rowcount
            else:
                after = int(getattr(conn, "total_changes", 0) or 0)
                deleted += max(0, after - before)
    return int(deleted)


async def preview_telegram_recreate_reimport(
    db: Database,
    *,
    results_path: str | Path,
) -> TelegramRecreateImportStats:
    scope = _build_recreate_scope_from_results(results_path)
    stats = TelegramRecreateImportStats(
        source_links_total=len(scope.source_links),
        source_usernames_total=len(scope.source_usernames),
        message_pairs_total=int(scope.message_pairs_total),
    )
    async with db.raw_conn() as conn:
        event_ids = await _select_event_ids_for_links(conn, scope.source_links)
        source_ids = await _select_source_ids_by_username(conn, scope.source_usernames)
        scanned_matches = await _count_scanned_matches(
            conn,
            source_ids_by_username=source_ids,
            message_ids_by_username=scope.message_ids_by_username,
        )
    stats.event_ids_found = len(event_ids)
    stats.scanned_matches_found = int(scanned_matches)
    return stats


async def recreate_telegram_events_from_results(
    db: Database,
    *,
    results_path: str | Path,
) -> TelegramRecreateImportStats:
    scope = _build_recreate_scope_from_results(results_path)
    stats = TelegramRecreateImportStats(
        source_links_total=len(scope.source_links),
        source_usernames_total=len(scope.source_usernames),
        message_pairs_total=int(scope.message_pairs_total),
    )

    async with db.raw_conn() as conn:
        event_ids = await _select_event_ids_for_links(conn, scope.source_links)
        source_ids = await _select_source_ids_by_username(conn, scope.source_usernames)
        scanned_matches = await _count_scanned_matches(
            conn,
            source_ids_by_username=source_ids,
            message_ids_by_username=scope.message_ids_by_username,
        )
        stats.event_ids_found = len(event_ids)
        stats.scanned_matches_found = int(scanned_matches)

        logger.info(
            "tg_monitor.dev_recreate.prepare path=%s links=%s pairs=%s event_ids=%s scanned_matches=%s",
            results_path,
            stats.source_links_total,
            stats.message_pairs_total,
            stats.event_ids_found,
            stats.scanned_matches_found,
        )

        await conn.execute("BEGIN")
        try:
            stats.joboutbox_deleted = await _delete_joboutbox_by_event_ids(conn, event_ids)
            stats.events_deleted = await _delete_events_by_ids(conn, event_ids)
            stats.scanned_deleted = await _delete_scanned_marks(
                conn,
                source_ids_by_username=source_ids,
                message_ids_by_username=scope.message_ids_by_username,
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    logger.info(
        "tg_monitor.dev_recreate.done path=%s events_deleted=%s joboutbox_deleted=%s scanned_deleted=%s",
        results_path,
        stats.events_deleted,
        stats.joboutbox_deleted,
        stats.scanned_deleted,
    )
    return stats


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
        "poll_error": "временная ошибка сети",
        "complete": "завершено",
        "failed": "ошибка",
        "timeout": "таймаут",
    }
    return labels.get(phase, phase)


def _is_sqlite_locked_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return "database is locked" in msg or "database table is locked" in msg


async def _import_results_with_retry(
    results_path: Path,
    db: Database,
    *,
    bot: Any | None,
    run_id: str,
    progress_callback,
    notify: Callable[[str], Awaitable[None]] | None = None,
) -> TelegramMonitorReport:
    last_exc: Exception | None = None
    for attempt in range(1, IMPORT_RETRY_ATTEMPTS + 1):
        try:
            return await process_telegram_results(
                results_path,
                db,
                bot=bot,
                progress_callback=progress_callback,
            )
        except OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            last_exc = exc
            if attempt >= IMPORT_RETRY_ATTEMPTS:
                raise
            delay = min(60.0, IMPORT_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)))
            logger.warning(
                "tg_monitor.import_retry_locked run_id=%s attempt=%s/%s delay=%.1fs err=%s",
                run_id,
                attempt,
                IMPORT_RETRY_ATTEMPTS,
                delay,
                str(exc)[:260],
            )
            if notify:
                await notify(
                    "⚠️ Импорт упёрся в блокировку SQLite (`database is locked`), "
                    f"повторяю попытку {attempt + 1}/{IMPORT_RETRY_ATTEMPTS} через {delay:.1f}s…"
                )
            await asyncio.sleep(delay)
        except Exception:
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("tg_monitor import retry failed without exception")


def _format_kaggle_status(status: dict | None) -> str:
    if not status:
        return "неизвестен"
    state = status.get("status")
    failure_msg = _extract_kaggle_failure_message(status)
    if not state:
        return "неизвестен"
    result = str(state)
    if failure_msg:
        result += f" ({failure_msg})"
    return result


def _extract_kaggle_failure_message(status: dict | None) -> str:
    if not status:
        return ""
    for key in (
        "failureMessage",
        "failure_message",
        "errorMessage",
        "error_message",
        "error",
    ):
        value = status.get(key)
        if value:
            return str(value)
    return ""


def _is_auth_key_duplicated_failure(status: dict | None) -> bool:
    msg = _extract_kaggle_failure_message(status)
    if not msg:
        return False
    msg_l = msg.lower()
    if "authkeyduplicatederror" in msg_l:
        return True
    if "authorization key" in msg_l and "different ip" in msg_l:
        return True
    return False


def _build_auth_key_duplicated_hint() -> str:
    bundle_env = _resolve_auth_bundle_env_key()
    src = bundle_env or "TG_SESSION"
    return (
        "❌ Telegram session стала недействительной (AuthKeyDuplicatedError).\n"
        "Одна и та же Telethon session использовалась одновременно с разных IP.\n"
        "Перевыпустите отдельную session и обновите переменную "
        f"<code>{html.escape(src)}</code> (или задайте <code>TG_MONITORING_AUTH_BUNDLE_ENV</code>), "
        "затем повторите /tg."
    )


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
        timeout_minutes = status.get("_poll_timeout_minutes")
        elapsed_seconds = status.get("_elapsed_seconds")
        try:
            timeout_i = int(timeout_minutes) if timeout_minutes is not None else None
        except Exception:
            timeout_i = None
        try:
            elapsed_f = float(elapsed_seconds) if elapsed_seconds is not None else None
        except Exception:
            elapsed_f = None
        if timeout_i:
            if elapsed_f is not None:
                lines.append(
                    f"Ожидание: до {timeout_i} мин (прошло {elapsed_f / 60.0:.1f} мин)"
                )
            else:
                lines.append(f"Ожидание: до {timeout_i} мин")
    return "\n".join(lines)


def _format_event_block(
    label: str,
    events: list[TelegramMonitorEventInfo],
    *,
    icon: str,
    bot_username: str | None = None,
    ctx: object | None = None,
) -> list[str]:
    lines = [f"{icon} <b>{html.escape(label)}</b>: {len(events)}", ""]

    tz = getattr(ctx, "tz", None)
    sources_by_eid = getattr(ctx, "sources_by_event_id", None) or {}
    video_counts = getattr(ctx, "video_count_by_event_id", None) or {}
    ticket_queue_by_eid = getattr(ctx, "ticket_queue_by_event_id", None) or {}
    fest_queue_by_src = getattr(ctx, "festival_queue_by_source_url", None) or {}

    def _ics_line(url: str | None, *, has_time: bool) -> str:
        value = (url or "").strip()
        if value:
            safe = html.escape(value, quote=True)
            return f'ICS: <a href="{safe}">ics</a>'
        return "ICS: ⏳" if has_time else "ICS: —"

    def _sources_lines(eid: int) -> list[str]:
        rows = list(sources_by_eid.get(int(eid)) or [])
        if not rows or not tz:
            return []
        from source_parsing.smart_update_report import format_dt_compact, short_url_label

        out: list[str] = ["Источники:"]
        limit = 24
        shown = rows[:limit]
        for imported_at, url in shown:
            stamp = format_dt_compact(imported_at, tz)
            label = short_url_label(url) or url
            if str(url).strip().startswith(("http://", "https://")):
                href = _preview_friendly_tg_post_url(str(url).strip()) or str(url).strip()
                safe_href = html.escape(href, quote=True)
                safe_label = html.escape(label)
                out.append(f"{stamp} <a href=\"{safe_href}\">{safe_label}</a>")
            else:
                out.append(f"{stamp} {html.escape(label)}")
        if len(rows) > limit:
            out.append(f"… ещё {len(rows) - limit}")
        return out

    def _queue_lines(eid: int, source_url: str | None) -> list[str]:
        out: list[str] = []
        src = (source_url or "").strip()
        if src:
            fest = fest_queue_by_src.get(src)
            if fest:
                name = (getattr(fest, "festival_name", None) or getattr(fest, "festival_full", None) or "").strip()
                ctx2 = (getattr(fest, "festival_context", None) or "").strip()
                status = (getattr(fest, "status", None) or "").strip()
                fid = getattr(fest, "id", None)
                tail = name or ctx2
                extra = f" {tail}" if tail else ""
                id_part = f" (id={int(fid)})" if isinstance(fid, int) and fid > 0 else ""
                st_part = f" {status}" if status else ""
                out.append(f"🎪 festival_queue:{st_part}{extra}{id_part}".strip())

        tickets = list(ticket_queue_by_eid.get(int(eid)) or [])
        if tickets:
            first = tickets[0]
            href = html.escape(str(getattr(first, 'url', '') or '').strip(), quote=True)
            label = html.escape(str(getattr(first, 'site_kind', '') or 'tickets').strip() or "tickets")
            extra = f" +{len(tickets)}" if len(tickets) > 1 else ""
            if href:
                out.append(f'🎟 ticket_site_queue:{extra} <a href="{href}">{label}</a>')
            else:
                out.append(f"🎟 ticket_site_queue:{extra}".strip())
        return out

    for item in events:
        pop = (getattr(item, "popularity", None) or "").strip()
        title = html.escape(item.title or "Без названия")
        if pop:
            title = f"{html.escape(pop)} {title}"
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
            href = _preview_friendly_tg_post_url(item.source_link) or item.source_link
            safe_href = html.escape(href, quote=True)
            safe_label = html.escape(item.source_link)
            lines.append(f'Источник: <a href="{safe_href}">{safe_label}</a>')
        if not item.telegraph_url:
            hint = (getattr(item, "telegraph_job_status", None) or "").strip()
            if hint:
                low = hint.lower()
                icon2 = "❌" if ("error" in low or "failed" in low) else "⏳"
                lines.append(f"Telegraph: {icon2} {html.escape(hint)}")
            else:
                lines.append("Telegraph: ⏳ в очереди")
        try:
            eid_i = int(getattr(item, "event_id", 0) or 0)
        except Exception:
            eid_i = 0
        if eid_i:
            lines.extend(_sources_lines(eid_i))
        if item.log_cmd:
            if bot_username:
                href = html.escape(_log_deeplink(bot_username, int(item.event_id)), quote=True)
                text = html.escape(item.log_cmd)
                lines.append(f"Лог: <a href=\"{href}\">{text}</a>")
            else:
                lines.append(f"Лог: {html.escape(item.log_cmd)}")
        lines.append(_ics_line(item.ics_url, has_time=bool((item.time or "").strip())))
        stats = item.fact_stats or {}
        try:
            photos = int(getattr(item, "photo_count", None) or 0)
        except Exception:
            photos = 0
        try:
            videos_total = int(video_counts.get(int(getattr(item, "event_id", 0) or 0), 0) or 0)
        except Exception:
            videos_total = 0
        added_videos_raw = getattr(item, "added_videos", None)
        try:
            added_videos = int(added_videos_raw) if added_videos_raw is not None else None
        except Exception:
            added_videos = None
        added_posters_raw = getattr(item, "added_posters", None)
        try:
            added_posters = int(added_posters_raw) if added_posters_raw is not None else None
        except Exception:
            added_posters = None
        if added_posters is None:
            photos_label = f"Иллюстрации: {'⚠️0' if photos == 0 else photos}"
        else:
            photos_label = f"Иллюстрации: +{added_posters}, всего {'⚠️0' if photos == 0 else photos}"
        videos_label = ""
        if added_videos is not None:
            videos_label = f" | Видео: +{added_videos}, всего {'⚠️0' if videos_total == 0 else videos_total}"
        elif videos_total > 0:
            videos_label = f" | Видео: {videos_total}"
        if stats:
            added = int(stats.get("added") or 0)
            dup = int(stats.get("duplicate") or 0)
            conf = int(stats.get("conflict") or 0)
            note = int(stats.get("note") or 0)
            lines.append(f"Факты: ✅{added} ↩️{dup} ⚠️{conf} ℹ️{note} | {photos_label}{videos_label}")
        else:
            lines.append(f"Факты: — | {photos_label}{videos_label}")
        if eid_i:
            for q in _queue_lines(eid_i, getattr(item, "source_link", None)):
                msg = str(q or "").strip()
                if msg:
                    lines.append(html.escape(msg))
        queue_notes = list(getattr(item, "queue_notes", None) or [])
        for note in queue_notes[:4]:
            msg = str(note or "").strip()
            if msg:
                lines.append(html.escape(msg))
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
            lines.append(html.escape(item.source_excerpt))
        lines.append("")
    return lines


def _format_skipped_posts_block(report: TelegramMonitorReport, *, limit: int = 16) -> list[str]:
    import html

    items = list(getattr(report, "skipped_posts", None) or [])
    if not items:
        return []
    total = len(items)
    lines: list[str] = [f"⏭️ <b>Пропущенные/частично обработанные посты</b>: {total}"]

    def _fmt_breakdown(b: dict[str, int] | None) -> str:
        if not b:
            return "—"
        parts: list[str] = []
        for k, v in sorted(b.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))):
            v_i = int(v or 0)
            if v_i <= 0:
                continue
            key = str(k or "").strip()
            if not key:
                continue
            parts.append(f"{key}={v_i}")
            if len(parts) >= 6:
                break
        return ", ".join(parts) if parts else "—"

    shown = items[: max(1, int(limit or 16))]
    for it in shown:
        username = (getattr(it, "source_username", None) or "").strip()
        title = (getattr(it, "source_title", None) or "").strip()
        head = f"@{html.escape(username)}"
        if title:
            head = f"<b>{html.escape(title)}</b> ({head})"
        link = (getattr(it, "source_link", None) or "").strip()
        if link:
            head = f'{head} — <a href="{html.escape(link)}">пост</a>'

        status = (getattr(it, "status", None) or "").strip()
        reason = (getattr(it, "reason", None) or "").strip()
        extracted = int(getattr(it, "events_extracted", 0) or 0)
        imported = int(getattr(it, "events_imported", 0) or 0)
        reason_tail = f"{status}" + (f":{reason}" if reason else "")
        lines.append(f"• {head}")
        lines.append(
            f"  events: {imported}/{extracted} | {html.escape(reason_tail or 'skipped')}"
        )
        breakdown = getattr(it, "skip_breakdown", None)
        if breakdown:
            lines.append(f"  причины: {html.escape(_fmt_breakdown(dict(breakdown)))}")
        titles = list(getattr(it, "event_titles", None) or [])
        if titles:
            joined = "; ".join(str(t).strip() for t in titles if str(t or '').strip())
            if joined:
                if len(joined) > 420:
                    joined = joined[:419].rstrip() + "…"
                lines.append(f"  события: {html.escape(joined)}")
        excerpt = (getattr(it, "source_excerpt", None) or "").strip()
        if excerpt and not titles:
            lines.append(f"  текст: {html.escape(excerpt)}")
        lines.append("")

    if total > len(shown):
        lines.append(f"… ещё {total - len(shown)}")
    return lines


def _format_metrics_only_summary(report: TelegramMonitorReport) -> list[str]:
    metrics_only = int(getattr(report, "messages_metrics_only", 0) or 0)
    if metrics_only <= 0:
        return []
    extracted = int(getattr(report, "events_extracted_metrics_only", 0) or 0)
    return [
        f"📈 <b>Метрики обновлены (без Smart Update)</b>: {metrics_only}",
        f"  событий (ранее извлечено): {extracted}" if extracted else "  событий (ранее извлечено): —",
        "",
    ]


def _format_popular_posts_block(
    report: TelegramMonitorReport,
    *,
    limit: int = 10,
    telegraph_map: dict[str, list[str]] | None = None,
) -> list[str]:
    import html

    items = list(getattr(report, "popular_posts", None) or [])
    if not items:
        return []

    def _score(it: Any) -> tuple[int, int, int]:
        pop = (getattr(it, "popularity", None) or "").strip()
        both = 1 if ("⭐" in pop and "👍" in pop) else 0
        views = 0
        likes = 0
        m = getattr(it, "metrics", None) or {}
        if isinstance(m, dict):
            v = m.get("views")
            l = m.get("likes")
            if isinstance(v, int):
                views = v
            if isinstance(l, int):
                likes = l
        return (both, views, likes)

    items_sorted = sorted(items, key=_score, reverse=True)
    shown = items_sorted[: max(1, int(limit or 10))]
    lines: list[str] = [f"🔥 <b>Популярные посты</b>: {len(items)}", ""]
    for it in shown:
        username = (getattr(it, "source_username", None) or "").strip()
        title = (getattr(it, "source_title", None) or "").strip()
        pop = (getattr(it, "popularity", None) or "").strip()
        head = f"@{html.escape(username)}"
        if title:
            head = f"<b>{html.escape(title)}</b> ({head})"
        link = (getattr(it, "source_link", None) or "").strip()
        if link:
            head = f'{head} — <a href="{html.escape(link)}">пост</a>'
        if pop:
            head = f"{html.escape(pop)} {head}"
        lines.append(f"• {head}")
        metrics = getattr(it, "metrics", None) or {}
        if isinstance(metrics, dict):
            v = metrics.get("views")
            l = metrics.get("likes")
            parts = []
            if isinstance(v, int) and v >= 0:
                parts.append(f"views={v}")
            if isinstance(l, int) and l >= 0:
                parts.append(f"likes={l}")
            if parts:
                lines.append(f"  метрики: {html.escape(' '.join(parts))}")
        extracted = int(getattr(it, "events_extracted", 0) or 0)
        imported = int(getattr(it, "events_imported", 0) or 0)
        if extracted or imported:
            lines.append(f"  events: {imported}/{extracted}")
        if link:
            urls = (telegraph_map or {}).get(link) or []
            if urls:
                first = str(urls[0]).strip()
                if first:
                    lines.append(f'  Telegraph: <a href="{html.escape(first)}">событие</a>')
                if len(urls) > 1:
                    lines.append(f"  … ещё {len(urls) - 1}")
        lines.append("")
    if len(items) > len(shown):
        lines.append(f"… ещё {len(items) - len(shown)}")
    return lines


def _render_tg_post_progress_text(
    icon: str,
    *,
    progress: TelegramMonitorImportProgress,
    extra_lines: list[str] | None = None,
) -> str:
    total_txt = str(int(progress.total_no)) if int(progress.total_no or 0) > 0 else "?"
    source_url = (progress.source_link or "").strip()
    if not source_url:
        source_url = f"https://t.me/{progress.source_username}/{int(progress.message_id)}"
    lines = [
        f"{icon} Разбираю Telegram пост {int(progress.current_no)}/{total_txt}: {source_url}",
    ]
    for line in (extra_lines or []):
        txt = str(line or "").strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines).strip()


def _format_metrics_line(metrics: dict[str, Any] | None) -> str | None:
    if not isinstance(metrics, dict):
        return None
    parts: list[str] = []
    views = metrics.get("views")
    likes = metrics.get("likes")
    if isinstance(views, int) and views >= 0:
        parts.append(f"views={views}")
    if isinstance(likes, int) and likes >= 0:
        parts.append(f"likes={likes}")
    return f"Метрики поста: {' '.join(parts)}" if parts else None


def _format_video_status_line(progress: TelegramMonitorImportProgress) -> str | None:
    if not getattr(progress, "post_has_video", None):
        return None
    status_raw = str(getattr(progress, "post_video_status", "") or "").strip().lower()
    photos_n = getattr(progress, "post_posters_total", None)
    photo_tail = " (фото=0)" if isinstance(photos_n, int) and photos_n <= 0 else ""
    if not status_raw:
        return f"Медиа поста: 🎬 видео{photo_tail}"
    status = status_raw
    if status.startswith("skipped:"):
        reason = status.split(":", 1)[1].strip() or "unknown"
        return f"Медиа поста: 🎬 видео (skipped: {reason}){photo_tail}"
    if status.startswith("partial:"):
        reason = status.split(":", 1)[1].strip() or "partial"
        return f"Медиа поста: 🎬 видео (partial: {reason}){photo_tail}"
    if status in {"supabase", "uploaded"}:
        return f"Медиа поста: 🎬 видео (supabase){photo_tail}"
    return f"Медиа поста: 🎬 видео ({status}){photo_tail}"


async def _upsert_progress_message(
    bot: Any,
    *,
    chat_id: int,
    message_id: int | None,
    text: str,
) -> int | None:
    payload = (text or "").strip()
    if not payload:
        return message_id
    if message_id and hasattr(bot, "edit_message_text"):
        try:
            await bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(message_id),
                text=payload,
                disable_web_page_preview=True,
            )
            return int(message_id)
        except Exception as exc:
            # If the bot isn't wrapped by SafeBot, Telegram can throw on "no-op" edits.
            msg = str(exc or "").lower()
            if "message is not modified" in msg:
                return int(message_id)
            logger.warning("tg_monitor: failed to edit import progress message", exc_info=True)
    try:
        sent = await bot.send_message(
            int(chat_id),
            payload,
            disable_web_page_preview=True,
        )
        return int(getattr(sent, "message_id", 0) or 0) or message_id
    except Exception:
        logger.warning("tg_monitor: failed to send import progress message", exc_info=True)
        return message_id


async def _send_per_post_event_details(
    bot: Any,
    *,
    chat_id: int,
    progress: TelegramMonitorImportProgress,
    db: Database | None = None,
) -> bool:
    created_events = list(progress.created_events or [])
    merged_events = list(progress.merged_events or [])
    if not created_events and not merged_events:
        return True
    bot_username = await _resolve_bot_username(bot) if bot else None
    ctx = None
    if db:
        try:
            from source_parsing.smart_update_report import build_smart_update_report_context

            all_events = created_events + merged_events
            eids = [int(getattr(e, "event_id", 0) or 0) for e in all_events]
            urls = [str(getattr(e, "source_link", "") or "").strip() for e in all_events]
            ctx = await build_smart_update_report_context(db, event_ids=eids, source_urls=urls)
        except Exception:
            logger.debug("tg_monitor: smart_update_report_context build failed", exc_info=True)
    lines: list[str] = ["<b>Smart Update (детали событий):</b>"]
    metrics_line = _format_metrics_line(progress.metrics)
    if metrics_line:
        lines.append(html.escape(metrics_line))
    video_line = _format_video_status_line(progress)
    if video_line:
        lines.append(video_line)
    lines.append("")
    if created_events:
        lines.extend(
            _format_event_block(
                "Созданные события",
                created_events,
                icon="✅",
                bot_username=bot_username,
                ctx=ctx,
            )
        )
    if merged_events:
        lines.extend(
            _format_event_block(
                "Обновлённые события",
                merged_events,
                icon="🔄",
                bot_username=bot_username,
                ctx=ctx,
            )
        )
    try:
        for chunk in _chunk_lines(lines):
            await bot.send_message(
                int(chat_id),
                chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        return True
    except Exception:
        logger.exception("tg_monitor: failed to send per-post event details")
        return False


async def _send_event_details(
    bot,
    chat_id: int,
    report: TelegramMonitorReport,
    db: Database | None,
    *,
    include_event_lists: bool = True,
) -> None:
    bot_username = await _resolve_bot_username(bot) if bot else None
    sections: list[list[str]] = []
    ctx = None
    if db and include_event_lists:
        try:
            from source_parsing.smart_update_report import build_smart_update_report_context

            all_events = list(report.created_events or []) + list(report.merged_events or [])
            eids = [int(getattr(e, "event_id", 0) or 0) for e in all_events]
            urls = [str(getattr(e, "source_link", "") or "").strip() for e in all_events]
            ctx = await build_smart_update_report_context(db, event_ids=eids, source_urls=urls)
        except Exception:
            logger.debug("tg_monitor: report_context build failed", exc_info=True)
    if include_event_lists:
        if report.created_events:
            sections.append(
                _format_event_block(
                    "Созданные события",
                    report.created_events,
                    icon="✅",
                    bot_username=bot_username,
                    ctx=ctx,
                )
            )
        if report.merged_events:
            sections.append(
                _format_event_block(
                    "Обновлённые события",
                    report.merged_events,
                    icon="🔄",
                    bot_username=bot_username,
                    ctx=ctx,
                )
            )
    telegraph_map: dict[str, list[str]] = {}
    if db and getattr(report, "popular_posts", None):
        try:
            from models import Event, EventSource

            def _event_telegraph_url(event: Any) -> str | None:
                url = getattr(event, "telegraph_url", None)
                if url and str(url).strip().startswith(("http://", "https://")):
                    return str(url).strip()
                path = getattr(event, "telegraph_path", None)
                if path and str(path).strip():
                    return f"https://telegra.ph/{str(path).strip().lstrip('/')}"
                return None

            links: list[str] = []
            for it in list(getattr(report, "popular_posts", None) or []):
                link = (getattr(it, "source_link", None) or "").strip()
                if link.startswith(("http://", "https://")):
                    links.append(link)
            # Keep deterministic order (same as report order) but ensure uniqueness.
            uniq_links = list(dict.fromkeys(links))
            if uniq_links:
                async with db.get_session() as session:
                    rows = (
                        await session.execute(
                            select(EventSource.source_url, EventSource.event_id).where(
                                EventSource.source_url.in_(uniq_links)
                            )
                        )
                    ).all()
                    url_to_event_ids: dict[str, list[int]] = {}
                    event_ids: set[int] = set()
                    for source_url, eid in rows:
                        if not source_url or eid is None:
                            continue
                        su = str(source_url).strip()
                        if not su:
                            continue
                        try:
                            ev_id = int(eid)
                        except Exception:
                            continue
                        url_to_event_ids.setdefault(su, [])
                        if ev_id not in url_to_event_ids[su]:
                            url_to_event_ids[su].append(ev_id)
                        event_ids.add(ev_id)
                    if event_ids:
                        events = (
                            await session.execute(select(Event).where(Event.id.in_(sorted(event_ids))))
                        ).scalars().all()
                        id_to_url: dict[int, str] = {}
                        for ev in events:
                            try:
                                ev_id = int(getattr(ev, "id", 0) or 0)
                            except Exception:
                                continue
                            turl = _event_telegraph_url(ev)
                            if turl:
                                id_to_url[ev_id] = turl
                        for su, eids in url_to_event_ids.items():
                            turls: list[str] = []
                            for ev_id in eids:
                                turl = id_to_url.get(int(ev_id))
                                if turl and turl not in turls:
                                    turls.append(turl)
                                if len(turls) >= 3:
                                    break
                            if turls:
                                telegraph_map[su] = turls
        except Exception:
            logger.exception("tg_monitor: failed to resolve popular post telegraph links")

    popular_lines = _format_popular_posts_block(report, telegraph_map=telegraph_map)
    if popular_lines:
        sections.append(popular_lines)
    metrics_only_lines = _format_metrics_only_summary(report)
    if metrics_only_lines:
        sections.append(metrics_only_lines)
    skipped_lines = _format_skipped_posts_block(report)
    if skipped_lines:
        sections.append(skipped_lines)
    if not sections:
        if not include_event_lists:
            # Interactive runs already send per-post Smart Update details. Do not spam an
            # extra "zero changes" block when the final report has no extra sections.
            return
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


def _make_import_progress_callback(
    *,
    db: Database,
    bot: Any | None,
    chat_id: int | None,
    send_progress: bool,
) -> Callable[[TelegramMonitorImportProgress], Awaitable[None]]:
    import_progress_message_id: int | None = None
    import_progress_key: tuple[str, int] | None = None
    seen_done_keys: set[tuple[str, int]] = set()
    drain_raw = (os.getenv("TG_MONITORING_DRAIN_EVENT_JOBS") or "").strip().lower()
    if drain_raw:
        drain_event_jobs = drain_raw in {"1", "true", "yes"}
    else:
        drain_event_jobs = bool(send_progress)
    inline_drain_timeout_raw = (os.getenv("TG_MONITORING_INLINE_DRAIN_TIMEOUT_SEC") or "").strip()
    try:
        inline_drain_timeout_sec = float(inline_drain_timeout_raw) if inline_drain_timeout_raw else 10.0
    except Exception:
        inline_drain_timeout_sec = 10.0
    inline_drain_timeout_sec = max(0.5, min(inline_drain_timeout_sec, 120.0))

    async def _handle_import_progress(progress: TelegramMonitorImportProgress) -> None:
        nonlocal import_progress_message_id, import_progress_key
        if not (send_progress and bot and chat_id):
            return
        key = (str(progress.source_username), int(progress.message_id))
        if progress.stage == "start":
            start_text = _render_tg_post_progress_text("⏳", progress=progress)
            active_mid = import_progress_message_id if import_progress_key == key else None
            import_progress_key = key
            import_progress_message_id = await _upsert_progress_message(
                bot,
                chat_id=int(chat_id),
                message_id=active_mid,
                text=start_text,
            )
            return
        if progress.stage != "done":
            return
        if key in seen_done_keys:
            # Avoid UI spam if the importer (or a concurrent bot instance) emits duplicate "done".
            return
        seen_done_keys.add(key)
        created_cnt = len(progress.created_event_ids or [])
        merged_cnt = len(progress.merged_event_ids or [])
        imported_total = int(progress.events_imported or 0)

        if progress.status == "metrics_only":
            icon = "📈"
        elif created_cnt and merged_cnt:
            icon = "✅🔄"
        elif created_cnt:
            icon = "✅"
        elif merged_cnt:
            icon = "🔄"
        elif progress.status == "error":
            icon = "❌"
        elif progress.status == "partial":
            icon = "⚠️"
        elif progress.status in {"filtered", "skipped"}:
            icon = "⏭️"
        else:
            icon = "ℹ️"

        ids = list(progress.created_event_ids or []) + list(progress.merged_event_ids or [])
        ids_preview = ", ".join(str(int(eid)) for eid in ids[:5])
        extra_lines: list[str] = []
        if progress.status == "metrics_only":
            extra_lines.append("Метрики обновлены (без Smart Update)")
            metrics_line = _format_metrics_line(progress.metrics)
            if metrics_line:
                extra_lines.append(metrics_line)
        else:
            extra_lines.append(f"Smart Update: ✅{created_cnt} 🔄{merged_cnt}")
            if ids_preview:
                suffix = "…" if len(ids) > 5 else ""
                extra_lines.append(f"event_ids: {ids_preview}{suffix}")
            extra_lines.append(f"Иллюстрации: +{int(progress.added_posters_total or 0)}")
            video_line = _format_video_status_line(progress)
            if video_line:
                extra_lines.append(video_line.replace("Медиа поста:", "Медиа:", 1))
            if imported_total > 0:
                extra_lines.append("Отчёт Smart Update: ⏳")
            else:
                extra_lines.append("Отчёт Smart Update: —")
        if progress.reason:
            extra_lines.append(f"Причина: {progress.reason}")
        if progress.skip_breakdown:
            parts: list[str] = []
            for key_name, val in list(progress.skip_breakdown.items())[:5]:
                try:
                    val_i = int(val or 0)
                except Exception:
                    val_i = 0
                if val_i <= 0:
                    continue
                parts.append(f"{key_name}={val_i}")
            if parts:
                extra_lines.append(f"Пропуски: {', '.join(parts)}")

        active_mid = import_progress_message_id if import_progress_key == key else None
        summary_text = _render_tg_post_progress_text(icon, progress=progress, extra_lines=extra_lines)
        import_progress_message_id = await _upsert_progress_message(
            bot,
            chat_id=int(chat_id),
            message_id=active_mid,
            text=summary_text,
        )

        details_ok = True
        if imported_total > 0:
            # Best-effort: build Telegraph/ICS before sending per-post details.
            # This keeps Smart Update behaviour aligned with VK auto-queue (links are actionable
            # right after merge/create) and avoids confusing "old" Telegraph pages in prod DB snapshots.
            try:
                if drain_event_jobs and ids:
                    from main import JobTask, run_event_update_jobs
                    from models import Event, JobOutbox, JobStatus

                    allowed = {JobTask.ics_publish, JobTask.telegraph_build}
                    drain_ids = sorted({int(v) for v in ids if v})

                    drain_needed = True
                    try:
                        now = datetime.now(timezone.utc)
                        async with db.get_session() as session:
                            drain_needed = bool(
                                (
                                    await session.execute(
                                        select(JobOutbox.id)
                                        .where(
                                            JobOutbox.event_id.in_(drain_ids),
                                            JobOutbox.task.in_(sorted(allowed, key=lambda x: x.value)),
                                            JobOutbox.status.in_(
                                                [
                                                    JobStatus.pending,
                                                    JobStatus.error,
                                                    JobStatus.running,
                                                ]
                                            ),
                                            JobOutbox.next_run_at <= now,
                                        )
                                        .limit(1)
                                    )
                                ).first()
                            )
                    except Exception:
                        # Best-effort: if the query fails, try draining anyway.
                        drain_needed = True

                    drain_ok = True
                    if drain_needed:
                        for eid in drain_ids:
                            try:
                                await asyncio.wait_for(
                                    run_event_update_jobs(
                                        db,
                                        bot,
                                        event_id=eid,
                                        allowed_tasks=allowed,
                                    ),
                                    timeout=inline_drain_timeout_sec,
                                )
                            except asyncio.TimeoutError:
                                drain_ok = False
                                logger.warning(
                                    "tg_monitor.inline_drain timeout event_id=%s timeout_sec=%.1f",
                                    eid,
                                    inline_drain_timeout_sec,
                                )
                                break
                            except OperationalError as exc:
                                if _is_sqlite_locked_error(exc):
                                    logger.warning(
                                        "tg_monitor.inline_drain sqlite_locked event_id=%s err=%s",
                                        eid,
                                        str(exc)[:260],
                                    )
                                else:
                                    raise
                            except Exception:
                                logger.warning(
                                    "tg_monitor.inline_drain failed event_id=%s",
                                    eid,
                                    exc_info=True,
                                )

                    # Another worker may be processing `telegraph_build` concurrently and leave the
                    # outbox status as `running` briefly. Wait a bit so operator reports contain
                    # actionable Telegraph links.
                    pending_ids = {int(v) for v in ids if v} if drain_ok else set()
                    deadline = time.monotonic() + 12.0
                    while pending_ids and time.monotonic() < deadline:
                        async with db.get_session() as session:
                            rows = (
                                await session.execute(
                                    select(
                                        Event.id,
                                        Event.telegraph_url,
                                        Event.telegraph_path,
                                    ).where(Event.id.in_(sorted(pending_ids)))
                                )
                            ).all()
                        for ev_id, url, path in rows:
                            if (str(url or "").strip()) or (str(path or "").strip()):
                                pending_ids.discard(int(ev_id))
                        if pending_ids:
                            await asyncio.sleep(0.4)

                    # If Telegraph is still missing, try a direct build as a last resort
                    # (best-effort, avoids confusing "pending" in operator details).
                    if pending_ids:
                        try:
                            from source_parsing.handlers import _ensure_telegraph_url  # local helper
                        except Exception:
                            _ensure_telegraph_url = None  # type: ignore[assignment]
                        if _ensure_telegraph_url:
                            for ev_id in sorted(pending_ids):
                                try:
                                    await _ensure_telegraph_url(db, int(ev_id))
                                except Exception:
                                    logger.debug(
                                        "tg_monitor.inline_drain ensure_telegraph_url failed event_id=%s",
                                        ev_id,
                                        exc_info=True,
                                    )
                            # Re-check after direct build.
                            async with db.get_session() as session:
                                rows = (
                                    await session.execute(
                                        select(
                                            Event.id,
                                            Event.telegraph_url,
                                            Event.telegraph_path,
                                        ).where(Event.id.in_(sorted(pending_ids)))
                                    )
                                ).all()
                            for ev_id, url, path in rows:
                                if (str(url or "").strip()) or (str(path or "").strip()):
                                    pending_ids.discard(int(ev_id))

                # Refresh URLs/fact stats after (possible) draining so operator sees up-to-date links.
                from source_parsing.telegram.handlers import (
                    refresh_telegram_monitor_event_info,
                )

                for info in list(progress.created_events or []) + list(progress.merged_events or []):
                    try:
                        await refresh_telegram_monitor_event_info(db, info)
                    except Exception:
                        logger.debug(
                            "tg_monitor.inline_drain refresh failed event_id=%s",
                            getattr(info, "event_id", None),
                            exc_info=True,
                        )
            except Exception:
                logger.warning("tg_monitor.inline_drain failed", exc_info=True)
            details_ok = await _send_per_post_event_details(
                bot,
                chat_id=int(chat_id),
                progress=progress,
                db=db,
            )
            extra_lines = list(extra_lines)
            for idx, line in enumerate(extra_lines):
                if line.startswith("Отчёт Smart Update:"):
                    extra_lines[idx] = f"Отчёт Smart Update: {'✅' if details_ok else '⚠️'}"
                    break
        if progress.took_sec is not None:
            extra_lines.append(f"took_sec: {float(progress.took_sec):.1f}")
        final_text = _render_tg_post_progress_text(icon, progress=progress, extra_lines=extra_lines)
        import_progress_message_id = await _upsert_progress_message(
            bot,
            chat_id=int(chat_id),
            message_id=import_progress_message_id,
            text=final_text,
        )
        import_progress_key = None
        import_progress_message_id = None

    return _handle_import_progress


async def _post_import_finalize(
    db: Database,
    *,
    bot: Any | None,
    chat_id: int | None,
    report: TelegramMonitorReport,
    send_progress: bool = False,
) -> None:
    # Optional: drain only the jobs for the affected events. This makes local/E2E runs
    # deterministic without enabling a global outbox worker that would try to process
    # the entire prod snapshot backlog.
    worker_raw = (os.getenv("ENABLE_JOB_OUTBOX_WORKER") or "1").strip().lower()
    worker_enabled = worker_raw in {"1", "true", "yes"}
    drain_raw = (os.getenv("TG_MONITORING_DRAIN_EVENT_JOBS") or "").strip().lower()
    if drain_raw:
        drain = drain_raw in {"1", "true", "yes"}
    else:
        # Interactive runs drain inline (per message) via the progress callback.
        # Non-interactive runs drain only when the outbox worker is disabled.
        if send_progress:
            drain = False
        else:
            drain = not worker_enabled

    event_ids = sorted(
        {
            int(info.event_id)
            for info in (report.created_events + report.merged_events)
            if getattr(info, "event_id", None)
        }
    )

    drain_max = int(os.getenv("TG_MONITORING_DRAIN_MAX_EVENTS", "12") or 12)
    if not worker_enabled:
        should_drain = bool(drain and bot and event_ids)
    else:
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
            f"Новые посты (message_id): {int(getattr(report, 'messages_new', 0) or 0)}",
            f"Форс-обработка постов: {int(getattr(report, 'messages_forced', 0) or 0)}",
            f"Посты только для метрик: {int(getattr(report, 'messages_metrics_only', 0) or 0)}",
            f"Сообщений пропущено: {report.messages_skipped}",
            f"Событий извлечено (Kaggle): {report.events_extracted}",
            f"Событий в новых постах: {int(getattr(report, 'events_extracted_new', 0) or 0)}",
            f"Создано: {report.events_created}",
            f"Смёрджено: {report.events_merged}",
            f"Пропущено: {report.events_skipped}",
        ]
    )
    breakdown = [
        ("прошедшие", getattr(report, "events_past", 0)),
        ("невалидные", getattr(report, "events_invalid", 0)),
        ("отклонены", getattr(report, "events_rejected", 0)),
        ("без изменений", getattr(report, "events_nochange", 0)),
        ("отфильтрованы", getattr(report, "events_filtered", 0)),
        ("ошибки", getattr(report, "events_errored", 0)),
    ]
    if any(int(v or 0) for _k, v in breakdown):
        for label, value in breakdown:
            value_i = int(value or 0)
            if value_i:
                lines.append(f"  └ {label}: {value_i}")
    if report.errors:
        lines.append("")
        lines.append("Ошибки:")
        for err in report.errors[:5]:
            lines.append(f"- {err}")
        if len(report.errors) > 5:
            lines.append(f"... ещё {len(report.errors) - 5}")
    return "\n".join(lines)


async def _run_telegram_import_locked(
    db: Database,
    *,
    results_path: str | Path,
    bot=None,
    chat_id: int | None = None,
    run_id: str,
    send_progress: bool = False,
) -> TelegramMonitorReport:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"telegram_results.json not found: {path}")

    async def _notify(text: str) -> None:
        if not (send_progress and bot and chat_id):
            return
        try:
            await bot.send_message(chat_id, text, parse_mode="HTML")
        except Exception:
            logger.exception("tg_monitor_import: failed to send progress update")

    if send_progress and bot and chat_id:
        await _notify("♻️ Перезапускаю импорт из последнего локального telegram_results.json…")
        await _notify(f"📂 Файл: <code>{html.escape(str(path))}</code>")

    progress_callback = _make_import_progress_callback(
        db=db,
        bot=bot,
        chat_id=chat_id,
        send_progress=send_progress,
    )
    report = await _import_results_with_retry(
        path,
        db,
        bot=bot,
        run_id=run_id,
        progress_callback=progress_callback,
        notify=_notify,
    )
    await _post_import_finalize(
        db,
        bot=bot,
        chat_id=chat_id,
        report=report,
        send_progress=send_progress,
    )
    if bot and chat_id:
        try:
            await bot.send_message(chat_id, format_report(report), parse_mode="HTML")
            # In interactive runs we already send per-post event details. Avoid repeating
            # the full event list in the final report; keep only popular/skipped blocks.
            final_lists_raw = (os.getenv("TG_MONITORING_FINAL_EVENT_LIST") or "").strip().lower()
            if final_lists_raw:
                include_event_lists = final_lists_raw in {"1", "true", "yes"}
            else:
                include_event_lists = not bool(send_progress)
            await _send_event_details(
                bot,
                chat_id,
                report,
                db,
                include_event_lists=include_event_lists,
            )
        except Exception:
            logger.exception("tg_monitor_import: failed to send report")
    return report


async def run_telegram_import_from_results(
    db: Database,
    *,
    results_path: str | Path,
    bot=None,
    chat_id: int | None = None,
    run_id: str | None = None,
    send_progress: bool = False,
    trigger: str = "manual_import_only",
    operator_id: int | None = None,
) -> TelegramMonitorReport:
    started_ts = time.monotonic()
    run_id = run_id or uuid.uuid4().hex
    logger.info("tg_monitor_import.start run_id=%s path=%s", run_id, results_path)
    ops_run_id = await start_ops_run(
        db,
        kind="tg_monitoring",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={"run_id": run_id, "mode": "import_only", "results_path": str(results_path)},
    )
    status = "success"
    report = TelegramMonitorReport(run_id=run_id)
    try:
        if _RUN_LOCK.locked():
            status = "skipped"
            logger.warning("tg_monitor_import.skip reason=already_running run_id=%s", run_id)
            if bot and chat_id:
                try:
                    await bot.send_message(
                        chat_id,
                        "⏳ Мониторинг/импорт уже запущен, ждём завершения.",
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("tg_monitor_import: failed to notify already-running")
            report = TelegramMonitorReport(run_id=run_id, errors=["already_running"])
            return report

        async with _RUN_LOCK:
            try:
                async with _tg_monitor_global_lock(
                    bot=bot,
                    chat_id=chat_id,
                    send_progress=send_progress,
                    run_id=run_id,
                    purpose="import_only",
                ):
                    report = await _run_telegram_import_locked(
                        db,
                        results_path=results_path,
                        bot=bot,
                        chat_id=chat_id,
                        run_id=run_id,
                        send_progress=send_progress,
                    )
                    return report
            except TgMonitoringAlreadyRunningError:
                status = "skipped"
                report = TelegramMonitorReport(run_id=run_id, errors=["already_running_global_lock"])
                return report
    except Exception:
        status = "error"
        raise
    finally:
        duration_sec = float(time.monotonic() - started_ts)
        await finish_ops_run(
            db,
            run_id=ops_run_id,
            status=status,
            metrics={
                "messages_processed": int(report.messages_scanned or 0),
                "messages_with_events": int(report.messages_with_events or 0),
                "messages_new": int(getattr(report, "messages_new", 0) or 0),
                "messages_forced": int(getattr(report, "messages_forced", 0) or 0),
                "messages_metrics_only": int(getattr(report, "messages_metrics_only", 0) or 0),
                "events_imported": int((report.events_created or 0) + (report.events_merged or 0)),
                "events_created": int(report.events_created or 0),
                "events_merged": int(report.events_merged or 0),
                "popular_posts": int(len(getattr(report, "popular_posts", None) or [])),
                "errors_count": int(len(report.errors or [])),
                "duration_sec": round(duration_sec, 3),
            },
            details={
                "run_id": run_id,
                "mode": "import_only",
                "results_path": str(results_path),
                "errors": list(report.errors or [])[:40],
            },
        )


async def run_telegram_dev_recreate_reimport(
    db: Database,
    *,
    results_path: str | Path,
    bot=None,
    chat_id: int | None = None,
    run_id: str | None = None,
    send_progress: bool = False,
    trigger: str = "manual_import_dev_recreate",
    operator_id: int | None = None,
) -> TelegramMonitorReport:
    """
    DEV-only helper: delete all events/sources referenced by the given JSON and then
    re-import from the same JSON, in a single global lock window.
    """
    started_ts = time.monotonic()
    run_id = run_id or uuid.uuid4().hex
    logger.info("tg_monitor_dev_recreate_reimport.start run_id=%s path=%s", run_id, results_path)
    ops_run_id = await start_ops_run(
        db,
        kind="tg_monitoring",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={"run_id": run_id, "mode": "dev_recreate_reimport", "results_path": str(results_path)},
    )
    status = "success"
    report = TelegramMonitorReport(run_id=run_id)
    try:
        if _RUN_LOCK.locked():
            status = "skipped"
            logger.warning("tg_monitor_dev_recreate_reimport.skip reason=already_running run_id=%s", run_id)
            if bot and chat_id:
                try:
                    await bot.send_message(
                        int(chat_id),
                        "⏳ Мониторинг/импорт уже запущен, ждём завершения.",
                        parse_mode="HTML",
                    )
                except Exception:
                    logger.exception("tg_monitor_dev_recreate_reimport: failed to notify already-running")
            report = TelegramMonitorReport(run_id=run_id, errors=["already_running"])
            return report

        async with _RUN_LOCK:
            try:
                async with _tg_monitor_global_lock(
                    bot=bot,
                    chat_id=chat_id,
                    send_progress=send_progress,
                    run_id=run_id,
                    purpose="dev_recreate_reimport",
                ):
                    stats = await recreate_telegram_events_from_results(
                        db,
                        results_path=results_path,
                    )
                    if send_progress and bot and chat_id:
                        await bot.send_message(
                            int(chat_id),
                            (
                                "🧪 DEV Recreate завершён:\n"
                                f"• событий найдено: {int(stats.event_ids_found)}\n"
                                f"• событий удалено: {int(stats.events_deleted)}\n"
                                f"• JobOutbox удалено: {int(stats.joboutbox_deleted)}\n"
                                f"• already_scanned очищено: {int(stats.scanned_deleted)}\n"
                                f"• source_url ссылок из JSON: {int(stats.source_links_total)}\n\n"
                                f"♻️ Запускаю реимпорт из:\n{results_path}"
                            ),
                            disable_web_page_preview=True,
                        )
                    report = await _run_telegram_import_locked(
                        db,
                        results_path=results_path,
                        bot=bot,
                        chat_id=chat_id,
                        run_id=run_id,
                        send_progress=send_progress,
                    )
                    return report
            except TgMonitoringAlreadyRunningError:
                status = "skipped"
                report = TelegramMonitorReport(run_id=run_id, errors=["already_running_global_lock"])
                return report
    except Exception:
        status = "error"
        raise
    finally:
        duration_sec = float(time.monotonic() - started_ts)
        await finish_ops_run(
            db,
            run_id=ops_run_id,
            status=status,
            metrics={
                "sources_scanned": int(report.sources_total or 0),
                "messages_processed": int(report.messages_scanned or 0),
                "messages_with_events": int(report.messages_with_events or 0),
                "messages_new": int(getattr(report, "messages_new", 0) or 0),
                "messages_forced": int(getattr(report, "messages_forced", 0) or 0),
                "messages_metrics_only": int(getattr(report, "messages_metrics_only", 0) or 0),
                "events_imported": int((report.events_created or 0) + (report.events_merged or 0)),
                "events_created": int(report.events_created or 0),
                "events_merged": int(report.events_merged or 0),
                "popular_posts": int(len(getattr(report, "popular_posts", None) or [])),
                "errors_count": int(len(report.errors or [])),
                "duration_sec": round(duration_sec, 3),
            },
            details={
                "run_id": run_id,
                "errors": list(report.errors or [])[:40],
            },
        )


async def run_telegram_monitor(
    db: Database,
    *,
    bot=None,
    chat_id: int | None = None,
    run_id: str | None = None,
    send_progress: bool = False,
    trigger: str = "manual",
    operator_id: int | None = None,
) -> TelegramMonitorReport:
    started_ts = time.monotonic()
    run_id = run_id or uuid.uuid4().hex
    logger.info("tg_monitor.start run_id=%s", run_id)
    ops_run_id = await start_ops_run(
        db,
        kind="tg_monitoring",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={"run_id": run_id},
    )
    status = "success"
    report = TelegramMonitorReport(run_id=run_id)
    try:
        if _RUN_LOCK.locked():
            status = "skipped"
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
            report = TelegramMonitorReport(run_id=run_id, errors=["already_running"])
            return report

        async with _RUN_LOCK:
            try:
                async with _tg_monitor_global_lock(
                    bot=bot,
                    chat_id=chat_id,
                    send_progress=send_progress,
                    run_id=run_id,
                    purpose="monitor",
                ):
                    report = await _run_telegram_monitor_locked(
                        db,
                        bot=bot,
                        chat_id=chat_id,
                        run_id=run_id,
                        send_progress=send_progress,
                    )
                    return report
            except TgMonitoringAlreadyRunningError:
                status = "skipped"
                report = TelegramMonitorReport(run_id=run_id, errors=["already_running_global_lock"])
                return report
    except Exception:
        status = "error"
        raise
    finally:
        duration_sec = float(time.monotonic() - started_ts)
        await finish_ops_run(
            db,
            run_id=ops_run_id,
            status=status,
            metrics={
                "sources_scanned": int(report.sources_total or 0),
                "messages_processed": int(report.messages_scanned or 0),
                "messages_with_events": int(report.messages_with_events or 0),
                "messages_new": int(getattr(report, "messages_new", 0) or 0),
                "messages_forced": int(getattr(report, "messages_forced", 0) or 0),
                "messages_metrics_only": int(getattr(report, "messages_metrics_only", 0) or 0),
                "events_imported": int((report.events_created or 0) + (report.events_merged or 0)),
                "events_created": int(report.events_created or 0),
                "events_merged": int(report.events_merged or 0),
                "popular_posts": int(len(getattr(report, "popular_posts", None) or [])),
                "errors_count": int(len(report.errors or [])),
                "duration_sec": round(duration_sec, 3),
            },
            details={
                "run_id": run_id,
                "errors": list(report.errors or [])[:40],
            },
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
    poll_timeout_minutes = _compute_kaggle_poll_timeout_minutes(sources_count=len(sources))
    logger.info(
        "tg_monitor.timeout_plan run_id=%s mode=%s sources=%d timeout_minutes=%d "
        "(base=%s per_source=%s safety=%s effective_per_source=%s max=%s floor=%s)",
        run_id,
        TIMEOUT_MODE,
        len(sources),
        poll_timeout_minutes,
        TIMEOUT_BASE_MINUTES,
        TIMEOUT_PER_SOURCE_MINUTES,
        TIMEOUT_SAFETY_MULTIPLIER,
        TIMEOUT_PER_SOURCE_MINUTES * TIMEOUT_SAFETY_MULTIPLIER,
        TIMEOUT_MAX_MINUTES,
        TIMEOUT_MINUTES,
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
    progress_callback = _make_import_progress_callback(
        db=db,
        bot=bot,
        chat_id=chat_id,
        send_progress=send_progress,
    )

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
            timeout_minutes=poll_timeout_minutes,
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
            failure = _extract_kaggle_failure_message(status_data)
            if _is_auth_key_duplicated_failure(status_data):
                await _notify(_build_auth_key_duplicated_hint())
                raise RuntimeError(
                    "Kaggle kernel failed: AuthKeyDuplicatedError (Telegram session duplicated across IPs)"
                )
            if status == "timeout":
                await _notify(
                    (
                        f"⏳ Не дождались завершения Kaggle kernel за {poll_timeout_minutes} мин (локальный таймаут ожидания). "
                        "Kernel мог продолжить выполнение в Kaggle — попробуйте увеличить таймаут ожидания."
                    ).strip()
                )
            else:
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
        report = await _import_results_with_retry(
            results_path,
            db,
            bot=bot,
            run_id=run_id,
            progress_callback=progress_callback,
            notify=_notify,
        )
        logger.info("tg_monitor: completed in %.1fs", duration)
        await _post_import_finalize(
            db,
            bot=bot,
            chat_id=chat_id,
            report=report,
            send_progress=send_progress,
        )

        if bot and chat_id:
            try:
                await bot.send_message(chat_id, format_report(report), parse_mode="HTML")
                await _send_event_details(bot, chat_id, report, db)
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
        await run_telegram_monitor(
            db,
            bot=bot,
            chat_id=chat_id,
            run_id=run_id,
            trigger="scheduled",
            operator_id=0,
        )
    except Exception:
        logger.exception("tg_monitor: scheduler failed run_id=%s", run_id)
