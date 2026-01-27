import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from db import Database
from models import TelegramSource
from source_parsing.telegram.deduplication import get_month_context_urls
from source_parsing.telegram.handlers import TelegramMonitorReport, process_telegram_results
from video_announce.kaggle_client import KaggleClient

from .split_secrets import encrypt_secret

logger = logging.getLogger(__name__)

CONFIG_DATASET_CIPHER = os.getenv("TG_MONITORING_CONFIG_CIPHER", "telegram-monitor-cipher")
CONFIG_DATASET_KEY = os.getenv("TG_MONITORING_CONFIG_KEY", "telegram-monitor-key")
KEEP_DATASETS = os.getenv("TG_MONITORING_KEEP_DATASETS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

KERNEL_REF = os.getenv("TG_MONITORING_KERNEL_REF", "artkoder/telegram-monitor-bot")
KERNEL_PATH = Path(os.getenv("TG_MONITORING_KERNEL_PATH", "kaggle/TelegramMonitor"))

DATASET_PROPAGATION_WAIT_SECONDS = int(os.getenv("TG_MONITORING_DATASET_WAIT", "30"))
POLL_INTERVAL_SECONDS = int(os.getenv("TG_MONITORING_POLL_INTERVAL", "30"))
TIMEOUT_MINUTES = int(os.getenv("TG_MONITORING_TIMEOUT_MINUTES", "90"))
KAGGLE_STARTUP_WAIT_SECONDS = int(os.getenv("TG_MONITORING_STARTUP_WAIT", "10"))


def _require_env(key: str) -> str:
    value = (os.getenv(key) or "").strip()
    if not value:
        raise RuntimeError(f"{key} is missing")
    return value


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
    payload = {
        "TG_SESSION": _require_env("TG_SESSION"),
        "TG_API_ID": _require_env("TG_API_ID"),
        "TG_API_HASH": _require_env("TG_API_HASH"),
        "GOOGLE_API_KEY": _require_env("GOOGLE_API_KEY"),
    }
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
) -> tuple[str, dict | None, float]:
    started = time.monotonic()
    deadline = started + TIMEOUT_MINUTES * 60
    last_status: dict | None = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status
        state = (status.get("status") or "").upper()
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
            return "complete", last_status, time.monotonic() - started
        if state in ("ERROR", "FAILED", "CANCELLED"):
            return "failed", last_status, time.monotonic() - started
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
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
) -> TelegramMonitorReport:
    run_id = run_id or uuid.uuid4().hex
    logger.info("tg_monitor.start run_id=%s", run_id)
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
        logger.info(
            "tg_monitor.kernel pushed run_id=%s kernel_ref=%s datasets=%s",
            run_id,
            kernel_ref,
            [dataset_cipher, dataset_key],
        )
        await asyncio.sleep(KAGGLE_STARTUP_WAIT_SECONDS)

        status, status_data, duration = await _poll_kaggle_kernel(
            client, kernel_ref, run_id=run_id
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
            raise RuntimeError(f"Kaggle kernel failed ({status}) {failure}".strip())

        results_path = await _download_results(client, kernel_ref, run_id)
        logger.info(
            "tg_monitor.results_downloaded run_id=%s kernel_ref=%s path=%s",
            run_id,
            kernel_ref,
            results_path,
        )
        report = await process_telegram_results(results_path, db)
        logger.info("tg_monitor: completed in %.1fs", duration)

        if bot and chat_id:
            try:
                await bot.send_message(chat_id, format_report(report), parse_mode="HTML")
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
