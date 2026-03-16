from __future__ import annotations

import asyncio
import base64
import contextlib
import html
import json
import logging
import math
import os
import re
import ssl
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiosqlite

from db import Database
from kaggle_registry import register_job, remove_job
from source_parsing.telegram.split_secrets import encrypt_secret
from video_announce.kaggle_client import KaggleClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNEL_REF = os.getenv("GUIDE_MONITORING_KERNEL_REF", "artkoder/guide-excursions-monitor")
_RAW_KERNEL_PATH = Path(os.getenv("GUIDE_MONITORING_KERNEL_PATH", "kaggle/GuideExcursionsMonitor"))
KERNEL_PATH = _RAW_KERNEL_PATH if _RAW_KERNEL_PATH.is_absolute() else PROJECT_ROOT / _RAW_KERNEL_PATH
GOOGLE_AI_PACKAGE_PATH = PROJECT_ROOT / "google_ai"
KERNEL_SLUG_OVERRIDE = (os.getenv("GUIDE_MONITORING_KERNEL_SLUG") or "").strip()
CONFIG_DATASET_CIPHER = os.getenv("GUIDE_MONITORING_CONFIG_CIPHER", "guide-excursions-cipher")
CONFIG_DATASET_KEY = os.getenv("GUIDE_MONITORING_CONFIG_KEY", "guide-excursions-key")
KEEP_DATASETS = (os.getenv("GUIDE_MONITORING_KEEP_DATASETS") or "").strip().lower() in {
    "1",
    "true",
    "yes",
}
DATASET_PROPAGATION_WAIT_SECONDS = int(os.getenv("GUIDE_MONITORING_DATASET_WAIT", "60"))
KAGGLE_STARTUP_WAIT_SECONDS = int(os.getenv("GUIDE_MONITORING_STARTUP_WAIT", "10"))
POLL_INTERVAL_SECONDS = int(os.getenv("GUIDE_MONITORING_POLL_INTERVAL", "30"))
TIMEOUT_MINUTES = int(os.getenv("GUIDE_MONITORING_TIMEOUT_MINUTES", "120"))
TIMEOUT_MODE = (os.getenv("GUIDE_MONITORING_TIMEOUT_MODE") or "dynamic").strip().lower()
TIMEOUT_BASE_MINUTES = int(os.getenv("GUIDE_MONITORING_TIMEOUT_BASE_MINUTES", "12"))
TIMEOUT_PER_SOURCE_MINUTES = float(os.getenv("GUIDE_MONITORING_TIMEOUT_PER_SOURCE_MINUTES", "2.8"))
TIMEOUT_SAFETY_MULTIPLIER = float(os.getenv("GUIDE_MONITORING_TIMEOUT_SAFETY_MULTIPLIER", "1.35"))
TIMEOUT_MAX_MINUTES = int(os.getenv("GUIDE_MONITORING_TIMEOUT_MAX_MINUTES", "240"))
STALE_COMPLETE_MIN_SECONDS = int(os.getenv("GUIDE_MONITORING_STALE_COMPLETE_MIN_SECONDS", "45"))
RESULTS_MATCH_RETRY_ATTEMPTS = max(
    8,
    min(int((os.getenv("GUIDE_MONITORING_RESULTS_MATCH_RETRY_ATTEMPTS") or "30") or 30), 120),
)
RESULTS_MATCH_RETRY_DELAY_SECONDS = max(
    2.0,
    min(float((os.getenv("GUIDE_MONITORING_RESULTS_MATCH_RETRY_DELAY_SECONDS") or "10") or 10), 60.0),
)
REMOTE_KERNEL_SHAPE_RETRY_ATTEMPTS = max(
    1,
    min(int((os.getenv("GUIDE_MONITORING_REMOTE_SHAPE_RETRY_ATTEMPTS") or "6") or 6), 20),
)
REMOTE_KERNEL_SHAPE_RETRY_DELAY_SECONDS = max(
    1.0,
    min(float((os.getenv("GUIDE_MONITORING_REMOTE_SHAPE_RETRY_DELAY_SECONDS") or "5") or 5), 30.0),
)
LOCAL_RESULTS_GLOB = (
    os.getenv("GUIDE_MONITORING_LOCAL_RESULTS_GLOB", "guide-excursions-*/guide_excursions_results.json")
    or "guide-excursions-*/guide_excursions_results.json"
).strip()


def _require_env_any(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    raise RuntimeError(f"Missing required env var, checked: {', '.join(names)}")


def _parse_auth_bundle(env_key: str) -> dict[str, Any]:
    bundle_b64 = _require_env_any(env_key)
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
    allow_non_s22 = (
        (os.getenv("GUIDE_MONITORING_ALLOW_NON_S22_AUTH") or "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    override = (os.getenv("GUIDE_MONITORING_AUTH_BUNDLE_ENV") or "").strip()
    if override:
        if override != "TELEGRAM_AUTH_BUNDLE_S22" and not allow_non_s22:
            raise RuntimeError(
                "GUIDE_MONITORING_AUTH_BUNDLE_ENV may use only TELEGRAM_AUTH_BUNDLE_S22 for Kaggle "
                "unless GUIDE_MONITORING_ALLOW_NON_S22_AUTH=1 is explicitly set"
            )
        return override
    if (os.getenv("TELEGRAM_AUTH_BUNDLE_S22") or "").strip():
        return "TELEGRAM_AUTH_BUNDLE_S22"
    if (os.getenv("TELEGRAM_AUTH_BUNDLE_E2E") or "").strip():
        if not allow_non_s22:
            raise RuntimeError(
                "Guide Kaggle monitoring requires TELEGRAM_AUTH_BUNDLE_S22; refusing to fall back to "
                "TELEGRAM_AUTH_BUNDLE_E2E without GUIDE_MONITORING_ALLOW_NON_S22_AUTH=1"
            )
        return "TELEGRAM_AUTH_BUNDLE_E2E"
    return None


def _require_kaggle_username() -> str:
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not set")
    return username


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
    slug = f"{safe_prefix}-{safe_run}" if safe_run else safe_prefix
    return slug[:60].rstrip("-")


def _read_kernel_metadata(kernel_path: Path) -> dict[str, Any]:
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


async def _enable_row_factory(conn: aiosqlite.Connection) -> None:
    conn.row_factory = aiosqlite.Row


async def _build_config_payload(
    db: Database,
    *,
    run_id: str,
    mode: str,
    limit: int,
    days_back: int,
) -> dict[str, Any]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            """
            SELECT
                gs.username,
                gs.title,
                gs.source_kind,
                gs.trust_level,
                gs.priority_weight,
                gs.flags_json,
                gs.base_region,
                gs.last_scanned_message_id,
                gp.display_name,
                gp.marketing_name
            FROM guide_source gs
            LEFT JOIN guide_profile gp ON gp.id = gs.primary_profile_id
            WHERE gs.platform='telegram' AND COALESCE(gs.enabled, 1)=1
            ORDER BY gs.priority_weight DESC, gs.username ASC
            """
        )
        rows = await cur.fetchall()
    sources: list[dict[str, Any]] = []
    for row in rows:
        try:
            flags = json.loads(row["flags_json"]) if row["flags_json"] else {}
        except Exception:
            flags = {}
        sources.append(
            {
                "username": str(row["username"] or "").strip(),
                "title": str(row["title"] or "").strip() or None,
                "source_kind": str(row["source_kind"] or "").strip() or None,
                "trust_level": str(row["trust_level"] or "").strip() or None,
                "priority_weight": float(row["priority_weight"] or 1.0),
                "flags": flags if isinstance(flags, dict) else {},
                "base_region": str(row["base_region"] or "").strip() or None,
                "last_scanned_message_id": int(row["last_scanned_message_id"]) if row["last_scanned_message_id"] is not None else None,
                "display_name": str(row["display_name"] or "").strip() or None,
                "marketing_name": str(row["marketing_name"] or "").strip() or None,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "scan_mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limit_per_source": int(limit),
        "days_back": int(days_back),
        "sources": sources,
    }


def _build_secrets_payload() -> str:
    auth_bundle_env = _resolve_auth_bundle_env_key()
    payload = {
        "TG_API_ID": _require_env_any("TG_API_ID", "TELEGRAM_API_ID"),
        "TG_API_HASH": _require_env_any("TG_API_HASH", "TELEGRAM_API_HASH"),
        "GOOGLE_API_KEY2": _require_env_any("GOOGLE_API_KEY2"),
    }
    if auth_bundle_env:
        bundle_raw = _require_env_any(auth_bundle_env)
        payload["TELEGRAM_AUTH_BUNDLE_S22"] = bundle_raw
        payload["GUIDE_MONITORING_AUTH_BUNDLE_ENV"] = auth_bundle_env
        try:
            bundle = _parse_auth_bundle(auth_bundle_env)
        except Exception:
            logger.warning("guide_monitor.secrets_payload invalid bundle env=%s", auth_bundle_env, exc_info=True)
        else:
            payload["TG_SESSION"] = str(bundle.get("session") or "").strip()
    else:
        payload["TG_SESSION"] = _require_env_any("TG_SESSION", "TELEGRAM_SESSION")
    for key in (
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_KEY",
        "SUPABASE_SCHEMA",
        "SUPABASE_DISABLED",
        "GUIDE_MONITORING_MODEL",
        "GUIDE_MONITORING_SCREEN_MODEL",
        "GUIDE_MONITORING_EXTRACT_MODEL",
        "GUIDE_MONITORING_LLM_TIMEOUT_SEC",
        "GUIDE_MONITORING_GOOGLE_KEY_ENV",
        "GOOGLE_API_LOCALNAME2",
        "GOOGLE_API_LOCALNAME",
        "GOOGLE_AI_ALLOW_RESERVE_FALLBACK",
        "GOOGLE_AI_LOCAL_LIMITER_FALLBACK",
        "GOOGLE_AI_LOCAL_LIMITER_ON_RESERVE_ERROR",
        "GOOGLE_AI_LOCAL_RPM",
        "GOOGLE_AI_LOCAL_TPM",
        "GOOGLE_AI_LOCAL_RPD",
        "GOOGLE_AI_RESERVE_RPC_RECHECK_SECONDS",
        "GOOGLE_AI_RESERVE_RPC_RETRY_ATTEMPTS",
        "GOOGLE_AI_RESERVE_RPC_RETRY_BASE_DELAY_MS",
        "GOOGLE_AI_INCIDENT_NOTIFICATIONS",
        "GOOGLE_AI_INCIDENT_COOLDOWN_SECONDS",
        "GOOGLE_AI_RESERVE_DIRECT_RETRY",
        "GOOGLE_AI_RESERVE_DIRECT_SCHEMA",
    ):
        value = (os.getenv(key) or "").strip()
        if value:
            payload[key] = value
    return json.dumps(payload, ensure_ascii=False)


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
            logger.exception("guide_monitor.dataset_create_failed dataset=%s", slug)
            try:
                client.create_dataset_version(
                    tmp_path,
                    version_notes=f"refresh {slug_suffix}",
                    quiet=True,
                    convert_to_csv=False,
                    dir_mode="zip",
                )
                return slug
            except Exception:
                logger.exception("guide_monitor.dataset_version_failed dataset=%s", slug)
            try:
                client.delete_dataset(slug, no_confirm=True)
            except Exception:
                logger.exception("guide_monitor.dataset_delete_failed dataset=%s", slug)
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
        f"Guide Excursions Cipher {slug_suffix}",
        write_cipher,
    )
    slug_key = _create_dataset(
        client,
        username,
        _build_dataset_slug(CONFIG_DATASET_KEY, run_id),
        f"Guide Excursions Key {slug_suffix}",
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
        return str(kernel_id)
    except Exception:
        return KERNEL_REF


def stage_repo_bundle(output_root: Path) -> Path:
    if not GOOGLE_AI_PACKAGE_PATH.exists():
        raise FileNotFoundError(f"Missing package: {GOOGLE_AI_PACKAGE_PATH}")
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GOOGLE_AI_PACKAGE_PATH, output_root / "google_ai")
    return output_root


def _embedded_google_ai_sources() -> dict[str, str]:
    files = ("__init__.py", "client.py", "exceptions.py", "secrets.py")
    payload: dict[str, str] = {}
    for name in files:
        path = GOOGLE_AI_PACKAGE_PATH / name
        if not path.exists():
            raise FileNotFoundError(f"Missing google_ai source for notebook embed: {path}")
        payload[name] = path.read_text(encoding="utf-8")
    return payload


def _build_notebook_payload_from_script(script_path: Path) -> dict[str, Any]:
    script_source = script_path.read_text(encoding="utf-8")
    script_source = re.sub(
        r"\nif __name__ == [\"']__main__[\"']:\n\s+asyncio\.run\(main\(\)\)\s*$",
        "\n",
        script_source,
        flags=re.MULTILINE,
    )
    embedded_google_ai = json.dumps(_embedded_google_ai_sources(), ensure_ascii=False)
    script_lines = script_source.splitlines(keepends=True)
    future_import_lines: list[str] = []
    while script_lines and script_lines[0].startswith("from __future__ import "):
        future_import_lines.append(script_lines.pop(0))
    if script_lines and not script_lines[0].strip():
        future_import_lines.append(script_lines.pop(0))
    source_lines = list(future_import_lines)
    source_lines.extend(
        [
            "from pathlib import Path as _GuideNotebookPath\n",
            "import sys as _GuideNotebookSys\n",
            "_GUIDE_EMBEDDED_GOOGLE_AI = " + embedded_google_ai + "\n",
            "_GUIDE_EMBEDDED_ROOT = (_GuideNotebookPath.cwd() / 'embedded_repo_bundle').resolve()\n",
            "_GUIDE_EMBEDDED_PACKAGE = _GUIDE_EMBEDDED_ROOT / 'google_ai'\n",
            "_GUIDE_EMBEDDED_PACKAGE.mkdir(parents=True, exist_ok=True)\n",
            "for _guide_name, _guide_body in _GUIDE_EMBEDDED_GOOGLE_AI.items():\n",
            "    (_GUIDE_EMBEDDED_PACKAGE / _guide_name).write_text(_guide_body, encoding='utf-8')\n",
            "if str(_GUIDE_EMBEDDED_ROOT) not in _GuideNotebookSys.path:\n",
            "    _GuideNotebookSys.path.insert(0, str(_GUIDE_EMBEDDED_ROOT))\n",
            "__file__ = str((_GuideNotebookPath.cwd() / 'guide_excursions_monitor.py').resolve())\n",
            "\n",
        ]
    )
    source_lines.extend(script_lines)
    if source_lines and not source_lines[-1].endswith("\n"):
        source_lines[-1] = f"{source_lines[-1]}\n"
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Guide Excursions Monitor\n",
                    "\n",
                    "Kaggle notebook for guide excursions fact-first monitoring and exporting `guide_excursions_results.json`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Guide notebook bootstrap complete', flush=True)\n",
                    "print(f'Guide notebook runner file={__file__}', flush=True)\n",
                    "print(f'Guide notebook embedded google_ai root={_GUIDE_EMBEDDED_ROOT}', flush=True)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import asyncio\n",
                    "import nest_asyncio\n",
                    "\n",
                    "def _guide_run_main_sync() -> None:\n",
                    "    try:\n",
                    "        loop = asyncio.get_event_loop()\n",
                    "    except RuntimeError:\n",
                    "        loop = asyncio.new_event_loop()\n",
                    "        asyncio.set_event_loop(loop)\n",
                    "    if loop.is_closed():\n",
                    "        loop = asyncio.new_event_loop()\n",
                    "        asyncio.set_event_loop(loop)\n",
                    "    if loop.is_running():\n",
                    "        nest_asyncio.apply(loop)\n",
                    "    loop.run_until_complete(main())\n",
                    "\n",
                    "_guide_run_main_sync()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def _sync_notebook_entrypoint(kernel_path: Path) -> None:
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    code_file = str(meta.get("code_file") or "").strip()
    if not code_file.endswith(".ipynb"):
        return
    notebook_path = kernel_path / code_file
    script_path = kernel_path / f"{Path(code_file).stem}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Guide Kaggle runner script missing for notebook build: {script_path}")
    notebook = _build_notebook_payload_from_script(script_path)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_kernel_slug_override(kernel_path: Path) -> None:
    if not KERNEL_SLUG_OVERRIDE:
        return
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if username:
        meta["id"] = f"{username}/{KERNEL_SLUG_OVERRIDE}"
    else:
        meta["id"] = KERNEL_SLUG_OVERRIDE
    meta["slug"] = KERNEL_SLUG_OVERRIDE
    meta["title"] = KERNEL_SLUG_OVERRIDE.replace("-", " ").title()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


@contextlib.contextmanager
def _prepared_kernel_path(kernel_path: Path) -> Path:
    if not _kernel_has_code(kernel_path):
        raise RuntimeError(f"Guide kernel code missing: {kernel_path}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        prepared = tmp_path / kernel_path.name
        shutil.copytree(kernel_path, prepared)
        stage_repo_bundle(prepared)
        _apply_kernel_slug_override(prepared)
        _sync_notebook_entrypoint(prepared)
        yield prepared


async def _push_kernel(client: KaggleClient, dataset_sources: list[str]) -> str:
    with _prepared_kernel_path(KERNEL_PATH) as prepared_path:
        expected_meta = _read_kernel_metadata(prepared_path)
        kernel_ref = _kernel_ref_from_meta(prepared_path)
        client.push_kernel(kernel_path=prepared_path, dataset_sources=dataset_sources)
        return kernel_ref, expected_meta


async def _pull_remote_kernel_metadata(
    client: KaggleClient,
    kernel_ref: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="guide-kernel-meta-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        await asyncio.to_thread(client.kernels_pull, kernel_ref, tmp_path, True)
        return _read_kernel_metadata(tmp_path)


async def _wait_for_remote_kernel_shape(
    client: KaggleClient,
    kernel_ref: str,
    *,
    expected_meta: dict[str, Any],
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    expected_type = str(expected_meta.get("kernel_type") or "").strip().lower()
    expected_code_file = str(expected_meta.get("code_file") or "").strip()
    expected_suffix = Path(expected_code_file).suffix.lower()
    last_meta: dict[str, Any] = {}

    for attempt in range(1, REMOTE_KERNEL_SHAPE_RETRY_ATTEMPTS + 1):
        meta = await _pull_remote_kernel_metadata(client, kernel_ref)
        last_meta = dict(meta or {})
        observed_type = str(last_meta.get("kernel_type") or "").strip().lower()
        observed_code_file = str(last_meta.get("code_file") or "").strip()
        observed_suffix = Path(observed_code_file).suffix.lower()
        if observed_type == expected_type and observed_suffix == expected_suffix:
            return last_meta

        if status_callback:
            failure = (
                "Жду канонический Kaggle notebook: "
                f"expected {expected_type or '-'} {expected_code_file or '-'}, "
                f"observed {observed_type or '-'} {observed_code_file or '-'}"
            )
            try:
                await status_callback(
                    "kernel_shape_wait",
                    kernel_ref,
                    {
                        "status": "PUSHED",
                        "failureMessage": failure,
                        "_kernel_shape_attempt": attempt,
                        "_kernel_shape_max_attempts": REMOTE_KERNEL_SHAPE_RETRY_ATTEMPTS,
                    },
                )
            except Exception:
                logger.exception(
                    "guide_monitor.status_callback_failed phase=kernel_shape_wait kernel=%s",
                    kernel_ref,
                )

        if attempt < REMOTE_KERNEL_SHAPE_RETRY_ATTEMPTS:
            await asyncio.sleep(REMOTE_KERNEL_SHAPE_RETRY_DELAY_SECONDS)

    observed_type = str(last_meta.get("kernel_type") or "").strip().lower() or "-"
    observed_code_file = str(last_meta.get("code_file") or "").strip() or "-"
    raise RuntimeError(
        "Guide Kaggle kernel shape mismatch: "
        f"expected kernel_type={expected_type or '-'} code_file={expected_code_file or '-'} "
        f"but observed kernel_type={observed_type} code_file={observed_code_file}. "
        "Recreate the canonical Kaggle kernel as a notebook."
    )


def _compute_poll_timeout_minutes(*, sources_count: int) -> int:
    if TIMEOUT_MODE in {"fixed", "static"}:
        return max(1, TIMEOUT_MINUTES)
    est = TIMEOUT_BASE_MINUTES + int(math.ceil(max(0, int(sources_count)) * TIMEOUT_PER_SOURCE_MINUTES * TIMEOUT_SAFETY_MULTIPLIER))
    return min(TIMEOUT_MAX_MINUTES, max(TIMEOUT_MINUTES, est))


def _extract_failure_message(status: dict | None) -> str:
    if not status:
        return ""
    for key in ("failureMessage", "failure_message", "errorMessage", "error_message", "error"):
        value = status.get(key)
        if value:
            return str(value)
    return ""


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

    async def _notify(phase: str, payload: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, payload)
        except Exception:
            logger.exception("guide_monitor.status_callback_failed phase=%s kernel=%s", phase, kernel_ref)

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
                "guide_monitor.kernel_poll_transient_error run_id=%s kernel_ref=%s attempt=%s consecutive=%s err=%s",
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

        payload = dict(status or {})
        payload["_poll_timeout_minutes"] = timeout_minutes
        payload["_elapsed_seconds"] = time.monotonic() - started
        await _notify("poll", payload)
        state = str((status or {}).get("status") or "").upper()
        if state == "COMPLETE":
            elapsed = time.monotonic() - started
            if elapsed < max(0, STALE_COMPLETE_MIN_SECONDS):
                payload["failureMessage"] = (
                    "Kaggle ещё показывает предыдущий COMPLETE; жду fresh session/output"
                )
                await _notify("stale_complete", payload)
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
            await _notify("complete", payload)
            return "complete", status, time.monotonic() - started
        if state in {"ERROR", "FAILED", "CANCELLED"}:
            await _notify("failed", payload)
            return "failed", status, time.monotonic() - started
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    try:
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status or last_status
        state = str((status or {}).get("status") or "").upper()
        if state == "COMPLETE":
            payload = dict(last_status or {})
            payload["_poll_timeout_minutes"] = timeout_minutes
            payload["_elapsed_seconds"] = time.monotonic() - started
            await _notify("complete", payload)
            return "complete", last_status, time.monotonic() - started
    except Exception:
        pass

    payload = dict(last_status or {})
    payload.setdefault("failureMessage", "локальный таймаут ожидания: Kaggle kernel мог продолжить выполнение")
    payload["_poll_timeout_minutes"] = timeout_minutes
    payload["_elapsed_seconds"] = time.monotonic() - started
    await _notify("timeout", payload)
    return "timeout", last_status, time.monotonic() - started


async def _download_results(
    client: KaggleClient,
    kernel_ref: str,
    run_id: str,
    *,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> Path:
    output_dir = Path(tempfile.gettempdir()) / f"guide-excursions-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = RESULTS_MATCH_RETRY_ATTEMPTS
    last_seen_run_id: str | None = None
    for attempt in range(1, max_attempts + 1):
        files = await asyncio.to_thread(
            client.download_kernel_output,
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        for name in files:
            path = output_dir / name
            if path.name == "guide_excursions_results.json":
                observed_run_id = _read_results_run_id(path)
                if observed_run_id == run_id:
                    return path
                last_seen_run_id = observed_run_id
                logger.warning(
                    "guide_monitor.results_run_id_mismatch kernel=%s attempt=%s expected=%s observed=%s path=%s",
                    kernel_ref,
                    attempt,
                    run_id,
                    observed_run_id,
                    path,
                )
                if status_callback:
                    try:
                        await status_callback(
                            "fresh_output_wait",
                            kernel_ref,
                            {
                                "status": "COMPLETE",
                                "failureMessage": (
                                    "Жду fresh Kaggle output: "
                                    f"expected run_id={run_id}, observed={observed_run_id or '-'}"
                                ),
                                "_fresh_output_attempt": attempt,
                                "_fresh_output_max_attempts": max_attempts,
                            },
                        )
                    except Exception:
                        logger.exception(
                            "guide_monitor.status_callback_failed phase=fresh_output_wait kernel=%s",
                            kernel_ref,
                        )
        if attempt < max_attempts:
            await asyncio.sleep(RESULTS_MATCH_RETRY_DELAY_SECONDS)
    if last_seen_run_id is not None:
        raise RuntimeError(
            "guide_excursions_results.json run_id mismatch: "
            f"expected={run_id} observed={last_seen_run_id}"
        )
    raise RuntimeError("guide_excursions_results.json not found in Kaggle output")


def _read_results_run_id(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = payload.get("run_id")
    return str(run_id).strip() or None if run_id is not None else None


async def download_guide_results(
    client: KaggleClient,
    kernel_ref: str,
    run_id: str,
) -> Path:
    return await _download_results(client, kernel_ref, run_id)


def extract_guide_failure_message(status: dict | None) -> str:
    return _extract_failure_message(status)


async def _cleanup_datasets(dataset_slugs: list[str]) -> None:
    if KEEP_DATASETS:
        return
    client = KaggleClient()
    for slug in dataset_slugs:
        if not slug:
            continue
        try:
            await asyncio.to_thread(client.delete_dataset, slug)
        except Exception:
            logger.exception("guide_monitor.dataset_cleanup_failed dataset=%s", slug)


def find_latest_guide_results_json(search_root: str | Path | None = None) -> Path:
    root = Path(search_root) if search_root is not None else Path(tempfile.gettempdir())
    candidates = [path for path in root.glob(LOCAL_RESULTS_GLOB) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No local guide_excursions_results.json found under {root} (pattern={LOCAL_RESULTS_GLOB})"
        )
    candidates.sort(key=lambda item: (item.stat().st_mtime, item.stat().st_ctime, str(item)), reverse=True)
    return candidates[0]


def format_kaggle_status_message(phase: str, kernel_ref: str, status: dict | None) -> str:
    labels = {
        "prepare": "подготовка",
        "pushed": "запуск в Kaggle",
        "kernel_shape_wait": "ожидание notebook kernel",
        "poll": "выполнение",
        "poll_error": "временная ошибка сети",
        "stale_complete": "ожидание fresh session",
        "fresh_output_wait": "ожидание fresh output",
        "complete": "завершено",
        "failed": "ошибка",
        "timeout": "таймаут",
    }
    lines = [
        "🛰️ Kaggle: Guide Excursions Monitor",
        f"Kernel: {html.escape(kernel_ref or '—')}",
        f"Этап: {labels.get(phase, phase)}",
    ]
    if status:
        state = str(status.get("status") or "UNKNOWN")
        failure = _extract_failure_message(status)
        lines.append(f"Статус Kaggle: {html.escape(state if not failure else f'{state} ({failure})')}")
        timeout_minutes = status.get("_poll_timeout_minutes")
        elapsed_seconds = status.get("_elapsed_seconds")
        if timeout_minutes:
            if elapsed_seconds is not None:
                lines.append(f"Ожидание: до {int(timeout_minutes)} мин (прошло {float(elapsed_seconds) / 60.0:.1f} мин)")
            else:
                lines.append(f"Ожидание: до {int(timeout_minutes)} мин")
        fresh_attempt = status.get("_fresh_output_attempt")
        fresh_total = status.get("_fresh_output_max_attempts")
        if fresh_attempt and fresh_total:
            lines.append(f"Fresh output: попытка {int(fresh_attempt)}/{int(fresh_total)}")
        shape_attempt = status.get("_kernel_shape_attempt")
        shape_total = status.get("_kernel_shape_max_attempts")
        if shape_attempt and shape_total:
            lines.append(f"Kernel shape: попытка {int(shape_attempt)}/{int(shape_total)}")
    return "\n".join(lines)


async def run_guide_monitor_kaggle(
    db: Database,
    *,
    run_id: str,
    mode: str,
    limit: int,
    days_back: int,
    chat_id: int | None = None,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    client = KaggleClient()
    config_payload = await _build_config_payload(db, run_id=run_id, mode=mode, limit=limit, days_back=days_back)
    sources_count = len(config_payload.get("sources") or [])
    timeout_minutes = _compute_poll_timeout_minutes(sources_count=sources_count)
    secrets_payload = _build_secrets_payload()
    dataset_slugs: list[str] = []
    kernel_ref = _kernel_ref_from_meta(KERNEL_PATH)
    remote_kernel_meta: dict[str, Any] = {}
    registered_recovery = False
    await asyncio.sleep(0)
    try:
        if status_callback:
            await status_callback("prepare", kernel_ref, None)
        slug_cipher, slug_key = await _prepare_kaggle_datasets(
            client=client,
            config_payload=config_payload,
            secrets_payload=secrets_payload,
            run_id=run_id,
        )
        dataset_slugs = [slug_cipher, slug_key]
        if DATASET_PROPAGATION_WAIT_SECONDS > 0:
            await asyncio.sleep(DATASET_PROPAGATION_WAIT_SECONDS)
        kernel_ref, expected_kernel_meta = await _push_kernel(client, dataset_sources=dataset_slugs)
        remote_kernel_meta = await _wait_for_remote_kernel_shape(
            client,
            kernel_ref,
            expected_meta=expected_kernel_meta,
            status_callback=status_callback,
        )
        try:
            await register_job(
                "guide_monitoring",
                kernel_ref,
                meta={
                    "run_id": run_id,
                    "mode": mode,
                    "chat_id": chat_id,
                    "pid": os.getpid(),
                    "dataset_slugs": list(dataset_slugs),
                },
            )
            registered_recovery = True
        except Exception:
            logger.warning("guide_monitor.register_job_failed kernel=%s", kernel_ref, exc_info=True)
        if status_callback:
            await status_callback("pushed", kernel_ref, None)
        if KAGGLE_STARTUP_WAIT_SECONDS > 0:
            await asyncio.sleep(KAGGLE_STARTUP_WAIT_SECONDS)
        status, status_data, duration = await _poll_kaggle_kernel(
            client,
            kernel_ref,
            run_id=run_id,
            timeout_minutes=timeout_minutes,
            status_callback=status_callback,
        )
        if status != "complete":
            failure = _extract_failure_message(status_data)
            raise RuntimeError(f"Guide Kaggle kernel failed ({status}) {failure}".strip())
        result_path = await _download_results(
            client,
            kernel_ref,
            run_id,
            status_callback=status_callback,
        )
        return result_path, {
            "kernel_ref": kernel_ref,
            "status": status,
            "status_data": status_data or {},
            "duration_sec": int(max(0, round(duration))),
            "timeout_minutes": timeout_minutes,
            "sources_count": sources_count,
            "dataset_slugs": list(dataset_slugs),
            "remote_kernel_meta": dict(remote_kernel_meta or {}),
        }
    finally:
        if registered_recovery:
            try:
                await remove_job("guide_monitoring", kernel_ref)
            except Exception:
                logger.exception("guide_monitor.remove_job_failed kernel=%s", kernel_ref)
        await _cleanup_datasets(dataset_slugs)
