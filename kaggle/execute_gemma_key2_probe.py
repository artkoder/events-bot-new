#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source_parsing.telegram.split_secrets import encrypt_secret
from video_announce.kaggle_client import KaggleClient

logger = logging.getLogger("gemma_key2_probe")

DEFAULT_KERNEL_PATH = PROJECT_ROOT / "kaggle" / "GemmaKey2Probe"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "codex" / "kaggle" / "gemma-key2-probe"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_SECRET_VAR = "GOOGLE_API_KEY2"
DEFAULT_MODEL = "models/gemma-3-27b-it"
DEFAULT_PROMPT = "Reply with exactly: OK"
DEFAULT_MAX_OUTPUT_TOKENS = 8
DEFAULT_TIMEOUT_MINUTES = 15
DEFAULT_POLL_INTERVAL_SECONDS = 20
DEFAULT_DATASET_WAIT_SECONDS = 30
DEFAULT_DATASET_CIPHER_PREFIX = "gemma-key2-probe-cipher"
DEFAULT_DATASET_KEY_PREFIX = "gemma-key2-probe-key"


def load_env_file_values(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def apply_env_file_defaults(paths: Sequence[Path]) -> list[Path]:
    loaded: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        for key, value in load_env_file_values(path).items():
            os.environ.setdefault(key, value)
        loaded.append(path)
    return loaded


def candidate_secret_names(secret_env_var: str) -> list[str]:
    normalized = (secret_env_var or "").strip()
    if not normalized:
        return []
    names = [normalized]
    if normalized == "GOOGLE_API_KEY2":
        names.append("GOOGLE_API_KEY_2")
    elif normalized == "GOOGLE_API_KEY_2":
        names.append("GOOGLE_API_KEY2")
    return names


def resolve_secret_value(
    secret_env_var: str,
    env_files: Sequence[Path],
) -> tuple[str, str, str]:
    lookup_names = candidate_secret_names(secret_env_var)
    for name in lookup_names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value, name, "environment"
    for env_file in env_files:
        values = load_env_file_values(env_file)
        for name in lookup_names:
            value = (values.get(name) or "").strip()
            if value:
                return value, name, str(env_file)
    raise RuntimeError(
        f"Secret not found for {secret_env_var}; checked env and {[str(p) for p in env_files]}"
    )


def normalize_model_name(model: str) -> str:
    raw = (model or "").strip()
    if not raw:
        raise ValueError("model must not be empty")
    if raw.startswith("models/"):
        return raw
    return f"models/{raw}"


def build_probe_config(
    *,
    secret_env_var: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    lookup_names = candidate_secret_names(secret_env_var)
    if not lookup_names:
        raise ValueError("secret_env_var must not be empty")
    return {
        "probe_name": "gemma-key2-probe",
        "secret_env_var": lookup_names[0],
        "secret_env_aliases": lookup_names[1:],
        "model": normalize_model_name(model),
        "prompt": str(prompt),
        "max_output_tokens": max(1, int(max_output_tokens)),
        "request_timeout_seconds": 90,
    }


def build_secret_payload(config_payload: dict[str, Any], secret_value: str) -> str:
    payload: dict[str, str] = {}
    for name in [
        str(config_payload.get("secret_env_var") or "").strip(),
        *[str(item).strip() for item in (config_payload.get("secret_env_aliases") or [])],
    ]:
        if name:
            payload[name] = secret_value
    local_name = (os.getenv("GOOGLE_API_LOCALNAME") or "").strip()
    if local_name:
        payload["GOOGLE_API_LOCALNAME"] = local_name
    return json.dumps(payload, ensure_ascii=False)


def _slugify(value: str, *, max_len: int = 60) -> str:
    raw = (value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = raw.strip("-")
    if not raw:
        raw = uuid.uuid4().hex[:8]
    return raw[:max_len].rstrip("-") or raw[:max_len] or uuid.uuid4().hex[:8]


def _build_dataset_slug(prefix: str, run_id: str) -> str:
    return f"{prefix}-{_slugify(run_id, max_len=16)}"


def _require_kaggle_username() -> str:
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not set")
    return username


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
            logger.exception("gemma_key2_probe.dataset_create_failed dataset=%s", slug)
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
                logger.exception("gemma_key2_probe.dataset_version_failed dataset=%s", slug)
            try:
                client.delete_dataset(slug, no_confirm=True)
            except Exception:
                logger.exception("gemma_key2_probe.dataset_delete_failed dataset=%s", slug)
            client.create_dataset(tmp_path)
    return slug


def prepare_probe_datasets(
    *,
    client: KaggleClient,
    config_payload: dict[str, Any],
    secrets_payload: str,
    run_id: str,
    cipher_prefix: str = DEFAULT_DATASET_CIPHER_PREFIX,
    key_prefix: str = DEFAULT_DATASET_KEY_PREFIX,
) -> tuple[str, str]:
    encrypted, fernet_key = encrypt_secret(secrets_payload)
    username = _require_kaggle_username()

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
        _build_dataset_slug(cipher_prefix, run_id),
        f"Gemma Key2 Probe Cipher {_slugify(run_id, max_len=16)}",
        write_cipher,
    )
    slug_key = _create_dataset(
        client,
        username,
        _build_dataset_slug(key_prefix, run_id),
        f"Gemma Key2 Probe Key {_slugify(run_id, max_len=16)}",
        write_key,
    )
    return slug_cipher, slug_key


def kernel_ref_from_meta(kernel_path: Path) -> str:
    meta_path = kernel_path / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kernel_id = str(meta.get("id") or meta.get("slug") or "").strip()
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if username and kernel_id:
        if "/" in kernel_id:
            owner, slug = kernel_id.split("/", 1)
            if slug and owner != username:
                return f"{username}/{slug}"
        else:
            return f"{username}/{kernel_id}"
    return kernel_id


def push_kernel(
    client: KaggleClient,
    *,
    kernel_path: Path = DEFAULT_KERNEL_PATH,
    dataset_sources: list[str],
) -> str:
    kernel_ref = kernel_ref_from_meta(kernel_path)
    client.push_kernel(kernel_path=kernel_path, dataset_sources=dataset_sources)
    return kernel_ref


def poll_kernel(
    client: KaggleClient,
    kernel_ref: str,
    *,
    timeout_minutes: int,
    poll_interval_seconds: int,
) -> tuple[str, dict[str, Any] | None, float]:
    started = time.monotonic()
    deadline = started + max(1, int(timeout_minutes)) * 60
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        status = client.get_kernel_status(kernel_ref)
        last_status = dict(status or {})
        state = str(last_status.get("status") or "").upper()
        logger.info(
            "gemma_key2_probe.kernel_status kernel=%s status=%s failure=%s",
            kernel_ref,
            state or "UNKNOWN",
            last_status.get("failureMessage") or last_status.get("failure_message"),
        )
        if state == "COMPLETE":
            return "complete", last_status, time.monotonic() - started
        if state in {"ERROR", "FAILED", "CANCELLED"}:
            return "failed", last_status, time.monotonic() - started
        time.sleep(max(5, int(poll_interval_seconds)))
    return "timeout", last_status, time.monotonic() - started


def download_probe_output(
    client: KaggleClient,
    kernel_ref: str,
    output_dir: Path,
) -> tuple[list[str], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = client.download_kernel_output(kernel_ref, path=output_dir, force=True)
    output_path = next(output_dir.rglob("output.json"), None)
    if output_path and output_path.exists():
        return files, json.loads(output_path.read_text(encoding="utf-8"))
    return files, None


def cleanup_datasets(client: KaggleClient, slugs: Sequence[str]) -> None:
    for slug in slugs:
        if not slug:
            continue
        try:
            client.delete_dataset(slug, no_confirm=True)
        except Exception:
            logger.exception("gemma_key2_probe.dataset_cleanup_failed dataset=%s", slug)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma GOOGLE_API_KEY2 probe on Kaggle.")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional env file(s) to load with setdefault semantics.",
    )
    parser.add_argument("--secret-var", default=DEFAULT_SECRET_VAR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--dataset-wait-seconds", type=int, default=DEFAULT_DATASET_WAIT_SECONDS)
    parser.add_argument("--keep-datasets", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    env_files = [DEFAULT_ENV_FILE]
    for raw in args.env_file:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate not in env_files:
            env_files.append(candidate)
    loaded_env_files = apply_env_file_defaults(env_files)
    logger.info("gemma_key2_probe.env_files loaded=%s", [str(path) for path in loaded_env_files])

    secret_value, resolved_name, secret_source = resolve_secret_value(args.secret_var, env_files)
    logger.info(
        "gemma_key2_probe.secret_resolved requested=%s resolved=%s source=%s",
        args.secret_var,
        resolved_name,
        secret_source,
    )

    config_payload = build_probe_config(
        secret_env_var=args.secret_var,
        model=args.model,
        prompt=args.prompt,
        max_output_tokens=args.max_output_tokens,
    )
    secrets_payload = build_secret_payload(config_payload, secret_value)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output_dir = DEFAULT_OUTPUT_ROOT / run_id
    client = KaggleClient()
    dataset_cipher = ""
    dataset_key = ""
    try:
        dataset_cipher, dataset_key = prepare_probe_datasets(
            client=client,
            config_payload=config_payload,
            secrets_payload=secrets_payload,
            run_id=run_id,
        )
        logger.info(
            "gemma_key2_probe.datasets cipher=%s key=%s wait_seconds=%s",
            dataset_cipher,
            dataset_key,
            args.dataset_wait_seconds,
        )
        if args.dataset_wait_seconds > 0:
            time.sleep(args.dataset_wait_seconds)

        kernel_ref = push_kernel(
            client,
            kernel_path=DEFAULT_KERNEL_PATH,
            dataset_sources=[dataset_cipher, dataset_key],
        )
        logger.info("gemma_key2_probe.kernel_pushed ref=%s", kernel_ref)
        status, status_data, duration = poll_kernel(
            client,
            kernel_ref,
            timeout_minutes=args.timeout_minutes,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        logger.info(
            "gemma_key2_probe.kernel_done ref=%s status=%s duration_sec=%.1f",
            kernel_ref,
            status,
            duration,
        )
        files, output_payload = download_probe_output(client, kernel_ref, output_dir)
        logger.info("gemma_key2_probe.output_files=%s", files)
        if output_payload is None:
            logger.error("gemma_key2_probe.output_missing dir=%s", output_dir)
            return 1
        logger.info(
            "gemma_key2_probe.summary ok=%s status_code=%s model=%s response_excerpt=%s",
            output_payload.get("ok"),
            output_payload.get("status_code"),
            output_payload.get("model"),
            (output_payload.get("response_excerpt") or output_payload.get("error_excerpt") or "")[:160],
        )
        if status != "complete":
            failure = ""
            if isinstance(status_data, dict):
                failure = str(status_data.get("failureMessage") or status_data.get("failure_message") or "")
            logger.error("gemma_key2_probe.kernel_terminal_failure status=%s failure=%s", status, failure)
            return 1
        return 0 if output_payload.get("ok") else 1
    finally:
        if not args.keep_datasets:
            cleanup_datasets(client, [dataset_cipher, dataset_key])
        else:
            logger.info(
                "gemma_key2_probe.datasets_kept cipher=%s key=%s",
                dataset_cipher,
                dataset_key,
            )


if __name__ == "__main__":
    raise SystemExit(main())
