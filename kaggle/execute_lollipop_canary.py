#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from source_parsing.telegram.split_secrets import encrypt_secret
from video_announce.kaggle_client import KaggleClient

logger = logging.getLogger("lollipop_canary")

DEFAULT_KERNEL_PATH = PROJECT_ROOT / "kaggle" / "LollipopCanary"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "codex" / "kaggle" / "lollipop-canary"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_DB_PATH = PROJECT_ROOT / "db_prod_snapshot.sqlite"
DEFAULT_SECRET_VAR = "GOOGLE_API_KEY2"
DEFAULT_CANARY_EVENT_IDS = [2657, 2673, 2747, 2759]
DEFAULT_DATE_TAG = "2026-03-12k"
DEFAULT_RUN_LABEL = "v2_16_2_iter9_kaggle_canary"
DEFAULT_TIMEOUT_MINUTES = 45
DEFAULT_POLL_INTERVAL_SECONDS = 20
DEFAULT_DATASET_WAIT_SECONDS = 45
DEFAULT_GEMMA_CALL_GAP_S = 6.0
DEFAULT_PAYLOAD_DATASET_PREFIX = "lollipop-canary-payload"
DEFAULT_CIPHER_DATASET_PREFIX = "lollipop-canary-cipher"
DEFAULT_KEY_DATASET_PREFIX = "lollipop-canary-key"
DEFAULT_ITER6_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "codex"
    / "smart_update_lollipop_writer_final_4o_family_v2_16_2_iter6_2026-03-11c.json"
)

ROOT_FILE_MANIFEST = [
    "db.py",
    "models.py",
    "sections.py",
    "smart_event_update.py",
]
CODEX_FILE_MANIFEST = [
    "experimental_pattern_atomic_step_tuning_v2_15_5_2026_03_08.py",
    "experimental_pattern_dryrun_v2_15_3_2026_03_08.py",
    "experimental_pattern_v2_15_4_common_2026_03_08.py",
    "smart_update_lollipop_editorial_layout_family_v2_16_2_iter1_2026_03_10.py",
    "smart_update_lollipop_facts_dedup_family_v2_16_2_iter3_2026_03_09.py",
    "smart_update_lollipop_facts_extract_family_v2_16_2_2026_03_09.py",
    "smart_update_lollipop_facts_merge_family_v2_16_2_iter2_2026_03_09.py",
    "smart_update_lollipop_facts_merge_family_v2_16_2_iter3_2026_03_09.py",
    "smart_update_lollipop_facts_merge_family_v2_16_2_iter5_2026_03_09.py",
    "smart_update_lollipop_facts_prioritize_family_v2_16_2_iter1_2026_03_10.py",
    "smart_update_lollipop_writer_final_4o_family_v2_16_2_iter1_2026_03_10.py",
    "smart_update_lollipop_writer_pack_compose_family_v2_16_2_iter1_2026_03_10.py",
]
SHIM_MAIN = """
from __future__ import annotations

from typing import Any


def get_supabase_client() -> None:
    return None


async def notify_llm_incident(kind: str, payload: dict[str, Any]) -> None:
    return None
""".strip() + "\n"


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
            logger.exception("lollipop_canary.dataset_create_failed dataset=%s", slug)
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
                logger.exception("lollipop_canary.dataset_version_failed dataset=%s", slug)
            try:
                client.delete_dataset(slug, no_confirm=True)
            except Exception:
                logger.exception("lollipop_canary.dataset_delete_failed dataset=%s", slug)
            client.create_dataset(tmp_path)
    return slug


def _copy_file(src: Path, dst_root: Path, relative_path: str) -> None:
    target = dst_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)


def stage_repo_tree(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for rel_path in ROOT_FILE_MANIFEST:
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Missing repo file: {src}")
        _copy_file(src, output_root, rel_path)

    google_ai_src = PROJECT_ROOT / "google_ai"
    if not google_ai_src.exists():
        raise FileNotFoundError(f"Missing package: {google_ai_src}")
    shutil.copytree(google_ai_src, output_root / "google_ai")

    for rel_name in CODEX_FILE_MANIFEST:
        src = PROJECT_ROOT / "artifacts" / "codex" / rel_name
        if not src.exists():
            raise FileNotFoundError(f"Missing codex file: {src}")
        _copy_file(src, output_root, str(Path("artifacts") / "codex" / rel_name))

    (output_root / "main.py").write_text(SHIM_MAIN, encoding="utf-8")


def build_repo_bundle(output_zip_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        stage_root = Path(tmp_dir) / "repo"
        stage_repo_tree(stage_root)
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in stage_root.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(stage_root))


def candidate_secret_names(secret_env_var: str) -> list[str]:
    normalized = (secret_env_var or "").strip()
    if not normalized:
        return []
    names = [normalized]
    if normalized == "GOOGLE_API_KEY2":
        names.extend(["GOOGLE_API_KEY_2", "GOOGLE_API_KEY"])
    elif normalized == "GOOGLE_API_KEY_2":
        names.extend(["GOOGLE_API_KEY2", "GOOGLE_API_KEY"])
    elif normalized == "GOOGLE_API_KEY":
        names.extend(["GOOGLE_API_KEY2", "GOOGLE_API_KEY_2"])
    return names


def resolve_secret_value(secret_names: Sequence[str], env_files: Sequence[Path]) -> tuple[str, str, str]:
    for name in secret_names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value, name, "environment"
    for env_file in env_files:
        values = load_env_file_values(env_file)
        for name in secret_names:
            value = (values.get(name) or "").strip()
            if value:
                return value, name, str(env_file)
    raise RuntimeError(f"Secret not found for any of {list(secret_names)}")


def parse_event_ids(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_CANARY_EVENT_IDS)
    event_ids: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        event_ids.append(int(chunk))
    return event_ids or list(DEFAULT_CANARY_EVENT_IDS)


def build_canary_config(
    *,
    run_id: str,
    event_ids: list[int],
    date_tag: str,
    run_label: str,
    gemma_call_gap_s: float,
    db_filename: str,
    repo_archive_filename: str,
) -> dict[str, Any]:
    return {
        "run_name": "lollipop-canary",
        "run_id": run_id,
        "event_ids": list(event_ids),
        "date_tag": date_tag,
        "run_label": run_label,
        "db_filename": db_filename,
        "repo_archive_filename": repo_archive_filename,
        "gemma_call_gap_s": max(0.0, float(gemma_call_gap_s)),
    }


def build_secret_payload(
    *,
    gemma_value: str,
    four_o_token: str,
    google_api_localname: str | None,
) -> str:
    payload = {
        "GOOGLE_API_KEY": gemma_value,
        "GOOGLE_API_KEY2": gemma_value,
        "GOOGLE_API_KEY_2": gemma_value,
        "FOUR_O_TOKEN": four_o_token,
    }
    if google_api_localname:
        payload["GOOGLE_API_LOCALNAME"] = google_api_localname
    return json.dumps(payload, ensure_ascii=False)


def prepare_run_datasets(
    *,
    client: KaggleClient,
    config_payload: dict[str, Any],
    secret_payload: str,
    db_path: Path,
    run_id: str,
    payload_prefix: str = DEFAULT_PAYLOAD_DATASET_PREFIX,
    cipher_prefix: str = DEFAULT_CIPHER_DATASET_PREFIX,
    key_prefix: str = DEFAULT_KEY_DATASET_PREFIX,
) -> tuple[str, str, str]:
    encrypted, fernet_key = encrypt_secret(secret_payload)
    username = _require_kaggle_username()

    def write_payload(path: Path) -> None:
        stage_repo_tree(path / "repo_bundle")
        shutil.copy2(db_path, path / str(config_payload["db_filename"]))
        (path / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_cipher(path: Path) -> None:
        (path / "secrets.enc").write_bytes(encrypted)

    def write_key(path: Path) -> None:
        (path / "fernet.key").write_bytes(fernet_key)

    payload_slug = _create_dataset(
        client,
        username,
        _build_dataset_slug(payload_prefix, run_id),
        f"Lollipop Canary Payload {_slugify(run_id, max_len=16)}",
        write_payload,
    )
    cipher_slug = _create_dataset(
        client,
        username,
        _build_dataset_slug(cipher_prefix, run_id),
        f"Lollipop Canary Cipher {_slugify(run_id, max_len=16)}",
        write_cipher,
    )
    key_slug = _create_dataset(
        client,
        username,
        _build_dataset_slug(key_prefix, run_id),
        f"Lollipop Canary Key {_slugify(run_id, max_len=16)}",
        write_key,
    )
    return payload_slug, cipher_slug, key_slug


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


@contextlib.contextmanager
def prepared_kernel_path(
    kernel_path: Path,
    *,
    kernel_slug: str | None,
) -> Path:
    if not kernel_slug:
        yield kernel_path
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copytree(kernel_path, tmp_path / kernel_path.name)
        prepared = tmp_path / kernel_path.name
        meta_path = prepared / "kernel-metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        username = (os.getenv("KAGGLE_USERNAME") or "").strip()
        if username:
            meta["id"] = f"{username}/{kernel_slug}"
        else:
            meta["id"] = kernel_slug
        meta["slug"] = kernel_slug
        meta["title"] = kernel_slug.replace("-", " ").title()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        yield prepared


def push_kernel(
    client: KaggleClient,
    *,
    kernel_path: Path,
    dataset_sources: list[str],
    kernel_slug: str | None = None,
) -> str:
    with prepared_kernel_path(kernel_path, kernel_slug=kernel_slug) as prepared_path:
        kernel_ref = kernel_ref_from_meta(prepared_path)
        client.push_kernel(kernel_path=prepared_path, dataset_sources=dataset_sources)
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
        try:
            status = client.get_kernel_status(kernel_ref)
        except Exception as exc:
            logger.warning("lollipop_canary.kernel_status_probe_failed kernel=%s error=%s", kernel_ref, exc)
            time.sleep(max(5, int(poll_interval_seconds)))
            continue
        last_status = dict(status or {})
        state = str(last_status.get("status") or "").upper()
        logger.info(
            "lollipop_canary.kernel_status kernel=%s status=%s failure=%s",
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


def download_output(
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


def cleanup_datasets(client: KaggleClient, slugs: Iterable[str]) -> None:
    for slug in slugs:
        if not slug:
            continue
        try:
            client.delete_dataset(slug, no_confirm=True)
        except Exception:
            logger.exception("lollipop_canary.dataset_cleanup_failed dataset=%s", slug)


def _heading_count(markdown_text: str) -> int:
    return len(re.findall(r"(?m)^###\s+", markdown_text or ""))


def _load_iter6_reference() -> dict[int, dict[str, Any]]:
    if not DEFAULT_ITER6_PATH.exists():
        return {}
    payload = json.loads(DEFAULT_ITER6_PATH.read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for item in payload.get("results") or []:
        event_id = int(item["event_id"])
        result = item.get("result") or {}
        applied = result.get("applied_output") or {}
        markdown_text = str(applied.get("description_md") or "")
        metrics = result.get("metrics") or {}
        out[event_id] = {
            "description_length": int(metrics.get("description_length") or len(markdown_text)),
            "heading_count": _heading_count(markdown_text),
            "warnings": len((item.get("audit") or {}).get("warnings") or []),
            "errors": len((item.get("audit") or {}).get("errors") or []),
        }
    return out


def build_local_report(output_payload: dict[str, Any], output_dir: Path) -> Path:
    reference = _load_iter6_reference()
    aggregate = output_payload.get("final_summary") or {}
    canaries = output_payload.get("canaries") or []
    lines = [
        "# Lollipop Kaggle Canary",
        "",
        f"- Run ID: `{output_payload.get('run_id')}`",
        f"- Event IDs: `{','.join(str(x) for x in (output_payload.get('event_ids') or []))}`",
        f"- Kernel status: `{output_payload.get('kernel_status')}`",
        f"- Avg description length: `{aggregate.get('avg_description_length')}`",
        f"- Total warnings: `{aggregate.get('warning_count')}`",
        f"- Total errors: `{aggregate.get('error_count')}`",
        "",
        "## Canary Verdict",
        "",
    ]
    for item in canaries:
        event_id = int(item.get("event_id") or 0)
        ref = reference.get(event_id) or {}
        length = int(item.get("description_length") or 0)
        delta = length - int(ref.get("description_length") or 0) if ref else None
        heading_delta = (
            int(item.get("heading_count") or 0) - int(ref.get("heading_count") or 0) if ref else None
        )
        lines.append(f"### {event_id}")
        lines.append(f"- length: `{length}`" + (f" (`{delta:+d}` vs iter6)" if delta is not None else ""))
        lines.append(
            f"- headings: `{item.get('heading_count')}`"
            + (f" (`{heading_delta:+d}` vs iter6)" if heading_delta is not None else "")
        )
        lines.append(f"- warnings/errors: `{item.get('warning_count')}` / `{item.get('error_count')}`")
        if "has_cookie_tea_leak" in item:
            lines.append(f"- cookies/tea leak: `{item.get('has_cookie_tea_leak')}`")
        if "has_koroleva_luiza_tail" in item:
            lines.append(f"- Koroleva Luiza tail: `{item.get('has_koroleva_luiza_tail')}`")
        lines.append(f"- title: `{item.get('title')}`")
        excerpt = str(item.get("description_excerpt") or "").strip()
        if excerpt:
            lines.append("")
            lines.append("```md")
            lines.append(excerpt)
            lines.append("```")
        lines.append("")
    report_path = output_dir / "lollipop-canary-report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lollipop canary pipeline on Kaggle.")
    parser.add_argument("--env-file", action="append", default=[], help="Additional env file(s) to load.")
    parser.add_argument("--secret-var", default=DEFAULT_SECRET_VAR)
    parser.add_argument("--events", default=",".join(str(x) for x in DEFAULT_CANARY_EVENT_IDS))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--gemma-call-gap-s", type=float, default=DEFAULT_GEMMA_CALL_GAP_S)
    parser.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--dataset-wait-seconds", type=int, default=DEFAULT_DATASET_WAIT_SECONDS)
    parser.add_argument("--keep-datasets", action="store_true")
    parser.add_argument("--kernel-slug", default="")
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
    logger.info("lollipop_canary.env_files loaded=%s", [str(path) for path in loaded_env_files])

    event_ids = parse_event_ids(args.events)
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not db_path.exists():
        raise FileNotFoundError(f"DB snapshot not found: {db_path}")

    gemma_names = candidate_secret_names(args.secret_var)
    gemma_value, gemma_name, gemma_source = resolve_secret_value(gemma_names, env_files)
    four_o_value, four_o_name, four_o_source = resolve_secret_value(["FOUR_O_TOKEN"], env_files)
    google_localname = (os.getenv("GOOGLE_API_LOCALNAME") or "").strip() or None
    logger.info(
        "lollipop_canary.secrets_resolved gemma=%s from=%s four_o=%s from=%s",
        gemma_name,
        gemma_source,
        four_o_name,
        four_o_source,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output_dir = DEFAULT_OUTPUT_ROOT / run_id
    config_payload = build_canary_config(
        run_id=run_id,
        event_ids=event_ids,
        date_tag=args.date_tag,
        run_label=args.run_label,
        gemma_call_gap_s=args.gemma_call_gap_s,
        db_filename=db_path.name,
        repo_archive_filename="repo_bundle.zip",
    )
    secret_payload = build_secret_payload(
        gemma_value=gemma_value,
        four_o_token=four_o_value,
        google_api_localname=google_localname,
    )

    client = KaggleClient()
    payload_slug = ""
    cipher_slug = ""
    key_slug = ""
    try:
        payload_slug, cipher_slug, key_slug = prepare_run_datasets(
            client=client,
            config_payload=config_payload,
            secret_payload=secret_payload,
            db_path=db_path,
            run_id=run_id,
        )
        logger.info(
            "lollipop_canary.datasets payload=%s cipher=%s key=%s wait_seconds=%s",
            payload_slug,
            cipher_slug,
            key_slug,
            args.dataset_wait_seconds,
        )
        if args.dataset_wait_seconds > 0:
            time.sleep(args.dataset_wait_seconds)

        kernel_ref = push_kernel(
            client,
            kernel_path=DEFAULT_KERNEL_PATH,
            dataset_sources=[payload_slug, cipher_slug, key_slug],
            kernel_slug=(args.kernel_slug or "").strip() or None,
        )
        logger.info("lollipop_canary.kernel_pushed ref=%s", kernel_ref)
        status, status_data, duration = poll_kernel(
            client,
            kernel_ref,
            timeout_minutes=args.timeout_minutes,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        logger.info(
            "lollipop_canary.kernel_done ref=%s status=%s duration_sec=%.1f",
            kernel_ref,
            status,
            duration,
        )
        files, output_payload = download_output(client, kernel_ref, output_dir)
        logger.info("lollipop_canary.output_files=%s", files)
        if output_payload is None:
            logger.error("lollipop_canary.output_missing dir=%s", output_dir)
            return 1
        output_payload["kernel_status"] = status
        output_payload["kernel_status_data"] = status_data or {}
        (output_dir / "output.json").write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_path = build_local_report(output_payload, output_dir)
        logger.info("lollipop_canary.report_path=%s", report_path)
        if status != "complete":
            failure = ""
            if isinstance(status_data, dict):
                failure = str(status_data.get("failureMessage") or status_data.get("failure_message") or "")
            logger.error("lollipop_canary.kernel_terminal_failure status=%s failure=%s", status, failure)
            return 1
        return 0 if output_payload.get("ok") else 1
    finally:
        if not args.keep_datasets:
            cleanup_datasets(client, [payload_slug, cipher_slug, key_slug])
        else:
            logger.info(
                "lollipop_canary.datasets_kept payload=%s cipher=%s key=%s",
                payload_slug,
                cipher_slug,
                key_slug,
            )


if __name__ == "__main__":
    raise SystemExit(main())
