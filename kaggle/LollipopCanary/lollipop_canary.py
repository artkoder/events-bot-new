#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any


INPUT_ROOT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
WORK_REPO = WORK_DIR / "repo"
STAGE_LOG_DIR = WORK_DIR / "stage_logs"
OUTPUT_JSON = WORK_DIR / "output.json"
INPUT_WAIT_SECONDS = 120
INPUT_POLL_SECONDS = 5


def _install_if_missing() -> None:
    packages = [
        ("cryptography", "cryptography>=41.0.0"),
        ("aiosqlite", "aiosqlite>=0.20.0"),
        ("sqlalchemy", "SQLAlchemy>=2.0.29"),
        ("sqlmodel", "sqlmodel>=0.0.22"),
        ("google.generativeai", "google-generativeai>=0.8.5"),
    ]
    missing: list[str] = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_install_if_missing()

from cryptography.fernet import Fernet  # noqa: E402


def _find_first(filename: str) -> Path | None:
    for path in INPUT_ROOT.rglob(filename):
        if path.is_file():
            return path
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_repo_tree() -> Path | None:
    for path in INPUT_ROOT.rglob("main.py"):
        if path.is_file() and path.parent.name == "repo_bundle":
            return path.parent
    return None


def _input_snapshot() -> list[str]:
    out: list[str] = []
    try:
        for path in sorted(INPUT_ROOT.rglob("*")):
            if path.is_file():
                out.append(str(path.relative_to(INPUT_ROOT)))
                if len(out) >= 200:
                    break
    except Exception:
        return out
    return out


def _load_inputs() -> tuple[dict[str, Any], dict[str, str], Path | None, Path | None, Path]:
    deadline = time.monotonic() + INPUT_WAIT_SECONDS
    last_snapshot: list[str] = []
    last_missing = "initial"
    while True:
        config_path = _find_first("config.json")
        enc_path = _find_first("secrets.enc")
        key_path = _find_first("fernet.key")
        repo_zip_path = _find_first("repo_bundle.zip")
        repo_tree_path = _find_repo_tree()
        last_snapshot = _input_snapshot()
        if config_path is None:
            last_missing = "config.json"
        elif enc_path is None or key_path is None:
            last_missing = "secrets.enc/fernet.key"
        elif repo_zip_path is None and repo_tree_path is None:
            last_missing = "repo bundle"
        else:
            config = _read_json(config_path)
            payload = Fernet(key_path.read_bytes().strip()).decrypt(enc_path.read_bytes())
            secrets_raw = json.loads(payload.decode("utf-8"))
            secrets = {str(k): str(v) for k, v in (secrets_raw or {}).items() if v is not None}
            db_filename = str(config.get("db_filename") or "").strip()
            if not db_filename:
                raise RuntimeError("db_filename missing in config.json")
            db_path = _find_first(db_filename)
            if db_path is not None:
                return config, secrets, repo_zip_path, repo_tree_path, db_path
            last_missing = db_filename
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"required input missing after wait: {last_missing}; visible={last_snapshot[:50]}"
            )
        time.sleep(INPUT_POLL_SECONDS)


def _prepare_repo(repo_zip_path: Path | None, repo_tree_path: Path | None, db_path: Path) -> Path:
    if WORK_REPO.exists():
        shutil.rmtree(WORK_REPO)
    WORK_REPO.mkdir(parents=True, exist_ok=True)
    if repo_tree_path is not None:
        for item in repo_tree_path.iterdir():
            target = WORK_REPO / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    elif repo_zip_path is not None:
        with zipfile.ZipFile(repo_zip_path) as zf:
            zf.extractall(WORK_REPO)
    else:
        raise RuntimeError("repo bundle not available")
    shutil.copy2(db_path, WORK_REPO / db_path.name)
    return WORK_REPO


def _stage_json_paths(repo_root: Path, run_label: str, date_tag: str) -> dict[str, Path]:
    return {
        "extract": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_facts_extract_family_{run_label}_{date_tag}.json",
        "dedup": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_facts_dedup_family_{run_label}_{date_tag}.json",
        "merge": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_facts_merge_family_{run_label}_{date_tag}.json",
        "prioritize": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_facts_prioritize_family_{run_label}_{date_tag}.json",
        "layout": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_editorial_layout_family_{run_label}_{date_tag}.json",
        "writer_pack": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_writer_pack_compose_family_{run_label}_{date_tag}.json",
        "writer_final": repo_root / "artifacts" / "codex" / f"smart_update_lollipop_writer_final_4o_family_{run_label}_{date_tag}.json",
    }


def _run_stage(
    name: str,
    script_path: Path,
    *,
    repo_root: Path,
    env: dict[str, str],
    output_json_path: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    STAGE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_sec)),
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        returncode = 124
    duration = time.monotonic() - started
    stdout_path = STAGE_LOG_DIR / f"{name}.stdout.log"
    stderr_path = STAGE_LOG_DIR / f"{name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "stage": name,
        "script": str(script_path),
        "returncode": returncode,
        "duration_sec": round(duration, 2),
        "timed_out": timed_out,
        "timeout_sec": int(timeout_sec),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "output_json_path": str(output_json_path),
    }


def _heading_count(markdown_text: str) -> int:
    return len(re.findall(r"(?m)^###\s+", markdown_text or ""))


def _summarize_final(final_payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = final_payload.get("results") or []
    canaries: list[dict[str, Any]] = []
    lengths: list[int] = []
    warnings_total = 0
    errors_total = 0
    for item in results:
        event_id = int(item["event_id"])
        result = item.get("result") or {}
        metrics = result.get("metrics") or {}
        applied = result.get("applied_output") or {}
        description_md = str(applied.get("description_md") or "")
        description_length = int(metrics.get("description_length") or len(description_md))
        lengths.append(description_length)
        audit = item.get("audit") or {}
        warnings = audit.get("warnings") or []
        errors = audit.get("errors") or []
        warnings_total += len(warnings)
        errors_total += len(errors)
        canaries.append(
            {
                "event_id": event_id,
                "title": str(applied.get("title") or item.get("title") or ""),
                "description_length": description_length,
                "heading_count": _heading_count(description_md),
                "warning_count": len(warnings),
                "error_count": len(errors),
                "has_cookie_tea_leak": bool(re.search(r"(?iu)(печенье|чай)", description_md)),
                "has_koroleva_luiza_tail": bool(
                    "Королева Луиза" in description_md
                    or re.search(r"(?m)^###\s+Организаторы\s*$", description_md)
                ),
                "description_excerpt": description_md[:1200],
            }
        )
    aggregate = {
        "event_count": len(results),
        "avg_description_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "warning_count": warnings_total,
        "error_count": errors_total,
    }
    return aggregate, canaries


def main() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "ok": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_runs": [],
    }
    try:
        config, secrets, repo_zip_path, repo_tree_path, db_path = _load_inputs()
        repo_root = _prepare_repo(repo_zip_path, repo_tree_path, db_path)
        output["run_id"] = str(config.get("run_id") or "")
        output["event_ids"] = list(config.get("event_ids") or [])
        output["date_tag"] = str(config.get("date_tag") or "")
        output["run_label"] = str(config.get("run_label") or "")
        output["repo_root"] = str(repo_root)
        output["db_path"] = str(repo_root / db_path.name)

        env = os.environ.copy()
        env.update(secrets)
        env.setdefault("GOOGLE_AI_ALLOW_RESERVE_FALLBACK", "1")
        env.setdefault("GOOGLE_AI_LOCAL_LIMITER_FALLBACK", "1")
        env.setdefault("GOOGLE_AI_LOCAL_LIMITER_ON_RESERVE_ERROR", "1")
        env.setdefault("GOOGLE_AI_INCIDENT_NOTIFICATIONS", "0")
        env["EVENT_IDS"] = ",".join(str(x) for x in (config.get("event_ids") or []))
        env["LOLLIPOP_DATE_TAG"] = str(config.get("date_tag") or "")
        env["LOLLIPOP_GEMMA_CALL_GAP_S"] = str(config.get("gemma_call_gap_s") or "6")
        env["DB_PATH"] = str(repo_root / db_path.name)
        run_label = str(config.get("run_label") or "").strip()
        env["LOLLIPOP_FACTS_EXTRACT_RUN_LABEL"] = run_label
        env["LOLLIPOP_FACTS_DEDUP_RUN_LABEL"] = run_label
        env["LOLLIPOP_FACTS_MERGE_RUN_LABEL"] = run_label
        env["LOLLIPOP_FACTS_PRIORITIZE_RUN_LABEL"] = run_label
        env["LOLLIPOP_EDITORIAL_LAYOUT_RUN_LABEL"] = run_label
        env["LOLLIPOP_WRITER_PACK_RUN_LABEL"] = run_label
        env["LOLLIPOP_WRITER_FINAL_RUN_LABEL"] = run_label

        stage_paths = _stage_json_paths(repo_root, run_label, env["LOLLIPOP_DATE_TAG"])
        stage_specs = [
            (
                "facts_extract",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_facts_extract_family_v2_16_2_2026_03_09.py",
                {},
                stage_paths["extract"],
                900,
            ),
            (
                "facts_dedup",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_facts_dedup_family_v2_16_2_iter3_2026_03_09.py",
                {"LOLLIPOP_EXTRACT_JSON_PATH": str(stage_paths["extract"])},
                stage_paths["dedup"],
                300,
            ),
            (
                "facts_merge",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_facts_merge_family_v2_16_2_iter5_2026_03_09.py",
                {"LOLLIPOP_DEDUP_JSON_PATH": str(stage_paths["dedup"])},
                stage_paths["merge"],
                300,
            ),
            (
                "facts_prioritize",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_facts_prioritize_family_v2_16_2_iter1_2026_03_10.py",
                {"LOLLIPOP_MERGE_JSON_PATH": str(stage_paths["merge"])},
                stage_paths["prioritize"],
                300,
            ),
            (
                "editorial_layout",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_editorial_layout_family_v2_16_2_iter1_2026_03_10.py",
                {"LOLLIPOP_PRIORITIZE_JSON_PATH": str(stage_paths["prioritize"])},
                stage_paths["layout"],
                600,
            ),
            (
                "writer_pack",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_writer_pack_compose_family_v2_16_2_iter1_2026_03_10.py",
                {
                    "LOLLIPOP_PRIORITIZE_JSON_PATH": str(stage_paths["prioritize"]),
                    "LOLLIPOP_LAYOUT_JSON_PATH": str(stage_paths["layout"]),
                },
                stage_paths["writer_pack"],
                300,
            ),
            (
                "writer_final",
                repo_root / "artifacts" / "codex" / "smart_update_lollipop_writer_final_4o_family_v2_16_2_iter1_2026_03_10.py",
                {"LOLLIPOP_WRITER_PACK_JSON_PATH": str(stage_paths["writer_pack"])},
                stage_paths["writer_final"],
                600,
            ),
        ]

        for stage_name, script_path, stage_env, output_json_path, timeout_sec in stage_specs:
            run_env = env.copy()
            run_env.update(stage_env)
            stage_result = _run_stage(
                stage_name,
                script_path,
                repo_root=repo_root,
                env=run_env,
                output_json_path=output_json_path,
                timeout_sec=timeout_sec,
            )
            output["stage_runs"].append(stage_result)
            if stage_result["returncode"] != 0:
                output["failed_stage"] = stage_name
                output["ok"] = False
                output["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                return 1

        final_payload = _read_json(stage_paths["writer_final"])
        aggregate, canaries = _summarize_final(final_payload)
        output["final_summary"] = aggregate
        output["canaries"] = canaries
        output["final_output_json_path"] = str(stage_paths["writer_final"])
        output["ok"] = aggregate["error_count"] == 0
    except Exception as exc:
        output["ok"] = False
        output["exception_type"] = exc.__class__.__name__
        output["exception"] = str(exc)[:1000]
        output["traceback_excerpt"] = traceback.format_exc()[:4000]
    output["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": output.get("ok"),
                "failed_stage": output.get("failed_stage"),
                "avg_description_length": (output.get("final_summary") or {}).get("avg_description_length"),
                "event_ids": output.get("event_ids"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if output.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
