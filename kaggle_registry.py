from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(os.getenv("KAGGLE_JOBS_PATH", "/data/kaggle_jobs.json"))
_LOCK = asyncio.Lock()


def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return {"jobs": []}
    try:
        raw = _REGISTRY_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {"jobs": []}
    if not isinstance(data, dict):
        return {"jobs": []}
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        data["jobs"] = []
    return data


def _save_registry(data: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _REGISTRY_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(_REGISTRY_PATH)


def _job_id(job_type: str, kernel_ref: str) -> str:
    return f"{job_type}:{kernel_ref}" if kernel_ref else job_type


async def register_job(
    job_type: str,
    kernel_ref: str,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    job = {
        "id": _job_id(job_type, kernel_ref),
        "type": job_type,
        "kernel_ref": kernel_ref,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    async with _LOCK:
        data = _load_registry()
        jobs = [j for j in data.get("jobs", []) if isinstance(j, dict)]
        job_ids = {j.get("id") for j in jobs}
        if job["id"] in job_ids:
            jobs = [j for j in jobs if j.get("id") != job["id"]]
        jobs.append(job)
        data["jobs"] = jobs
        _save_registry(data)


async def remove_job(job_type: str, kernel_ref: str) -> None:
    job_id = _job_id(job_type, kernel_ref)
    async with _LOCK:
        data = _load_registry()
        jobs = [j for j in data.get("jobs", []) if isinstance(j, dict)]
        jobs = [j for j in jobs if j.get("id") != job_id]
        data["jobs"] = jobs
        _save_registry(data)


async def list_jobs(job_type: str | None = None) -> list[dict[str, Any]]:
    async with _LOCK:
        data = _load_registry()
        jobs = [j for j in data.get("jobs", []) if isinstance(j, dict)]
    if job_type:
        return [j for j in jobs if j.get("type") == job_type]
    return jobs
