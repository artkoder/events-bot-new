from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, Mapping
from urllib.parse import urlparse

import asyncio

_MB = 1024 * 1024

# In-process best-effort cache to avoid repeated full bucket listings.
# Keyed by bucket name.
_BUCKET_USAGE_CACHE: dict[str, tuple[float, int]] = {}


def resolve_bucket_env(*, primary: str, fallback: str, default: str) -> str:
    """Resolve a bucket name from environment variables.

    Backward-compat rule:
    - prefer primary env var if set
    - else use fallback env var if set
    - else use default
    """

    import os

    v = (os.getenv(primary) or "").strip()
    if v:
        return v
    v = (os.getenv(fallback) or "").strip()
    if v:
        return v
    return (default or "").strip() or default


def parse_storage_object_url(url: str | None) -> tuple[str, str] | None:
    """Parse a Supabase Storage object URL into (bucket, object_path).

    Supports common Supabase Storage URL formats:
    - /storage/v1/object/public/<bucket>/<path...>
    - /storage/v1/object/sign/<bucket>/<path...> (signed URLs)
    - /storage/v1/object/<bucket>/<path...> (rare; no public/sign segment)
    """

    if not url:
        return None
    raw = str(url).strip()
    if not raw:
        return None

    try:
        u = urlparse(raw)
    except Exception:
        return None

    parts = [p for p in (u.path or "").split("/") if p]
    if not parts:
        return None

    for i, part in enumerate(parts):
        if part != "object":
            continue

        # ".../object/<mode>/<bucket>/<path...>".
        if i + 2 < len(parts) and parts[i + 1] in ("public", "sign"):
            if i + 3 >= len(parts):
                return None
            bucket = parts[i + 2]
            obj = "/".join(parts[i + 3 :])
            if bucket and obj:
                return bucket, obj
            return None

        # Fallback: ".../object/<bucket>/<path...>" (no explicit mode).
        if i + 2 < len(parts):
            bucket = parts[i + 1]
            if bucket in ("public", "sign"):
                # Avoid mis-parsing ".../object/public/<bucket>" when <path...> is missing.
                return None
            obj = "/".join(parts[i + 2 :])
            if bucket and obj:
                return bucket, obj
            return None

    return None


def storage_object_exists_http(
    *,
    supabase_url: str | None,
    supabase_key: str | None,
    bucket: str,
    object_path: str,
    timeout_sec: float = 12.0,
) -> bool | None:
    """Return whether a Storage object exists via the REST API (best-effort).

    Uses a service-role key (or other key with Storage access) to probe:
      HEAD /storage/v1/object/<bucket>/<path>

    Returns:
    - True/False when a definitive answer is available
    - None when the check failed (network error, unexpected status, etc.)
    """

    base = (supabase_url or "").strip().rstrip("/")
    key = (supabase_key or "").strip()
    b = (bucket or "").strip()
    p = (object_path or "").strip().lstrip("/")
    if not (base and key and b and p):
        return None

    url = f"{base}/storage/v1/object/{b}/{p}"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    timeout = max(1.0, float(timeout_sec or 0.0))

    try:
        import requests

        resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
    except Exception:
        return None

    if resp.status_code in (200, 206):
        return True
    if resp.status_code == 404:
        return False

    # Some environments may not allow HEAD; try a tiny ranged GET.
    if resp.status_code in (400, 405):
        try:
            import requests

            headers2 = dict(headers)
            headers2["Range"] = "bytes=0-0"
            resp2 = requests.get(url, headers=headers2, timeout=timeout, allow_redirects=True)
            if resp2.status_code in (200, 206):
                return True
            if resp2.status_code == 404:
                return False
        except Exception:
            return None

    return None


def _parse_int(value: Any, fallback: int = 0) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed


def _storage_list(storage: Any, path: str, *, limit: int, offset: int) -> list[dict[str, Any]]:
    """List storage objects for a path (compatible across supabase-py versions)."""

    options = {"limit": limit, "offset": offset, "sortBy": {"column": "name", "order": "asc"}}
    try:
        rows = storage.list(path=path, options=options)
    except TypeError:
        try:
            rows = storage.list(path, options)
        except TypeError:
            rows = storage.list(path)
    return list(rows or [])


def compute_bucket_size_bytes(supabase_client: Any, bucket: str) -> int:
    """Compute total size (bytes) in a Supabase Storage bucket via recursive listing.

    Notes:
    - This is O(N) in the number of objects and can be slow for large buckets.
    - Some list APIs return folders without `metadata.size`; we traverse into them.
    """

    b = (bucket or "").strip()
    if not b:
        raise ValueError("bucket is required")

    storage = supabase_client.storage.from_(b)
    total = 0
    seen_paths: set[str] = set()
    stack = [""]
    while stack:
        path = stack.pop()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        offset = 0
        limit = 1000
        while True:
            rows = _storage_list(storage, path, limit=limit, offset=offset)
            if not rows:
                break
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("name") or "").strip().strip("/")
                if not name:
                    continue
                metadata = row.get("metadata")
                size_val: int | None = None
                if isinstance(metadata, Mapping):
                    size_val = _parse_int(
                        metadata.get("size")
                        or metadata.get("contentLength")
                        or metadata.get("content_length")
                    )
                if not size_val:
                    size_val = _parse_int(row.get("size") or row.get("bytes"))
                if size_val and size_val > 0:
                    total += size_val
                    continue
                is_dir = metadata is None and row.get("id") is None
                if is_dir:
                    child = f"{path.rstrip('/')}/{name}".strip("/")
                    if child and child not in seen_paths:
                        stack.append(child)
            if len(rows) < limit:
                break
            offset += limit
    return int(total)


def get_bucket_usage_bytes(
    supabase_client: Any,
    bucket: str,
    *,
    cache_sec: int = 600,
    now_ts: float | None = None,
) -> tuple[int, bool]:
    """Return (used_bytes, is_cached) for a bucket (best-effort cache)."""

    b = (bucket or "").strip()
    if not b:
        raise ValueError("bucket is required")
    cache_sec = max(0, int(cache_sec or 0))
    now = float(time.time() if now_ts is None else now_ts)

    if cache_sec > 0:
        cached = _BUCKET_USAGE_CACHE.get(b)
        if cached:
            cached_at, cached_bytes = cached
            if cached_at and (now - cached_at) <= cache_sec and cached_bytes >= 0:
                return int(cached_bytes), True

    used = compute_bucket_size_bytes(supabase_client, b)
    if cache_sec > 0:
        _BUCKET_USAGE_CACHE[b] = (now, int(used))
    return int(used), False


@dataclass(frozen=True)
class BucketUsageCheck:
    ok: bool
    used_mb: float | None
    max_used_mb: float
    additional_mb: float
    cached: bool
    reason: str


def check_bucket_usage_limit(
    supabase_client: Any | None,
    bucket: str,
    *,
    additional_bytes: int = 0,
    max_used_mb: float = 490.0,
    cache_sec: int = 600,
    on_error: str = "deny",
) -> BucketUsageCheck:
    """Check whether uploading `additional_bytes` keeps bucket usage below `max_used_mb`.

    `on_error`:
    - "deny": return ok=False when usage cannot be computed (safer).
    - "allow": return ok=True when usage cannot be computed (fail-open).
    """

    b = (bucket or "").strip()
    if not b:
        raise ValueError("bucket is required")

    additional = max(0, int(additional_bytes or 0))
    max_used = float(max_used_mb or 0.0)
    if max_used <= 0:
        raise ValueError("max_used_mb must be > 0")

    on_err = (on_error or "deny").strip().lower()
    if on_err not in {"deny", "allow"}:
        raise ValueError("on_error must be 'deny' or 'allow'")

    if supabase_client is None:
        return BucketUsageCheck(
            ok=(on_err == "allow"),
            used_mb=None,
            max_used_mb=max_used,
            additional_mb=round(float(additional) / _MB, 3),
            cached=False,
            reason="supabase_unavailable",
        )

    try:
        used_bytes, cached = get_bucket_usage_bytes(
            supabase_client, b, cache_sec=cache_sec
        )
        used_mb = round(float(used_bytes) / _MB, 3)
        ok = (used_bytes + additional) <= int(max_used * _MB)
        return BucketUsageCheck(
            ok=bool(ok),
            used_mb=used_mb,
            max_used_mb=max_used,
            additional_mb=round(float(additional) / _MB, 3),
            cached=bool(cached),
            reason="ok" if ok else "limit_exceeded",
        )
    except Exception:
        return BucketUsageCheck(
            ok=(on_err == "allow"),
            used_mb=None,
            max_used_mb=max_used,
            additional_mb=round(float(additional) / _MB, 3),
            cached=False,
            reason="usage_unavailable",
        )


def require_bucket_usage_limit(
    supabase_client: Any | None,
    bucket: str,
    *,
    additional_bytes: int = 0,
    max_used_mb: float = 490.0,
    cache_sec: int = 600,
    on_error: str = "deny",
) -> BucketUsageCheck:
    """Same as `check_bucket_usage_limit`, but raises when not allowed."""

    res = check_bucket_usage_limit(
        supabase_client,
        bucket,
        additional_bytes=additional_bytes,
        max_used_mb=max_used_mb,
        cache_sec=cache_sec,
        on_error=on_error,
    )
    if not res.ok:
        raise RuntimeError(
            f"Supabase bucket usage guard blocked upload: bucket={bucket} "
            f"used_mb={res.used_mb} additional_mb={res.additional_mb} "
            f"max_used_mb={res.max_used_mb} reason={res.reason}"
        )
    return res


def check_bucket_usage_limit_from_env(
    supabase_client: Any | None,
    bucket: str,
    *,
    additional_bytes: int = 0,
) -> BucketUsageCheck:
    """Env-driven wrapper for `check_bucket_usage_limit`.

    ENV:
    - SUPABASE_BUCKET_USAGE_GUARD_MAX_USED_MB (default: 490)
    - SUPABASE_BUCKET_USAGE_GUARD_CACHE_SEC (default: 600)
    - SUPABASE_BUCKET_USAGE_GUARD_ON_ERROR ("deny"|"allow", default: "deny")
    """

    import os

    def _read_float(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    def _read_int(name: str, default: int) -> int:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    max_used_mb = _read_float("SUPABASE_BUCKET_USAGE_GUARD_MAX_USED_MB", 490.0)
    cache_sec = _read_int("SUPABASE_BUCKET_USAGE_GUARD_CACHE_SEC", 600)
    on_error = (os.getenv("SUPABASE_BUCKET_USAGE_GUARD_ON_ERROR") or "deny").strip().lower()
    if on_error not in {"deny", "allow"}:
        on_error = "deny"
    return check_bucket_usage_limit(
        supabase_client,
        bucket,
        additional_bytes=additional_bytes,
        max_used_mb=max_used_mb,
        cache_sec=cache_sec,
        on_error=on_error,
    )


async def flush_supabase_delete_queue(
    db: Any,
    *,
    supabase_client: Any | None,
    limit: int = 2000,
    chunk_size: int = 1000,
    path_filter: Callable[[str], bool] | None = None,
) -> int:
    """Try to delete queued storage objects from Supabase (best-effort).

    The queue is stored in SQLite table `supabase_delete_queue` and is intended to make
    cleanup durable: if Supabase is temporarily unavailable, we keep delete targets and
    retry on the next scheduler run.
    """

    if supabase_client is None:
        return 0
    limit = max(1, int(limit or 0))
    chunk_size = max(1, min(1000, int(chunk_size or 0)))

    rows: list[tuple[Any, Any, Any]] = []
    scan_page = max(200, min(5000, int(limit or 0)))
    offset = 0
    async with db.raw_conn() as conn:
        while len(rows) < limit:
            cur = await conn.execute(
                "SELECT id, bucket, path FROM supabase_delete_queue "
                "ORDER BY id ASC LIMIT ? OFFSET ?",
                (scan_page, offset),
            )
            batch = await cur.fetchall()
            if not batch:
                break
            offset += len(batch)
            if path_filter is None:
                rows.extend(batch)
                if len(rows) >= limit:
                    rows = rows[:limit]
                    break
                continue
            for row in batch:
                if len(rows) >= limit:
                    break
                try:
                    path_raw = str(row[2] or "")
                except Exception:
                    continue
                try:
                    if not path_filter(path_raw):
                        continue
                except Exception:
                    continue
                rows.append(row)

    if not rows:
        return 0

    # Group by bucket for fewer client calls.
    by_bucket: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        try:
            qid, bucket, path = int(row[0]), str(row[1]), str(row[2])
        except Exception:
            continue
        bucket = (bucket or "").strip()
        path = (path or "").strip().lstrip("/")
        if not bucket or not path:
            continue
        by_bucket.setdefault(bucket, []).append((qid, path))

    removed_ids: list[int] = []
    failed_ids: list[int] = []

    for bucket, items in by_bucket.items():
        paths = [p for _, p in items]
        ids = [i for i, _ in items]
        for start in range(0, len(paths), chunk_size):
            chunk_paths = paths[start : start + chunk_size]
            chunk_ids = ids[start : start + chunk_size]
            try:
                await asyncio.to_thread(
                    supabase_client.storage.from_(bucket).remove,
                    chunk_paths,
                )
                removed_ids.extend(chunk_ids)
            except Exception as exc:
                failed_ids.extend(chunk_ids)
                # Don't keep hammering Supabase; leave the rest for the next run.
                remaining = ids[start + chunk_size :]
                failed_ids.extend(remaining)
                # Also mark other buckets we haven't attempted yet.
                for b2, items2 in by_bucket.items():
                    if b2 == bucket:
                        continue
                    failed_ids.extend([i for i, _ in items2])

                err = str(exc)
                if len(err) > 800:
                    err = err[:800]
                now = datetime.now(timezone.utc).isoformat()
                async with db.raw_conn() as conn:
                    await conn.executemany(
                        "UPDATE supabase_delete_queue "
                        "SET attempts = attempts + 1, last_attempt_at = ?, last_error = ? "
                        "WHERE id = ?",
                        [(now, err, int(i)) for i in set(failed_ids)],
                    )
                    await conn.commit()
                # Stop after the first failure.
                break
        else:
            continue
        break

    if removed_ids:
        async with db.raw_conn() as conn:
            await conn.executemany(
                "DELETE FROM supabase_delete_queue WHERE id = ?",
                [(int(i),) for i in set(removed_ids)],
            )
            await conn.commit()

    return len(set(removed_ids))
