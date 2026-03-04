from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Mapping

import httpx
from sqlmodel import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import Database
from models import Event
from supabase_storage import parse_storage_object_url, resolve_bucket_env


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _storage_list(storage: Any, path: str, *, limit: int, offset: int) -> list[dict[str, Any]]:
    options = {"limit": int(limit), "offset": int(offset), "sortBy": {"column": "name", "order": "asc"}}
    try:
        rows = storage.list(path=path, options=options)
    except TypeError:
        try:
            rows = storage.list(path, options)
        except TypeError:
            rows = storage.list(path)
    return list(rows or [])


def _parse_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bucket_item_size_bytes(item: Mapping[str, Any]) -> int:
    meta = item.get("metadata")
    if isinstance(meta, Mapping):
        size = meta.get("size") or meta.get("contentLength") or meta.get("content_length")
        parsed = _parse_int(size)
        if parsed > 0:
            return parsed
    parsed = _parse_int(item.get("size") or item.get("bytes"))
    return parsed if parsed > 0 else 0


def _list_object_paths(storage: Any, *, prefix: str) -> list[str]:
    root = (prefix or "").strip().strip("/")
    if not root:
        raise ValueError("prefix is required")

    out: list[str] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
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
                child_path = f"{path}/{name}" if path else name
                if _bucket_item_size_bytes(row) > 0:
                    out.append(child_path)
                    continue
                # Supabase list returns folders without metadata; recurse into them.
                is_dir = row.get("id") is None and row.get("updated_at") is None
                if is_dir and child_path not in seen:
                    stack.append(child_path)
            if len(rows) < limit:
                break
            offset += limit
    return sorted(set(out))


def _get_supabase_client() -> Any:
    if (os.getenv("SUPABASE_DISABLED") or "").strip() in {"1", "true", "yes"}:
        raise RuntimeError("SUPABASE_DISABLED=1")
    supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
    supabase_key = ((os.getenv("SUPABASE_SERVICE_KEY") or "").strip() or (os.getenv("SUPABASE_KEY") or "").strip())
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY are not set")
    supabase_schema = (os.getenv("SUPABASE_SCHEMA") or "public").strip() or "public"

    from supabase import create_client  # local import
    from supabase.client import ClientOptions

    options = ClientOptions()
    options.schema = supabase_schema
    options.httpx_client = httpx.Client(timeout=30)
    return create_client(supabase_url.rstrip("/"), supabase_key, options=options)


async def _load_referenced_preview_paths(db: Database, *, bucket: str, prefix: str) -> set[str]:
    referenced: set[str] = set()
    pref = (prefix or "").strip().strip("/") + "/"
    async with db.get_session() as session:
        urls = (
            await session.execute(
                select(Event.preview_3d_url).where(
                    (Event.preview_3d_url.is_not(None)) & (Event.preview_3d_url != "")
                )
            )
        ).scalars().all()
    for url in urls or []:
        parsed = parse_storage_object_url(str(url))
        if not parsed:
            continue
        b, p = parsed
        if b != bucket:
            continue
        if not str(p).startswith(pref):
            continue
        referenced.add(str(p))
    return referenced


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete orphaned /3di (preview3d) images from Supabase Storage (best-effort)."
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DB_PATH") or ("/tmp/db.sqlite" if os.path.exists("/tmp/db.sqlite") else "db.sqlite"),
        help="Path to SQLite DB (default: DB_PATH or /tmp/db.sqlite or db.sqlite)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Supabase Storage bucket (default: SUPABASE_MEDIA_BUCKET/SUPABASE_BUCKET resolution)",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("SUPABASE_PREVIEW3D_PREFIX") or "p3d",
        help="Preview3D prefix inside the bucket (default: SUPABASE_PREVIEW3D_PREFIX or p3d)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete objects (default: dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max objects to delete (0 = unlimited)",
    )
    args = parser.parse_args()

    bucket = (
        (args.bucket or "").strip()
        or resolve_bucket_env(primary="SUPABASE_MEDIA_BUCKET", fallback="SUPABASE_BUCKET", default="events-ics")
    )
    prefix = (args.prefix or "").strip().strip("/")
    if not prefix:
        raise SystemExit("--prefix is required")

    db = Database(str(args.db))
    await db.init()

    referenced = await _load_referenced_preview_paths(db, bucket=bucket, prefix=prefix)

    client = _get_supabase_client()
    storage = client.storage.from_(bucket)

    all_paths = _list_object_paths(storage, prefix=prefix)
    orphaned = [p for p in all_paths if p not in referenced]
    if int(args.limit or 0) > 0:
        orphaned = orphaned[: int(args.limit)]

    print(f"bucket={bucket} prefix={prefix}")
    print(f"referenced={len(referenced)} total_in_bucket={len(all_paths)} orphaned={len(orphaned)}")
    if orphaned:
        sample = orphaned[:10]
        print(f"sample_orphans={len(sample)} first={sample[0]}")

    if not args.apply:
        print("dry_run=1 (use --apply to delete)")
        return 0

    removed = 0
    for chunk in _chunks(orphaned, 100):
        if not chunk:
            continue
        # supabase-py returns a response-like object; any exception means failure.
        storage.remove(chunk)
        removed += len(chunk)

    print(f"removed={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
