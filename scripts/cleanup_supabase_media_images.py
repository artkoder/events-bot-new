from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

import httpx
from sqlmodel import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import Database
from models import Event, EventMediaAsset, EventPoster, Festival
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
    if (os.getenv("SUPABASE_DISABLED") or "").strip().lower() in {"1", "true", "yes", "on"}:
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


def _matches_prefix(path: str, prefixes: list[str]) -> bool:
    raw = (path or "").strip().lstrip("/")
    if not raw:
        return False
    for pfx in prefixes:
        p = (pfx or "").strip().strip("/")
        if not p:
            continue
        if raw == p or raw.startswith(p + "/"):
            return True
    return False


def _parse_url_path(url: str | None, *, bucket: str) -> str | None:
    parsed = parse_storage_object_url(url)
    if not parsed:
        return None
    b, p = parsed
    if b != bucket:
        return None
    return str(p)


async def _load_referenced_paths(db: Database, *, bucket: str, prefixes: list[str]) -> set[str]:
    referenced: set[str] = set()
    async with db.get_session() as session:
        poster_rows = (
            await session.execute(
                select(EventPoster.supabase_url, EventPoster.supabase_path).where(
                    (EventPoster.supabase_url.is_not(None)) | (EventPoster.supabase_path.is_not(None))
                )
            )
        ).all()
        media_rows = (
            await session.execute(
                select(EventMediaAsset.supabase_url, EventMediaAsset.supabase_path).where(
                    (EventMediaAsset.supabase_url.is_not(None)) | (EventMediaAsset.supabase_path.is_not(None))
                )
            )
        ).all()
        event_rows = (
            await session.execute(
                select(Event.photo_urls, Event.preview_3d_url).where(
                    (Event.photo_urls.is_not(None)) | (Event.preview_3d_url.is_not(None))
                )
            )
        ).all()
        festival_rows = (
            await session.execute(
                select(Festival.photo_url, Festival.photo_urls).where(
                    (Festival.photo_url.is_not(None)) | (Festival.photo_urls.is_not(None))
                )
            )
        ).all()

    for supabase_url, supabase_path in poster_rows or []:
        if supabase_path:
            path = str(supabase_path).strip().lstrip("/")
            if _matches_prefix(path, prefixes):
                referenced.add(path)
        url_path = _parse_url_path(supabase_url, bucket=bucket)
        if url_path and _matches_prefix(url_path, prefixes):
            referenced.add(url_path)

    for supabase_url, supabase_path in media_rows or []:
        if supabase_path:
            path = str(supabase_path).strip().lstrip("/")
            if _matches_prefix(path, prefixes):
                referenced.add(path)
        url_path = _parse_url_path(supabase_url, bucket=bucket)
        if url_path and _matches_prefix(url_path, prefixes):
            referenced.add(url_path)

    for photo_urls, preview_3d_url in event_rows or []:
        if isinstance(photo_urls, list):
            for u in photo_urls:
                url_path = _parse_url_path(str(u), bucket=bucket)
                if url_path and _matches_prefix(url_path, prefixes):
                    referenced.add(url_path)
        url_path = _parse_url_path(preview_3d_url, bucket=bucket)
        if url_path and _matches_prefix(url_path, prefixes):
            referenced.add(url_path)

    for photo_url, photo_urls in festival_rows or []:
        url_path = _parse_url_path(photo_url, bucket=bucket)
        if url_path and _matches_prefix(url_path, prefixes):
            referenced.add(url_path)
        if isinstance(photo_urls, list):
            for u in photo_urls:
                url_path = _parse_url_path(str(u), bucket=bucket)
                if url_path and _matches_prefix(url_path, prefixes):
                    referenced.add(url_path)

    return referenced


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete old/orphaned images from Supabase Storage (media bucket). "
            "Default mode deletes only objects under the selected prefixes that are not referenced by the DB."
        )
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
        action="append",
        default=[],
        help="Prefix inside the bucket to scan/delete (repeatable). Defaults to posters + /3di prefixes.",
    )
    parser.add_argument(
        "--mode",
        choices=["orphans", "purge"],
        default="orphans",
        help="orphans: delete only non-referenced objects; purge: delete everything under prefixes",
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

    prefixes: list[str] = [str(p).strip().strip("/") for p in (args.prefix or []) if str(p).strip().strip("/")]
    if not prefixes:
        posters_prefix = (
            os.getenv("SUPABASE_POSTERS_PREFIX")
            or os.getenv("TG_MONITORING_POSTERS_PREFIX")
            or "p"
        ).strip()
        preview3d_prefix = (os.getenv("SUPABASE_PREVIEW3D_PREFIX") or "p3d").strip()
        prefixes = [posters_prefix, preview3d_prefix]
    prefixes = [p.strip().strip("/") for p in prefixes if p.strip().strip("/")]
    prefixes = sorted(set(prefixes))
    if not prefixes:
        raise SystemExit("At least one --prefix is required (or configure SUPABASE_POSTERS_PREFIX/SUPABASE_PREVIEW3D_PREFIX)")

    db = Database(str(args.db))
    await db.init()

    referenced = await _load_referenced_paths(db, bucket=bucket, prefixes=prefixes)

    client = _get_supabase_client()
    storage = client.storage.from_(bucket)

    all_paths: set[str] = set()
    for pfx in prefixes:
        all_paths.update(_list_object_paths(storage, prefix=pfx))
    all_sorted = sorted(all_paths)

    if args.mode == "purge":
        candidates = list(all_sorted)
    else:
        candidates = [p for p in all_sorted if p not in referenced]

    if int(args.limit or 0) > 0:
        candidates = candidates[: int(args.limit)]

    print(f"bucket={bucket} prefixes={','.join(prefixes)} mode={args.mode}")
    print(f"referenced={len(referenced)} total_in_bucket={len(all_sorted)} to_delete={len(candidates)}")
    if candidates:
        sample = candidates[:10]
        print(f"sample_delete={len(sample)} first={sample[0]}")

    if not args.apply:
        print("dry_run=1 (use --apply to delete)")
        return 0

    removed = 0
    for chunk in _chunks(candidates, 1000):
        if not chunk:
            continue
        storage.remove(chunk)
        removed += len(chunk)

    print(f"removed={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
