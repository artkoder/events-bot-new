#!/usr/bin/env python3
"""Audit Supabase media deduplication (posters + videos) for the last N hours.

This script inspects the local SQLite DB to verify:
  - poster paths are canonical (WebP-only, dHash16-based, stable across PROD/TEST)
  - video assets are content-addressed when possible (sha256-based), and no obvious duplicates appear

Optionally it can HEAD-check referenced objects in Supabase Storage to ensure they exist and have
expected content types (image/webp for posters, video/* for videos).

Usage (typical):
  python scripts/inspect/audit_media_dedup.py --db /tmp/db.sqlite --hours 24
  python scripts/inspect/audit_media_dedup.py --db /tmp/db.sqlite --hours 24 --check-storage
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_POSTER_PATH_RE = re.compile(
    r"^(?P<prefix>[^/]+)/dh16/(?P<first2>[0-9a-f]{2})/(?P<hash>[0-9a-f]{64})\.webp$",
    re.IGNORECASE,
)

_VIDEO_SHA256_PATH_RE = re.compile(
    r"^(?P<prefix>[^/]+)/sha256/(?P<first2>[0-9a-f]{2})/(?P<sha>[0-9a-f]{64})\.(?P<ext>[a-z0-9]{1,8})$",
    re.IGNORECASE,
)
_VIDEO_TG_PATH_RE = re.compile(
    r"^(?P<prefix>[^/]+)/tg/(?P<docid>[0-9]{6,32})\.(?P<ext>[a-z0-9]{1,8})$",
    re.IGNORECASE,
)
_VIDEO_LEGACY_SHA_PATH_RE = re.compile(
    r"^(?P<prefix>[^/]+)/(?P<sha>[0-9a-f]{64})\.(?P<ext>[a-z0-9]{1,8})$",
    re.IGNORECASE,
)


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _get_env(key: str, env_file: dict[str, str]) -> str:
    return (os.getenv(key) or env_file.get(key) or "").strip()


def _parse_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    # Normalize common ISO variants.
    raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_dt(dt: datetime | None) -> str:
    if not dt:
        return "—"
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


@dataclass(frozen=True)
class PosterRow:
    id: int
    event_id: int | None
    poster_hash: str | None
    phash: str | None
    supabase_path: str | None
    supabase_url: str | None
    updated_at: datetime | None


@dataclass(frozen=True)
class VideoRow:
    id: int
    event_id: int | None
    kind: str | None
    supabase_path: str | None
    supabase_url: str | None
    sha256: str | None
    size_bytes: int | None
    mime_type: str | None
    created_at: datetime | None


def _open_db_ro(path: str) -> sqlite3.Connection:
    # Read-only to reduce the chance of interfering with long-running bots.
    # Note: not all sqlite builds support uri mode, so fall back to normal open.
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        return sqlite3.connect(path)


def _fetch_posters(con: sqlite3.Connection) -> list[PosterRow]:
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA table_info('eventposter')")
    if not cur.fetchall():
        return []
    cur.execute(
        """
        select
          id,
          event_id,
          poster_hash,
          phash,
          supabase_path,
          supabase_url,
          updated_at
        from eventposter
        where supabase_path is not null or supabase_url is not null or phash is not null
        """
    )
    out: list[PosterRow] = []
    for r in cur.fetchall():
        out.append(
            PosterRow(
                id=int(r["id"]),
                event_id=int(r["event_id"]) if r["event_id"] is not None else None,
                poster_hash=str(r["poster_hash"] or "").strip() or None,
                phash=str(r["phash"] or "").strip() or None,
                supabase_path=str(r["supabase_path"] or "").strip() or None,
                supabase_url=str(r["supabase_url"] or "").strip() or None,
                updated_at=_parse_dt(r["updated_at"]),
            )
        )
    return out


def _fetch_videos(con: sqlite3.Connection) -> list[VideoRow]:
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA table_info('event_media_asset')")
    if not cur.fetchall():
        return []
    cur.execute(
        """
        select
          id,
          event_id,
          kind,
          supabase_url,
          supabase_path,
          sha256,
          size_bytes,
          mime_type,
          created_at
        from event_media_asset
        where kind = 'video' and (supabase_path is not null or supabase_url is not null)
        """
    )
    out: list[VideoRow] = []
    for r in cur.fetchall():
        out.append(
            VideoRow(
                id=int(r["id"]),
                event_id=int(r["event_id"]) if r["event_id"] is not None else None,
                kind=str(r["kind"] or "").strip() or None,
                supabase_path=str(r["supabase_path"] or "").strip() or None,
                supabase_url=str(r["supabase_url"] or "").strip() or None,
                sha256=str(r["sha256"] or "").strip() or None,
                size_bytes=int(r["size_bytes"]) if r["size_bytes"] is not None else None,
                mime_type=str(r["mime_type"] or "").strip() or None,
                created_at=_parse_dt(r["created_at"]),
            )
        )
    return out


def _scan_poster_rows(rows: list[PosterRow], *, near_threshold: int, near_max_pairs: int) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    stats["rows"] = len(rows)

    paths = [r.supabase_path for r in rows if r.supabase_path]
    phashes = [str(r.phash or "").lower() for r in rows if r.phash and str(r.phash).strip()]
    stats["distinct_supabase_paths"] = len(set(paths))
    stats["distinct_phash"] = len(set(phashes))

    non_webp = sorted({p for p in paths if not p.lower().endswith(".webp")})
    stats["non_webp_paths"] = non_webp

    canonical: list[tuple[PosterRow, re.Match[str]]] = []
    noncanonical: list[PosterRow] = []
    for r in rows:
        p = (r.supabase_path or "").strip().lstrip("/")
        if not p:
            continue
        m = _POSTER_PATH_RE.match(p)
        if m:
            canonical.append((r, m))
        else:
            noncanonical.append(r)
    stats["canonical_paths"] = len(canonical)
    stats["noncanonical_paths"] = len(noncanonical)
    stats["noncanonical_path_samples"] = [
        (r.supabase_path or "")[:200] for r in noncanonical[:20]
    ]

    prefix_counts = Counter(m.group("prefix").strip().lower() for _r, m in canonical)
    stats["prefix_counts"] = dict(prefix_counts)

    first2_mismatch: list[str] = []
    phash_mismatch: list[str] = []
    phash_missing: list[str] = []
    by_phash_paths: dict[str, set[str]] = defaultdict(set)
    by_poster_hash_paths: dict[str, set[str]] = defaultdict(set)
    by_poster_hash_phash: dict[str, set[str]] = defaultdict(set)
    for r, m in canonical:
        first2 = m.group("first2").lower()
        h = m.group("hash").lower()
        p = (r.supabase_path or "").strip().lstrip("/")
        if h[:2] != first2:
            first2_mismatch.append(p)
        if r.phash:
            if str(r.phash).strip().lower() != h:
                phash_mismatch.append(f"id={r.id} phash={r.phash} path_hash={h} path={p}")
        else:
            phash_missing.append(f"id={r.id} path={p}")
        by_phash_paths[h].add(p)
        if r.poster_hash:
            by_poster_hash_paths[r.poster_hash].add(p)
            by_poster_hash_phash[r.poster_hash].add(h)

    stats["first2_mismatch"] = first2_mismatch[:20]
    stats["phash_mismatch"] = phash_mismatch[:20]
    stats["phash_missing"] = phash_missing[:20]

    multi_path_same_phash = {
        ph: sorted(list(paths))
        for ph, paths in by_phash_paths.items()
        if len(paths) > 1
    }
    stats["multi_path_same_phash"] = {
        k: v[:10] for k, v in sorted(multi_path_same_phash.items())[:20]
    }

    poster_hash_conflicts: list[str] = []
    for digest, pset in by_poster_hash_paths.items():
        if len(pset) > 1:
            poster_hash_conflicts.append(f"poster_hash={digest} paths={sorted(pset)[:4]}")
    stats["poster_hash_multi_paths"] = poster_hash_conflicts[:20]

    poster_hash_phash_conflicts: list[str] = []
    for digest, hset in by_poster_hash_phash.items():
        if len(hset) > 1:
            poster_hash_phash_conflicts.append(f"poster_hash={digest} phash={sorted(hset)[:4]}")
    stats["poster_hash_multi_phash"] = poster_hash_phash_conflicts[:20]

    usage = Counter(phashes)
    stats["top_phash_usage"] = usage.most_common(12)

    # Near-duplicate scan (extremely low false-positive rate at 256 bits).
    unique = sorted({h for h in phashes if re.fullmatch(r"[0-9a-f]{64}", h)})
    near_pairs: list[tuple[int, str, str]] = []
    if len(unique) <= 1800 and near_threshold >= 0 and near_max_pairs > 0:
        vals = [int(h, 16) for h in unique]
        for i in range(len(unique)):
            a = vals[i]
            for j in range(i + 1, len(unique)):
                d = _hamming(a, vals[j])
                if d <= near_threshold:
                    near_pairs.append((d, unique[i], unique[j]))
        near_pairs.sort(key=lambda t: t[0])
    stats["near_pairs_threshold"] = near_threshold
    stats["near_pairs"] = [
        {
            "dist": d,
            "a": a,
            "b": b,
            "a_count": int(usage.get(a, 0)),
            "b_count": int(usage.get(b, 0)),
        }
        for d, a, b in near_pairs[:near_max_pairs]
    ]

    return stats


def _scan_video_rows(rows: list[VideoRow]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    stats["rows"] = len(rows)

    paths = [r.supabase_path for r in rows if r.supabase_path]
    stats["distinct_supabase_paths"] = len(set(paths))
    sha_values = [str(r.sha256 or "").lower() for r in rows if r.sha256 and str(r.sha256).strip()]
    stats["distinct_sha256"] = len(set(sha_values))
    stats["missing_sha256_rows"] = sum(1 for r in rows if not (r.sha256 and str(r.sha256).strip()))

    kind_counts = Counter((r.kind or "").strip().lower() or "—" for r in rows)
    stats["kind_counts"] = dict(kind_counts)

    path_kinds: Counter[str] = Counter()
    sha_mismatches: list[str] = []
    noncanonical: list[str] = []
    sha_to_paths: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        p = (r.supabase_path or "").strip().lstrip("/")
        sha = (r.sha256 or "").strip().lower() or None
        if sha:
            sha_to_paths[sha].add(p or (r.supabase_url or "").strip())

        if not p:
            continue
        m_sha = _VIDEO_SHA256_PATH_RE.match(p)
        if m_sha:
            path_kinds["sha256"] += 1
            extracted = m_sha.group("sha").lower()
            if sha and sha != extracted:
                sha_mismatches.append(
                    f"id={r.id} sha256_col={sha} sha256_path={extracted} path={p}"
                )
            continue
        m_tg = _VIDEO_TG_PATH_RE.match(p)
        if m_tg:
            path_kinds["tg_fastpath"] += 1
            continue
        m_legacy = _VIDEO_LEGACY_SHA_PATH_RE.match(p)
        if m_legacy:
            path_kinds["legacy_sha"] += 1
            extracted = m_legacy.group("sha").lower()
            if sha and sha != extracted:
                sha_mismatches.append(
                    f"id={r.id} sha256_col={sha} sha256_path={extracted} path={p}"
                )
            continue
        path_kinds["other"] += 1
        noncanonical.append(p)

    stats["path_kind_counts"] = dict(path_kinds)
    stats["sha_mismatches"] = sha_mismatches[:20]
    stats["noncanonical_paths"] = noncanonical[:20]

    sha_multi_paths: list[str] = []
    for sha, pset in sha_to_paths.items():
        if len(pset) > 1:
            sha_multi_paths.append(f"sha256={sha} paths={sorted(pset)[:4]}")
    stats["sha256_multi_paths"] = sha_multi_paths[:30]

    # Flag tg_fastpath rows missing sha256: they are not content-addressed and can cause duplicates.
    tg_missing_sha: list[str] = []
    for r in rows:
        p = (r.supabase_path or "").strip().lstrip("/")
        if not p:
            continue
        if _VIDEO_TG_PATH_RE.match(p) and not (r.sha256 and str(r.sha256).strip()):
            tg_missing_sha.append(f"id={r.id} event_id={r.event_id} path={p} url={(r.supabase_url or '')[:120]}")
    stats["tg_paths_missing_sha256"] = tg_missing_sha[:30]

    return stats


def _head_storage_object(
    *,
    supabase_url: str,
    supabase_key: str,
    bucket: str,
    path: str,
    timeout_sec: float = 15.0,
) -> tuple[int | None, str | None, int | None, datetime | None]:
    base = (supabase_url or "").strip().rstrip("/")
    key = (supabase_key or "").strip()
    b = (bucket or "").strip()
    p = (path or "").strip().lstrip("/")
    if not (base and key and b and p):
        return None, None, None, None

    import requests

    url = f"{base}/storage/v1/object/{b}/{p}"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    timeout = max(1.0, float(timeout_sec or 0.0))

    def _try_head() -> requests.Response:
        return requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)

    def _try_range_get() -> requests.Response:
        headers2 = dict(headers)
        headers2["Range"] = "bytes=0-0"
        return requests.get(url, headers=headers2, timeout=timeout, allow_redirects=True)

    try:
        resp = _try_head()
    except Exception:
        return None, None, None, None

    if resp.status_code in (400, 405):
        try:
            resp = _try_range_get()
        except Exception:
            return None, None, None, None

    ct = resp.headers.get("Content-Type")
    clen = resp.headers.get("Content-Length")
    size = None
    if clen:
        try:
            size = int(clen)
        except Exception:
            size = None
    last_mod = resp.headers.get("Last-Modified") or ""
    last_dt = None
    if last_mod.strip():
        try:
            from email.utils import parsedate_to_datetime

            last_dt = parsedate_to_datetime(last_mod.strip())
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            last_dt = last_dt.astimezone(timezone.utc)
        except Exception:
            last_dt = None
    return int(resp.status_code), (ct or None), size, last_dt


def _storage_check(
    *,
    posters: list[PosterRow],
    videos: list[VideoRow],
    max_items: int,
    cutoff: datetime | None,
) -> dict[str, Any]:
    env_file = _load_dotenv(Path(".env"))
    supabase_url = _get_env("SUPABASE_URL", env_file)
    if not supabase_url:
        return {"enabled": False, "reason": "SUPABASE_URL missing"}

    key = _get_env("SUPABASE_SERVICE_KEY", env_file) or _get_env("SUPABASE_KEY", env_file)
    if not key:
        return {"enabled": False, "reason": "SUPABASE_SERVICE_KEY/SUPABASE_KEY missing"}

    # Prefer explicit media bucket, fall back to legacy bucket.
    bucket_default = _get_env("SUPABASE_MEDIA_BUCKET", env_file) or _get_env("SUPABASE_BUCKET", env_file) or "events-ics"
    bucket_default = bucket_default.strip() or "events-ics"

    # Build a de-duplicated set of objects to check.
    items: dict[tuple[str, str], str] = {}  # (bucket, path) -> expected_kind

    try:
        from supabase_storage import parse_storage_object_url
    except Exception:
        parse_storage_object_url = None  # type: ignore

    def _add(url: str | None, path: str | None, expected: str) -> None:
        u = (url or "").strip()
        p = (path or "").strip().lstrip("/")
        bucket = bucket_default
        obj = p
        if u and parse_storage_object_url is not None:
            parsed = parse_storage_object_url(u)
            if parsed:
                bucket, obj = parsed
        if not obj:
            return
        items[(bucket, obj)] = expected

    for r in posters:
        _add(r.supabase_url, r.supabase_path, "poster")
    for r in videos:
        _add(r.supabase_url, r.supabase_path, "video")

    targets = list(items.items())[: max(0, int(max_items or 0))] if max_items else list(items.items())
    checked = 0
    missing: list[str] = []
    wrong_type: list[str] = []
    recent_legacy_videos: list[str] = []
    errors: int = 0

    for (bucket, path), expected in targets:
        checked += 1
        status, ct, _size, last_dt = _head_storage_object(
            supabase_url=supabase_url,
            supabase_key=key,
            bucket=bucket,
            path=path,
        )
        if status is None:
            errors += 1
            continue
        if status == 404:
            missing.append(f"{bucket}/{path}")
            continue
        if status >= 400:
            errors += 1
            continue
        ct_norm = (ct or "").split(";", 1)[0].strip().lower()
        if expected == "poster":
            if ct_norm and ct_norm != "image/webp":
                wrong_type.append(f"{bucket}/{path} content_type={ct_norm}")
        elif expected == "video":
            if ct_norm and not ct_norm.startswith("video/"):
                wrong_type.append(f"{bucket}/{path} content_type={ct_norm}")
            if cutoff is not None and _VIDEO_TG_PATH_RE.match(path) and last_dt and last_dt >= cutoff:
                recent_legacy_videos.append(
                    f"{bucket}/{path} last_modified={last_dt.isoformat(timespec='seconds')}"
                )

    return {
        "enabled": True,
        "supabase_url_set": bool(supabase_url),
        "bucket_default": bucket_default,
        "unique_targets": len(items),
        "checked": checked,
        "head_errors": errors,
        "missing": missing[:40],
        "wrong_content_type": wrong_type[:40],
        "recent_legacy_video_tg": recent_legacy_videos[:40],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="SQLite DB path (default: $DB_PATH, /tmp/db.sqlite, or db_prod_snapshot.sqlite)")
    ap.add_argument("--hours", type=int, default=24, help="Time window (last N hours)")
    ap.add_argument("--check-storage", action="store_true", help="HEAD-check referenced objects in Supabase Storage")
    ap.add_argument("--max-storage-items", type=int, default=600, help="Limit number of Storage objects to check (0 = no limit)")
    ap.add_argument("--near-threshold", type=int, default=6, help="Poster near-duplicate scan threshold (Hamming distance)")
    ap.add_argument("--near-max-pairs", type=int, default=50, help="Max near-duplicate pairs to print")
    args = ap.parse_args()

    db_path = (args.db or "").strip()
    if not db_path:
        db_path = (os.getenv("DB_PATH") or "").strip()
    if not db_path:
        db_path = "/tmp/db.sqlite" if Path("/tmp/db.sqlite").exists() else "db_prod_snapshot.sqlite"

    con = _open_db_ro(db_path)
    posters_all = _fetch_posters(con)
    videos_all = _fetch_videos(con)
    con.close()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, int(args.hours or 24)))

    posters = [r for r in posters_all if (r.updated_at or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    videos = [r for r in videos_all if (r.created_at or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]

    print("# Media Dedup Audit")
    print("")
    print(f"- db: `{db_path}`")
    print(f"- window: last {int(args.hours or 24)}h")
    print(f"- cutoff_utc: `{_fmt_dt(cutoff)}`")
    print(f"- now_utc: `{_fmt_dt(now)}`")
    print("")

    poster_stats = _scan_poster_rows(
        posters,
        near_threshold=int(args.near_threshold or 0),
        near_max_pairs=int(args.near_max_pairs or 0),
    )
    video_stats = _scan_video_rows(videos)

    print("## Posters (eventposter)")
    print("")
    print(f"- rows_in_window: {poster_stats['rows']}")
    print(f"- canonical_paths: {poster_stats['canonical_paths']} / {poster_stats['rows']}")
    print(f"- noncanonical_paths: {poster_stats['noncanonical_paths']}")
    print(f"- distinct_paths: {poster_stats['distinct_supabase_paths']}")
    print(f"- distinct_phash: {poster_stats['distinct_phash']}")
    print(f"- prefix_counts: {poster_stats['prefix_counts']}")
    if poster_stats["non_webp_paths"]:
        print(f"- NON_WEBP paths: {len(poster_stats['non_webp_paths'])}")
        for p in poster_stats["non_webp_paths"][:12]:
            print(f"  - `{p}`")
    if poster_stats["noncanonical_path_samples"]:
        print(f"- noncanonical_samples: {len(poster_stats['noncanonical_path_samples'])}")
        for p in poster_stats["noncanonical_path_samples"][:10]:
            if p.strip():
                print(f"  - `{p}`")
    if poster_stats["first2_mismatch"]:
        print(f"- first2_mismatch: {len(poster_stats['first2_mismatch'])}")
        for p in poster_stats["first2_mismatch"][:10]:
            print(f"  - `{p}`")
    if poster_stats["phash_mismatch"]:
        print(f"- PHASH_MISMATCH rows: {len(poster_stats['phash_mismatch'])}")
        for s in poster_stats["phash_mismatch"][:10]:
            print(f"  - `{s}`")
    if poster_stats["multi_path_same_phash"]:
        print(f"- MULTI_PATH_SAME_PHASH: {len(poster_stats['multi_path_same_phash'])}")
        for ph, paths in list(poster_stats["multi_path_same_phash"].items())[:10]:
            print(f"  - `{ph}`:")
            for p in list(paths)[:6]:
                print(f"    - `{p}`")
    if poster_stats["poster_hash_multi_paths"]:
        print(f"- POSTER_HASH_MULTI_PATHS: {len(poster_stats['poster_hash_multi_paths'])}")
        for s in poster_stats["poster_hash_multi_paths"][:10]:
            print(f"  - `{s}`")
    if poster_stats["poster_hash_multi_phash"]:
        print(f"- POSTER_HASH_MULTI_PHASH: {len(poster_stats['poster_hash_multi_phash'])}")
        for s in poster_stats["poster_hash_multi_phash"][:10]:
            print(f"  - `{s}`")
    if poster_stats["top_phash_usage"]:
        print("- top_phash_usage:")
        for ph, c in poster_stats["top_phash_usage"][:12]:
            print(f"  - `{ph}`: {c}")
    if poster_stats["near_pairs"]:
        print(f"- near_pairs (<= {poster_stats['near_pairs_threshold']}): {len(poster_stats['near_pairs'])}")
        for it in poster_stats["near_pairs"][:20]:
            print(f"  - dist={it['dist']} a={it['a']} ({it['a_count']}) b={it['b']} ({it['b_count']})")
    print("")

    print("## Videos (event_media_asset kind=video)")
    print("")
    print(f"- rows_in_window: {video_stats['rows']}")
    print(f"- distinct_paths: {video_stats['distinct_supabase_paths']}")
    print(f"- distinct_sha256: {video_stats['distinct_sha256']}")
    print(f"- missing_sha256_rows: {video_stats['missing_sha256_rows']}")
    print(f"- path_kind_counts: {video_stats['path_kind_counts']}")
    if video_stats["sha_mismatches"]:
        print(f"- SHA_MISMATCH rows: {len(video_stats['sha_mismatches'])}")
        for s in video_stats["sha_mismatches"][:10]:
            print(f"  - `{s}`")
    if video_stats["sha256_multi_paths"]:
        print(f"- SHA256_MULTI_PATHS: {len(video_stats['sha256_multi_paths'])}")
        for s in video_stats["sha256_multi_paths"][:10]:
            print(f"  - `{s}`")
    if video_stats["noncanonical_paths"]:
        print(f"- noncanonical_paths: {len(video_stats['noncanonical_paths'])}")
        for p in video_stats["noncanonical_paths"][:10]:
            print(f"  - `{p}`")
    if video_stats["tg_paths_missing_sha256"]:
        print(f"- TG_PATHS_MISSING_SHA256: {len(video_stats['tg_paths_missing_sha256'])}")
        for s in video_stats["tg_paths_missing_sha256"][:10]:
            print(f"  - `{s}`")
    print("")

    if args.check_storage:
        print("## Supabase Storage HEAD checks")
        print("")
        storage = _storage_check(
            posters=posters,
            videos=videos,
            max_items=int(args.max_storage_items or 0),
            cutoff=cutoff,
        )
        if not storage.get("enabled"):
            print(f"- enabled: false ({storage.get('reason')})")
        else:
            print(f"- enabled: true")
            print(f"- bucket_default: `{storage.get('bucket_default')}`")
            print(f"- unique_targets: {storage.get('unique_targets')}")
            print(f"- checked: {storage.get('checked')}")
            print(f"- head_errors: {storage.get('head_errors')}")
            if storage.get("missing"):
                print(f"- MISSING objects: {len(storage.get('missing') or [])}")
                for p in list(storage.get("missing") or [])[:20]:
                    print(f"  - `{p}`")
            if storage.get("wrong_content_type"):
                print(f"- WRONG content-type: {len(storage.get('wrong_content_type') or [])}")
                for p in list(storage.get("wrong_content_type") or [])[:20]:
                    print(f"  - `{p}`")
            if storage.get("recent_legacy_video_tg"):
                print(f"- RECENT legacy `v/tg/*` objects (should be rare): {len(storage.get('recent_legacy_video_tg') or [])}")
                for p in list(storage.get("recent_legacy_video_tg") or [])[:20]:
                    print(f"  - `{p}`")
        print("")

    anomalies = 0
    anomalies += 1 if poster_stats["non_webp_paths"] else 0
    anomalies += 1 if poster_stats["first2_mismatch"] else 0
    anomalies += 1 if poster_stats["phash_mismatch"] else 0
    anomalies += 1 if poster_stats["multi_path_same_phash"] else 0
    anomalies += 1 if poster_stats["poster_hash_multi_paths"] else 0
    anomalies += 1 if poster_stats["poster_hash_multi_phash"] else 0
    anomalies += 1 if video_stats["sha_mismatches"] else 0
    anomalies += 1 if video_stats["sha256_multi_paths"] else 0

    if args.check_storage:
        storage = _storage_check(
            posters=posters,
            videos=videos,
            max_items=int(args.max_storage_items or 0),
            cutoff=cutoff,
        )
        anomalies += 1 if storage.get("enabled") and storage.get("missing") else 0
        anomalies += 1 if storage.get("enabled") and storage.get("wrong_content_type") else 0
        anomalies += 1 if storage.get("enabled") and storage.get("recent_legacy_video_tg") else 0

    return 2 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())
