#!/usr/bin/env python3
"""Inspect attached event videos and (optionally) verify Telegraph pages render them.

Usage:
  DB_PATH=artifacts/db/db_e2e.sqlite python scripts/inspect/inspect_event_videos.py --upcoming
  DB_PATH=artifacts/db/db_e2e.sqlite python scripts/inspect/inspect_event_videos.py --upcoming --check-telegraph
  python scripts/inspect/inspect_event_videos.py --db artifacts/db/db_e2e.sqlite --today 2026-02-26 --limit 20
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import requests


@dataclass(frozen=True)
class VideoItem:
    url: str | None
    path: str | None


def _telegraph_url(url: str | None, path: str | None) -> str | None:
    u = (url or "").strip()
    if u:
        return u
    p = (path or "").strip().lstrip("/")
    return f"https://telegra.ph/{p}" if p else None


def _build_supabase_public_url(path: str | None) -> str | None:
    p = (path or "").strip().lstrip("/")
    if not p:
        return None
    base = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    if not base:
        return None
    bucket = (os.getenv("SUPABASE_MEDIA_BUCKET") or os.getenv("SUPABASE_BUCKET") or "events-ics").strip()
    if not bucket:
        bucket = "events-ics"
    return f"{base}/storage/v1/object/public/{bucket}/{p}"


def _start_end_dates(date_value: str | None, end_date_value: str | None) -> tuple[str | None, str | None]:
    raw = (date_value or "").strip()
    if not raw:
        return None, (end_date_value or "").strip() or None
    if ".." in raw:
        a, b = raw.split("..", 1)
        start = a.strip() or None
        end = (end_date_value or "").strip() or (b.strip() or None)
        return start, end
    return raw, (end_date_value or "").strip() or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Path to sqlite DB (default: $DB_PATH or db_prod_snapshot.sqlite)")
    ap.add_argument("--today", default=None, help="Override today's date (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=50, help="Max events to print")
    ap.add_argument("--upcoming", action="store_true", help="Show only upcoming/active events (default: off)")
    ap.add_argument("--check-telegraph", action="store_true", help="Fetch Telegraph pages and verify video links")
    args = ap.parse_args()

    db_path = args.db or (os.getenv("DB_PATH") or "").strip() or "db_prod_snapshot.sqlite"
    today = (args.today or "").strip() or date.today().isoformat()

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("PRAGMA table_info('event_media_asset')")
    if not cur.fetchall():
        print(f"event_media_asset table is missing in db={db_path}")
        return 1

    cur.execute(
        """
        select
          e.id as event_id,
          e.title as title,
          e.date as date,
          e.end_date as end_date,
          e.telegraph_url as telegraph_url,
          e.telegraph_path as telegraph_path,
          ema.supabase_url as supabase_url,
          ema.supabase_path as supabase_path,
          ema.created_at as created_at
        from event_media_asset ema
        join event e on e.id = ema.event_id
        where ema.kind = 'video'
        order by e.date asc, e.id asc, ema.created_at asc
        """
    )
    rows = cur.fetchall()
    con.close()

    if not rows:
        print(f"No event_media_asset(kind=video) rows found in db={db_path}")
        return 0

    grouped: dict[int, dict] = {}
    videos_by_event: dict[int, list[VideoItem]] = defaultdict(list)
    for r in rows:
        eid = int(r["event_id"])
        if eid not in grouped:
            grouped[eid] = {
                "event_id": eid,
                "title": str(r["title"] or ""),
                "date": str(r["date"] or "") or None,
                "end_date": str(r["end_date"] or "") or None,
                "telegraph_url": str(r["telegraph_url"] or "") or None,
                "telegraph_path": str(r["telegraph_path"] or "") or None,
            }
        videos_by_event[eid].append(
            VideoItem(
                url=str(r["supabase_url"] or "").strip() or None,
                path=str(r["supabase_path"] or "").strip() or None,
            )
        )

    event_ids = sorted(grouped.keys())
    if args.upcoming:
        filtered: list[int] = []
        for eid in event_ids:
            meta = grouped[eid]
            start, end = _start_end_dates(meta.get("date"), meta.get("end_date"))
            if end and end >= today:
                filtered.append(eid)
                continue
            if start and start >= today:
                filtered.append(eid)
        event_ids = filtered

    event_ids = event_ids[: max(0, int(args.limit or 0))] if args.limit else event_ids

    print(f"db={db_path}")
    print(f"today={today}")
    print(f"events_with_videos={len(event_ids)} (total_rows={len(rows)})")
    print("")

    mismatches = 0
    checked = 0
    for eid in event_ids:
        meta = grouped[eid]
        t_url = _telegraph_url(meta.get("telegraph_url"), meta.get("telegraph_path"))
        start, end = _start_end_dates(meta.get("date"), meta.get("end_date"))
        date_label = start or meta.get("date") or "—"
        if end and end != start:
            date_label = f"{date_label}..{end}"
        print(f"- id={eid} date={date_label} title={meta.get('title') or ''}")
        print(f"  telegraph={t_url or 'MISSING'}")

        resolved_urls: list[str] = []
        for it in videos_by_event.get(eid, []):
            u = (it.url or "").strip()
            if not u:
                u = _build_supabase_public_url(it.path) or ""
            if u and u not in resolved_urls:
                resolved_urls.append(u)
        for idx, u in enumerate(resolved_urls[:6], start=1):
            print(f"  video{idx}={u}")
        if len(resolved_urls) > 6:
            print(f"  ... +{len(resolved_urls) - 6} more")

        if args.check_telegraph:
            checked += 1
            if not t_url:
                print("  check=SKIP (no telegraph url)")
            else:
                try:
                    resp = requests.get(t_url, timeout=25)
                    if resp.status_code >= 400:
                        print(f"  check=ERROR status={resp.status_code}")
                    else:
                        ok = any(u in resp.text for u in resolved_urls) if resolved_urls else False
                        if ok:
                            print("  check=OK")
                        else:
                            mismatches += 1
                            print("  check=MISSING (no video urls found in telegraph html)")
                except Exception as exc:
                    mismatches += 1
                    print(f"  check=ERROR {type(exc).__name__}: {exc}")
        print("")

    if args.check_telegraph:
        print(f"checked={checked} mismatches={mismatches}")
        return 2 if mismatches else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

