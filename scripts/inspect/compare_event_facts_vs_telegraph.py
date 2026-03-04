#!/usr/bin/env python3
"""Compare EventSourceFact items against current Telegraph page text.

This is a best-effort diagnostic tool. Since descriptions are rewritten (not verbatim),
string matching can have false negatives. Still useful to quickly spot obvious drops
or leaks (e.g. a wrong title line from another event).

Usage:
  python scripts/inspect/compare_event_facts_vs_telegraph.py --db artifacts/db/db_prod_snapshot.sqlite --event-id 2151
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from html import unescape

import requests


def _strip_html(html: str) -> str:
    # Remove scripts/styles and tags; keep whitespace reasonably stable.
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to sqlite DB")
    ap.add_argument("--event-id", type=int, required=True)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("select id,title,telegraph_url,telegraph_path,description from event where id=?", (args.event_id,))
    ev = cur.fetchone()
    if not ev:
        raise SystemExit(f"event not found: {args.event_id}")

    url = (ev["telegraph_url"] or "").strip()
    if not url and (ev["telegraph_path"] or "").strip():
        url = "https://telegra.ph/" + str(ev["telegraph_path"]).lstrip("/")

    cur.execute(
        "PRAGMA table_info('event_source_fact')"
    )
    cols = {str(r[1] or "") for r in cur.fetchall()}
    if "status" in cols:
        cur.execute(
            """
            select esf.fact
            from event_source_fact esf
            where esf.event_id=? and coalesce(esf.status,'added')='added'
            order by esf.created_at asc, esf.id asc
            """,
            (args.event_id,),
        )
    else:
        cur.execute(
            """
            select esf.fact
            from event_source_fact esf
            where esf.event_id=?
            order by esf.created_at asc, esf.id asc
            """,
            (args.event_id,),
        )
    facts = [str(r["fact"] or "").strip() for r in cur.fetchall() if str(r["fact"] or "").strip()]
    con.close()

    print(f"event_id={args.event_id} title={ev['title']}")
    print(f"telegraph_url={url or 'MISSING'}")
    print(f"description_len={len(ev['description'] or '')}")
    print(f"facts={len(facts)}")

    if not url:
        print("No telegraph url; nothing to compare.")
        return 0

    resp = requests.get(url, timeout=25)
    print(f"telegraph_status={resp.status_code}")
    if resp.status_code >= 400:
        print(resp.text[:300])
        return 1

    page_text = _norm(_strip_html(resp.text))
    desc_text = _norm(str(ev["description"] or ""))

    missing = []
    for f in facts:
        f_norm = _norm(f)
        if not f_norm:
            continue
        # URLs are expected to be present as-is if included.
        if "http://" in f_norm or "https://" in f_norm:
            if f_norm not in page_text:
                missing.append(f)
            continue
        # For other facts, check against description first (primary content), then whole page.
        if f_norm in desc_text:
            continue
        if f_norm in page_text:
            continue
        missing.append(f)

    if missing:
        print("\nNOT FOUND (best-effort substring check):")
        for f in missing[:30]:
            print("-", f)
        if len(missing) > 30:
            print("...", len(missing) - 30, "more")
        return 2

    print("\nOK: all facts matched by substring (note: this does not guarantee semantic coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
