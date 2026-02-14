from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


_PROMO_RE = re.compile(
    r"\b(акци(?:я|и|ю|ях)|скидк\w*|промокод\w*|купон\w*|sale)\b|%",
    re.IGNORECASE,
)


def _telegraph_link(row: sqlite3.Row) -> str | None:
    url = (row["telegraph_url"] or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    path = (row["telegraph_path"] or "").strip()
    if path:
        return f"https://telegra.ph/{path.lstrip('/')}"
    src = (row["source_post_url"] or "").strip()
    if src.startswith(("http://", "https://")):
        return src
    return None


def _sentences(text: str) -> list[str]:
    chunks = re.split(r"[.!?…]\s+|\n{2,}|\n", text)
    out: list[str] = []
    for chunk in chunks:
        c = re.sub(r"\s+", " ", chunk).strip()
        if c:
            out.append(c)
    return out


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower().rstrip(".!?…")


def _find_new_sentences(source_text: str, description: str) -> list[str]:
    desc_norm = _normalize(description)
    out: list[str] = []
    for sent in _sentences(source_text):
        if len(sent) < 35:
            continue
        if _normalize(sent) not in desc_norm:
            out.append(sent)
    return out


def _find_duplicate_sentences(description: str) -> list[str]:
    counts: dict[str, int] = {}
    originals: dict[str, str] = {}
    for sent in _sentences(description):
        key = _normalize(sent)
        if len(key) < 35:
            continue
        counts[key] = counts.get(key, 0) + 1
        originals.setdefault(key, sent)
    return [originals[k] for k, v in counts.items() if v >= 2]


@dataclass(frozen=True)
class EventAudit:
    event_id: int
    title: str
    date: str
    time: str
    location_name: str
    telegraph: str | None
    sources_total: int
    missing_from_description: list[tuple[str, str]]  # (source_url, sentence)
    duplicated_in_description: list[str]
    promo_posters: list[tuple[str, str | None]]  # (url, ocr_title)


def _render_markdown(audits: list[EventAudit], *, usernames: list[str]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Merge audit report",
        "",
        f"Generated: {now}",
        f"Telegram sources: {', '.join('@' + u for u in usernames)}",
        "",
    ]
    interesting = [
        a
        for a in audits
        if a.missing_from_description or a.duplicated_in_description or a.promo_posters
    ]
    if not interesting:
        lines.append("No issues detected for the selected sources.")
        return "\n".join(lines).strip() + "\n"

    for a in interesting:
        lines.append(f"## #{a.event_id} — {a.title}")
        lines.append(f"- When/where: {a.date} {a.time} · {a.location_name}")
        lines.append(f"- Sources: {a.sources_total}")
        if a.telegraph:
            lines.append(f"- Telegraph: {a.telegraph}")
        if a.missing_from_description:
            lines.append("- Missing facts from telegram source_text (first 3 per source):")
            for source_url, sent in a.missing_from_description[:6]:
                lines.append(f"  - {source_url}: {sent}")
        if a.duplicated_in_description:
            lines.append("- Possible duplicate sentences in description:")
            for sent in a.duplicated_in_description[:6]:
                lines.append(f"  - {sent}")
        if a.promo_posters:
            lines.append("- Posters with promo-like OCR (review relevance):")
            for url, title in a.promo_posters:
                suffix = f" — {title}" if title else ""
                lines.append(f"  - {url}{suffix}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _audit(
    db_path: str,
    *,
    usernames: list[str],
    date_from: str,
    date_to: str,
    limit_events: int,
) -> list[EventAudit]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute("PRAGMA table_info('eventposter')")
        poster_cols = {r[1] for r in cur.fetchall()}
        has_supabase_url = "supabase_url" in poster_cols

        placeholders = ",".join("?" for _ in usernames) or "?"
        params = [date_from, date_to, *usernames]
        query = f"""
        SELECT DISTINCT e.id AS event_id
        FROM event e
        JOIN event_source s ON s.event_id = e.id
        WHERE e.date >= ? AND e.date <= ?
          AND s.source_type = 'telegram'
          AND s.source_chat_username IN ({placeholders})
        ORDER BY e.date ASC, e.id ASC
        """
        cur.execute(query, params)
        event_ids = [int(r["event_id"]) for r in cur.fetchmany(limit_events or 10_000)]
        if not event_ids:
            return []

        ev_placeholders = ",".join("?" for _ in event_ids)
        cur.execute(
            f"""
            SELECT id, title, date, time, location_name, description, telegraph_url, telegraph_path, source_post_url
            FROM event
            WHERE id IN ({ev_placeholders})
            """,
            event_ids,
        )
        events = {int(r["id"]): r for r in cur.fetchall()}

        cur.execute(
            f"""
            SELECT event_id, source_type, source_url, source_chat_username, source_text
            FROM event_source
            WHERE event_id IN ({ev_placeholders})
            ORDER BY imported_at ASC, id ASC
            """,
            event_ids,
        )
        sources_by_event: dict[int, list[sqlite3.Row]] = {}
        for row in cur.fetchall():
            sources_by_event.setdefault(int(row["event_id"]), []).append(row)

        cur.execute(
            f"""
            SELECT event_id, catbox_url, {('supabase_url' if has_supabase_url else 'NULL')} AS supabase_url, ocr_text, ocr_title
            FROM eventposter
            WHERE event_id IN ({ev_placeholders})
            ORDER BY updated_at DESC, id DESC
            """,
            event_ids,
        )
        posters_by_event: dict[int, list[sqlite3.Row]] = {}
        for row in cur.fetchall():
            posters_by_event.setdefault(int(row["event_id"]), []).append(row)

        audits: list[EventAudit] = []
        for event_id in event_ids:
            ev = events.get(int(event_id))
            if ev is None:
                continue
            description = ev["description"] or ""
            all_sources = sources_by_event.get(int(event_id), [])
            telegram_sources = [
                s
                for s in all_sources
                if (s["source_type"] == "telegram" and (s["source_chat_username"] or "") in usernames)
            ]
            missing: list[tuple[str, str]] = []
            for src in telegram_sources:
                src_text = (src["source_text"] or "").strip()
                if not src_text:
                    continue
                for sent in _find_new_sentences(src_text, description)[:3]:
                    missing.append((src["source_url"] or "", sent))

            duplicated = _find_duplicate_sentences(description)

            promo_posters: list[tuple[str, str | None]] = []
            for poster in posters_by_event.get(int(event_id), []):
                ocr = (poster["ocr_text"] or "").strip()
                if not ocr:
                    continue
                if not _PROMO_RE.search(ocr):
                    continue
                url = (poster["catbox_url"] or poster["supabase_url"] or "").strip()
                if not url:
                    continue
                promo_posters.append((url, (poster["ocr_title"] or None)))

            audits.append(
                EventAudit(
                    event_id=int(event_id),
                    title=ev["title"],
                    date=ev["date"],
                    time=ev["time"],
                    location_name=ev["location_name"],
                    telegraph=_telegraph_link(ev),
                    sources_total=len(all_sources),
                    missing_from_description=missing,
                    duplicated_in_description=duplicated,
                    promo_posters=promo_posters[:5],
                )
            )

        return audits
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument(
        "--usernames",
        default="dramteatr39,meowafisha,kaliningradlibrary",
        help="Comma-separated Telegram usernames to audit",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--limit-events", type=int, default=400)
    args = parser.parse_args()

    db_path = str(Path(args.db_path))
    usernames = [u.strip().lstrip("@") for u in str(args.usernames).split(",") if u.strip()]

    today = datetime.now(timezone.utc).date()
    default_from = (today - timedelta(days=30)).isoformat()
    default_to = (today + timedelta(days=120)).isoformat()
    date_from = (args.date_from or default_from).strip()
    date_to = (args.date_to or default_to).strip()

    audits = _audit(
        db_path,
        usernames=usernames,
        date_from=date_from,
        date_to=date_to,
        limit_events=int(args.limit_events or 0),
    )
    md = _render_markdown(audits, usernames=usernames)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
