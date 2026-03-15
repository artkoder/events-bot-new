from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


_TIME_RE = re.compile(r"(\d{1,2})[:.](\d{2})")
_NONWORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _is_missing_relation_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "no such table" in msg or "no such column" in msg


def _time_anchor(raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    m = _TIME_RE.search(text)
    if not m:
        return ""
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return ""
    if hh == 0 and mm == 0:
        return ""
    return f"{hh:02d}:{mm:02d}"


def _is_placeholder_time(raw: str | None) -> bool:
    text = str(raw or "").strip()
    if not text:
        return True
    return _time_anchor(text) == ""


def _norm_title(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\"«»]", "", text)
    text = _NONWORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _norm_loc(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", " ").replace("—", " ")
    text = re.sub(r"[\"«»']", " ", text)
    text = _NONWORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _loc_matches(a: str | None, b: str | None) -> bool:
    na = _norm_loc(a)
    nb = _norm_loc(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return na in nb or nb in na


def _load_json_list(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list):
        out: list[str] = []
        for item in data:
            s = str(item or "").strip()
            if s:
                out.append(s)
        return out
    return []


@dataclass(frozen=True)
class EventRow:
    id: int
    title: str
    date: str
    time: str
    location_name: str
    location_address: str | None
    telegraph_url: str | None
    description: str
    search_digest: str | None
    topics_json: str | None
    ticket_link: str | None
    source_post_url: str | None
    source_vk_post_url: str | None
    source_chat_id: int | None
    source_message_id: int | None
    added_at: str | None
    lifecycle_status: str
    silent: int

    @property
    def title_norm(self) -> str:
        return _norm_title(self.title)

    @property
    def loc_norm(self) -> str:
        return _norm_loc(self.location_name)

    @property
    def time_anchor(self) -> str:
        return _time_anchor(self.time)

    @property
    def topics(self) -> list[str]:
        return _load_json_list(self.topics_json)


def _backup_sqlite(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src_path) as src, sqlite3.connect(dst_path) as dst:
        src.execute("PRAGMA busy_timeout=30000")
        src.backup(dst)


def _fetch_events(con: sqlite3.Connection) -> list[EventRow]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT
            id,
            title,
            date,
            time,
            location_name,
            location_address,
            telegraph_url,
            description,
            search_digest,
            topics,
            ticket_link,
            source_post_url,
            source_vk_post_url,
            source_chat_id,
            source_message_id,
            added_at,
            lifecycle_status,
            silent
        FROM event
        WHERE lifecycle_status = 'active'
        """
    )
    out: list[EventRow] = []
    for row in cur.fetchall():
        out.append(
            EventRow(
                id=int(row[0]),
                title=str(row[1] or ""),
                date=str(row[2] or "").strip(),
                time=str(row[3] or "").strip(),
                location_name=str(row[4] or "").strip(),
                location_address=(str(row[5]) if row[5] is not None else None),
                telegraph_url=(str(row[6]) if row[6] is not None else None),
                description=str(row[7] or ""),
                search_digest=(str(row[8]) if row[8] is not None else None),
                topics_json=(str(row[9]) if row[9] is not None else None),
                ticket_link=(str(row[10]) if row[10] is not None else None),
                source_post_url=(str(row[11]) if row[11] is not None else None),
                source_vk_post_url=(str(row[12]) if row[12] is not None else None),
                source_chat_id=(int(row[13]) if row[13] is not None else None),
                source_message_id=(int(row[14]) if row[14] is not None else None),
                added_at=(str(row[15]) if row[15] is not None else None),
                lifecycle_status=str(row[16] or "active"),
                silent=int(row[17] or 0),
            )
        )
    return out


def _event_source_count(con: sqlite3.Connection) -> dict[int, int]:
    cur = con.cursor()
    cur.execute("SELECT event_id, COUNT(*) FROM event_source GROUP BY event_id")
    return {int(eid): int(cnt) for eid, cnt in cur.fetchall()}


def _score_event(ev: EventRow, *, source_count: int) -> tuple[int, int]:
    score = 0
    if (ev.telegraph_url or "").strip():
        score += 6
    if ev.description.strip():
        score += 2
    if (ev.search_digest or "").strip():
        score += 2
    if ev.topics:
        score += 1
    if (ev.ticket_link or "").strip():
        score += 1
    if ev.location_address and ev.location_address.strip():
        score += 1
    if ev.time_anchor:
        score += 2
    score += min(3, max(0, int(source_count)))
    return score, -int(ev.id)


def _union_find_components(items: list[EventRow]) -> list[list[EventRow]]:
    if len(items) <= 1:
        return [items]
    parent: dict[int, int] = {ev.id: ev.id for ev in items}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            if _loc_matches(items[i].location_name, items[j].location_name):
                union(items[i].id, items[j].id)

    grouped: dict[int, list[EventRow]] = defaultdict(list)
    for ev in items:
        grouped[find(ev.id)].append(ev)
    return [g for g in grouped.values() if g]


def _choose_best_time(cluster: Iterable[EventRow]) -> str | None:
    candidates: list[str] = []
    for ev in cluster:
        t = str(ev.time or "").strip()
        if not _is_placeholder_time(t):
            candidates.append(t)
    if not candidates:
        return None
    # Prefer longer strings (often include ranges like "16:00..17:30").
    return max(candidates, key=lambda s: (len(s), s))


def _choose_best_address(cluster: Iterable[EventRow]) -> str | None:
    candidates: list[str] = []
    for ev in cluster:
        a = str(ev.location_address or "").strip()
        if a:
            candidates.append(a)
    if not candidates:
        return None
    return max(candidates, key=lambda s: (len(s), s))


def _merge_event_sources(con: sqlite3.Connection, *, keep_id: int, drop_id: int) -> tuple[int, int]:
    cur = con.cursor()
    cur.execute("SELECT id, source_url FROM event_source WHERE event_id=?", (keep_id,))
    keep_urls = {str(url or "").strip(): int(src_id) for src_id, url in cur.fetchall() if str(url or "").strip()}

    moved = 0
    removed = 0
    cur.execute(
        "SELECT id, source_url FROM event_source WHERE event_id=? ORDER BY id",
        (drop_id,),
    )
    rows = [(int(i), str(u or "").strip()) for i, u in cur.fetchall()]
    for source_id, url in rows:
        if not url:
            continue
        if url in keep_urls:
            cur.execute("DELETE FROM event_source WHERE id=?", (source_id,))
            removed += 1
            continue
        cur.execute("UPDATE event_source SET event_id=? WHERE id=?", (keep_id, source_id))
        cur.execute(
            "UPDATE event_source_fact SET event_id=? WHERE source_id=?",
            (keep_id, source_id),
        )
        keep_urls[url] = source_id
        moved += 1
    return moved, removed


def _merge_posters(con: sqlite3.Connection, *, keep_id: int, drop_id: int) -> tuple[int, int]:
    cur = con.cursor()
    cur.execute(
        "SELECT poster_hash FROM eventposter WHERE event_id=?",
        (keep_id,),
    )
    keep_hashes = {str(h or "").strip() for (h,) in cur.fetchall() if str(h or "").strip()}

    moved = 0
    removed = 0
    cur.execute(
        "SELECT id, poster_hash FROM eventposter WHERE event_id=? ORDER BY id",
        (drop_id,),
    )
    rows = [(int(i), str(h or "").strip()) for i, h in cur.fetchall()]
    for poster_id, phash in rows:
        if not phash or phash in keep_hashes:
            cur.execute("DELETE FROM eventposter WHERE id=?", (poster_id,))
            removed += 1
            continue
        cur.execute("UPDATE eventposter SET event_id=? WHERE id=?", (keep_id, poster_id))
        keep_hashes.add(phash)
        moved += 1
    return moved, removed


def _merge_session_event_unique(
    con: sqlite3.Connection,
    table: str,
    *,
    keep_id: int,
    drop_id: int,
) -> tuple[int, int]:
    """Merge tables that have UNIQUE(session_id, event_id)."""

    cur = con.cursor()
    try:
        cur.execute(f"SELECT session_id FROM {table} WHERE event_id=?", (keep_id,))
    except sqlite3.OperationalError as exc:
        if _is_missing_relation_error(exc):
            return 0, 0
        raise
    keep_sessions = {int(sid) for (sid,) in cur.fetchall() if sid is not None}

    moved = 0
    removed = 0
    try:
        cur.execute(
            f"SELECT id, session_id FROM {table} WHERE event_id=? ORDER BY id",
            (drop_id,),
        )
    except sqlite3.OperationalError as exc:
        if _is_missing_relation_error(exc):
            return 0, 0
        raise
    rows = [(int(rid), int(sid)) for rid, sid in cur.fetchall() if sid is not None]
    for row_id, session_id in rows:
        if session_id in keep_sessions:
            cur.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
            removed += 1
            continue
        cur.execute(f"UPDATE {table} SET event_id=? WHERE id=?", (keep_id, row_id))
        keep_sessions.add(session_id)
        moved += 1
    return moved, removed


def _repoint_event_id_simple(
    con: sqlite3.Connection, table: str, *, keep_id: int, drop_id: int
) -> int:
    cur = con.cursor()
    try:
        cur.execute(f"UPDATE {table} SET event_id=? WHERE event_id=?", (keep_id, drop_id))
    except sqlite3.OperationalError as exc:
        if _is_missing_relation_error(exc):
            return 0
        raise
    return int(cur.rowcount or 0)


def _repoint_vk_inbox_import_event(con: sqlite3.Connection, *, keep_id: int, drop_id: int) -> tuple[int, int]:
    cur = con.cursor()
    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO vk_inbox_import_event(inbox_id, event_id, created_at)
            SELECT inbox_id, ?, created_at
            FROM vk_inbox_import_event
            WHERE event_id=?
            """,
            (keep_id, drop_id),
        )
    except sqlite3.OperationalError as exc:
        if _is_missing_relation_error(exc):
            return 0, 0
        raise
    inserted = int(cur.rowcount or 0)
    cur.execute("DELETE FROM vk_inbox_import_event WHERE event_id=?", (drop_id,))
    deleted = int(cur.rowcount or 0)
    return inserted, deleted


def _repoint_vk_inbox_imported_event(
    con: sqlite3.Connection, *, keep_id: int, drop_id: int
) -> int:
    cur = con.cursor()
    try:
        cur.execute(
            "UPDATE vk_inbox SET imported_event_id=? WHERE imported_event_id=?",
            (keep_id, drop_id),
        )
    except sqlite3.OperationalError as exc:
        if _is_missing_relation_error(exc):
            return 0
        raise
    return int(cur.rowcount or 0)


def _rewrite_linked_event_ids(
    con: sqlite3.Connection, *, keep_id: int, drop_id: int
) -> int:
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, linked_event_ids
        FROM event
        WHERE linked_event_ids IS NOT NULL
          AND linked_event_ids != '[]'
        """
    )
    changed = 0
    for row_id_raw, raw_links in cur.fetchall():
        row_id = int(row_id_raw)
        if row_id == drop_id:
            continue
        try:
            data = json.loads(str(raw_links or "").strip() or "[]")
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        updated: list[int] = []
        seen: set[int] = set()
        touched = False
        for item in data:
            try:
                link_id = int(item)
            except Exception:
                continue
            if link_id == drop_id:
                link_id = keep_id
                touched = True
            if link_id == row_id:
                touched = True
                continue
            if link_id in seen:
                touched = True
                continue
            seen.add(link_id)
            updated.append(link_id)
        if touched:
            cur.execute(
                "UPDATE event SET linked_event_ids=? WHERE id=?",
                (json.dumps(updated), row_id),
            )
            changed += 1
    return changed


def _merge_event_row_fields(
    con: sqlite3.Connection, *, keep: EventRow, cluster: list[EventRow]
) -> dict[str, object]:
    best_time = _choose_best_time(cluster)
    best_addr = _choose_best_address(cluster)

    patch: dict[str, object] = {}

    if _is_placeholder_time(keep.time) and best_time:
        patch["time"] = best_time
        patch["time_is_default"] = 0

    if (not (keep.location_address or "").strip()) and best_addr:
        patch["location_address"] = best_addr

    if not (keep.search_digest or "").strip():
        for ev in cluster:
            if (ev.search_digest or "").strip():
                patch["search_digest"] = ev.search_digest
                break

    if not keep.description.strip():
        for ev in cluster:
            if ev.description.strip():
                patch["description"] = ev.description
                break

    if not keep.topics and any(ev.topics for ev in cluster):
        merged_topics: set[str] = set()
        for ev in cluster:
            merged_topics.update(ev.topics)
        patch["topics"] = json.dumps(sorted(merged_topics))

    if not (keep.ticket_link or "").strip():
        for ev in cluster:
            if (ev.ticket_link or "").strip():
                patch["ticket_link"] = ev.ticket_link
                break

    if not (keep.source_post_url or "").strip():
        for ev in cluster:
            if (ev.source_post_url or "").strip():
                patch["source_post_url"] = ev.source_post_url
                break

    if not (keep.source_vk_post_url or "").strip():
        for ev in cluster:
            if (ev.source_vk_post_url or "").strip():
                patch["source_vk_post_url"] = ev.source_vk_post_url
                break

    if keep.source_chat_id is None:
        for ev in cluster:
            if ev.source_chat_id is not None:
                patch["source_chat_id"] = ev.source_chat_id
                break

    if keep.source_message_id is None:
        for ev in cluster:
            if ev.source_message_id is not None:
                patch["source_message_id"] = ev.source_message_id
                break

    return patch


def _apply_event_patch(con: sqlite3.Connection, *, event_id: int, patch: dict[str, object]) -> None:
    if not patch:
        return
    cur = con.cursor()
    columns = sorted(patch.keys())
    assigns = ", ".join(f"{c}=?" for c in columns)
    values = [patch[c] for c in columns] + [event_id]
    cur.execute(f"UPDATE event SET {assigns} WHERE id=?", values)


def _find_duplicate_clusters(events: list[EventRow]) -> list[list[EventRow]]:
    by_date_title: dict[tuple[str, str], list[EventRow]] = defaultdict(list)
    for ev in events:
        if not ev.date:
            continue
        tn = ev.title_norm
        if not tn:
            continue
        by_date_title[(ev.date, tn)].append(ev)

    clusters: list[list[EventRow]] = []
    for _key, items in by_date_title.items():
        if len(items) < 2:
            continue
        for comp in _union_find_components(items):
            if len(comp) < 2:
                continue
            explicit_times = {ev.time_anchor for ev in comp if ev.time_anchor}
            if len(explicit_times) > 1:
                continue
            clusters.append(sorted(comp, key=lambda e: e.id))
    # Prefer bigger clusters first for readability.
    clusters.sort(key=lambda c: (-len(c), c[0].date, c[0].title_norm, c[0].loc_norm))
    return clusters


def main() -> int:
    ap = argparse.ArgumentParser(description="Find and merge high-confidence duplicate events in sqlite DB.")
    ap.add_argument(
        "--db",
        default=None,
        help="Path to sqlite DB (default: $DB_PATH or db_prod_snapshot.sqlite)",
    )
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup when applying.",
    )
    ap.add_argument(
        "--merge",
        action="append",
        default=[],
        help=(
            "Explicit merge cluster (comma-separated event IDs). "
            "Can be passed multiple times. Example: --merge 3006,3007,3008"
        ),
    )
    ap.add_argument("--max-clusters", type=int, default=200, help="Max clusters to print/process.")
    args = ap.parse_args()

    db_path = Path(
        args.db
        or (os.getenv("DB_PATH") or "").strip()
        or "db_prod_snapshot.sqlite"
    )
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    if args.apply and not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = Path("artifacts/db") / f"{db_path.stem}.backup_{ts}.sqlite"
        print(f"Backup: {backup}")
        _backup_sqlite(db_path, backup)

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")

    try:
        events = _fetch_events(con)
        src_counts = _event_source_count(con)

        events_by_id: dict[int, EventRow] = {ev.id: ev for ev in events}

        explicit_clusters: list[list[EventRow]] = []
        used_ids: set[int] = set()
        for raw in list(args.merge or []):
            parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
            ids: list[int] = []
            for p in parts:
                try:
                    ids.append(int(p))
                except Exception as exc:
                    raise SystemExit(f"Bad --merge id {p!r}: {exc}") from exc
            ids = [i for i in ids if i > 0]
            # De-dupe while keeping order.
            seen_local: set[int] = set()
            ids = [i for i in ids if not (i in seen_local or seen_local.add(i))]
            if len(ids) < 2:
                continue
            for eid in ids:
                if eid in used_ids:
                    raise SystemExit(f"Event id {eid} appears in multiple --merge clusters")
                used_ids.add(eid)
            cluster: list[EventRow] = []
            missing: list[int] = []
            for eid in ids:
                ev = events_by_id.get(eid)
                if ev is None:
                    missing.append(eid)
                else:
                    cluster.append(ev)
            if missing:
                raise SystemExit(f"--merge references missing/inactive event IDs: {missing}")
            explicit_clusters.append(sorted(cluster, key=lambda e: e.id))

        if explicit_clusters:
            print(f"Explicit merge clusters: {len(explicit_clusters)}")

        auto_clusters = _find_duplicate_clusters(events)
        if auto_clusters:
            print(f"Duplicate clusters (merge candidates): {len(auto_clusters)}")
        else:
            print("Duplicate clusters (merge candidates): 0")

        def _process_clusters(clusters: list[list[EventRow]]) -> None:
            nonlocal src_counts
            clusters = clusters[: max(0, int(args.max_clusters))]
            for idx, cluster in enumerate(clusters, 1):
                head = cluster[0]
                date = head.date
                title = head.title_norm
                loc = head.loc_norm
                anchors = sorted({ev.time_anchor or "" for ev in cluster})
                print(
                    f"\n[{idx}] size={len(cluster)} date={date} time_anchors={anchors} title='{title}' loc='{loc}'"
                )
                for ev in cluster:
                    src = ev.source_post_url or ev.source_vk_post_url or ""
                    print(
                        f"  - #{ev.id} time='{ev.time}' loc='{ev.location_name}' "
                        f"addr='{ev.location_address or ''}' src='{src}'"
                    )

                if not args.apply:
                    continue

                keep = max(cluster, key=lambda ev: _score_event(ev, source_count=src_counts.get(ev.id, 0)))
                drop_ids = [ev.id for ev in cluster if ev.id != keep.id]
                print(f"  => KEEP #{keep.id}; DROP {drop_ids}")

                con.execute("BEGIN IMMEDIATE")
                try:
                    patch = _merge_event_row_fields(con, keep=keep, cluster=cluster)
                    _apply_event_patch(con, event_id=keep.id, patch=patch)

                    moved_sources = removed_sources = 0
                    moved_posters = removed_posters = 0
                    repointed_jobs = repointed_vkq = repointed_tsq = 0
                    repointed_vk_inbox = linked_rows_updated = 0
                    inserted_inbox = deleted_inbox = 0

                    for drop_id in drop_ids:
                        moved, removed = _merge_event_sources(con, keep_id=keep.id, drop_id=drop_id)
                        moved_sources += moved
                        removed_sources += removed

                        pmoved, premoved = _merge_posters(con, keep_id=keep.id, drop_id=drop_id)
                        moved_posters += pmoved
                        removed_posters += premoved

                        _ = _repoint_event_id_simple(con, "event_media_asset", keep_id=keep.id, drop_id=drop_id)
                        _merge_session_event_unique(con, "videoannounce_item", keep_id=keep.id, drop_id=drop_id)
                        _merge_session_event_unique(con, "videoannounce_eventhit", keep_id=keep.id, drop_id=drop_id)

                        repointed_jobs += _repoint_event_id_simple(con, "joboutbox", keep_id=keep.id, drop_id=drop_id)
                        repointed_tsq += _repoint_event_id_simple(
                            con, "ticket_site_queue", keep_id=keep.id, drop_id=drop_id
                        )
                        repointed_vkq += _repoint_event_id_simple(
                            con, "vk_publish_queue", keep_id=keep.id, drop_id=drop_id
                        )

                        ins, dele = _repoint_vk_inbox_import_event(con, keep_id=keep.id, drop_id=drop_id)
                        inserted_inbox += ins
                        deleted_inbox += dele

                        repointed_vk_inbox += _repoint_vk_inbox_imported_event(
                            con, keep_id=keep.id, drop_id=drop_id
                        )
                        linked_rows_updated += _rewrite_linked_event_ids(
                            con, keep_id=keep.id, drop_id=drop_id
                        )

                        con.execute("DELETE FROM event WHERE id=?", (drop_id,))

                    con.commit()
                except Exception:
                    con.rollback()
                    raise

                print(
                    "  merged:"
                    f" sources moved={moved_sources} removed={removed_sources};"
                    f" posters moved={moved_posters} removed={removed_posters};"
                    f" joboutbox repointed={repointed_jobs};"
                    f" ticket_site_queue repointed={repointed_tsq};"
                    f" vk_publish_queue repointed={repointed_vkq};"
                    f" vk_inbox repointed={repointed_vk_inbox};"
                    f" vk_inbox_import_event inserted={inserted_inbox} deleted={deleted_inbox}"
                    f"; linked_event_ids updated={linked_rows_updated}"
                )

        if args.apply and explicit_clusters:
            _process_clusters(explicit_clusters)
            # Refresh clusters after explicit merges to avoid stale IDs.
            events = _fetch_events(con)
            src_counts = _event_source_count(con)
            auto_clusters = _find_duplicate_clusters(events)
            if auto_clusters:
                print(f"\nDuplicate clusters (merge candidates) after explicit merges: {len(auto_clusters)}")
            else:
                print("\nDuplicate clusters (merge candidates) after explicit merges: 0")
            _process_clusters(auto_clusters)
        else:
            _process_clusters(explicit_clusters + auto_clusters)

        if args.apply:
            # Re-check quickly after modifications.
            after = _find_duplicate_clusters(_fetch_events(con))
            print(f"\nAfter merge: remaining duplicate clusters={len(after)}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
