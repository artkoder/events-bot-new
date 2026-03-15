#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко", "у", "о", "об", "от", "за",
    "для", "из", "или", "а", "но", "что", "как", "это", "эта", "этот", "эти",
    "мы", "вы", "он", "она", "они", "его", "ее", "их", "при", "над", "под", "до",
    "после", "без", "через", "уже", "будет", "будут", "приглашаем", "присоединяйтесь",
    "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
    "ноября", "декабря", "января", "февраля", "субботу", "воскресенье", "воскресенье,",
    "среду", "четверг", "пятницу", "суббота", "воскресенье", "пятница", "понедельник",
}


@dataclass(slots=True)
class OccurrenceRow:
    id: int
    title: str
    summary: str
    date: str
    time: str
    price_text: str
    booking_url: str
    channel_url: str
    source_username: str
    source_title: str
    source_kind: str


def collapse_ws(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def token_set(*parts: str) -> set[str]:
    payload = " ".join(collapse_ws(part).lower().replace("ё", "е") for part in parts if part)
    tokens = set(re.findall(r"[a-zа-я0-9]{3,}", payload))
    return {token for token in tokens if token not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_occurrences(conn: sqlite3.Connection) -> list[OccurrenceRow]:
    cur = conn.execute(
        """
        SELECT
            go.id,
            go.canonical_title,
            go.summary_one_liner,
            go.date,
            COALESCE(go.time, '') AS time,
            COALESCE(go.price_text, '') AS price_text,
            COALESCE(go.booking_url, '') AS booking_url,
            COALESCE(go.channel_url, '') AS channel_url,
            COALESCE(gs.username, '') AS source_username,
            COALESCE(gs.title, '') AS source_title,
            COALESCE(gs.source_kind, '') AS source_kind
        FROM guide_occurrence go
        LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
        WHERE go.date IS NOT NULL
        ORDER BY go.date, go.id
        """
    )
    rows: list[OccurrenceRow] = []
    for row in cur.fetchall():
        rows.append(
            OccurrenceRow(
                id=int(row[0]),
                title=collapse_ws(row[1]),
                summary=collapse_ws(row[2]),
                date=collapse_ws(row[3]),
                time=collapse_ws(row[4]),
                price_text=collapse_ws(row[5]),
                booking_url=collapse_ws(row[6]),
                channel_url=collapse_ws(row[7]),
                source_username=collapse_ws(row[8]),
                source_title=collapse_ws(row[9]),
                source_kind=collapse_ws(row[10]),
            )
        )
    return rows


def pair_record(left: OccurrenceRow, right: OccurrenceRow) -> dict[str, object]:
    left_tokens = token_set(left.title, left.summary)
    right_tokens = token_set(right.title, right.summary)
    shared = sorted(left_tokens & right_tokens)
    token_score = jaccard(left_tokens, right_tokens)
    same_source = left.source_username == right.source_username
    same_booking = bool(left.booking_url and left.booking_url == right.booking_url)
    same_price = bool(left.price_text and left.price_text == right.price_text)
    same_time = bool(left.time and left.time == right.time)
    aggregator_involved = "aggregator" in {left.source_kind, right.source_kind}

    suspicion = token_score
    reasons: list[str] = []
    if same_source:
        suspicion += 0.22
        reasons.append("same_source")
    if same_time:
        suspicion += 0.08
        reasons.append("same_time")
    if same_booking:
        suspicion += 0.10
        reasons.append("same_booking")
    if same_price:
        suspicion += 0.05
        reasons.append("same_price")
    if aggregator_involved:
        reasons.append("aggregator_involved")

    is_candidate = False
    if same_source and token_score >= 0.12:
        is_candidate = True
    if same_source and len(shared) >= 3:
        is_candidate = True
    if same_source and same_booking:
        is_candidate = True
    if aggregator_involved and token_score >= 0.22:
        is_candidate = True
    if aggregator_involved and same_booking and same_time:
        is_candidate = True

    return {
        "date": left.date,
        "candidate": is_candidate,
        "suspicion_score": round(suspicion, 3),
        "token_score": round(token_score, 3),
        "shared_tokens": shared[:14],
        "reasons": reasons,
        "left": {
            "id": left.id,
            "source_username": left.source_username,
            "source_title": left.source_title,
            "source_kind": left.source_kind,
            "title": left.title,
            "time": left.time,
            "booking_url": left.booking_url,
            "channel_url": left.channel_url,
        },
        "right": {
            "id": right.id,
            "source_username": right.source_username,
            "source_title": right.source_title,
            "source_kind": right.source_kind,
            "title": right.title,
            "time": right.time,
            "booking_url": right.booking_url,
            "channel_url": right.channel_url,
        },
    }


def find_candidates(rows: list[OccurrenceRow]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    by_source_date: dict[tuple[str, str], list[OccurrenceRow]] = defaultdict(list)
    by_date: dict[str, list[OccurrenceRow]] = defaultdict(list)
    for row in rows:
        by_source_date[(row.source_username, row.date)].append(row)
        by_date[row.date].append(row)

    for group in by_source_date.values():
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            rec = pair_record(left, right)
            if rec["candidate"]:
                candidates.append(rec)

    for group in by_date.values():
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            if left.source_username == right.source_username:
                continue
            if "aggregator" not in {left.source_kind, right.source_kind}:
                continue
            rec = pair_record(left, right)
            if rec["candidate"]:
                candidates.append(rec)

    candidates.sort(
        key=lambda item: (
            -float(item["suspicion_score"]),
            -float(item["token_score"]),
            str(item["date"]),
            int(item["left"]["id"]),
            int(item["right"]["id"]),
        )
    )
    return candidates


def render_text(candidates: list[dict[str, object]], *, limit: int) -> str:
    lines = [f"Guide duplicate audit: {min(limit, len(candidates))} candidate pairs shown / {len(candidates)} total"]
    for idx, item in enumerate(candidates[:limit], start=1):
        left = item["left"]
        right = item["right"]
        lines.append("")
        lines.append(
            f"{idx}. {item['date']} score={item['suspicion_score']} token={item['token_score']} reasons={','.join(item['reasons'])}"
        )
        lines.append(f"   L#{left['id']} @{left['source_username']}: {left['title']}")
        lines.append(f"      {left['channel_url']}")
        lines.append(f"   R#{right['id']} @{right['source_username']}: {right['title']}")
        lines.append(f"      {right['channel_url']}")
        if item["shared_tokens"]:
            lines.append(f"   shared: {', '.join(item['shared_tokens'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit likely duplicate guide occurrences in SQLite.")
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--limit", type=int, default=20, help="How many pairs to print")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = load_occurrences(conn)
        candidates = find_candidates(rows)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(candidates[: args.limit], ensure_ascii=False, indent=2))
    else:
        print(render_text(candidates, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
