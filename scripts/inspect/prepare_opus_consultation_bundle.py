#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _norm_url(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        scheme = (parts.scheme or "https").lower()
        netloc = (parts.netloc or "").lower()
        path = parts.path or ""
        if path and path != "/":
            path = path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return raw.split("?", 1)[0].split("#", 1)[0].rstrip("/")


def _parse_tg_post(url: str) -> tuple[str, int] | None:
    norm = _norm_url(url)
    if not norm:
        return None
    try:
        parts = urlsplit(norm)
        netloc = (parts.netloc or "").lower()
        if not netloc.endswith("t.me"):
            return None
        path_parts = [p for p in (parts.path or "").split("/") if p]
        if len(path_parts) < 2:
            return None
        username = path_parts[-2].lstrip("@").strip().lower()
        mid = int(path_parts[-1])
        if not username or mid <= 0:
            return None
        return username, mid
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_tg_results(limit: int, search_root: Path) -> list[Path]:
    files = [p for p in search_root.glob("tg-monitor-*/telegram_results.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return files[: max(1, limit)]


def _extract_casepack_targets(casepack: dict[str, Any]) -> tuple[set[int], set[str], dict[str, list[int]]]:
    event_ids: set[int] = set()
    source_urls: set[str] = set()
    case_to_event_ids: dict[str, list[int]] = {}

    for case in casepack.get("cases") or []:
        key = str(case.get("key") or "case")
        bucket: list[int] = []
        for ev in case.get("events") or []:
            eid = ev.get("event_id")
            if isinstance(eid, int):
                event_ids.add(eid)
                bucket.append(eid)

            for field in ("source_post_url", "source_vk_post_url", "source_telegram_post_url"):
                norm = _norm_url(ev.get(field))
                if norm:
                    source_urls.add(norm)

            for src in ev.get("source_material") or []:
                norm = _norm_url(src.get("source_url"))
                if norm:
                    source_urls.add(norm)
        case_to_event_ids[key] = sorted(set(bucket))
    return event_ids, source_urls, case_to_event_ids


def _query_rows(
    con: sqlite3.Connection,
    query: str,
    values: list[Any],
    *,
    chunk: int = 500,
) -> list[sqlite3.Row]:
    if not values:
        return []
    out: list[sqlite3.Row] = []
    for i in range(0, len(values), chunk):
        part = values[i : i + chunk]
        ph = ",".join("?" for _ in part)
        sql = query.replace("__IN__", ph)
        out.extend(con.execute(sql, part).fetchall())
    return out


def _load_db_payload(db_path: Path, event_ids: set[int], source_urls: set[str], max_source_chars: int) -> dict[str, Any]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        event_rows = _query_rows(
            con,
            """
            select
              id, title, date, time, time_is_default, end_date,
              city, location_name, location_address,
              ticket_link, source_post_url, source_vk_post_url,
              telegraph_url, lifecycle_status, added_at
            from event
            where id in (__IN__)
            order by id
            """,
            sorted(event_ids),
        )
        events = [dict(r) for r in event_rows]

        event_source_rows = _query_rows(
            con,
            """
            select
              id, event_id, source_type, source_url,
              source_chat_username, source_chat_id, source_message_id,
              source_text, imported_at, trust_level
            from event_source
            where event_id in (__IN__)
            order by event_id, id
            """,
            sorted(event_ids),
        )
        event_sources: list[dict[str, Any]] = []
        source_ids: list[int] = []
        for row in event_source_rows:
            item = dict(row)
            text = str(item.get("source_text") or "")
            if max_source_chars > 0 and len(text) > max_source_chars:
                item["source_text"] = text[:max_source_chars]
                item["source_text_truncated"] = True
            else:
                item["source_text_truncated"] = False
            source_ids.append(int(item["id"]))
            event_sources.append(item)

        event_source_fact_rows = _query_rows(
            con,
            """
            select id, source_id, fact, status, created_at
            from event_source_fact
            where source_id in (__IN__)
            order by source_id, id
            """,
            source_ids,
        )
        event_source_facts = [dict(r) for r in event_source_fact_rows]

        event_poster_rows = _query_rows(
            con,
            """
            select id, event_id, catbox_url, supabase_url, supabase_path, poster_hash, ocr_title, ocr_text, phash
            from eventposter
            where event_id in (__IN__)
            order by event_id, id
            """,
            sorted(event_ids),
        )
        event_posters = [dict(r) for r in event_poster_rows]

        source_url_hits_rows = _query_rows(
            con,
            """
            select id, event_id, source_type, source_url, source_chat_username, source_message_id, imported_at
            from event_source
            where source_url in (__IN__)
            order by source_url, imported_at, id
            """,
            sorted(source_urls),
        )
        source_url_hits = [dict(r) for r in source_url_hits_rows]
    finally:
        con.close()

    return {
        "events": events,
        "event_sources": event_sources,
        "event_source_facts": event_source_facts,
        "event_posters": event_posters,
        "source_url_hits": source_url_hits,
    }


def _extract_tg_messages(
    tg_json_paths: list[Path],
    target_urls: set[str],
    *,
    max_text_chars: int,
) -> dict[str, Any]:
    target_norm = {_norm_url(u) for u in target_urls if _norm_url(u)}
    target_tg_pairs = {_parse_tg_post(u) for u in target_norm}
    target_tg_pairs.discard(None)

    matched_messages: list[dict[str, Any]] = []
    scanned = 0
    for path in tg_json_paths:
        try:
            payload = _read_json(path)
        except Exception:
            continue
        run_id = str(payload.get("run_id") or path.parent.name)
        messages = payload.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            scanned += 1
            source_link = _norm_url(msg.get("source_link"))
            uname = str(msg.get("source_username") or "").strip().lstrip("@").lower()
            mid_raw = msg.get("message_id")
            try:
                mid = int(mid_raw)
            except Exception:
                mid = 0
            pair = (uname, mid) if uname and mid > 0 else None
            if source_link not in target_norm and pair not in target_tg_pairs:
                continue

            text = str(msg.get("text") or "")
            if max_text_chars > 0 and len(text) > max_text_chars:
                text_out = text[:max_text_chars]
                truncated = True
            else:
                text_out = text
                truncated = False

            matched_messages.append(
                {
                    "results_file": str(path),
                    "run_id": run_id,
                    "generated_at": payload.get("generated_at"),
                    "source_link": source_link or None,
                    "source_username": uname or None,
                    "message_id": mid if mid > 0 else None,
                    "message_date": msg.get("message_date"),
                    "text": text_out,
                    "text_truncated": truncated,
                    "events": msg.get("events") or [],
                    "posters": msg.get("posters") or [],
                    "metrics": msg.get("metrics") or {},
                }
            )

    matched_messages.sort(
        key=lambda x: (
            str(x.get("source_link") or ""),
            str(x.get("message_date") or ""),
            str(x.get("results_file") or ""),
        )
    )
    return {
        "searched_files": [str(p) for p in tg_json_paths],
        "scanned_messages": scanned,
        "matched_messages_count": len(matched_messages),
        "matched_messages": matched_messages,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = payload["meta"]
    lines.append("# Opus Consultation Bundle")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{meta['generated_at_utc']}`")
    lines.append(f"- DB snapshot: `{meta['db_path']}`")
    lines.append(f"- Casepack: `{meta['casepack_path']}`")
    lines.append(f"- Cases: `{meta['cases_total']}`")
    lines.append(f"- Target events: `{meta['target_event_ids_count']}`")
    lines.append(f"- Target source URLs: `{meta['target_source_urls_count']}`")
    lines.append(f"- Telegram results files scanned: `{meta['tg_results_files_count']}`")
    lines.append(f"- Telegram messages matched: `{meta['tg_matched_messages_count']}`")
    lines.append("")

    lines.append("## Cases")
    lines.append("")
    for case in payload.get("cases") or []:
        lines.append(
            f"- `{case['key']}` [{case.get('kind')}] -> event_ids={case.get('event_ids')} "
            f"| target_urls={case.get('target_source_urls_count')}"
        )
    lines.append("")

    lines.append("## DB Summary")
    lines.append("")
    dbs = payload["db_extract"]["summary"]
    lines.append(f"- events: `{dbs['events']}`")
    lines.append(f"- event_sources: `{dbs['event_sources']}`")
    lines.append(f"- event_source_facts: `{dbs['event_source_facts']}`")
    lines.append(f"- event_posters: `{dbs['event_posters']}`")
    lines.append(f"- source_url_hits: `{dbs['source_url_hits']}`")
    lines.append("")

    lines.append("## Telegram Results Summary")
    lines.append("")
    tgs = payload["tg_extract"]["summary"]
    lines.append(f"- searched_files: `{tgs['searched_files']}`")
    lines.append(f"- scanned_messages: `{tgs['scanned_messages']}`")
    lines.append(f"- matched_messages: `{tgs['matched_messages']}`")
    lines.append("")

    lines.append("## Files For Opus")
    lines.append("")
    lines.append("- This bundle JSON (full payload)")
    lines.append("- This bundle MD (summary)")
    lines.append("- `docs/reports/smart-update-opus-session2-brief.md`")
    lines.append("- `docs/reports/smart-update-opus-session2-material-map.md`")
    lines.append("- `artifacts/codex/opus_session2_casepack_20260306.json`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified Opus consultation bundle from casepack + DB + telegram_results.json")
    parser.add_argument("--db", required=True, help="Path to SQLite snapshot (prod copy)")
    parser.add_argument(
        "--casepack",
        default="artifacts/codex/opus_session2_casepack_20260306.json",
        help="Casepack JSON path",
    )
    parser.add_argument(
        "--tg-results",
        action="append",
        default=[],
        help="Path to telegram_results.json (repeatable). If omitted, autodiscover in /tmp.",
    )
    parser.add_argument(
        "--tg-search-root",
        default=str(Path(tempfile.gettempdir())),
        help="Root for autodiscovery (default: system tmp dir)",
    )
    parser.add_argument(
        "--tg-limit",
        type=int,
        default=12,
        help="How many newest telegram_results.json files to scan when --tg-results is not provided",
    )
    parser.add_argument("--max-source-chars", type=int, default=12000, help="Max source_text chars from DB")
    parser.add_argument("--max-tg-text-chars", type=int, default=12000, help="Max message text chars from telegram_results")
    parser.add_argument("--out-json", help="Output JSON path")
    parser.add_argument("--out-md", help="Output Markdown path")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    casepack_path = Path(args.casepack).resolve()
    casepack = _read_json(casepack_path)
    event_ids, source_urls, case_to_event_ids = _extract_casepack_targets(casepack)

    tg_paths = [Path(p).resolve() for p in (args.tg_results or []) if Path(p).exists()]
    if not tg_paths:
        tg_paths = _discover_tg_results(limit=args.tg_limit, search_root=Path(args.tg_search_root).resolve())

    db_payload = _load_db_payload(db_path, event_ids, source_urls, args.max_source_chars)
    tg_payload = _extract_tg_messages(tg_paths, source_urls, max_text_chars=args.max_tg_text_chars)

    cases_summary: list[dict[str, Any]] = []
    for case in casepack.get("cases") or []:
        key = str(case.get("key") or "case")
        kinds = str(case.get("kind") or "")
        eids = case_to_event_ids.get(key) or []
        case_urls: set[str] = set()
        for ev in case.get("events") or []:
            for field in ("source_post_url", "source_vk_post_url", "source_telegram_post_url"):
                norm = _norm_url(ev.get(field))
                if norm:
                    case_urls.add(norm)
            for src in ev.get("source_material") or []:
                norm = _norm_url(src.get("source_url"))
                if norm:
                    case_urls.add(norm)
        cases_summary.append(
            {
                "key": key,
                "kind": kinds,
                "event_ids": sorted(set(eids)),
                "target_source_urls_count": len(case_urls),
            }
        )

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "casepack_path": str(casepack_path),
            "cases_total": len(casepack.get("cases") or []),
            "target_event_ids_count": len(event_ids),
            "target_source_urls_count": len(source_urls),
            "tg_results_files_count": len(tg_paths),
            "tg_matched_messages_count": tg_payload["matched_messages_count"],
        },
        "cases": cases_summary,
        "db_extract": {
            "summary": {
                "events": len(db_payload["events"]),
                "event_sources": len(db_payload["event_sources"]),
                "event_source_facts": len(db_payload["event_source_facts"]),
                "event_posters": len(db_payload["event_posters"]),
                "source_url_hits": len(db_payload["source_url_hits"]),
            },
            "events": db_payload["events"],
            "event_sources": db_payload["event_sources"],
            "event_source_facts": db_payload["event_source_facts"],
            "event_posters": db_payload["event_posters"],
            "source_url_hits": db_payload["source_url_hits"],
        },
        "tg_extract": {
            "summary": {
                "searched_files": len(tg_payload["searched_files"]),
                "scanned_messages": tg_payload["scanned_messages"],
                "matched_messages": tg_payload["matched_messages_count"],
            },
            "searched_files": tg_payload["searched_files"],
            "matched_messages": tg_payload["matched_messages"],
        },
    }

    stamp = _utc_stamp()
    out_json = Path(args.out_json) if args.out_json else Path(f"artifacts/codex/opus_consultation_bundle_{stamp}.json")
    out_md = Path(args.out_md) if args.out_md else Path(f"artifacts/codex/opus_consultation_bundle_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_md": str(out_md),
                "db_path": str(db_path),
                "casepack_path": str(casepack_path),
                "cases_total": payload["meta"]["cases_total"],
                "target_event_ids": payload["meta"]["target_event_ids_count"],
                "target_source_urls": payload["meta"]["target_source_urls_count"],
                "tg_results_files_scanned": payload["meta"]["tg_results_files_count"],
                "tg_matched_messages": payload["meta"]["tg_matched_messages_count"],
                "db_events": payload["db_extract"]["summary"]["events"],
                "db_event_sources": payload["db_extract"]["summary"]["event_sources"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
