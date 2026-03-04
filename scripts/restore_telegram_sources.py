import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from db import Database  # noqa: E402
from telegram_sources import normalize_tg_username  # noqa: E402


async def _ensure_filters_column(conn) -> None:
    try:
        await conn.execute("ALTER TABLE telegram_source ADD COLUMN filters_json TEXT")
    except Exception as exc:
        if "duplicate column name" not in str(exc).lower():
            # Best-effort: older/newer schemas might differ; do not block restore.
            return


def _read_usernames(args) -> list[str]:
    raw: list[str] = []
    for u in list(args.usernames or []):
        if u:
            raw.append(u)
    if args.from_file:
        p = Path(args.from_file)
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            raw.append(s)
    if (not raw) and (not sys.stdin.isatty()):
        raw.extend([line.strip() for line in sys.stdin.read().splitlines() if line.strip()])
    # normalize + dedup
    out: list[str] = []
    seen: set[str] = set()
    for s in raw:
        key = normalize_tg_username(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


async def _run(db_path: str, usernames: list[str]) -> dict[str, int]:
    metrics: dict[str, int] = {"input": len(usernames), "inserted": 0, "re_enabled": 0, "total_sources": 0}
    if not usernames:
        return metrics

    db = Database(db_path)
    await db.init()
    async with db.raw_conn() as conn:
        await _ensure_filters_column(conn)
        tables = {
            str(r[0] or "").strip()
            for r in await conn.execute_fetchall("SELECT name FROM sqlite_master WHERE type='table'")
            if r and r[0]
        }
        if "telegram_source" not in tables:
            raise RuntimeError("telegram_source table not found (DB schema not initialized?)")

        rows = await conn.execute_fetchall("PRAGMA table_info(telegram_source)")
        cols = {str(r[1] or "").strip() for r in rows if r and len(r) >= 2}

        for username in usernames:
            # Re-enable existing row (idempotent).
            before = int(getattr(conn, "total_changes", 0) or 0)
            await conn.execute(
                "UPDATE telegram_source SET enabled=1 WHERE username=? AND (enabled IS NULL OR enabled=0)",
                (username,),
            )
            after = int(getattr(conn, "total_changes", 0) or 0)
            metrics["re_enabled"] += max(0, after - before)
            # Insert missing row.
            before = after
            if "enabled" in cols:
                await conn.execute(
                    "INSERT OR IGNORE INTO telegram_source(username, enabled) VALUES(?, 1)",
                    (username,),
                )
            else:
                await conn.execute(
                    "INSERT OR IGNORE INTO telegram_source(username) VALUES(?)",
                    (username,),
                )
            after2 = int(getattr(conn, "total_changes", 0) or 0)
            metrics["inserted"] += max(0, after2 - before)

        rows = await conn.execute_fetchall("SELECT COUNT(*) FROM telegram_source")
        metrics["total_sources"] = int(rows[0][0] or 0) if rows else 0
        await conn.commit()
    await db.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore Telegram sources by upserting @usernames into telegram_source (non-destructive)."
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=os.getenv("DB_PATH", "/data/db.sqlite"),
        help="SQLite DB path (default: $DB_PATH or /data/db.sqlite)",
    )
    parser.add_argument(
        "--from-file",
        dest="from_file",
        default=None,
        help="Text file with one @username / t.me link per line",
    )
    parser.add_argument(
        "usernames",
        nargs="*",
        help="Usernames or t.me links (e.g. @dramteatr39 https://t.me/ambermuseum)",
    )
    args = parser.parse_args()
    usernames = _read_usernames(args)
    metrics = asyncio.run(_run(args.db_path, usernames))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
