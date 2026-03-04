import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from db import Database  # noqa: E402
from telegram_sources_seed import seed_telegram_sources  # noqa: E402


async def _run(db_path: str) -> dict[str, int]:
    db = Database(db_path)
    # Avoid double-seeding inside Database.init(); this script calls the seed explicitly.
    os.environ.setdefault("DB_INIT_SKIP_TG_SOURCES_SEED", "1")
    await db.init()
    async with db.raw_conn() as conn:
        metrics = await seed_telegram_sources(conn)
        rows = await conn.execute_fetchall("SELECT COUNT(*) FROM telegram_source")
        total = int(rows[0][0] or 0) if rows else 0
        metrics["total_sources"] = total
        await conn.commit()
    await db.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed canonical telegram_source rows (idempotent).")
    parser.add_argument(
        "--db",
        dest="db_path",
        default=os.getenv("DB_PATH", "/data/db.sqlite"),
        help="SQLite DB path (default: $DB_PATH or /data/db.sqlite)",
    )
    args = parser.parse_args()
    metrics = asyncio.run(_run(args.db_path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
