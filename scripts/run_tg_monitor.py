#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from db import Database
from models import TelegramSource
from source_parsing.telegram.service import run_telegram_monitor


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


async def _ensure_source(db: Database, username: str) -> None:
    async with db.get_session() as session:
        res = await session.execute(
            select(TelegramSource).where(TelegramSource.username == username)
        )
        src = res.scalar_one_or_none()
        if not src:
            src = TelegramSource(username=username, enabled=True)
            session.add(src)
            await session.commit()
            logging.info("tg_monitor.prep source_added=%s", username)
            return
        if not src.enabled:
            src.enabled = True
            session.add(src)
            await session.commit()
        logging.info("tg_monitor.prep source_exists=%s enabled=%s", username, bool(src.enabled))


async def _run(args: argparse.Namespace) -> int:
    _load_env_file(Path(args.env_file))
    db_path = args.db_path or os.environ.get("DB_PATH", "/data/db.sqlite")
    logging.info(
        "tg_monitor.local start channel=%s db_path=%s env_file=%s",
        args.channel,
        db_path,
        args.env_file,
    )
    db = Database(db_path)
    await db.init()

    await _ensure_source(db, args.channel)

    run_id = args.run_id or f"tgtest_{uuid.uuid4().hex[:8]}"
    logging.info("tg_monitor.local run_id=%s", run_id)
    report = await run_telegram_monitor(db, run_id=run_id)

    results_path = Path(tempfile.gettempdir()) / f"tg-monitor-{run_id}" / "telegram_results.json"
    print(f"run_id={run_id}")
    print(f"telegram_results_path={results_path}")
    print(f"report={report}")
    if args.print_json:
        if results_path.exists():
            print(results_path.read_text(encoding="utf-8"))
        else:
            print("telegram_results.json not found after run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Telegram monitoring via Kaggle")
    parser.add_argument("--channel", default="meowafisha", help="Telegram channel username")
    parser.add_argument("--db-path", default=None, help="DB path (defaults to DB_PATH or /data/db.sqlite)")
    parser.add_argument("--run-id", default=None, help="Custom run_id")
    parser.add_argument("--print-json", action="store_true", help="Print telegram_results.json")
    parser.add_argument("--env-file", default=".env", help="Path to .env file to load")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
