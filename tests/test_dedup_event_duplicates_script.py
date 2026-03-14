import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from db import Database
from models import Event, EventSource, EventSourceFact, JobOutbox, JobStatus, JobTask, VKInbox


@pytest.mark.asyncio
async def test_dedup_script_repoints_vk_inbox_and_linked_event_ids(tmp_path):
    db_path = tmp_path / "db.sqlite"
    db = Database(str(db_path))
    await db.init()

    async with db.get_session() as session:
        session.add_all(
            [
                Event(
                    id=1,
                    title="Презентация книги",
                    description="Подробное описание.",
                    source_text="Источник keep.",
                    date="2026-03-16",
                    time="14:00",
                    location_name="Библиотека им. Лунина",
                    location_address="Калинина 4",
                    city="Черняховск",
                    telegraph_url="https://telegra.ph/keep",
                    linked_event_ids=[2, 3],
                ),
                Event(
                    id=2,
                    title="Презентация книги",
                    description="",
                    source_text="Источник drop.",
                    date="2026-03-16",
                    time="14:00",
                    location_name="Библиотека им. Лунина, Калинина 4, Черняховск",
                    city="Черняховск",
                    linked_event_ids=[1, 3],
                ),
                Event(
                    id=3,
                    title="Связанное событие",
                    description="Описание связанного события.",
                    source_text="Источник linked.",
                    date="2026-03-17",
                    time="18:00",
                    location_name="Библиотека им. Лунина",
                    city="Черняховск",
                    linked_event_ids=[1, 2],
                ),
                JobOutbox(
                    event_id=2,
                    task=JobTask.month_pages,
                    status=JobStatus.pending,
                ),
                VKInbox(
                    group_id=1,
                    post_id=1,
                    date=1,
                    text="post text",
                    has_date=1,
                    imported_event_id=2,
                ),
            ]
        )
        await session.flush()

        source = EventSource(
            event_id=2,
            source_type="vk",
            source_url="https://vk.com/wall-1_1",
            source_text="post text",
        )
        session.add(source)
        await session.flush()

        session.add(
            EventSourceFact(
                event_id=2,
                source_id=int(source.id or 0),
                fact="Факт из drop-события",
                status="added",
            )
        )
        await session.commit()

    await db.exec_driver_sql(
        "INSERT INTO vk_inbox_import_event(inbox_id, event_id) VALUES (?, ?)",
        (1, 2),
    )
    await db.close()

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/inspect/dedup_event_duplicates.py"),
            "--db",
            str(db_path),
            "--apply",
            "--no-backup",
            "--merge",
            "1,2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "vk_inbox repointed=1" in result.stdout
    assert "linked_event_ids updated=2" in result.stdout

    con = sqlite3.connect(db_path)
    try:
        assert con.execute("SELECT COUNT(*) FROM event WHERE id=2").fetchone()[0] == 0
        assert (
            con.execute(
                "SELECT event_id FROM event_source WHERE source_url='https://vk.com/wall-1_1'"
            ).fetchone()[0]
            == 1
        )
        assert con.execute("SELECT event_id FROM event_source_fact").fetchone()[0] == 1
        assert con.execute("SELECT event_id FROM joboutbox").fetchone()[0] == 1
        assert con.execute("SELECT imported_event_id FROM vk_inbox").fetchone()[0] == 1
        assert con.execute("SELECT event_id FROM vk_inbox_import_event").fetchone()[0] == 1
        assert json.loads(con.execute("SELECT linked_event_ids FROM event WHERE id=1").fetchone()[0]) == [3]
        assert json.loads(con.execute("SELECT linked_event_ids FROM event WHERE id=3").fetchone()[0]) == [1]
    finally:
        con.close()
