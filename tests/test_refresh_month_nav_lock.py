import importlib
from pathlib import Path
import pytest
import main


@pytest.mark.asyncio
async def test_sync_month_page_refresh_nav_outside_lock(tmp_path: Path, monkeypatch):
    m = importlib.reload(main)
    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            m.Event(
                title="E1",
                description="d",
                source_text="s",
                date="2025-07-01",
                time="10:00",
                location_name="L",
            )
        )
        await session.commit()

    class DummyTG:
        def create_page(self, title, content=None, html_content=None, **kwargs):
            return {"url": "u1", "path": "p1"}

        def edit_page(self, path, title=None, content=None, html_content=None, **kwargs):
            pass

    monkeypatch.setattr(m, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(m, "Telegraph", lambda access_token=None, domain=None: DummyTG())

    called = {"flag": False}

    async def fake_refresh(db_obj):
        called["flag"] = True
        assert not m.HEAVY_SEMAPHORE.locked()

    monkeypatch.setattr(m, "refresh_month_nav", fake_refresh)

    await m.sync_month_page(db, "2025-07")

    assert called["flag"]
