import pytest
import main
from models import Festival, JobOutbox, JobStatus
from markup import FEST_NAV_START
from sqlalchemy import select
from datetime import date, timedelta


@pytest.mark.asyncio
async def test_rebuild_festival_nav_updates_only_upcoming(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # prepare festivals
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    async with db.get_session() as session:
        for i in range(3):
            session.add(
                Festival(
                    name=f"Fest{i+1}",
                    telegraph_path=f"p{i+1}",
                    vk_post_url=f"u{i+1}",
                    start_date=today,
                    end_date=today,
                )
            )
        session.add(
            Festival(
                name="Past",
                telegraph_path="p_old",
                vk_post_url="u_old",
                start_date=yesterday,
                end_date=yesterday,
            )
        )
        await session.commit()

    tg_pages = {f"p{i+1}": {"html": "<p>start</p>", "title": f"Fest{i+1}"} for i in range(3)}
    tg_pages["p_old"] = {"html": "<p>start</p>", "title": "Past"}

    class DummyTelegraph:
        def __init__(self, *_, **__):
            pass

        def get_page(self, path, return_html=True):
            return {"content_html": tg_pages[path]["html"], "title": tg_pages[path]["title"]}

        def edit_page(self, path, title, html_content):
            tg_pages[path] = {"html": html_content, "title": title}
            return {}

    async def fake_telegraph_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    vk_base = {f"Fest{i+1}": f"base{i+1}\n" for i in range(3)}
    vk_base["Past"] = "base_old\n"
    vk_posts = vk_base.copy()

    async def fake_sync_festival_vk_post(db, name, bot, nav_only=False, nav_lines=None):
        assert nav_only
        _, lines = await main._build_festival_nav_block(db, exclude=name)
        nav = "\n".join(lines)
        base = vk_base[name]
        cur = vk_posts.get(name, base)
        new = base + nav
        if cur == new:
            return False
        vk_posts[name] = new
        return True

    monkeypatch.setattr(main, "get_vk_group_id", lambda db: 1)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_festival_vk_post)
    called_index = False

    async def fake_sync_index(db):
        nonlocal called_index
        called_index = True

    monkeypatch.setattr(main, "sync_festivals_index_page", fake_sync_index)

    changed = await main.rebuild_fest_nav_if_changed(db)
    assert changed
    async with db.get_session() as session:
        jobs = (await session.execute(select(JobOutbox))).all()
    assert len(jobs) == 1
    while await main._run_due_jobs_once(db, None):
        pass

    for key in ["p1", "p2", "p3"]:
        page = tg_pages[key]
        assert FEST_NAV_START in page["html"]
        assert page["html"].endswith(main.FOOTER_LINK_HTML)
    assert tg_pages["p_old"]["html"] == "<p>start</p>"

    for name in ["Fest1", "Fest2", "Fest3"]:
        assert vk_posts[name] != vk_base[name]
    assert vk_posts["Past"] == vk_base["Past"]
    assert called_index

    changed2 = await main.rebuild_fest_nav_if_changed(db)
    assert not changed2

    async with db.get_session() as session:
        session.add(
            Festival(
                name="Fest4",
                telegraph_path="p4",
                vk_post_url="u4",
                start_date=today,
                end_date=today,
            )
        )
        await session.commit()
    tg_pages["p4"] = {"html": "<p>start</p>", "title": "Fest4"}
    vk_base["Fest4"] = "base4\n"
    vk_posts["Fest4"] = vk_base["Fest4"]

    changed3 = await main.rebuild_fest_nav_if_changed(db)
    assert changed3
    while await main._run_due_jobs_once(db, None):
        pass
    for name in ["Fest1", "Fest2", "Fest3", "Fest4"]:
        assert vk_posts[name] != vk_base[name]
    assert vk_posts["Past"] == vk_base["Past"]


@pytest.mark.asyncio
async def test_vk_failure_does_not_block_tg(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        for i in range(2):
            session.add(
                Festival(
                    name=f"Fest{i+1}",
                    telegraph_path=f"p{i+1}",
                    vk_post_url=f"u{i+1}",
                    start_date=today,
                    end_date=today,
                )
            )
        await session.commit()

    tg_pages = {f"p{i+1}": {"html": "<p>start</p>", "title": f"Fest{i+1}"} for i in range(2)}

    class DummyTelegraph:
        def __init__(self, *_, **__):
            pass

        def get_page(self, path, return_html=True):
            return {"content_html": tg_pages[path]["html"], "title": tg_pages[path]["title"]}

        def edit_page(self, path, title, html_content):
            tg_pages[path] = {"html": html_content, "title": title}
            return {}

    async def fake_telegraph_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    vk_base = {f"Fest{i+1}": f"base{i+1}\n" for i in range(2)}
    vk_posts = vk_base.copy()
    fail_name = "Fest1"

    async def fake_sync_festival_vk_post(db, name, bot, nav_only=False, nav_lines=None):
        assert nav_only
        if name == fail_name:
            raise RuntimeError("vk failure")
        _, lines = await main._build_festival_nav_block(db, exclude=name)
        nav = "\n".join(lines)
        base = vk_base[name]
        vk_posts[name] = base + nav
        return True

    monkeypatch.setattr(main, "get_vk_group_id", lambda db: 1)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_festival_vk_post)

    async def fake_sync_index2(db):
        return None

    monkeypatch.setattr(main, "sync_festivals_index_page", fake_sync_index2)

    await main.rebuild_fest_nav_if_changed(db)
    for _ in range(3):
        if not await main._run_due_jobs_once(db, None):
            break

    for page in tg_pages.values():
        assert FEST_NAV_START in page["html"]
    assert vk_posts["Fest2"] != vk_base["Fest2"]
    assert vk_posts["Fest1"] == vk_base["Fest1"]

    async with db.get_session() as session:
        rows = (
            await session.execute(
                select(JobOutbox.event_id, JobOutbox.task, JobOutbox.status)
            )
        ).all()
    statuses = [r.status for r in rows if r.task.value == "fest_nav:update_all"]
    assert JobStatus.error in statuses

