import logging
import os
from datetime import date
from pathlib import Path

import pytest

import main
from db import Database
from models import Festival
from sections import content_hash


@pytest.mark.asyncio
async def test_rebuild_fest_nav_if_changed_logs_nav_hash(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        session.add(Festival(name="Fest", start_date=today, end_date=today))
        await session.commit()

    async def fake_sync_index(db, telegraph=None, force: bool = False):
        return "built", ""

    monkeypatch.setattr(main, "rebuild_festivals_index_if_needed", fake_sync_index)

    with caplog.at_level(logging.INFO):
        changed = await main.rebuild_fest_nav_if_changed(db)
    assert changed
    nav_hash = await main.get_setting_value(db, "fest_nav_hash")
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "scheduled")
    assert rec.count == 1
    assert rec.nav_hash == nav_hash[:6]


@pytest.mark.asyncio
async def test_rebuild_festivals_index_logs_img_counts(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        session.add_all(
            [
                Festival(
                    name="WithImg",
                    start_date=today,
                    end_date=today,
                    photo_url="https://example.com/i.jpg",
                ),
                Festival(name="NoImg", start_date=today, end_date=today),
            ]
        )
        await session.commit()

    class DummyTelegraph:
        def __init__(self, *a, **k):
            pass

        def create_page(self, title, html_content):
            return {"url": "https://telegra.ph/f", "path": "f"}

    async def fake_call(func, *a, **k):
        return func(*a, **k)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_call)

    async def fake_create_page(tg, *a, **k):
        return tg.create_page(*a, **k)

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    with caplog.at_level(logging.INFO):
        await main.rebuild_festivals_index_if_needed(db)

    rec = next(r for r in caplog.records if getattr(r, "action", None) in {"built", "updated"})
    assert rec.with_img == 1
    assert rec.without_img == 1


@pytest.mark.asyncio
async def test_update_festival_tg_nav_logs_edited(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        fest = Festival(name="Fest1", telegraph_path="p1", start_date=today, end_date=today)
        session.add(fest)
        await session.commit()
        fid = fest.id
    await main.set_setting_value(db, "fest_nav_hash", "abc")
    await main.set_setting_value(db, "fest_nav_html", "<p>nav</p>")

    tg_pages = {"p1": {"html": "<p>start</p>", "title": "Fest1"}}

    class DummyTelegraph:
        def __init__(self, *_, **__):
            pass

        def get_page(self, path, return_html=True):
            return {"content_html": tg_pages[path]["html"], "title": tg_pages[path]["title"]}

        def edit_page(self, path, title, html_content, **kwargs):
            tg_pages[path] = {"html": html_content, "title": title}
            return {}

    async def fake_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_call)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    with caplog.at_level(logging.INFO):
        changed = await main.update_festival_tg_nav(-fid * main.FEST_JOB_MULT, db, None)
    assert changed
    h = content_hash("<p>nav</p>")
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "edited")
    assert rec.target == "tg"
    assert rec.path == "p1"
    assert rec.fest == "Fest1"
    assert rec.nav_old == ""
    assert rec.nav_new == h
    assert rec.removed_legacy_blocks == 0
    assert rec.legacy_markers_replaced is False


@pytest.mark.asyncio
async def test_update_festival_tg_nav_logs_skipped(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        fest = Festival(name="Fest1", telegraph_path="p1", start_date=today, end_date=today)
        session.add(fest)
        await session.commit()
        fid = fest.id
    await main.set_setting_value(db, "fest_nav_hash", "abc")
    nav_html = "<p>nav</p>"
    await main.set_setting_value(db, "fest_nav_html", nav_html)

    existing_html, *_ = main.apply_festival_nav("<p>start</p>", nav_html)
    tg_pages = {"p1": {"html": existing_html, "title": "Fest1"}}

    class DummyTelegraph:
        def __init__(self, *_, **__):
            pass

        def get_page(self, path, return_html=True):
            return {"content_html": tg_pages[path]["html"], "title": tg_pages[path]["title"]}

        def edit_page(self, path, title, html_content, **kwargs):
            tg_pages[path] = {"html": html_content, "title": title}
            return {}

    async def fake_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "Telegraph", DummyTelegraph)
    monkeypatch.setattr(main, "telegraph_call", fake_call)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    with caplog.at_level(logging.INFO):
        changed = await main.update_festival_tg_nav(-fid * main.FEST_JOB_MULT, db, None)
    assert not changed
    h = content_hash(nav_html)
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "skipped_nochange")
    assert rec.target == "tg"
    assert rec.path == "p1"
    assert rec.fest == "Fest1"
    assert rec.nav_old == h
    assert rec.nav_new == h
    assert rec.removed_legacy_blocks == 0
    assert rec.legacy_markers_replaced is False


@pytest.mark.asyncio
async def test_update_all_festival_nav_skips_same_hash_logging(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        session.add(
            Festival(
                name="Fest1",
                telegraph_path="p1",
                vk_post_url="u1",
                start_date=today,
                end_date=today,
                nav_hash="abc",
            )
        )
        await session.commit()
    await main.set_setting_value(db, "fest_nav_hash", "abc")

    calls = {"tg": 0, "vk": 0}

    async def fake_tg(eid, db, bot):
        calls["tg"] += 1
        return True

    async def fake_vk(eid, db, bot):
        calls["vk"] += 1
        return True

    monkeypatch.setattr(main, "update_festival_tg_nav", fake_tg)
    monkeypatch.setattr(main, "update_festival_vk_nav", fake_vk)

    with caplog.at_level(logging.INFO):
        changed = await main.update_all_festival_nav(0, db, None)
    assert not changed
    assert calls["tg"] == 0
    assert calls["vk"] == 0
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "skipped_same_hash")
    assert rec.fest == "Fest1"
    rec2 = next(r for r in caplog.records if getattr(r, "action", None) == "done")
    assert rec2.changed is False


@pytest.mark.asyncio
async def test_sync_festival_vk_post_nav_only_no_change(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest1",
            vk_post_url="https://vk.com/wall-1_1",
            start_date=today,
            end_date=today,
        )
        session.add(fest)
        await session.commit()

    monkeypatch.setenv("VK_USER_TOKEN", "tok")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)

    async def fake_group_id(db):
        return "1"

    monkeypatch.setattr(main, "get_vk_group_id", fake_group_id)

    async def fake_nav_block(db, exclude=None, today=None, items=None):
        return [], [main.VK_BLANK_LINE, "Ближайшие фестивали", "nav"]

    async def fake_vk_api(method, params, db, bot, token=None, token_kind=None):
        text = f"base\n{main.VK_BLANK_LINE}\nБлижайшие фестивали\nnav"
        return {"response": [{"text": text}]}

    edit_called = {"called": False}

    async def fake_edit(url, message, db, bot, attachments):
        edit_called["called"] = True

    async def fake_post(*args, **kwargs):
        assert False, "should not post"

    monkeypatch.setattr(main, "_build_festival_nav_block", fake_nav_block)
    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    monkeypatch.setattr(main, "edit_vk_post", fake_edit)
    monkeypatch.setattr(main, "post_to_vk", fake_post)

    with caplog.at_level(logging.INFO):
        res = await main.sync_festival_vk_post(db, "Fest1", nav_only=True)
    assert res is False
    assert not edit_called["called"]
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "skipped_nochange")
    assert rec.target == "vk"
    assert rec.url == "https://vk.com/wall-1_1"
    assert rec.fest == "Fest1"


@pytest.mark.asyncio
async def test_festivals_fix_nav_force(tmp_path: Path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        f1 = Festival(name="Fest1", start_date=today, end_date=today, nav_hash="abc")
        f2 = Festival(name="Fest2", start_date=today, end_date=today, nav_hash="def")
        f3 = Festival(name="Fest3", telegraph_path="f3", nav_hash="ghi")
        session.add_all([f1, f2, f3])
        await session.commit()
    await main.set_setting_value(db, "fest_nav_hash", "abc")
    calls = {"tg": 0, "vk": 0}

    async def fake_tg(eid, db_obj, bot_obj):
        calls["tg"] += 1
        return True

    async def fake_vk(eid, db_obj, bot_obj):
        calls["vk"] += 1
        return True

    monkeypatch.setattr(main, "update_festival_tg_nav", fake_tg)
    monkeypatch.setattr(main, "update_festival_vk_nav", fake_vk)

    with caplog.at_level(logging.INFO):
        pages, changed, dup = await main.festivals_fix_nav(db, None)

    assert pages == 3
    assert changed == 6
    assert dup == 1
    assert calls["tg"] == 3
    assert calls["vk"] == 3
    rec_start = next(
        r
        for r in caplog.records
        if r.message == "fest_nav_force_rebuild" and getattr(r, "action", None) == "start"
    )
    rec_finish = next(
        r
        for r in caplog.records
        if r.message == "fest_nav_force_rebuild" and getattr(r, "action", None) == "finish"
    )
    assert rec_finish.pages == 3
    assert rec_finish.duplicates_removed == 1


@pytest.mark.asyncio
async def test_sync_festival_vk_post_nav_only_edit_success(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest1",
            vk_post_url="https://vk.com/wall-1_1",
            start_date=today,
            end_date=today,
        )
        session.add(fest)
        await session.commit()

    monkeypatch.setenv("VK_USER_TOKEN", "tok")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)

    async def fake_group_id(db):
        return "1"

    monkeypatch.setattr(main, "get_vk_group_id", fake_group_id)

    async def fake_nav_block(db, exclude=None, today=None, items=None):
        return [], ["nav"]

    async def fake_vk_api(method, params, db, bot, token=None, token_kind=None):
        return {"response": [{"text": "base"}]}

    edit_called = {"called": False}

    async def fake_edit(url, message, db, bot, attachments):
        edit_called["called"] = True

    async def fake_post(*args, **kwargs):
        assert False, "should not post"

    monkeypatch.setattr(main, "_build_festival_nav_block", fake_nav_block)
    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    monkeypatch.setattr(main, "edit_vk_post", fake_edit)
    monkeypatch.setattr(main, "post_to_vk", fake_post)

    with caplog.at_level(logging.INFO):
        res = await main.sync_festival_vk_post(db, "Fest1", nav_only=True)
    assert res is True
    assert edit_called["called"]
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "edited")
    assert rec.target == "vk"
    assert rec.url == "https://vk.com/wall-1_1"
    assert rec.fest == "Fest1"


@pytest.mark.asyncio
async def test_sync_festival_vk_post_nav_only_skip_edit(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    today = date.today().isoformat()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest1",
            vk_post_url="https://vk.com/wall-1_1",
            start_date=today,
            end_date=today,
        )
        session.add(fest)
        await session.commit()

    monkeypatch.setenv("VK_USER_TOKEN", "tok")
    monkeypatch.setenv("VK_NAV_FALLBACK", "skip")
    monkeypatch.setattr(main, "_vk_user_token_bad", None)

    async def fake_group_id(db):
        return "1"

    monkeypatch.setattr(main, "get_vk_group_id", fake_group_id)

    async def fake_nav_block(db, exclude=None, today=None, items=None):
        return [], ["nav"]

    async def fake_vk_api(method, params, db, bot, token=None, token_kind=None):
        return {"response": [{"text": "base"}]}

    async def fake_edit(url, message, db, bot, attachments):
        raise main.VKAPIError(214, "edit time expired")

    async def fake_post(*args, **kwargs):
        assert False, "should not post"

    monkeypatch.setattr(main, "_build_festival_nav_block", fake_nav_block)
    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    monkeypatch.setattr(main, "edit_vk_post", fake_edit)
    monkeypatch.setattr(main, "post_to_vk", fake_post)

    with caplog.at_level(logging.INFO):
        res = await main.sync_festival_vk_post(db, "Fest1", nav_only=True)
    assert res is False
    rec = next(r for r in caplog.records if getattr(r, "action", None) == "vk_nav_skip_edit")
    assert rec.target == "vk"
    assert rec.url == "https://vk.com/wall-1_1"
    assert rec.fest == "Fest1"
    os.environ.pop("VK_NAV_FALLBACK", None)

