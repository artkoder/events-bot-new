import pytest


@pytest.mark.asyncio
async def test_vk_auto_queue_rate_limit_deferred_to_pending(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from db import Database
    from google_ai.exceptions import RateLimitError
    import vk_auto_queue

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    inbox_id = 1
    group_id = 123
    post_id = 456

    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, locked_by, locked_at, review_batch)
            VALUES(?, ?, ?, ?, ?, NULL, 1, 'locked', 777, CURRENT_TIMESTAMP, 'batch-x')
            """,
            (inbox_id, group_id, post_id, 0, "text"),
        )
        await conn.commit()

    async def fake_fetch_vk_post_text_and_photos(_group_id, _post_id, *, db, bot):  # noqa: ARG001
        return "text", [], None, {"views": 10, "likes": 1}, vk_auto_queue.VkFetchStatus(True, "ok")

    async def fake_build_event_drafts(*_args, **_kwargs):
        raise RateLimitError(blocked_reason="tpm", retry_after_ms=3000)

    async def noop_sleep(_sec):  # noqa: ARG001
        return None

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch_vk_post_text_and_photos)
    monkeypatch.setattr(vk_auto_queue.vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_auto_queue.asyncio, "sleep", noop_sleep)
    monkeypatch.setenv("VK_AUTO_IMPORT_RATE_LIMIT_MAX_WAIT_SEC", "0")

    report = vk_auto_queue.VkAutoImportReport(batch_id="batch-x")
    post = SimpleNamespace(
        id=inbox_id,
        group_id=group_id,
        post_id=post_id,
        date=0,
        text="text",
        event_ts_hint=None,
    )

    class DummyBot:
        pass

    await vk_auto_queue._process_vk_inbox_row(  # type: ignore[attr-defined]
        db,
        DummyBot(),
        chat_id=1,
        operator_id=1,
        batch_id="batch-x",
        post=post,
        source_url="https://vk.com/wall-123_456",
        report=report,
        festival_names=None,
        festival_alias_pairs=None,
        progress_message_id=None,
        progress_current_no=1,
        progress_total_txt="1",
    )

    assert report.inbox_failed == 0
    assert report.inbox_deferred == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, locked_by, locked_at, review_batch FROM vk_inbox WHERE id=?",
            (inbox_id,),
        )
        row = await cur.fetchone()
    assert row[0] == "pending"
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None


@pytest.mark.asyncio
async def test_vk_auto_queue_rate_limit_waits_then_continues(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from db import Database
    from google_ai.exceptions import RateLimitError
    import vk_auto_queue

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    inbox_id = 1
    group_id = 123
    post_id = 456

    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, locked_by, locked_at, review_batch)
            VALUES(?, ?, ?, ?, ?, NULL, 1, 'locked', 777, CURRENT_TIMESTAMP, 'batch-x')
            """,
            (inbox_id, group_id, post_id, 0, "text"),
        )
        await conn.commit()

    async def fake_fetch_vk_post_text_and_photos(_group_id, _post_id, *, db, bot):  # noqa: ARG001
        return "text", [], None, {"views": 10, "likes": 1}, vk_auto_queue.VkFetchStatus(True, "ok")

    calls = {"n": 0}

    async def fake_build_event_drafts(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError(blocked_reason="tpm", retry_after_ms=10)
        return [], None

    async def noop_sleep(_sec):  # noqa: ARG001
        return None

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch_vk_post_text_and_photos)
    monkeypatch.setattr(vk_auto_queue.vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_auto_queue.asyncio, "sleep", noop_sleep)
    monkeypatch.setenv("VK_AUTO_IMPORT_RATE_LIMIT_MAX_WAIT_SEC", "5")

    report = vk_auto_queue.VkAutoImportReport(batch_id="batch-x")
    post = SimpleNamespace(
        id=inbox_id,
        group_id=group_id,
        post_id=post_id,
        date=0,
        text="text",
        event_ts_hint=None,
    )

    class DummyBot:
        pass

    await vk_auto_queue._process_vk_inbox_row(  # type: ignore[attr-defined]
        db,
        DummyBot(),
        chat_id=1,
        operator_id=1,
        batch_id="batch-x",
        post=post,
        source_url="https://vk.com/wall-123_456",
        report=report,
        festival_names=None,
        festival_alias_pairs=None,
        progress_message_id=None,
        progress_current_no=1,
        progress_total_txt="1",
    )

    assert calls["n"] >= 2
    assert report.inbox_deferred == 0
    assert report.inbox_failed == 0

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE id=?", (inbox_id,))
        row = await cur.fetchone()
    assert row[0] in {"rejected", "imported", "failed", "pending"}


@pytest.mark.asyncio
async def test_vk_auto_queue_disables_inner_event_parse_wait(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from db import Database
    from google_ai.exceptions import RateLimitError
    import vk_auto_queue

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    inbox_id = 1
    group_id = 123
    post_id = 456

    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, locked_by, locked_at, review_batch)
            VALUES(?, ?, ?, ?, ?, NULL, 1, 'locked', 777, CURRENT_TIMESTAMP, 'batch-x')
            """,
            (inbox_id, group_id, post_id, 0, "text"),
        )
        await conn.commit()

    async def fake_fetch_vk_post_text_and_photos(_group_id, _post_id, *, db, bot):  # noqa: ARG001
        return "text", [], None, {"views": 10, "likes": 1}, vk_auto_queue.VkFetchStatus(True, "ok")

    seen = {"rate_limit_max_wait_sec": None}

    async def fake_build_event_drafts(*_args, **kwargs):
        seen["rate_limit_max_wait_sec"] = kwargs.get("rate_limit_max_wait_sec")
        raise RateLimitError(blocked_reason="tpm", retry_after_ms=3000)

    async def noop_sleep(_sec):  # noqa: ARG001
        return None

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch_vk_post_text_and_photos)
    monkeypatch.setattr(vk_auto_queue.vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_auto_queue.asyncio, "sleep", noop_sleep)
    monkeypatch.setenv("VK_AUTO_IMPORT_RATE_LIMIT_MAX_WAIT_SEC", "0")

    report = vk_auto_queue.VkAutoImportReport(batch_id="batch-x")
    post = SimpleNamespace(
        id=inbox_id,
        group_id=group_id,
        post_id=post_id,
        date=0,
        text="text",
        event_ts_hint=None,
    )

    class DummyBot:
        pass

    await vk_auto_queue._process_vk_inbox_row(  # type: ignore[attr-defined]
        db,
        DummyBot(),
        chat_id=1,
        operator_id=1,
        batch_id="batch-x",
        post=post,
        source_url="https://vk.com/wall-123_456",
        report=report,
        festival_names=None,
        festival_alias_pairs=None,
        progress_message_id=None,
        progress_current_no=1,
        progress_total_txt="1",
    )

    assert seen["rate_limit_max_wait_sec"] == 0
