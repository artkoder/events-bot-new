import hashlib
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
import main
import poster_ocr
from models import PosterOcrCache
from _helpers.no_network import no_network  # noqa: F401


@pytest.fixture(autouse=True)
def _reset_run_due_jobs_lock():
    main._reset_run_due_jobs_locks()


@pytest.fixture(autouse=True)
def _mock_telegraph(monkeypatch, request):
    if "get_telegraph_token" not in request.node.nodeid:
        monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")
    async def fake_create_page(tg, *args, **kwargs):
        return {"path": "test", "url": "https://t.me/test"}
    # Skip for new test that needs real implementation to verify calls
    if "test_split_month_requires_many_pages" not in request.node.nodeid:
        monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)

    async def fake_update(event_id, db_obj, bot_obj):
        async with db_obj.get_session() as session:
            ev = await session.get(main.Event, event_id)
        if not ev:
            return None
        res = await main.create_source_page(
            ev.title or "Event",
            ev.source_text,
            ev.source_post_url,
            db=db_obj,
        )
        if res:
            url, path, *_ = res
            async with db_obj.get_session() as session:
                obj = await session.get(main.Event, event_id)
                if obj:
                    obj.telegraph_url = url
                    obj.telegraph_path = path
                    session.add(obj)
                    await session.commit()
            return url
        return None
    monkeypatch.setattr(main, "update_telegraph_event_page", fake_update)
    monkeypatch.setattr(main, "update_source_post_keyboard", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _mock_poster_ocr(monkeypatch, request):
    if "test_poster_ocr" in str(getattr(request.node, "fspath", "")):
        return

    async def fake_recognize(
        db,
        items,
        detail="auto",
        *,
        count_usage=True,
        log_context=None,
        **kwargs,
    ):
        results = []
        for item in items:
            if isinstance(item, tuple) and item:
                data = item[0]
            elif isinstance(item, (bytes, bytearray, memoryview)):
                data = bytes(item)
            else:
                data = getattr(item, "data", b"")
            digest = hashlib.sha256(bytes(data)).hexdigest()
            results.append(
                PosterOcrCache(
                    hash=digest,
                    detail=detail,
                    model="mock",
                    text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
            )
        return results, 0, poster_ocr.DAILY_TOKEN_LIMIT

    monkeypatch.setattr(poster_ocr, "recognize_posters", fake_recognize)
