import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import main


@pytest.fixture(autouse=True)
def _mock_telegraph(monkeypatch, request):
    if "get_telegraph_token" not in request.node.nodeid:
        monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")
    async def fake_create_page(tg, *args, **kwargs):
        return {"path": "test", "url": "https://t.me/test"}
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
