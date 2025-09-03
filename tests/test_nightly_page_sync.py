import pytest
from unittest.mock import AsyncMock

import pytest
import main


class DummyResult:
    def all(self):
        return [("2024-05-11",)]


class DummySession:
    async def execute(self, stmt):
        return DummyResult()

    async def get(self, model, key):
        return DummyPage()

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyPage:
    def __init__(self):
        self.content_hash = ""
        self.content_hash2 = None
        self.url = ""
        self.path = ""
        self.url2 = None
        self.path2 = None


class DummyDB:
    def get_session(self):
        return DummySession()


@pytest.mark.asyncio
async def test_nightly_page_sync_updates_links(monkeypatch):
    db = DummyDB()
    month_mock = AsyncMock()
    weekend_mock = AsyncMock()
    monkeypatch.setattr(main, "sync_month_page", month_mock)
    monkeypatch.setattr(main, "sync_weekend_page", weekend_mock)

    await main.nightly_page_sync(db)

    month_mock.assert_awaited_once_with(db, "2024-05", update_links=True)
    weekend_mock.assert_awaited_once_with(db, "2024-05-11", update_links=True, post_vk=False)
