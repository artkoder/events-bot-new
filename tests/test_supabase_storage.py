from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from db import Database
from supabase_storage import (
    check_bucket_usage_limit,
    flush_supabase_delete_queue,
    get_bucket_usage_bytes,
    storage_object_exists_http,
)


class _FakeStorageBucket:
    def __init__(self, tree: dict[str, list[dict[str, Any]]], *, calls: dict[str, int]):
        self._tree = tree
        self._calls = calls

    def list(self, *, path: str, options: dict | None = None):
        self._calls["list"] = self._calls.get("list", 0) + 1
        rows = list(self._tree.get(path or "", []))
        limit = int((options or {}).get("limit", 1000))
        offset = int((options or {}).get("offset", 0))
        return rows[offset : offset + limit]

    def remove(self, paths: list[str]):
        self._calls["remove"] = self._calls.get("remove", 0) + 1
        # Remove is best-effort in production; for the test just accept.
        return [{"name": p} for p in paths]


class _FakeStorage:
    def __init__(self, tree: dict[str, list[dict[str, Any]]], *, calls: dict[str, int]):
        self._tree = tree
        self._calls = calls

    def from_(self, bucket: str):
        # Each bucket can have a different tree in real life; tests only need one.
        return _FakeStorageBucket(self._tree, calls=self._calls)


@dataclass
class _FakeSupabaseClient:
    storage: _FakeStorage


def test_bucket_usage_guard_computes_and_caches_usage():
    calls: dict[str, int] = {}
    mb = 1024 * 1024
    tree = {
        "": [
            {"name": "dir", "metadata": None, "id": None},
            {"name": "a.jpg", "metadata": {"size": 50 * mb}},
        ],
        "dir": [
            {"name": "b.mp4", "metadata": {"size": 100 * mb}},
        ],
    }
    client = _FakeSupabaseClient(storage=_FakeStorage(tree, calls=calls))

    used1, cached1 = get_bucket_usage_bytes(client, "bucket", cache_sec=3600, now_ts=1000.0)
    calls_after_first = dict(calls)
    used2, cached2 = get_bucket_usage_bytes(client, "bucket", cache_sec=3600, now_ts=1001.0)

    assert used1 == 150 * mb
    assert cached1 is False
    assert used2 == 150 * mb
    assert cached2 is True
    assert calls.get("list", 0) >= 2  # recursive listing hits multiple paths
    assert calls == calls_after_first  # cached => no extra list calls


def test_bucket_usage_guard_blocks_when_limit_exceeded():
    calls: dict[str, int] = {}
    mb = 1024 * 1024
    tree = {"": [{"name": "x.bin", "metadata": {"size": 120 * mb}}]}
    client = _FakeSupabaseClient(storage=_FakeStorage(tree, calls=calls))

    res = check_bucket_usage_limit(
        client,
        "bucket",
        additional_bytes=10 * mb,
        max_used_mb=100.0,
        cache_sec=0,
        on_error="deny",
    )
    assert res.ok is False
    assert res.used_mb is not None
    assert res.reason == "limit_exceeded"


@pytest.mark.asyncio
async def test_flush_supabase_delete_queue_deletes_rows(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    calls: dict[str, int] = {}
    client = _FakeSupabaseClient(storage=_FakeStorage(tree={}, calls=calls))

    async with db.raw_conn() as conn:
        await conn.executemany(
            "INSERT OR IGNORE INTO supabase_delete_queue(bucket, path) VALUES(?, ?)",
            [
                ("b1", "a/1.mp4"),
                ("b1", "a/2.mp4"),
                ("b2", "x/1.jpg"),
            ],
        )
        await conn.commit()

    removed = await flush_supabase_delete_queue(db, supabase_client=client, limit=100)
    assert removed == 3
    assert calls.get("remove", 0) >= 2  # grouped by bucket

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM supabase_delete_queue")
        (count,) = await cur.fetchone()
    assert int(count) == 0


@pytest.mark.asyncio
async def test_flush_supabase_delete_queue_keeps_rows_on_failure(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class _FailingBucket(_FakeStorageBucket):
        def remove(self, paths: list[str]):
            self._calls["remove"] = self._calls.get("remove", 0) + 1
            raise RuntimeError("boom")

    class _FailingStorage(_FakeStorage):
        def from_(self, bucket: str):
            return _FailingBucket(tree={}, calls=self._calls)

    calls: dict[str, int] = {}
    client = _FakeSupabaseClient(storage=_FailingStorage(tree={}, calls=calls))

    async with db.raw_conn() as conn:
        await conn.executemany(
            "INSERT OR IGNORE INTO supabase_delete_queue(bucket, path) VALUES(?, ?)",
            [
                ("b1", "a/1.mp4"),
                ("b1", "a/2.mp4"),
            ],
        )
        await conn.commit()

    removed = await flush_supabase_delete_queue(db, supabase_client=client, limit=100)
    assert removed == 0
    assert calls.get("remove", 0) >= 1

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT attempts, last_error FROM supabase_delete_queue ORDER BY id ASC"
        )
        rows = await cur.fetchall()
    assert len(rows) == 2
    assert all(int(r[0]) >= 1 for r in rows)
    assert all((r[1] or "").strip() for r in rows)


def test_storage_object_exists_http_returns_none_on_missing_config():
    assert (
        storage_object_exists_http(
            supabase_url=None,
            supabase_key="k",
            bucket="b",
            object_path="x",
        )
        is None
    )


def test_storage_object_exists_http_head_true(monkeypatch):
    import requests

    class _Resp:
        status_code = 200

    def _head(*_args, **_kwargs):
        return _Resp()

    monkeypatch.setattr(requests, "head", _head)

    assert (
        storage_object_exists_http(
            supabase_url="https://supa.example",
            supabase_key="k",
            bucket="b",
            object_path="p/1.webp",
        )
        is True
    )


def test_storage_object_exists_http_head_false(monkeypatch):
    import requests

    class _Resp:
        status_code = 404

    def _head(*_args, **_kwargs):
        return _Resp()

    monkeypatch.setattr(requests, "head", _head)

    assert (
        storage_object_exists_http(
            supabase_url="https://supa.example",
            supabase_key="k",
            bucket="b",
            object_path="p/missing.webp",
        )
        is False
    )


def test_storage_object_exists_http_head_fallbacks_to_ranged_get(monkeypatch):
    import requests

    class _HeadResp:
        status_code = 405

    class _GetResp:
        status_code = 206

    def _head(*_args, **_kwargs):
        return _HeadResp()

    def _get(*_args, **_kwargs):
        return _GetResp()

    monkeypatch.setattr(requests, "head", _head)
    monkeypatch.setattr(requests, "get", _get)

    assert (
        storage_object_exists_http(
            supabase_url="https://supa.example",
            supabase_key="k",
            bucket="b",
            object_path="p/1.webp",
        )
        is True
    )
