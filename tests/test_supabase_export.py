import sys
import types

import pytest

import main
from supabase_export import SBExporter


def _clear_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_EXPORT_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("VK_MISSES_SAMPLE_RATE", raising=False)


def test_supabase_export_defaults_enabled(monkeypatch):
    _clear_env(monkeypatch)

    exporter = SBExporter(lambda: object())

    assert exporter.enabled is True
    assert exporter._retention_days == 60
    assert exporter._miss_sample_rate == 0.1


def test_supabase_export_disable_via_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_EXPORT_ENABLED", "0")

    exporter = SBExporter(lambda: object())

    assert exporter.enabled is False


class _DummyTable:
    def __init__(self, client):
        self._client = client

    def insert(self, payload):
        self._client.last_payload = payload
        return self

    def upsert(self, payload, on_conflict=None):
        self._client.last_payload = payload
        self._client.last_on_conflict = on_conflict
        return self

    def execute(self):
        return self


class _DummyClient:
    def __init__(self):
        self.table_name = None
        self.last_payload = None
        self.last_on_conflict = None

    def table(self, name):
        self.table_name = name
        return _DummyTable(self)


def test_write_snapshot_includes_expected_counters(monkeypatch):
    _clear_env(monkeypatch)

    client = _DummyClient()
    exporter = SBExporter(lambda: client)

    exporter.write_snapshot(
        group_id=123,
        group_title="Example",
        group_screen_name="example",
        ts=1700000000,
        match_rate=0.25,
        errors=3,
        counters={
            "posts_scanned": 42,
            "matched": 7,
            "duplicates": 2,
            "pages_loaded": 5,
            "ignored": None,
        },
    )

    assert client.table_name == "vk_crawl_snapshots"
    payload = client.last_payload
    assert payload is not None
    assert payload["group_id"] == 123
    assert payload["group_title"] == "Example"
    assert payload["group_screen_name"] == "example"
    assert payload["ts"] == "2023-11-14T22:13:20+00:00"
    assert payload["match_rate"] == 0.25
    assert payload["errors"] == 3
    assert payload["posts_scanned"] == 42
    assert payload["matched"] == 7
    assert payload["duplicates"] == 2
    assert payload["pages_loaded"] == 5
    assert "ignored" not in payload


def test_log_miss_includes_expected_fields(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("VK_MISSES_SAMPLE_RATE", "1")

    client = _DummyClient()
    exporter = SBExporter(lambda: client)

    exporter.log_miss(
        group_id=99,
        post_id=123,
        url="https://vk.com/wall99_123",
        reason="no_date",
        matched_kw=["music", "festival"],
        ts=1700000100,
        kw_ok=True,
        has_date=False,
        group_title="Example Club",
        group_screen_name="exampleclub",
    )

    assert client.table_name == "vk_misses_sample"
    payload = client.last_payload
    assert payload is not None
    assert payload["group_title"] == "Example Club"
    assert payload["group_screen_name"] == "exampleclub"
    assert payload["matched_kw"] == ["music", "festival"]
    assert payload["kw_ok"] is True
    assert payload["has_date"] is False
    assert payload["ts"] == "2023-11-14T22:15:00+00:00"
    assert "post_ts" not in payload
    assert "event_ts_hint" not in payload
    assert "flags" not in payload


def test_get_supabase_client_normalizes_url(monkeypatch):
    monkeypatch.setattr(main, "SUPABASE_URL", "https://proj.supabase.co/rest/v1")
    monkeypatch.setattr(main, "SUPABASE_KEY", "secret")
    monkeypatch.setattr(main, "_supabase_client", None)
    monkeypatch.setattr(main, "_normalized_supabase_url", None)
    monkeypatch.setattr(main, "_normalized_supabase_url_source", None)
    monkeypatch.setattr(main.atexit, "register", lambda func: None)

    class DummyHttpxClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(main.httpx, "Client", DummyHttpxClient)

    fake_client = object()

    def fake_create_client(url, key, options):
        assert url == "https://proj.supabase.co"
        assert key == "secret"
        assert isinstance(options.httpx_client, DummyHttpxClient)
        return fake_client

    class FakeClientOptions:
        def __init__(self):
            self.httpx_client = None

    supabase_module = types.ModuleType("supabase")
    supabase_module.create_client = fake_create_client
    supabase_module.Client = object
    supabase_client_module = types.ModuleType("supabase.client")
    supabase_client_module.ClientOptions = FakeClientOptions

    monkeypatch.setitem(sys.modules, "supabase", supabase_module)
    monkeypatch.setitem(sys.modules, "supabase.client", supabase_client_module)

    try:
        client = main.get_supabase_client()
        assert client is fake_client
        assert main._normalized_supabase_url == "https://proj.supabase.co"
        assert (
            main._normalized_supabase_url_source
            == "https://proj.supabase.co/rest/v1"
        )
    finally:
        main._supabase_client = None


class _DummyResponse:
    def __init__(self, url):
        self._url = url

    def raise_for_status(self):
        pass

    def json(self):
        return []


class _DummyAsyncClient:
    instance: "_DummyAsyncClient | None" = None

    def __init__(self, *args, **kwargs):
        _DummyAsyncClient.instance = self
        self.calls: list[tuple[str, dict | None, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - unused
        return False

    async def get(self, url, headers=None, params=None):
        self.calls.append((url, headers, params))
        return _DummyResponse(url)


@pytest.mark.asyncio
async def test_fetch_vk_import_view_uses_normalized_base(monkeypatch):
    monkeypatch.setattr(main, "SUPABASE_URL", "https://proj.supabase.co/storage/v1")
    monkeypatch.setattr(main, "SUPABASE_KEY", "test-key")
    monkeypatch.setattr(main, "_supabase_client", None)
    monkeypatch.setattr(main, "_normalized_supabase_url", None)
    monkeypatch.setattr(main, "_normalized_supabase_url_source", None)
    monkeypatch.delenv("SUPABASE_DISABLED", raising=False)
    monkeypatch.setattr(main.httpx, "AsyncClient", _DummyAsyncClient)

    result = await main._fetch_vk_import_view(
        "vk_import_by_group",
        days=7,
        client=None,
    )

    assert result == []
    instance = _DummyAsyncClient.instance
    assert instance is not None
    assert instance.calls, "HTTP client was not invoked"
    url, headers, params = instance.calls[0]
    assert url == "https://proj.supabase.co/rest/v1/vk_import_by_group"
    assert headers == {
        "apikey": "test-key",
        "Authorization": "Bearer test-key",
    }
    assert params is not None
    assert params["select"] == "*"
