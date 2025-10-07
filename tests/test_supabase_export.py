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


def test_log_miss_preserves_flags_and_keywords(monkeypatch):
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
        post_ts=1700000000,
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
