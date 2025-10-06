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
