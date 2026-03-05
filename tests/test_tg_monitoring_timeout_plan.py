import source_parsing.telegram.service as tg_service


def test_dynamic_timeout_uses_source_baseline_and_safety(monkeypatch):
    monkeypatch.setattr(tg_service, "TIMEOUT_MODE", "dynamic")
    monkeypatch.setattr(tg_service, "TIMEOUT_MINUTES", 90)
    monkeypatch.setattr(tg_service, "TIMEOUT_BASE_MINUTES", 15)
    monkeypatch.setattr(tg_service, "TIMEOUT_PER_SOURCE_MINUTES", 3.64)
    monkeypatch.setattr(tg_service, "TIMEOUT_SAFETY_MULTIPLIER", 1.3)
    monkeypatch.setattr(tg_service, "TIMEOUT_MAX_MINUTES", 360)

    timeout = tg_service._compute_kaggle_poll_timeout_minutes(sources_count=29)  # noqa: SLF001
    assert timeout == 153


def test_dynamic_timeout_applies_floor_and_max(monkeypatch):
    monkeypatch.setattr(tg_service, "TIMEOUT_MODE", "dynamic")
    monkeypatch.setattr(tg_service, "TIMEOUT_MINUTES", 120)
    monkeypatch.setattr(tg_service, "TIMEOUT_BASE_MINUTES", 15)
    monkeypatch.setattr(tg_service, "TIMEOUT_PER_SOURCE_MINUTES", 3.64)
    monkeypatch.setattr(tg_service, "TIMEOUT_SAFETY_MULTIPLIER", 1.3)
    monkeypatch.setattr(tg_service, "TIMEOUT_MAX_MINUTES", 140)

    low_sources_timeout = tg_service._compute_kaggle_poll_timeout_minutes(sources_count=0)  # noqa: SLF001
    high_sources_timeout = tg_service._compute_kaggle_poll_timeout_minutes(sources_count=200)  # noqa: SLF001

    assert low_sources_timeout == 120
    assert high_sources_timeout == 140


def test_fixed_timeout_mode_ignores_dynamic_components(monkeypatch):
    monkeypatch.setattr(tg_service, "TIMEOUT_MODE", "fixed")
    monkeypatch.setattr(tg_service, "TIMEOUT_MINUTES", 77)
    monkeypatch.setattr(tg_service, "TIMEOUT_BASE_MINUTES", 15)
    monkeypatch.setattr(tg_service, "TIMEOUT_PER_SOURCE_MINUTES", 3.64)
    monkeypatch.setattr(tg_service, "TIMEOUT_SAFETY_MULTIPLIER", 1.3)
    monkeypatch.setattr(tg_service, "TIMEOUT_MAX_MINUTES", 360)

    timeout = tg_service._compute_kaggle_poll_timeout_minutes(sources_count=999)  # noqa: SLF001
    assert timeout == 77
