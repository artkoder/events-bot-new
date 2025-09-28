import pytest
from aiohttp.test_utils import make_mocked_request

import vk_intake
import main


@pytest.mark.asyncio
async def test_vk_intake_processing_time_metric(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    vk_intake.processing_time_seconds_total = 0.0

    async def fake_build(text, **kwargs):
        assert kwargs.get("festival_names") == []
        assert kwargs.get("festival_alias_pairs") is None
        assert kwargs.get("festival_hint") is False
        return [vk_intake.EventDraft(title="T")]

    async def fake_persist(draft, photos, db):
        return vk_intake.PersistResult(
            event_id=1,
            telegraph_url="t",
            ics_supabase_url="s",
            ics_tg_url="tg",
            event_date="2025-01-01",
            event_end_date=None,
            event_time="10:00",
            event_type=None,
            is_free=False,
        )

    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)

    times = iter([1.0, 2.0])
    monkeypatch.setattr(vk_intake.time, "perf_counter", lambda: next(times))

    results = await vk_intake.process_event("text", photos=[], db=db)

    assert len(results) == 1

    assert vk_intake.processing_time_seconds_total == pytest.approx(1.0)

    req = make_mocked_request("GET", "/metrics")
    resp = await main.metrics_handler(req)
    assert (
        "vk_intake_processing_time_seconds_total 1" in resp.text
    )
