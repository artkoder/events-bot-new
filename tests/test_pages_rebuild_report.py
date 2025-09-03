import pytest
import main


@pytest.mark.asyncio
async def test_pages_rebuild_report_errors(monkeypatch):
    async def fake_rebuild_pages(db, months, weekends, force=False):
        return {
            "months": {"updated": {}, "failed": {"2025-09": "boom"}},
            "weekends": {"updated": {}, "failed": {"2025-09-06": "fail"}},
        }

    monkeypatch.setattr(main, "rebuild_pages", fake_rebuild_pages)
    monkeypatch.setattr(
        main,
        "_weekends_for_months",
        lambda months: (['2025-09-06'], {"2025-09": ['2025-09-06']})
    )
    report = await main._perform_pages_rebuild(None, ["2025-09"], force=True)
    assert "❌ 2025-09 — ошибка: boom" in report
    assert "❌ 2025-09 — ошибка: 2025-09-06: fail" in report
