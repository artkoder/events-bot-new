import pytest
from datetime import date
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
    label = main.format_weekend_range(date(2025, 9, 6))
    assert "❌ 2025-09 — ошибка: boom" in report
    assert "❌ 2025-09 — ошибка:" in report
    assert f"• {label}: ❌ fail" in report
    assert "без изменений" not in report
