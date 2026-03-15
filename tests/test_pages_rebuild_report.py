import pytest
from datetime import date
import main
import main_part2


def test_weekends_for_months_helper_has_defaultdict_import():
    weekends, mapping = main_part2._weekends_for_months(["2025-09"])
    assert weekends == ["2025-09-06", "2025-09-13", "2025-09-20", "2025-09-27"]
    assert mapping["2025-09"] == weekends


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
