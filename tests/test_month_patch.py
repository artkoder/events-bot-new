import pytest
from datetime import date

import main
from markup import DAY_START, DAY_END, PERM_START, PERM_END


@pytest.mark.asyncio
async def test_patch_month_page_inserts_chronologically(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    # insert month page and sample event for 15th
    async with db.get_session() as session:
        session.add(main.MonthPage(month="2025-08", url="u", path="p"))
        session.add(
            main.Event(
                title="Concert",
                description="desc",
                date="2025-08-15",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    html = (
        "Intro"
        + DAY_START("2025-08-14")
        + "14"
        + DAY_END("2025-08-14")
        + DAY_START("2025-08-17")
        + "17"
        + DAY_END("2025-08-17")
        + PERM_START
        + "perm"
        + PERM_END
    )

    class FakeTelegraph:
        def __init__(self):
            self.edited_html = None

        def get_page(self, path, return_html=True):
            assert path == "p"
            return {"content_html": html, "title": "Title"}

        def edit_page(self, path, title, html_content):
            self.edited_html = html_content
            return {"path": path}

    tg = FakeTelegraph()
    changed = await main.patch_month_page_for_date(db, tg, "2025-08", date(2025, 8, 15))
    assert changed is True
    result = tg.edited_html
    # ensure new day inserted before 17th and before permanent section
    assert result.index(DAY_START("2025-08-14")) < result.index(DAY_START("2025-08-15")) < result.index(DAY_START("2025-08-17"))
    assert result.index(DAY_START("2025-08-15")) < result.index(PERM_START)
