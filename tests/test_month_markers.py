import pytest
from datetime import date
from telegraph.utils import nodes_to_html

import main
from models import Event
from markup import DAY_START, DAY_END, PERM_START, PERM_END


@pytest.mark.asyncio
async def test_month_page_has_markers(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            Event(
                title="Event",
                description="d",
                source_text="s",
                date="2025-08-24",
                time="18:00",
                location_name="Hall",
            )
        )
        session.add(
            Event(
                title="Expo",
                description="d",
                source_text="s",
                date="2025-08-01",
                time="10:00",
                location_name="Hall",
                end_date="2025-08-30",
                event_type="выставка",
            )
        )
        await session.commit()

    _, content, _ = await main.build_month_page_content(db, "2025-08")
    html = main.unescape_html_comments(nodes_to_html(content))
    assert DAY_START("2025-08-24") in html
    assert DAY_END("2025-08-24") in html
    assert PERM_START in html and PERM_END in html
    assert "&lt;!--" not in html
