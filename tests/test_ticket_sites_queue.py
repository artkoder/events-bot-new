import pytest

from db import Database
from models import TicketSiteQueueItem
from ticket_sites_queue import enqueue_ticket_site_urls, extract_ticket_site_urls
from sqlalchemy import select


def test_extract_ticket_site_urls_pyramida():
    text = "Билеты: https://pyramida.info/tickets/kino-tribyut_55746464."
    urls = extract_ticket_site_urls(text=text, links_payload=None, events_payload=None)
    assert urls == ["https://pyramida.info/tickets/kino-tribyut_55746464"]


@pytest.mark.asyncio
async def test_enqueue_ticket_site_urls_idempotent(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    url = "https://pyramida.info/tickets/kino-tribyut_55746464"
    n1 = await enqueue_ticket_site_urls(
        db,
        urls=[url],
        event_id=123,
        source_post_url="https://t.me/meowafisha/6746",
        source_chat_username="meowafisha",
        source_message_id=6746,
    )
    assert n1 == 1

    n2 = await enqueue_ticket_site_urls(
        db,
        urls=[url],
        event_id=456,
        source_post_url="https://t.me/meowafisha/6746",
        source_chat_username="meowafisha",
        source_message_id=6746,
    )
    assert n2 == 1

    async with db.get_session() as session:
        rows = (await session.execute(select(TicketSiteQueueItem))).scalars().all()
    assert len(rows) == 1
    assert rows[0].url == url
    # First event_id should be kept (do not overwrite on re-enqueue).
    assert rows[0].event_id == 123
