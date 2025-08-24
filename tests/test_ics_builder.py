from datetime import datetime, timezone
import hashlib
import uuid

import pytest

import main
from main import Database, Event


@pytest.mark.asyncio
async def test_build_ics_content_golden(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    event = Event(
        id=1,
        title="T",
        description="d",
        source_text="s",
        date="2025-07-18",
        time="19:00",
        location_name="Hall",
        city="Town",
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 8, 24, 19, 2, 46, tzinfo=timezone.utc)

    monkeypatch.setattr(main, "datetime", FixedDateTime)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID("12345678-1234-1234-1234-1234567890ab"))

    content = await main.build_ics_content(db, event)
    expected = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//events-bot//RU\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:PUBLISH\r\n"
        f"X-WR-CALNAME:{main.ICS_CALNAME}\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:12345678-1234-1234-1234-1234567890ab@1\r\n"
        "DTSTAMP:20250824T190246Z\r\n"
        "DTSTART:20250718T190000\r\n"
        "DTEND:20250718T200000\r\n"
        "SUMMARY:T в Hall\r\n"
        "DESCRIPTION:d\r\n"
        "LOCATION:Town\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    assert content == expected


@pytest.mark.asyncio
async def test_build_ics_content_folding(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    title = (
        "Очень длинный заголовок события с русским текстом 🚀 и ссылкой "
        "https://example.com/12345"
    )
    desc = "Описание 😊, со запятыми; и ссылкой https://example.com/67890"
    event = Event(
        id=1,
        title=title,
        description=desc,
        source_text="s",
        date="2025-07-18",
        time="10:00",
        location_name="Hall",
    )
    content = await main.build_ics_content(db, event)
    lines = content.split("\r\n")
    for line in lines:
        if line:
            assert len(line.encode("utf-8")) <= 75 or line.startswith(" ")
    assert "DESCRIPTION:Описание" in content
    assert "\," in content
    assert "\;" in content
    assert "\n" in content
    assert "🚀" in content
    assert "😊" in content


@pytest.mark.asyncio
async def test_build_ics_dtstart_dtend(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    event = Event(
        id=1,
        title="T",
        description="d",
        source_text="s",
        date="2025-07-18",
        time="19:30",
        location_name="Hall",
    )
    content = await main.build_ics_content(db, event)
    assert "DTSTART:20250718T193000" in content
    assert "DTEND:20250718T203000" in content


@pytest.mark.asyncio
async def test_hash_independent_from_uid_dtstamp(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    event = Event(
        id=1,
        title="T",
        description="d",
        source_text="s",
        date="2025-07-18",
        time="19:00",
        location_name="Hall",
    )

    class FixedDateTime1(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 1, 1, tzinfo=timezone.utc)

    class FixedDateTime2(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 1, 2, tzinfo=timezone.utc)

    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID("11111111-1111-1111-1111-111111111111"))
    monkeypatch.setattr(main, "datetime", FixedDateTime1)
    c1 = await main.build_ics_content(db, event)

    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID("22222222-2222-2222-2222-222222222222"))
    monkeypatch.setattr(main, "datetime", FixedDateTime2)
    c2 = await main.build_ics_content(db, event)

    def norm_hash(txt: str) -> str:
        src = "\r\n".join(
            l for l in txt.split("\r\n") if not l.startswith(("UID:", "DTSTAMP:"))
        ).encode("utf-8")
        return hashlib.sha256(src).hexdigest()

    assert norm_hash(c1) == norm_hash(c2)
