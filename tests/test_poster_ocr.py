import hashlib
from dataclasses import dataclass

import pytest

from db import Database
from models import OcrUsage, PosterOcrCache
import poster_ocr
from vision_test.ocr import OcrResult, OcrUsage as OcrUsageStats


@dataclass
class DummyPoster:
    data: bytes


@pytest.mark.asyncio
async def test_recognize_posters_uses_cache(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    call_count = 0

    async def fake_run_ocr(data, *, model, detail):
        nonlocal call_count
        call_count += 1
        return OcrResult(
            text="hello",
            usage=OcrUsageStats(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)

    items = [DummyPoster(b"data")]
    result1, spent1, remaining1 = await poster_ocr.recognize_posters(db, items)
    result2, spent2, remaining2 = await poster_ocr.recognize_posters(db, items)

    assert call_count == 1
    assert result1[0].text == "hello"
    assert result2[0].text == "hello"
    assert spent1 == 3
    assert spent2 == 0
    assert remaining1 <= poster_ocr.DAILY_TOKEN_LIMIT
    assert remaining2 == remaining1


@pytest.mark.asyncio
async def test_recognize_posters_usage_resets_by_date(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_run_ocr(data, *, model, detail):
        return OcrResult(
            text="ok",
            usage=OcrUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=100),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-01")

    _, spent1, remaining1 = await poster_ocr.recognize_posters(db, [DummyPoster(b"one")])

    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-02")
    _, spent2, remaining2 = await poster_ocr.recognize_posters(db, [DummyPoster(b"two")])

    async with db.get_session() as session:
        usage_first = await session.get(OcrUsage, "2024-06-01")
        usage_second = await session.get(OcrUsage, "2024-06-02")

    assert usage_first is not None
    assert usage_second is not None
    assert usage_first.spent_tokens == 100
    assert usage_second.spent_tokens == 100
    assert spent1 == 100
    assert spent2 == 100
    assert remaining1 == poster_ocr.DAILY_TOKEN_LIMIT - 100
    assert remaining2 == poster_ocr.DAILY_TOKEN_LIMIT - 100


@pytest.mark.asyncio
async def test_recognize_posters_limit_exceeded(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_run_ocr(data, *, model, detail):
        return OcrResult(
            text="limit",
            usage=OcrUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=100),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-03")

    async with db.get_session() as session:
        session.add(OcrUsage(date="2024-06-03", spent_tokens=poster_ocr.DAILY_TOKEN_LIMIT - 50))
        await session.commit()

    item = DummyPoster(b"exceed")
    digest = hashlib.sha256(item.data).hexdigest()

    with pytest.raises(RuntimeError):
        await poster_ocr.recognize_posters(db, [item])

    async with db.get_session() as session:
        usage_row = await session.get(OcrUsage, "2024-06-03")
        cached = await session.get(PosterOcrCache, digest)

    assert usage_row is not None
    assert usage_row.spent_tokens == poster_ocr.DAILY_TOKEN_LIMIT - 50
    assert cached is None
