import hashlib
import os
from dataclasses import dataclass

import pytest

from db import Database
from models import OcrUsage, PosterOcrCache
import poster_ocr
import vision_test.ocr
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
async def test_recognize_posters_limit_exhausted(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    call_count = 0

    async def fake_run_ocr(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise AssertionError("run_ocr should not be called when limit is exhausted")

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-03")

    async with db.get_session() as session:
        session.add(OcrUsage(date="2024-06-03", spent_tokens=poster_ocr.DAILY_TOKEN_LIMIT))
        await session.commit()

    item = DummyPoster(b"exceed")

    with pytest.raises(poster_ocr.PosterOcrLimitExceededError) as excinfo:
        await poster_ocr.recognize_posters(db, [item])

    assert excinfo.value.spent_tokens == 0
    assert excinfo.value.remaining == 0
    assert call_count == 0

    async with db.get_session() as session:
        usage_row = await session.get(OcrUsage, "2024-06-03")

    assert usage_row is not None
    assert usage_row.spent_tokens == poster_ocr.DAILY_TOKEN_LIMIT


@pytest.mark.asyncio
async def test_recognize_posters_stops_after_reaching_limit(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    call_count = 0

    async def fake_run_ocr(data, *, model, detail):
        nonlocal call_count
        call_count += 1
        return OcrResult(
            text=f"text{call_count}",
            usage=OcrUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=80),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-04")

    async with db.get_session() as session:
        session.add(OcrUsage(date="2024-06-04", spent_tokens=poster_ocr.DAILY_TOKEN_LIMIT - 50))
        await session.commit()

    items = [DummyPoster(b"one"), DummyPoster(b"two")]
    digests = [hashlib.sha256(item.data).hexdigest() for item in items]

    results, spent_tokens, remaining = await poster_ocr.recognize_posters(db, items)

    assert call_count == 1
    assert len(results) == 1
    assert spent_tokens == 50
    assert remaining == 0

    async with db.get_session() as session:
        usage_row = await session.get(OcrUsage, "2024-06-04")
        model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")
        cached_first = await session.get(
            PosterOcrCache, (digests[0], "auto", model)
        )
        cached_second = await session.get(
            PosterOcrCache, (digests[1], "auto", model)
        )

    assert usage_row is not None
    assert usage_row.spent_tokens == poster_ocr.DAILY_TOKEN_LIMIT
    assert cached_first is not None
    assert cached_second is None


@pytest.mark.asyncio
async def test_recognize_posters_configures_http(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    vision_test.ocr.clear_http()
    monkeypatch.setattr(poster_ocr, "_HTTP_CONFIGURED", False)

    async def fake_run_ocr(data, *, model, detail):
        return OcrResult(
            text="configured",
            usage=OcrUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)

    result, spent, remaining = await poster_ocr.recognize_posters(
        db, [DummyPoster(b"payload")]
    )

    assert result[0].text == "configured"
    assert spent == 0
    assert remaining == poster_ocr.DAILY_TOKEN_LIMIT


@pytest.mark.asyncio
async def test_recognize_posters_detail_changes(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    call_details: list[str] = []

    async def fake_run_ocr(data, *, model, detail):
        call_details.append(detail)
        return OcrResult(
            text=f"{detail}-text",
            usage=OcrUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=5),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)

    item = DummyPoster(b"payload")
    digest = hashlib.sha256(item.data).hexdigest()
    model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")

    result_auto, spent_auto, remaining_auto = await poster_ocr.recognize_posters(
        db, [item], detail="auto"
    )
    result_high, spent_high, remaining_high = await poster_ocr.recognize_posters(
        db, [item], detail="high"
    )
    result_high_cached, spent_high_cached, remaining_high_cached = (
        await poster_ocr.recognize_posters(db, [item], detail="high")
    )

    assert call_details == ["auto", "high"]
    assert result_auto[0].text == "auto-text"
    assert result_high[0].text == "high-text"
    assert result_high_cached[0].text == "high-text"
    assert spent_auto == 5
    assert spent_high == 5
    assert spent_high_cached == 0
    assert remaining_high_cached == remaining_high

    async with db.get_session() as session:
        cached_auto = await session.get(PosterOcrCache, (digest, "auto", model))
        cached_high = await session.get(PosterOcrCache, (digest, "high", model))

    assert cached_auto is not None
    assert cached_high is not None
    assert cached_high.text == "high-text"
