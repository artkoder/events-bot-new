import asyncio
import hashlib
import os
from dataclasses import dataclass

import aiosqlite
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
async def test_poster_ocr_cache_migrates_to_composite_pk(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE posterocrcache(
                hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO posterocrcache (
                hash, text, prompt_tokens, completion_tokens, total_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("digest-old", "old text", 1, 2, 3, "2024-01-01 00:00:00"),
        )
        await conn.commit()

    db = Database(str(db_path))
    await db.init()

    async with db.raw_conn() as conn:
        cursor = await conn.execute("PRAGMA table_info('posterocrcache')")
        columns = await cursor.fetchall()
        await cursor.close()
    column_names = {col[1] for col in columns}
    pk_columns = [col[1] for col in sorted(columns, key=lambda col: col[5]) if col[5]]

    assert {"hash", "detail", "model", "created_at"}.issubset(column_names)
    assert pk_columns == ["hash", "detail", "model"]

    call_count = 0

    async def fake_run_ocr(data, *, model, detail):
        nonlocal call_count
        call_count += 1
        return OcrResult(
            text=f"{detail}-{model}-{call_count}",
            usage=OcrUsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)

    poster = DummyPoster(b"poster-bytes")
    result_auto, spent_auto, remaining_auto = await poster_ocr.recognize_posters(
        db, [poster], detail="auto"
    )
    result_high, spent_high, remaining_high = await poster_ocr.recognize_posters(
        db, [poster], detail="high"
    )

    assert call_count == 2
    assert result_auto[0].text.startswith("auto-")
    assert result_high[0].text.startswith("high-")
    assert spent_auto == 0
    assert spent_high == 0
    assert remaining_auto == remaining_high

    async with db.get_session() as session:
        model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")
        digest = hashlib.sha256(poster.data).hexdigest()
        cached_auto = await session.get(PosterOcrCache, (digest, "auto", model))
        cached_high = await session.get(PosterOcrCache, (digest, "high", model))
        migrated = await session.get(PosterOcrCache, ("digest-old", "auto", model))

    assert cached_auto is not None
    assert cached_high is not None
    assert cached_auto.text.startswith("auto-")
    assert cached_high.text.startswith("high-")
    assert migrated is not None
    assert migrated.text == "old text"
    assert str(migrated.created_at).startswith("2024-01-01 00:00:00")


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

    with pytest.raises(poster_ocr.PosterOcrLimitExceededError) as excinfo:
        await poster_ocr.recognize_posters(db, items)

    assert call_count == 1
    assert len(excinfo.value.results) == 1
    assert excinfo.value.results[0].text == "text1"
    assert excinfo.value.spent_tokens == 50
    assert excinfo.value.remaining == 0

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
async def test_recognize_posters_returns_cached_when_first_blocked(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    digest_cached = hashlib.sha256(b"cached").hexdigest()
    model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")

    async with db.get_session() as session:
        session.add(
            PosterOcrCache(
                hash=digest_cached,
                detail="auto",
                model=model,
                text="cached-text",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        )
        session.add(
            OcrUsage(date="2024-06-06", spent_tokens=poster_ocr.DAILY_TOKEN_LIMIT)
        )
        await session.commit()

    call_count = 0

    async def fake_run_ocr(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise AssertionError("run_ocr should not be called when limit is reached")

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-06")

    items = [DummyPoster(b"new"), DummyPoster(b"cached")]

    with pytest.raises(poster_ocr.PosterOcrLimitExceededError) as excinfo:
        await poster_ocr.recognize_posters(db, items)

    assert call_count == 0
    assert len(excinfo.value.results) == 1
    assert excinfo.value.results[0].hash == digest_cached
    assert excinfo.value.results[0].text == "cached-text"
    assert excinfo.value.spent_tokens == 0
    assert excinfo.value.remaining == 0


@pytest.mark.asyncio
async def test_recognize_posters_uses_cache_when_limit_zero(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    cached_item = DummyPoster(b"cached")
    new_item = DummyPoster(b"new")
    model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")
    digest_cached = hashlib.sha256(cached_item.data).hexdigest()
    digest_new = hashlib.sha256(new_item.data).hexdigest()

    async with db.get_session() as session:
        session.add(
            PosterOcrCache(
                hash=digest_cached,
                detail="auto",
                model=model,
                text="cached-text",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        )
        session.add(
            OcrUsage(date="2024-06-05", spent_tokens=poster_ocr.DAILY_TOKEN_LIMIT)
        )
        await session.commit()

    call_count = 0

    async def fake_run_ocr(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise AssertionError("run_ocr should not be called when limit is exhausted")

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-05")

    with pytest.raises(poster_ocr.PosterOcrLimitExceededError) as excinfo:
        await poster_ocr.recognize_posters(db, [cached_item, new_item])

    assert call_count == 0
    assert len(excinfo.value.results) == 1
    assert excinfo.value.results[0].hash == digest_cached
    assert excinfo.value.results[0].text == "cached-text"
    assert excinfo.value.spent_tokens == 0
    assert excinfo.value.remaining == 0

    async with db.get_session() as session:
        cached_first = await session.get(
            PosterOcrCache, (digest_cached, "auto", model)
        )
        cached_second = await session.get(
            PosterOcrCache, (digest_new, "auto", model)
        )

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


@pytest.mark.asyncio
async def test_recognize_posters_concurrent_usage_upsert(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    items = [DummyPoster(b"first"), DummyPoster(b"second")]
    token_map = {items[0].data: 5, items[1].data: 7}

    async def fake_run_ocr(data, *, model, detail):
        await asyncio.sleep(0)
        spent = token_map[data]
        return OcrResult(
            text=f"text-{data.decode()}",
            usage=OcrUsageStats(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=spent,
            ),
        )

    monkeypatch.setattr(poster_ocr, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(poster_ocr, "_today_key", lambda: "2024-06-07")

    async def recognize(item: DummyPoster):
        return await poster_ocr.recognize_posters(db, [item])

    results = await asyncio.gather(*[recognize(item) for item in items])

    for (result_items, _, _), item in zip(results, items):
        assert result_items[0].text == f"text-{item.data.decode()}"

    total_spent = sum(spent for _, spent, _ in results)
    assert total_spent == sum(token_map.values())

    async with db.get_session() as session:
        usage_row = await session.get(OcrUsage, "2024-06-07")

    assert usage_row is not None
    assert usage_row.spent_tokens == sum(token_map.values())
