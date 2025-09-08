import os
import logging
import asyncio
import hashlib
from cachetools import TTLCache
import httpx
from typing import Optional

from db import Database
from models import Setting

_cache_ttl_days = int(os.getenv("URL_SHORTENER_CACHE_TTL_DAYS", "30"))
_cache: TTLCache[str, str] = TTLCache(maxsize=2048, ttl=_cache_ttl_days * 86400)
_db: Optional[Database] = None


def init(db: Database) -> None:
    global _db
    _db = db


async def _get_from_db(key: str) -> Optional[str]:
    if _db is None:
        return None
    async with _db.get_session() as session:
        setting = await session.get(Setting, key)
        return setting.value if setting else None


async def _set_in_db(key: str, value: str) -> None:
    if _db is None:
        return
    async with _db.get_session() as session:
        setting = await session.get(Setting, key)
        if setting:
            setting.value = value
        else:
            session.add(Setting(key=key, value=value))
        await session.commit()


async def shorten_url(url: str) -> str:
    mode = os.getenv("URL_SHORTENER", "isgd").lower()
    if mode == "off":
        return url

    key = hashlib.sha1(url.encode()).hexdigest()
    cached = _cache.get(key)
    if cached:
        logging.info("digest.shortener.cache hit=mem url=%s", url)
        return cached

    db_key = f"short:{key}"
    db_val = await _get_from_db(db_key)
    if db_val:
        logging.info("digest.shortener.cache hit=db url=%s", url)
        _cache[key] = db_val
        return db_val

    providers = ["isgd", "cleanuri"]
    if mode == "cleanuri":
        providers.reverse()

    for provider in providers:
        for attempt in range(2):
            try:
                logging.info(
                    "digest.shortener.request provider=%s attempt=%s url=%s",
                    provider,
                    attempt + 1,
                    url,
                )
                async with httpx.AsyncClient(timeout=3.0) as client:
                    if provider == "isgd":
                        resp = await client.get(
                            "https://is.gd/create.php",
                            params={"format": "simple", "url": url},
                        )
                        resp.raise_for_status()
                        short = resp.text.strip()
                    else:
                        resp = await client.post(
                            "https://cleanuri.com/api/v1/shorten",
                            data={"url": url},
                        )
                        resp.raise_for_status()
                        short = resp.json()["result_url"]
                _cache[key] = short
                await _set_in_db(db_key, short)
                logging.info(
                    "digest.shortener.response provider=%s ok=1 url=%s short=%s",
                    provider,
                    url,
                    short,
                )
                return short
            except Exception as e:
                logging.warning(
                    "digest.shortener.response provider=%s ok=0 url=%s error=%r",
                    provider,
                    url,
                    e,
                )
                if attempt == 0:
                    await asyncio.sleep(0.5)
                continue
    return url
