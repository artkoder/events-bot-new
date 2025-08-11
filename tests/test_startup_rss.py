import asyncio
import gc

import pytest

from db import Database
from main import print_current_rss, _current_rss_mb


@pytest.mark.skip("RSS check is intended for manual runs")
@pytest.mark.asyncio
async def test_startup_rss(tmp_path, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    with caplog.at_level("INFO"):
        gc.collect()
        print_current_rss()
    rss = _current_rss_mb()
    assert rss < 130
    assert any("Peak RSS" in r.message for r in caplog.records)
