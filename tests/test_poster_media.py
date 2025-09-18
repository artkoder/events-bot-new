from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

import poster_media as poster_media_module


@pytest.mark.asyncio
async def test_process_media_uses_active_main(monkeypatch, caplog):
    original_main = sys.modules.get("main")
    original_dunder_main = sys.modules.get("__main__")

    script_main = types.ModuleType("__main__")
    upload_calls: list[tuple[bytes, str]] | None = None

    async def fake_upload_images(images, *args, **kwargs):
        nonlocal upload_calls
        upload_calls = list(images)
        logging.info("fake catbox upload")
        return [f"https://cat.box/{idx}" for idx, _ in enumerate(images)], "ok"

    script_main.upload_images = fake_upload_images  # type: ignore[attr-defined]
    script_main.CATBOX_ENABLED = True  # type: ignore[attr-defined]
    script_main.get_http_session = lambda: "session"  # type: ignore[attr-defined]
    script_main.HTTP_SEMAPHORE = "semaphore"  # type: ignore[attr-defined]

    sys.modules["__main__"] = script_main
    sys.modules.pop("main", None)

    try:
        module = importlib.reload(poster_media_module)
        monkeypatch.setattr(module, "_MAIN_MODULE", None, raising=False)

        caplog.set_level(logging.INFO)

        images = [(b"data", "poster.jpg")]
        posters, catbox_msg = await module.process_media(
            images,
            need_catbox=True,
            need_ocr=False,
        )

        assert upload_calls == images
        assert all(p.catbox_url for p in posters)
        assert catbox_msg == "ok"
        assert "CATBOX disabled" not in caplog.text
        start_log = next(
            record
            for record in caplog.records
            if "poster_media upload start" in record.getMessage()
        )
        assert start_log.levelno == logging.INFO
        message = start_log.getMessage()
        assert "need_catbox=True" in message
        assert "catbox_enabled=True" in message
        assert "raw_count=1" in message
        assert "preprocessed=False" in message

        complete_log = next(
            record
            for record in caplog.records
            if "poster_media upload complete" in record.getMessage()
        )
        assert complete_log.levelno == logging.INFO
        complete_message = complete_log.getMessage()
        assert "url_count=1" in complete_message
        assert "catbox_msg=ok" in complete_message
    finally:
        if original_main is not None:
            sys.modules["main"] = original_main
        else:
            sys.modules.pop("main", None)

        if original_dunder_main is not None:
            sys.modules["__main__"] = original_dunder_main
        else:
            sys.modules.pop("__main__", None)

        importlib.reload(poster_media_module)
