from __future__ import annotations

import hashlib
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Iterable, Mapping, Sequence

from vision_test.ocr import OcrResult, configure_http as _configure_ocr_http, run_ocr

__all__ = [
    "PosterMedia",
    "process_media",
    "collect_poster_texts",
    "build_poster_summary",
    "apply_ocr_results_to_media",
]


@dataclass(slots=True)
class PosterMedia:
    """Container for processed poster information."""

    data: bytes = field(repr=False)
    name: str
    catbox_url: str | None = None
    digest: str | None = None
    ocr_text: str | None = None
    ocr_title: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.digest is None:
            self.digest = hashlib.sha256(self.data).hexdigest()

    def clear_payload(self) -> None:
        """Release in-memory payload after processing."""

        if self.data:
            self.data = b""


_OCR_CONFIGURED = False
_MAIN_MODULE: ModuleType | None = None


def _get_main_module() -> ModuleType:
    """Return the active main module, preferring the live ``__main__`` module.

    The running script is registered as ``__main__`` which carries feature
    toggles like ``CATBOX_ENABLED``. Importing ``main`` directly can return a
    fresh module that does not reflect runtime state, so we only fall back to it
    when the live module is unavailable.
    """
    global _MAIN_MODULE
    if _MAIN_MODULE is not None:
        return _MAIN_MODULE

    module = None

    live_module = sys.modules.get("__main__")
    if live_module is not None and hasattr(live_module, "CATBOX_ENABLED"):
        module = live_module

    if module is None:
        module = sys.modules.get("main")

    if module is None:
        module = import_module("main")

    _MAIN_MODULE = module
    return module


def _ensure_ocr_http() -> None:
    global _OCR_CONFIGURED
    if _OCR_CONFIGURED:
        return
    main_mod = _get_main_module()
    session = main_mod.get_http_session()
    semaphore = main_mod.HTTP_SEMAPHORE
    _configure_ocr_http(session=session, semaphore=semaphore)
    _OCR_CONFIGURED = True


async def _run_ocr(poster: PosterMedia, model: str, detail: str) -> None:
    try:
        result: OcrResult = await run_ocr(poster.data, model=model, detail=detail)
    except Exception as exc:  # pragma: no cover - network/remote failures
        logging.warning("poster ocr failed name=%s error=%s", poster.name, exc)
        return
    poster.ocr_text = result.text
    poster.ocr_title = result.title
    logging.info(
        "poster_ocr success hash=%s ocr_title=%r source=from_llm",
        poster.digest,
        (poster.ocr_title or "")[:120],
    )
    usage = result.usage
    poster.prompt_tokens = usage.prompt_tokens
    poster.completion_tokens = usage.completion_tokens
    poster.total_tokens = usage.total_tokens


async def process_media(
    images: Iterable[tuple[bytes, str]] | None,
    *,
    need_catbox: bool,
    need_ocr: bool,
) -> tuple[list[PosterMedia], str]:
    """Upload media to Catbox and optionally run OCR over them."""

    raw = list(images or [])
    if not raw:
        return [], ""

    posters = [PosterMedia(data=data, name=name) for data, name in raw]
    catbox_msg = ""

    if need_catbox:
        main_mod = _get_main_module()
        catbox_enabled = getattr(main_mod, "CATBOX_ENABLED", None)
        preprocessed_provided = any(p.catbox_url for p in posters)
        logging.info(
            "poster_media upload start: need_catbox=%s catbox_enabled=%s raw_count=%d preprocessed=%s",
            need_catbox,
            catbox_enabled,
            len(raw),
            preprocessed_provided,
        )
        upload_images = main_mod.upload_images
        catbox_urls, catbox_msg = await upload_images(raw)
        logging.info(
            "poster_media upload complete: url_count=%d catbox_msg=%s",
            len(catbox_urls),
            catbox_msg,
        )
        for poster, url in zip(posters, catbox_urls):
            poster.catbox_url = url

    if need_ocr:
        _ensure_ocr_http()
        model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")
        detail = os.getenv("POSTER_OCR_DETAIL", "auto")
        for poster in posters:
            await _run_ocr(poster, model=model, detail=detail)

    for poster in posters:
        poster.clear_payload()

    return posters, catbox_msg


def collect_poster_texts(poster_media: Sequence[PosterMedia]) -> list[str]:
    """Return cleaned OCR texts from processed posters."""

    texts: list[str] = []
    for poster in poster_media:
        if poster.ocr_text:
            text = poster.ocr_text.strip()
            if text:
                texts.append(text)
    return texts


def build_poster_summary(poster_media: Sequence[PosterMedia]) -> str | None:
    """Return a short summary describing OCR token usage."""

    if not poster_media:
        return None

    prompt = sum(p.prompt_tokens or 0 for p in poster_media)
    completion = sum(p.completion_tokens or 0 for p in poster_media)
    total = sum(p.total_tokens or 0 for p in poster_media)

    if prompt == completion == total == 0:
        return f"Posters processed: {len(poster_media)}."

    return (
        f"Posters processed: {len(poster_media)}. "
        f"Tokens — prompt: {prompt}, completion: {completion}, total: {total}."
    )


def apply_ocr_results_to_media(
    poster_media: list[PosterMedia],
    ocr_results: Sequence[object],
    *,
    hash_to_indices: Mapping[str, list[int]] | None = None,
) -> None:
    """Populate poster metadata with OCR cache entries."""

    if not ocr_results:
        return

    index_map = {
        key: deque(indices)
        for key, indices in (hash_to_indices or {}).items()
    }
    used_indices: set[int] = set()

    for cache in ocr_results:
        idx: int | None = None
        cache_hash = getattr(cache, "hash", None)
        if cache_hash is not None:
            queue = index_map.get(cache_hash)
            while queue:
                candidate = queue.popleft()
                if 0 <= candidate < len(poster_media) and candidate not in used_indices:
                    idx = candidate
                    break

        if idx is None:
            for candidate in range(len(poster_media)):
                if candidate not in used_indices:
                    idx = candidate
                    break

        if idx is None:
            poster = PosterMedia(data=b"", name=str(cache_hash or ""))
            poster_media.append(poster)
            idx = len(poster_media) - 1

        poster = poster_media[idx]
        used_indices.add(idx)
        if cache_hash:
            poster.digest = cache_hash
        poster.ocr_text = getattr(cache, "text", None)
        poster.ocr_title = getattr(cache, "title", None)
        logging.info(
            "poster_ocr success hash=%s ocr_title=%r source=from_cache",
            cache_hash,
            (poster.ocr_title or "")[:120],
        )
        poster.prompt_tokens = getattr(cache, "prompt_tokens", None)
        poster.completion_tokens = getattr(cache, "completion_tokens", None)
        poster.total_tokens = getattr(cache, "total_tokens", None)
