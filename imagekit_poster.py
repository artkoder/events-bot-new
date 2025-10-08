"""Utility helpers for transforming posters through ImageKit."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

import requests

try:  # pragma: no cover - optional dependency for tests
    from imagekitio import ImageKit
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    ImageKit = None  # type: ignore[assignment]
    _IMAGEKIT_IMPORT_ERROR = exc
else:
    _IMAGEKIT_IMPORT_ERROR = None

__all__ = [
    "PosterProcessingMode",
    "PosterGravity",
    "process_poster",
]


class PosterProcessingMode(str, Enum):
    """Supported poster processing flows."""

    SMART_CROP = "smart_crop"
    EXTEND_GENFILL = "extend_genfill"


class PosterGravity(str, Enum):
    """Focus point used during cropping operations."""

    CENTER = "center"
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTH_EAST = "north_east"
    NORTH_WEST = "north_west"
    SOUTH_EAST = "south_east"
    SOUTH_WEST = "south_west"
    AUTO = "auto"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be configured for ImageKit")
    return value


def _build_transformation(
    mode: PosterProcessingMode,
    *,
    width: int,
    height: int,
    prompt: str | None,
    gravity: PosterGravity | str | None,
) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    focus = None
    if gravity is not None:
        focus = gravity.value if isinstance(gravity, PosterGravity) else str(gravity)

    if mode is PosterProcessingMode.SMART_CROP:
        payload: dict[str, Any] = {"w": width, "h": height, "fo": focus or PosterGravity.AUTO.value}
    elif mode is PosterProcessingMode.EXTEND_GENFILL:
        raw = "bg-genfill"
        if prompt:
            raw = f"{raw}:prompt={prompt}"
        payload = {
            "w": width,
            "h": height,
            "cm": "pad_resize",
            "fo": focus or PosterGravity.AUTO.value,
            "raw": raw,
        }
    else:  # pragma: no cover - defensive for future enum members
        raise ValueError(f"Unsupported poster processing mode: {mode}")

    return payload


def process_poster(
    image_bytes: bytes,
    *,
    mode: PosterProcessingMode,
    width: int,
    height: int,
    prompt: str | None = None,
    gravity: PosterGravity | str | None = None,
    file_name: str | None = None,
    request_timeout: float | tuple[float, float] = 10.0,
) -> bytes:
    """Upload a poster to ImageKit, transform it, and download the result."""

    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError("image_bytes must be raw bytes")

    if ImageKit is None:  # pragma: no cover - configuration error
        raise RuntimeError("imagekitio package is required to process posters") from _IMAGEKIT_IMPORT_ERROR

    client = ImageKit(
        public_key=_require_env("IMAGEKIT_PUBLIC_KEY"),
        private_key=_require_env("IMAGEKIT_PRIVATE_KEY"),
        url_endpoint=_require_env("IMAGEKIT_URL_ENDPOINT"),
    )

    resolved_name = file_name or "poster.jpg"
    upload_response = client.upload_file(file=bytes(image_bytes), file_name=resolved_name)

    file_path = upload_response.get("file_path")
    if not file_path:
        raise RuntimeError("ImageKit upload response did not include file_path")

    transformation = _build_transformation(
        mode,
        width=width,
        height=height,
        prompt=prompt,
        gravity=gravity,
    )

    url = client.url(
        {
            "path": file_path,
            "transformation_position": "path",
            "transformation": [transformation],
        }
    )

    response = requests.get(url, timeout=request_timeout)
    response.raise_for_status()
    return response.content
