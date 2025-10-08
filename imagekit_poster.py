"""Utility helpers for uploading and transforming posters through ImageKit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from imagekitio import ImageKit

__all__ = [
    "PosterResizeMode",
    "PosterGravity",
    "PosterTransformation",
    "process_poster",
]


class PosterResizeMode(str, Enum):
    """Resize behaviour supported by ImageKit transformations."""

    MAINTAIN_RATIO = "maintain_ratio"
    PAD = "pad_resize"
    CROP = "crop"
    SCALE = "scale"


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


@dataclass(slots=True)
class PosterTransformation:
    """ImageKit transformation description."""

    name: str
    width: int | None = None
    height: int | None = None
    mode: PosterResizeMode = PosterResizeMode.MAINTAIN_RATIO
    gravity: PosterGravity | None = None
    quality: int | None = None
    background: str | None = None
    raw: Mapping[str, Any] | str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert the transformation to ImageKit API format."""

        payload: dict[str, Any] = {}
        if self.width is not None:
            payload["width"] = self.width
        if self.height is not None:
            payload["height"] = self.height
        if self.mode is not None:
            payload["crop"] = self.mode.value
        if self.gravity is not None:
            payload["focus"] = self.gravity.value
        if self.quality is not None:
            payload["quality"] = self.quality
        if self.background is not None:
            payload["background"] = self.background
        if isinstance(self.raw, Mapping):
            payload.update(self.raw)
        elif self.raw is not None:
            payload["raw"] = self.raw
        return payload


def _ensure_bytes(source: str | Path | bytes) -> tuple[bytes, str]:
    """Load poster contents from disk when needed."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.name
    return source, "poster"


def _build_transformations(
    transforms: Sequence[PosterTransformation] | None,
) -> list[dict[str, Any]]:
    if not transforms:
        raise ValueError("at least one transformation must be provided")
    return [transform.as_dict() for transform in transforms]


def process_poster(
    source: str | Path | bytes,
    *,
    public_key: str,
    private_key: str,
    url_endpoint: str,
    file_name: str | None = None,
    transformations: Sequence[PosterTransformation] | None = None,
    upload_options: Mapping[str, Any] | None = None,
    url_options: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Upload the poster to ImageKit and return URLs for requested variants.

    Parameters
    ----------
    source:
        Path to a file on disk or raw bytes to upload.
    public_key, private_key, url_endpoint:
        Credentials for connecting to ImageKit.
    file_name:
        Optional name to use when uploading. If omitted and ``source`` is a
        filesystem path, the filename from the path is used.
    transformations:
        Sequence of transformations to apply. Each item is expected to include a
        unique ``name`` that will be used as a key in the returned dictionary.
    upload_options:
        Extra options forwarded to :meth:`ImageKit.upload_file`.
    url_options:
        Additional flags forwarded to :meth:`ImageKit.url`.

    Returns
    -------
    dict[str, str]
        Mapping of transformation names to generated URLs.
    """

    payload, default_name = _ensure_bytes(source)
    resolved_name = file_name or default_name

    client = ImageKit(
        public_key=public_key,
        private_key=private_key,
        url_endpoint=url_endpoint,
    )

    options: dict[str, Any] = {"file_name": resolved_name}
    if upload_options:
        options.update(upload_options)

    upload_response = client.upload_file(file=payload, **options)

    file_path = upload_response.get("file_path")
    if not file_path:
        raise RuntimeError("ImageKit upload response did not include file_path")

    transforms = list(transformations or [])
    transform_payload = _build_transformations(transforms)

    base_options: dict[str, Any] = {
        "path": file_path,
        "transformation_position": "path",
        "transformation": [],
    }
    if url_options:
        base_options.update(url_options)

    urls: dict[str, str] = {}
    for transform, payload in zip(transforms, transform_payload):
        options = dict(base_options)
        options["transformation"] = [payload]
        urls[transform.name] = client.url(options)

    return urls

