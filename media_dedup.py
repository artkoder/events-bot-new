from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


@dataclass(frozen=True, slots=True)
class PreparedImage:
    dhash_hex: str
    webp_bytes: bytes


def prepare_image_for_supabase(
    image_bytes: bytes,
    *,
    dhash_size: int = 16,
    webp_quality: int = 82,
) -> PreparedImage | None:
    """Decode bytes once, compute perceptual dHash, and re-encode as WebP.

    This is used for cross-environment deduplication in Supabase Storage:
    - key by perceptual hash (stable across re-encodes / different resolutions)
    - store as WebP only
    """

    if not image_bytes:
        return None

    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception:
        return None

    try:
        with Image.open(BytesIO(image_bytes)) as im:
            im = ImageOps.exif_transpose(im)

            resampling = getattr(Image, "Resampling", None)
            lanczos = resampling.LANCZOS if resampling else Image.LANCZOS

            # dHash (horizontal gradients) on grayscale, fixed size.
            gray = im.convert("L")
            small = gray.resize((dhash_size + 1, dhash_size), lanczos)
            get_flat = getattr(small, "get_flattened_data", None)
            pixels = list(get_flat() if callable(get_flat) else small.getdata())
            # Quantize to reduce sensitivity to minor resampling differences between resolutions.
            pixels = [p >> 3 for p in pixels]
            diff_bits: list[int] = []
            row_w = dhash_size + 1
            for row in range(dhash_size):
                off = row * row_w
                for col in range(dhash_size):
                    left = pixels[off + col]
                    right = pixels[off + col + 1]
                    diff_bits.append(1 if left > right else 0)
            value = 0
            for bit in diff_bits:
                value = (value << 1) | bit
            width = (dhash_size * dhash_size) // 4
            dhash_hex = f"{value:0{width}x}"

            # WebP encoding.
            out = BytesIO()
            if im.mode in {"RGBA", "LA"} or (
                im.mode == "P" and "transparency" in (im.info or {})
            ):
                im2 = im.convert("RGBA")
            else:
                im2 = im.convert("RGB")
            im2.save(
                out,
                format="WEBP",
                quality=int(webp_quality),
                method=6,
            )
            webp_bytes = out.getvalue()
            if not webp_bytes:
                return None
            return PreparedImage(dhash_hex=dhash_hex, webp_bytes=webp_bytes)
    except Exception:
        return None


def build_supabase_poster_object_path(
    dhash_hex: str,
    *,
    prefix: str = "p",
    dhash_size: int = 16,
) -> str:
    """Build a deterministic Storage path for a poster.

    Format:
      <prefix>/dh<dhash_size>/<first2>/<dhash>.webp
    """

    pfx = (prefix or "").strip().strip("/")
    if not pfx:
        pfx = "p"
    h = (dhash_hex or "").strip().lower()
    if not h:
        raise ValueError("dhash_hex is required")
    algo = f"dh{int(dhash_size)}"
    return f"{pfx}/{algo}/{h[:2]}/{h}.webp"
