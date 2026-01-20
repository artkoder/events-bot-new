from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "video_announce"
REF_DIR = VIDEO_DIR / "crumple_references"
PRIMARY_FONT_DIR = VIDEO_DIR / "test_afisha"
FALLBACK_FONT_DIR = VIDEO_DIR / "assets"
DEFAULT_OUT_DIR = VIDEO_DIR / "test_output"

CANVAS_SIZE = (1080, 1572)
DEFAULT_MONTH_ROTATE_DEGREES = -90  # 90° clockwise (PIL uses CCW-positive angles)


@dataclass(frozen=True)
class IntroStyle:
    name: str
    ref_path: Path
    date_text: str
    month_text: str
    title_text: str
    cities_text: str
    date_region: tuple[float, float, float, float]


STYLE_DAY = IntroStyle(
    name="day",
    ref_path=REF_DIR / "intro ref (day).png",
    date_text="19",
    month_text="ЯНВАРЯ",
    title_text="ПОНЕДЕЛЬНИК",
    cities_text="КАЛИНИНГРАД\nСВЕТЛОГОРСК\nЗЕЛЕНОГРАДСК",
    date_region=(0.50, 0.05, 0.98, 0.33),
)

STYLE_WEEKEND = IntroStyle(
    name="weekend",
    ref_path=REF_DIR / "intro ref (weekend).png",
    date_text="24-25",
    month_text="ДЕКАБРЯ",
    title_text="ВЫХОДНЫЕ",
    cities_text="КАЛИНИНГРАД\nСВЕТЛОГОРСК\nЗЕЛЕНОГРАДСК",
    date_region=(0.02, 0.05, 0.98, 0.33),
)


def _find_font_file(filename: str) -> Path:
    candidates = [
        PRIMARY_FONT_DIR / filename,
        PRIMARY_FONT_DIR / "fonts" / filename,
        FALLBACK_FONT_DIR / filename,
        FALLBACK_FONT_DIR / "fonts" / filename,
        ROOT / "3d_intro" / "assets" / "fonts" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Font not found: {filename} (looked in {candidates}).")


def _load_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), size)


def _set_variable_weight(font: ImageFont.FreeTypeFont, weight: int | None) -> None:
    if weight is None:
        return
    if not hasattr(font, "get_variation_axes"):
        return
    if not hasattr(font, "set_variation_by_axes"):
        return
    try:
        axes = font.get_variation_axes() or []
    except OSError:
        return
    if not axes:
        return
    axis = axes[0]
    min_w = int(axis.get("minimum", weight))
    max_w = int(axis.get("maximum", weight))
    clamped = max(min_w, min(max_w, int(weight)))
    try:
        font.set_variation_by_axes([clamped])
    except OSError:
        return


def _trim_transparency(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


def _render_text_rgba(
    *,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    align: str = "left",
    spacing: int = 4,
    padding: int = 8,
) -> Image.Image:
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align=align, spacing=spacing)
    left, top, right, bottom = bbox
    left_i = int(math.floor(left))
    top_i = int(math.floor(top))
    right_i = int(math.ceil(right))
    bottom_i = int(math.ceil(bottom))
    width = max(1, right_i - left_i)
    height = max(1, bottom_i - top_i)

    img = Image.new("RGBA", (width + padding * 2, height + padding * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (padding - left_i, padding - top_i),
        text,
        font=font,
        fill=fill,
        align=align,
        spacing=spacing,
    )
    return _trim_transparency(img)


def _sample_mode_rgba(im: Image.Image, step: int = 4) -> tuple[int, int, int, int]:
    im = im.convert("RGBA")
    px = im.load()
    w, h = im.size
    counts: Counter[tuple[int, int, int, int]] = Counter()
    for y in range(0, h, step):
        for x in range(0, w, step):
            counts[px[x, y]] += 1
    return counts.most_common(1)[0][0]


def _text_mask(
    im: Image.Image,
    *,
    bg_rgb: tuple[int, int, int],
    dist: int = 45,
    lum_thr: int = 200,
) -> np.ndarray:
    arr = np.array(im.convert("RGBA"))
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    bg = np.array(bg_rgb, dtype=np.int16)
    diff = rgb - bg
    dist2 = np.sum(diff * diff, axis=-1)
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    return (alpha > 0) & (dist2 > dist * dist) & (lum < lum_thr)


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def _bbox_in_region(mask: np.ndarray, region: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = region
    sub = mask[y0:y1, x0:x1]
    tight = _tight_bbox(sub)
    if tight is None:
        return None
    tx0, ty0, tx1, ty1 = tight
    return (x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1)


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    margin: int,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w, h = canvas_size
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(w, x1 + margin),
        min(h, y1 + margin),
    )


def _luma_u8(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32
    )


def _bg_luma(bg: tuple[int, int, int, int]) -> float:
    r, g, b, _ = bg
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def _estimate_text_color(
    reference: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    bg: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    arr = np.asarray(reference.crop(roi).convert("RGBA"), dtype=np.uint8)
    rgb = arr[..., :3].astype(np.int32)
    alpha = arr[..., 3] > 0
    if not alpha.any():
        return (0, 0, 0, 255)
    bg_luma = float(_luma_u8(np.array([[list(bg[:3]) + [255]]], dtype=np.uint8))[0, 0])
    luma = _luma_u8(arr)
    ink = alpha & (luma < (bg_luma - 25.0))
    if ink.sum() < 50:
        return (0, 0, 0, 255)
    med = np.median(rgb[ink], axis=0).astype(int)
    return (int(med[0]), int(med[1]), int(med[2]), 255)


def _fit_font_size_to_box(
    *,
    text: str,
    font_path: Path,
    fill: tuple[int, int, int, int],
    max_w: int,
    max_h: int,
    min_size: int,
    max_size: int,
    rotate: int | None = None,
    align: str = "left",
    spacing_for_size: Callable[[int], int] | None = None,
    variable_weight: int | None = None,
) -> int:
    lo, hi = min_size, max_size
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        if variable_weight is not None:
            _set_variable_weight(font, variable_weight)
        spacing = spacing_for_size(mid) if spacing_for_size is not None else 4
        glyph = _render_text_rgba(text=text, font=font, fill=fill, align=align, spacing=spacing)
        if rotate is not None:
            glyph = _trim_transparency(glyph.rotate(rotate, expand=True))
        if glyph.width <= max_w and glyph.height <= max_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _score_roi_mae(ref_roi_rgba: np.ndarray, cand_roi_rgba: np.ndarray) -> float:
    ref_l = _luma_u8(ref_roi_rgba)
    cand_l = _luma_u8(cand_roi_rgba)
    return float(np.mean(np.abs(ref_l - cand_l)))


def _hill_climb(
    *,
    initial: tuple[int, int, int],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    score_fn: Callable[[int, int, int], float],
) -> tuple[tuple[int, int, int], float]:
    (xmin, xmax), (ymin, ymax), (smin, smax) = bounds
    x, y, size = initial
    x = int(np.clip(x, xmin, xmax))
    y = int(np.clip(y, ymin, ymax))
    size = int(np.clip(size, smin, smax))
    best_score = float(score_fn(x, y, size))
    steps = [16, 8, 4, 2, 1]
    for pos_step in steps:
        size_step = max(1, pos_step // 2)
        for _ in range(40):
            improved = False
            for dx, dy, ds in (
                (-pos_step, 0, 0),
                (pos_step, 0, 0),
                (0, -pos_step, 0),
                (0, pos_step, 0),
                (0, 0, -size_step),
                (0, 0, size_step),
            ):
                cx, cy, cs = x + dx, y + dy, size + ds
                if not (xmin <= cx <= xmax and ymin <= cy <= ymax and smin <= cs <= smax):
                    continue
                s = float(score_fn(cx, cy, cs))
                if s < best_score:
                    x, y, size = cx, cy, cs
                    best_score = s
                    improved = True
                    break
            if not improved:
                break
    return (x, y, size), float(best_score)


def _optimize_with_restarts(
    *,
    roi: tuple[int, int, int, int],
    initial: tuple[int, int, int],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    score_fn: Callable[[int, int, int], float],
    glyph_for_size: Callable[[int], Image.Image],
) -> tuple[tuple[int, int, int], float]:
    x0, y0, x1, y1 = roi
    (xmin, xmax), (ymin, ymax), (smin, smax) = bounds
    _, _, initial_size = initial
    raw_sizes = {initial_size, initial_size - 24, initial_size - 12, initial_size + 12, initial_size + 24}
    size_candidates = sorted(s for s in raw_sizes if smin <= s <= smax)

    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for size in size_candidates:
        glyph = glyph_for_size(size)
        x_candidates = [x0, x1 - glyph.width, x0 + (x1 - x0 - glyph.width) // 2]
        y_candidates = [y0, y1 - glyph.height, y0 + (y1 - y0 - glyph.height) // 2]
        for ix in x_candidates:
            for iy in y_candidates:
                ix = int(np.clip(ix, xmin, xmax))
                iy = int(np.clip(iy, ymin, ymax))
                candidates.append((float(score_fn(ix, iy, size)), (ix, iy, size)))

    rng = np.random.default_rng(0)
    for _ in range(140):
        rs = int(rng.integers(smin, smax + 1))
        rx = int(rng.integers(xmin, xmax + 1))
        ry = int(rng.integers(ymin, ymax + 1))
        candidates.append((float(score_fn(rx, ry, rs)), (rx, ry, rs)))

    candidates.sort(key=lambda t: t[0])
    top = [c for _, c in candidates[:6]] or [initial]

    best_params: tuple[int, int, int] | None = None
    best_score: float | None = None
    for init in top:
        params, score = _hill_climb(initial=init, bounds=bounds, score_fn=score_fn)
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
    assert best_params is not None and best_score is not None
    return best_params, best_score


def _make_element_score_fns(
    *,
    ref: Image.Image,
    roi: tuple[int, int, int, int],
    bg: tuple[int, int, int, int],
    text: str,
    font_path: Path,
    fill: tuple[int, int, int, int],
    rotate: int | None,
    align: str,
    spacing_for_size: Callable[[int], int] | None,
    variable_weight: int | None,
) -> tuple[Callable[[int, int, int], float], Callable[[int], Image.Image]]:
    x0, y0, x1, y1 = roi
    ref_roi = np.asarray(ref.crop(roi).convert("RGBA"), dtype=np.uint8)
    ref_luma = _luma_u8(ref_roi)
    bg_luma = _bg_luma(bg)
    thr = bg_luma - 25.0
    ref_alpha = ref_roi[..., 3] > 0
    ref_ink = ref_alpha & (ref_luma < thr)
    ref_ink_count = int(ref_ink.sum())
    glyph_cache: dict[int, Image.Image] = {}

    def glyph_for_size(size: int) -> Image.Image:
        g = glyph_cache.get(size)
        if g is not None:
            return g
        font = _load_font(font_path, size)
        if variable_weight is not None:
            _set_variable_weight(font, variable_weight)
        spacing = spacing_for_size(size) if spacing_for_size is not None else 4
        g = _render_text_rgba(text=text, font=font, fill=fill, align=align, spacing=spacing)
        if rotate is not None:
            g = _trim_transparency(g.rotate(rotate, expand=True))
        glyph_cache[size] = g
        return g

    def score_fn(px: int, py: int, psize: int) -> float:
        roi_img = Image.new("RGBA", (x1 - x0, y1 - y0), bg)
        glyph = glyph_for_size(psize)
        dest = (px - x0, py - y0)
        gx0, gy0 = dest
        gx1, gy1 = gx0 + glyph.width, gy0 + glyph.height
        if not (gx1 <= 0 or gy1 <= 0 or gx0 >= roi_img.width or gy0 >= roi_img.height):
            roi_img.alpha_composite(glyph, dest=dest)

        cand_roi = np.asarray(roi_img, dtype=np.uint8)
        cand_luma = _luma_u8(cand_roi)
        cand_ink = (cand_roi[..., 3] > 0) & (cand_luma < thr)

        union = ref_ink | cand_ink
        if union.any():
            luma_mae = float(np.mean(np.abs(ref_luma[union] - cand_luma[union])))
        else:
            luma_mae = float(np.mean(np.abs(ref_luma - cand_luma)))

        if ref_ink_count <= 0:
            return luma_mae

        missing = ref_ink & (~cand_ink)
        extra = cand_ink & (~ref_ink)
        missing_rate = float(missing.sum()) / float(ref_ink_count)
        extra_rate = float(extra.sum()) / float(ref_ink_count)
        return luma_mae + (255.0 * (2.0 * missing_rate + 0.5 * extra_rate))

    return score_fn, glyph_for_size


def _render_full_intro(
    *,
    bg: tuple[int, int, int, int],
    date: tuple[str, Path, tuple[int, int, int, int], tuple[int, int, int]],
    month: tuple[str, Path, tuple[int, int, int, int], tuple[int, int, int], int],
    title: tuple[str, Path, tuple[int, int, int, int], tuple[int, int, int]],
    cities: tuple[str, Path, tuple[int, int, int, int], tuple[int, int, int], int | None, float],
) -> Image.Image:
    img = Image.new("RGBA", CANVAS_SIZE, bg)
    cache: dict[tuple[str, int, int | None, str, int, int | None], Image.Image] = {}

    def place(
        *,
        text: str,
        font_path: Path,
        fill: tuple[int, int, int, int],
        x: int,
        y: int,
        size: int,
        rotate: int | None,
        align: str,
        spacing: int,
        variable_weight: int | None,
    ) -> None:
        key = (str(font_path), size, rotate, align, spacing, variable_weight)
        glyph = cache.get(key)
        if glyph is None:
            font = _load_font(font_path, size)
            if variable_weight is not None:
                _set_variable_weight(font, variable_weight)
            glyph = _render_text_rgba(text=text, font=font, fill=fill, align=align, spacing=spacing)
            if rotate is not None:
                glyph = _trim_transparency(glyph.rotate(rotate, expand=True))
            cache[key] = glyph
        img.alpha_composite(glyph, dest=(x, y))

    date_text, date_font, date_fill, (dx, dy, ds) = date
    place(
        text=date_text,
        font_path=date_font,
        fill=date_fill,
        x=dx,
        y=dy,
        size=ds,
        rotate=None,
        align="left",
        spacing=4,
        variable_weight=None,
    )

    month_text, month_font, month_fill, (mx, my, ms), mrot = month
    place(
        text=month_text,
        font_path=month_font,
        fill=month_fill,
        x=mx,
        y=my,
        size=ms,
        rotate=mrot,
        align="left",
        spacing=4,
        variable_weight=None,
    )

    title_text, title_font, title_fill, (tx, ty, ts) = title
    place(
        text=title_text,
        font_path=title_font,
        fill=title_fill,
        x=tx,
        y=ty,
        size=ts,
        rotate=None,
        align="left",
        spacing=4,
        variable_weight=None,
    )

    cities_text, cities_font, cities_fill, (cx, cy, cs), cities_weight, cities_spacing_scale = cities
    place(
        text=cities_text,
        font_path=cities_font,
        fill=cities_fill,
        x=cx,
        y=cy,
        size=cs,
        rotate=None,
        align="center",
        spacing=max(0, int(round(cs * cities_spacing_scale))),
        variable_weight=cities_weight,
    )
    return img


def _style_by_name(name: str) -> IntroStyle:
    n = name.strip().lower()
    if n in {"day", "weekday"}:
        return STYLE_DAY
    if n in {"weekend", "weekends"}:
        return STYLE_WEEKEND
    raise ValueError(f"Unknown style: {name!r}")


def optimize_style(*, style: IntroStyle, out_dir: Path) -> dict:
    ref = Image.open(style.ref_path).convert("RGBA")
    if ref.size != CANVAS_SIZE:
        raise SystemExit(f"Unexpected reference size for {style.ref_path}: {ref.size} != {CANVAS_SIZE}")

    bg = _sample_mode_rgba(ref)
    mask = _text_mask(ref, bg_rgb=bg[:3])
    w, h = ref.size

    dx0 = int(w * style.date_region[0])
    dy0 = int(h * style.date_region[1])
    dx1 = int(w * style.date_region[2])
    dy1 = int(h * style.date_region[3])
    date_box = _bbox_in_region(mask, (dx0, dy0, dx1, dy1))
    if date_box is None:
        raise RuntimeError(f"Failed to locate date bbox in reference: {style.ref_path}")

    # Month: restrict search to be below the detected date to avoid date contamination.
    month_y0 = max(int(h * 0.25), date_box[3] + 10)
    month_box = _bbox_in_region(mask, (int(w * 0.70), month_y0, int(w * 0.95), int(h * 0.90)))
    if month_box is None:
        raise RuntimeError(f"Failed to locate month bbox in reference: {style.ref_path}")

    title_right_limit = max(0, month_box[0] - 10)
    # Title: keep it above the cities block to avoid contamination.
    title_box = _bbox_in_region(mask, (0, int(h * 0.50), title_right_limit, int(h * 0.68)))
    if title_box is None:
        raise RuntimeError(f"Failed to locate title bbox in reference: {style.ref_path}")

    cities_y0 = max(int(h * 0.67), title_box[3] + 10)
    cities_box = _bbox_in_region(mask, (int(w * 0.25), cities_y0, int(w * 0.75), int(h * 0.90)))
    if cities_box is None:
        raise RuntimeError(f"Failed to locate cities bbox in reference: {style.ref_path}")

    # Expand slightly to include anti-aliasing halo, but keep ROIs tight to avoid overlaps.
    date_roi = _expand_box(date_box, margin=24, canvas_size=ref.size)
    month_roi = _expand_box(month_box, margin=24, canvas_size=ref.size)
    title_roi = _expand_box(title_box, margin=24, canvas_size=ref.size)
    cities_roi = _expand_box(cities_box, margin=24, canvas_size=ref.size)

    font_date = _find_font_file("Benzin-Bold.ttf")
    font_month = _find_font_file("BebasNeue-Bold.ttf")
    font_title = _find_font_file("DrukCyr-Bold.ttf")

    date_fill = _estimate_text_color(ref, date_roi, bg=bg)
    month_fill = _estimate_text_color(ref, month_roi, bg=bg)
    title_fill = _estimate_text_color(ref, title_roi, bg=bg)
    cities_fill = _estimate_text_color(ref, cities_roi, bg=bg)

    def _try_font(filename: str) -> Path | None:
        try:
            return _find_font_file(filename)
        except FileNotFoundError:
            return None

    # Cities: pick best font/weight/line-spacing combo by scoring against the reference ROI.
    cities_candidates: list[tuple[Path, int | None]] = []
    if (p := _try_font("Oswald-VariableFont_wght.ttf")) is not None:
        for wght in (350, 400, 500, 600, 700):
            cities_candidates.append((p, wght))
    for fname in (
        "Akrobat-Bold.otf",
        "Akrobat-SemiBold.otf",
        "Akrobat-Regular.otf",
        "BebasNeue-Bold.ttf",
        "BebasNeue-Regular.ttf",
        "DrukCyr-Bold.ttf",
    ):
        if (p := _try_font(fname)) is not None:
            cities_candidates.append((p, None))
    if not cities_candidates:
        raise RuntimeError("No usable font found for cities.")

    best_cities: dict | None = None
    for font_path, weight in cities_candidates:
        for spacing_scale in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
            spacing_for_size = lambda s, sc=spacing_scale: max(0, int(round(s * sc)))
            size0 = _fit_font_size_to_box(
                text=style.cities_text,
                font_path=font_path,
                fill=cities_fill,
                max_w=cities_box[2] - cities_box[0],
                max_h=cities_box[3] - cities_box[1],
                min_size=14,
                max_size=260,
                align="center",
                spacing_for_size=spacing_for_size,
                variable_weight=weight,
            )
            font0 = _load_font(font_path, size0)
            if weight is not None:
                _set_variable_weight(font0, weight)
            glyph0 = _render_text_rgba(
                text=style.cities_text,
                font=font0,
                fill=cities_fill,
                align="center",
                spacing=spacing_for_size(size0),
            )
            x0 = (cities_box[0] + cities_box[2] - glyph0.width) // 2
            y0 = cities_box[1]
            score_fn0, _ = _make_element_score_fns(
                ref=ref,
                roi=cities_roi,
                bg=bg,
                text=style.cities_text,
                font_path=font_path,
                fill=cities_fill,
                rotate=None,
                align="center",
                spacing_for_size=spacing_for_size,
                variable_weight=weight,
            )
            score0 = float(score_fn0(int(x0), int(y0), int(size0)))
            if best_cities is None or score0 < float(best_cities["score0"]):
                best_cities = {
                    "font_path": font_path,
                    "weight": weight,
                    "spacing_scale": float(spacing_scale),
                    "size0": int(size0),
                    "x0": int(x0),
                    "y0": int(y0),
                    "score0": float(score0),
                }

    assert best_cities is not None
    font_cities: Path = best_cities["font_path"]
    cities_weight: int | None = best_cities["weight"]
    cities_spacing_scale = float(best_cities["spacing_scale"])
    cities_spacing_for_size = lambda s: max(0, int(round(s * cities_spacing_scale)))

    date_size0 = _fit_font_size_to_box(
        text=style.date_text,
        font_path=font_date,
        fill=date_fill,
        max_w=date_box[2] - date_box[0],
        max_h=date_box[3] - date_box[1],
        min_size=40,
        max_size=700,
    )
    month_size0 = _fit_font_size_to_box(
        text=style.month_text,
        font_path=font_month,
        fill=month_fill,
        max_w=month_box[2] - month_box[0],
        max_h=month_box[3] - month_box[1],
        min_size=40,
        max_size=700,
        rotate=DEFAULT_MONTH_ROTATE_DEGREES,
    )
    title_size0 = _fit_font_size_to_box(
        text=style.title_text,
        font_path=font_title,
        fill=title_fill,
        max_w=title_box[2] - title_box[0],
        max_h=title_box[3] - title_box[1],
        min_size=40,
        max_size=700,
    )
    cities_size0 = int(best_cities["size0"])

    date_glyph0 = _render_text_rgba(
        text=style.date_text,
        font=_load_font(font_date, date_size0),
        fill=date_fill,
    )
    date_x0 = date_box[2] - date_glyph0.width
    date_y0 = date_box[1]

    month_rotation = DEFAULT_MONTH_ROTATE_DEGREES
    month_font0 = _load_font(font_month, month_size0)
    month_glyph0 = _render_text_rgba(text=style.month_text, font=month_font0, fill=month_fill)
    month_glyph0 = _trim_transparency(month_glyph0.rotate(month_rotation, expand=True))
    month_x0 = month_box[2] - month_glyph0.width
    month_y0 = month_box[1]

    title_x0 = title_box[0]
    title_y0 = title_box[1]

    cities_font0 = _load_font(font_cities, cities_size0)
    if cities_weight is not None:
        _set_variable_weight(cities_font0, cities_weight)
    cities_glyph0 = _render_text_rgba(
        text=style.cities_text,
        font=cities_font0,
        fill=cities_fill,
        align="center",
        spacing=cities_spacing_for_size(cities_size0),
    )
    cities_x0 = int(best_cities["x0"])
    cities_y0 = int(best_cities["y0"])

    def bounds_for_box(box: tuple[int, int, int, int], size0: int, pad_xy: int, pad_s: int) -> tuple:
        x0, y0, x1, y1 = box
        return (
            (max(-500, x0 - pad_xy), min(CANVAS_SIZE[0] + 500, x1 + pad_xy)),
            (max(-500, y0 - pad_xy), min(CANVAS_SIZE[1] + 500, y1 + pad_xy)),
            (max(8, size0 - pad_s), min(900, size0 + pad_s)),
        )

    date_bounds = bounds_for_box(date_box, date_size0, pad_xy=120, pad_s=120)
    month_bounds = bounds_for_box(month_box, month_size0, pad_xy=140, pad_s=160)
    title_bounds = bounds_for_box(title_box, title_size0, pad_xy=140, pad_s=140)
    cities_bounds = bounds_for_box(cities_box, cities_size0, pad_xy=120, pad_s=100)

    score_fn, glyph_for_size = _make_element_score_fns(
        ref=ref,
        roi=date_roi,
        bg=bg,
        text=style.date_text,
        font_path=font_date,
        fill=date_fill,
        rotate=None,
        align="left",
        spacing_for_size=None,
        variable_weight=None,
    )
    best_date, date_score = _optimize_with_restarts(
        roi=date_roi,
        initial=(int(date_x0), int(date_y0), int(date_size0)),
        bounds=date_bounds,
        score_fn=score_fn,
        glyph_for_size=glyph_for_size,
    )

    score_fn, glyph_for_size = _make_element_score_fns(
        ref=ref,
        roi=month_roi,
        bg=bg,
        text=style.month_text,
        font_path=font_month,
        fill=month_fill,
        rotate=month_rotation,
        align="left",
        spacing_for_size=None,
        variable_weight=None,
    )
    best_month, month_score = _optimize_with_restarts(
        roi=month_roi,
        initial=(int(month_x0), int(month_y0), int(month_size0)),
        bounds=month_bounds,
        score_fn=score_fn,
        glyph_for_size=glyph_for_size,
    )

    score_fn, glyph_for_size = _make_element_score_fns(
        ref=ref,
        roi=title_roi,
        bg=bg,
        text=style.title_text,
        font_path=font_title,
        fill=title_fill,
        rotate=None,
        align="left",
        spacing_for_size=None,
        variable_weight=None,
    )
    best_title, title_score = _optimize_with_restarts(
        roi=title_roi,
        initial=(int(title_x0), int(title_y0), int(title_size0)),
        bounds=title_bounds,
        score_fn=score_fn,
        glyph_for_size=glyph_for_size,
    )

    score_fn, glyph_for_size = _make_element_score_fns(
        ref=ref,
        roi=cities_roi,
        bg=bg,
        text=style.cities_text,
        font_path=font_cities,
        fill=cities_fill,
        rotate=None,
        align="center",
        spacing_for_size=cities_spacing_for_size,
        variable_weight=cities_weight,
    )
    best_cities, cities_score = _optimize_with_restarts(
        roi=cities_roi,
        initial=(int(cities_x0), int(cities_y0), int(cities_size0)),
        bounds=cities_bounds,
        score_fn=score_fn,
        glyph_for_size=glyph_for_size,
    )

    img = _render_full_intro(
        bg=bg,
        date=(style.date_text, font_date, date_fill, best_date),
        month=(style.month_text, font_month, month_fill, best_month, month_rotation),
        title=(style.title_text, font_title, title_fill, best_title),
        cities=(
            style.cities_text,
            font_cities,
            cities_fill,
            best_cities,
            cities_weight,
            cities_spacing_scale,
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = out_dir / f"codex_intro_{style.name}_aligned.png"
    img.save(aligned_path)

    ref_arr = np.asarray(ref.convert("RGBA"), dtype=np.int16)
    gen_arr = np.asarray(img.convert("RGBA"), dtype=np.int16)
    diff = np.abs(ref_arr[..., :3] - gen_arr[..., :3]).max(axis=-1).astype(np.uint8)
    diff_img = Image.fromarray(np.clip(diff.astype(np.int16) * 6, 0, 255).astype(np.uint8), mode="L")
    diff_path = out_dir / f"debug_diff_{style.name}.png"
    diff_img.save(diff_path)

    result = {
        "style": style.name,
        "reference": str(style.ref_path),
        "output": str(aligned_path),
        "diff": str(diff_path),
        "bg": bg,
        "month_rotation_degrees": month_rotation,
        "date": {
            "text": style.date_text,
            "font": font_date.name,
            "fill": date_fill,
            "x": best_date[0],
            "y": best_date[1],
            "size": best_date[2],
            "score": float(date_score),
        },
        "month": {
            "text": style.month_text,
            "font": font_month.name,
            "fill": month_fill,
            "x": best_month[0],
            "y": best_month[1],
            "size": best_month[2],
            "score": float(month_score),
        },
        "title": {
            "text": style.title_text,
            "font": font_title.name,
            "fill": title_fill,
            "x": best_title[0],
            "y": best_title[1],
            "size": best_title[2],
            "score": float(title_score),
        },
        "cities": {
            "text": style.cities_text,
            "font": font_cities.name,
            "fill": cities_fill,
            "x": best_cities[0],
            "y": best_cities[1],
            "size": best_cities[2],
            "score": float(cities_score),
            "spacing_scale": float(cities_spacing_scale),
            "variable_weight": cities_weight,
        },
    }

    print(f"\n== {style.name.upper()} ==")
    print(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize intro text placement/sizing to match reference PNGs.")
    parser.add_argument("--style", choices=["day", "weekend", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.style in {"day", "weekend"}:
        optimize_style(style=_style_by_name(args.style), out_dir=args.out_dir)
        return 0

    optimize_style(style=STYLE_DAY, out_dir=args.out_dir)
    optimize_style(style=STYLE_WEEKEND, out_dir=args.out_dir)

    day_diff = Image.open(args.out_dir / "debug_diff_day.png").convert("L")
    weekend_diff = Image.open(args.out_dir / "debug_diff_weekend.png").convert("L")
    combined = Image.new("L", (day_diff.width * 2, day_diff.height))
    combined.paste(day_diff, (0, 0))
    combined.paste(weekend_diff, (day_diff.width, 0))
    combined.save(args.out_dir / "debug_diff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
