from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


@dataclass(frozen=True)
class OverlayPlacement:
    x: int
    y: int
    w: int
    h: int
    score: float


_FONT_PATH: Path | None = None
_FONT_LOGGED = False
_MONTH_NAMES = (
    "январ",
    "феврал",
    "март",
    "апрел",
    "мая",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
)


def _pick_font_path(*, search_roots: list[Path] | None = None) -> Path | None:
    candidates: list[Path] = []
    roots = list(search_roots or [])
    roots.extend([Path.cwd(), Path("/kaggle/working")])

    for root in roots:
        for name in (
            "BebasNeue-Bold.ttf",
            "BebasNeue-Regular.ttf",
            "Oswald-VariableFont_wght.ttf",
        ):
            p = root / name
            if p.exists():
                candidates.append(p)

    inp = Path("/kaggle/input")
    if inp.exists():
        for pat in (
            "*/BebasNeue-Bold.ttf",
            "*/BebasNeue-Regular.ttf",
            "*/assets/BebasNeue-Bold.ttf",
            "*/assets/BebasNeue-Regular.ttf",
            "*/Oswald-VariableFont_wght.ttf",
            "*/assets/Oswald-VariableFont_wght.ttf",
        ):
            candidates.extend(inp.glob(pat))

    # System fallback: guaranteed Cyrillic.
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )

    for p in candidates:
        if p.exists():
            return p
    return None


def _font_supports_cyrillic(font: ImageFont.FreeTypeFont) -> bool:
    for ch in ("Я", "Ж", "Ю", "П"):
        try:
            bbox = font.getbbox(ch)
        except Exception:
            return False
        if not bbox or (bbox[2] - bbox[0]) <= 0:
            return False
    return True


def _load_font(size: int, *, search_roots: list[Path] | None = None) -> ImageFont.FreeTypeFont:
    global _FONT_PATH
    if _FONT_PATH is None:
        _FONT_PATH = _pick_font_path(search_roots=search_roots)

    if _FONT_PATH is not None:
        try:
            f = ImageFont.truetype(str(_FONT_PATH), int(size))
            if _font_supports_cyrillic(f):
                return f
        except Exception:
            pass

    # Hard fallback.
    for p in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if p.exists():
            return ImageFont.truetype(str(p), int(size))
    return ImageFont.load_default()


def _wrap_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_w: int,
    *,
    max_lines: int,
) -> list[str]:
    words = [w for w in (text or "").split(" ") if w]
    out: list[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip() if cur else w
        bbox = draw.textbbox((0, 0), cand, font=font)
        if (bbox[2] - bbox[0]) <= max_w:
            cur = cand
            continue
        if cur:
            out.append(cur)
            if len(out) >= max_lines:
                return out
        cur = w
    if cur:
        out.append(cur)
    return out[:max_lines]


def _looks_like_fact_line(text: str) -> bool:
    lowered = (text or "").strip().casefold()
    if not lowered:
        return False
    if "," in lowered:
        return True
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", lowered):
        return True
    if any(month in lowered for month in _MONTH_NAMES) and re.search(r"\b\d{1,2}\b", lowered):
        return True
    return False


def _edge_density(gray: np.ndarray, x0: int, y0: int, bw: int, bh: int) -> float:
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    roi = edges[y0 : y0 + bh, x0 : x0 + bw]
    return float(np.mean(roi > 0))


def _build_text_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    base = float(min(w, h))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    gradient = cv2.morphologyEx(
        blur,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    _, gradient_bin = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edges = cv2.Canny(blur, 50, 150)
    mask = cv2.bitwise_or(adaptive, gradient_bin)
    mask = cv2.bitwise_or(mask, edges)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(3, int(round(base * 0.010))),
            max(3, int(round(base * 0.006))),
        ),
    )
    expand_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(15, int(round(base * 0.032))),
            max(9, int(round(base * 0.020))),
        ),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, expand_kernel, iterations=1)
    return mask


def _mask_fill_ratio(mask_integral: np.ndarray, x0: int, y0: int, bw: int, bh: int) -> float:
    x1 = x0 + bw
    y1 = y0 + bh
    filled = (
        int(mask_integral[y1, x1])
        - int(mask_integral[y0, x1])
        - int(mask_integral[y1, x0])
        + int(mask_integral[y0, x0])
    )
    return filled / float(max(bw * bh, 1))


def _find_best_placement(img_bgr: np.ndarray, box_w: int, box_h: int) -> OverlayPlacement:
    h, w = img_bgr.shape[:2]
    base = float(min(w, h))
    margin = int(max(18, min(42, base * 0.024)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    text_mask = _build_text_mask(gray)
    mask_integral = cv2.integral((text_mask > 0).astype(np.uint8), sdepth=cv2.CV_32S)

    max_x = max(margin, w - box_w - margin)
    max_y = max(margin, h - box_h - margin)
    step_x = max(margin, box_w // 4)
    step_y = max(margin, box_h // 4)

    x_positions = set(range(margin, max_x + 1, step_x))
    y_positions = set(range(margin, max_y + 1, step_y))
    x_positions.update({margin, max_x, max(margin, (w - box_w) // 2)})
    y_positions.update({margin, max_y, max(margin, (h - box_h) // 2), max(margin, int(h * 0.62))})

    best: OverlayPlacement | None = None
    for y0 in sorted(y_positions):
        for x0 in sorted(x_positions):
            x0 = int(max(margin, min(max_x, x0)))
            y0 = int(max(margin, min(max_y, y0)))
            fill_ratio = _mask_fill_ratio(mask_integral, x0, y0, box_w, box_h)
            edge_score = _edge_density(gray, x0, y0, box_w, box_h)
            variance_score = float(np.std(gray[y0 : y0 + box_h, x0 : x0 + box_w])) / 255.0
            center_x = (x0 + (box_w / 2.0)) / float(max(w, 1))
            center_y = (y0 + (box_h / 2.0)) / float(max(h, 1))
            preference = abs(center_y - 0.72) * 0.10 + (0.5 - abs(center_x - 0.5)) * 0.05
            hard_overlap_penalty = 0.0
            if fill_ratio > 0.015:
                hard_overlap_penalty = fill_ratio * 40.0
            score = (
                fill_ratio * 12.0
                + hard_overlap_penalty
                + edge_score * 2.5
                + variance_score * 0.8
                + preference
            )
            placement = OverlayPlacement(x=x0, y=y0, w=box_w, h=box_h, score=score)
            if best is None or placement.score < best.score:
                best = placement

    if best is None:
        return OverlayPlacement(
            x=margin,
            y=h - box_h - margin,
            w=box_w,
            h=box_h,
            score=1e9,
        )
    return best


def apply_poster_overlay(
    input_path: str | Path,
    *,
    text: str,
    out_dir: str | Path,
    search_roots: list[str | Path] | None = None,
    highlight_title: bool | None = None,
) -> Path:
    """Draw a modern badge using a Cyrillic-capable TTF font (BebasNeue if available)."""

    global _FONT_LOGGED

    in_path = Path(input_path)
    text = (text or "").strip()
    if not text:
        return in_path

    bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return in_path

    h, w = bgr.shape[:2]
    base = float(min(w, h))
    roots: list[Path] = []
    if search_roots:
        for r in search_roots:
            try:
                roots.append(Path(r))
            except Exception:
                pass

    # Typography
    title_size = int(max(42, min(98, base * 0.065)))
    body_size = int(max(30, min(68, base * 0.046)))
    font_title = _load_font(title_size, search_roots=roots)
    font_body = _load_font(body_size, search_roots=roots)

    if not _FONT_LOGGED:
        _FONT_LOGGED = True
        print(f"✅ Overlay font: {_FONT_PATH} (title_ok={_font_supports_cyrillic(font_title)} body_ok={_font_supports_cyrillic(font_body)})")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(img)

    raw = [" ".join(l.split()).strip() for l in text.splitlines() if " ".join(l.split()).strip()]
    if not raw:
        return in_path

    if highlight_title is None:
        highlight_title = len(raw) > 1 and not _looks_like_fact_line(raw[0])

    max_box_w = int(w * (0.66 if highlight_title else 0.54))
    pad_x = int(max(36, min(78, base * 0.050)))
    pad_y = int(max(26, min(64, base * 0.040)))
    max_text_w = max_box_w - pad_x * 2

    # Title can wrap to 2 lines; fact-only overlays use a more compact body-only layout.
    lines: list[tuple[str, ImageFont.FreeTypeFont]] = []
    remaining = list(raw)
    body_wrap_lines = 2 if highlight_title else 3

    if highlight_title and remaining:
        for part in _wrap_line(draw, remaining[0], font_title, max_text_w, max_lines=2):
            lines.append((part, font_title))
            if len(lines) >= 6:
                break
        remaining = remaining[1:]

    for extra in remaining:
        for part in _wrap_line(draw, extra, font_body, max_text_w, max_lines=body_wrap_lines):
            lines.append((part, font_body))
            if len(lines) >= 6:
                break
        if len(lines) >= 6:
            break

    stroke_w = 2
    line_boxes = [draw.textbbox((0, 0), t, font=f, stroke_width=stroke_w) for t, f in lines]
    line_heights = [b[3] - b[1] for b in line_boxes]
    line_widths = [b[2] - b[0] for b in line_boxes]
    gap = int(max(10, min(22, base * 0.014)))
    text_h = sum(line_heights) + gap * (len(lines) - 1)
    box_w = min(max_box_w, max(line_widths) + pad_x * 2)
    box_h = min(int(h * 0.44), text_h + pad_y * 2)

    placement = _find_best_placement(bgr, box_w, box_h)
    x0, y0 = placement.x, placement.y

    radius = int(max(16, min(40, base * 0.03)))
    border = max(2, int(round(base * 0.0022)))
    shadow_off = int(max(6, min(14, base * 0.01)))
    shadow_blur = int(max(10, min(28, base * 0.02)))

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [x0 + shadow_off, y0 + shadow_off, x0 + box_w + shadow_off, y0 + box_h + shadow_off],
        radius=radius,
        fill=(0, 0, 0, 110),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    img = Image.alpha_composite(img, shadow)

    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=radius, fill=(12, 12, 14, 215))
    pd.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=radius,
        outline=(255, 255, 255, 150),
        width=border,
    )
    img = Image.alpha_composite(img, panel)

    draw = ImageDraw.Draw(img)
    tx = x0 + pad_x
    ty = y0 + pad_y
    for (t, f), lh in zip(lines, line_heights):
        draw.text(
            (tx, ty),
            t,
            font=f,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_w,
            stroke_fill=(0, 0, 0, 140),
        )
        ty += lh + gap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}__overlay.png"
    img.convert("RGB").save(out_path, format="PNG")
    return out_path
