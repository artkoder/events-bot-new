from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def _edge_density(gray: np.ndarray, x0: int, y0: int, bw: int, bh: int) -> float:
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    roi = edges[y0 : y0 + bh, x0 : x0 + bw]
    return float(np.mean(roi > 0))


def _find_best_placement(img_bgr: np.ndarray, box_w: int, box_h: int) -> OverlayPlacement:
    h, w = img_bgr.shape[:2]
    base = float(min(w, h))
    margin = int(max(18, min(32, base * 0.022)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cand = [
        (margin, int(h * 0.62)),
        (w - box_w - margin, int(h * 0.62)),
        (margin, h - box_h - margin),
        (w - box_w - margin, h - box_h - margin),
        ((w - box_w) // 2, h - box_h - margin),
    ]

    best: OverlayPlacement | None = None
    for x0, y0 in cand:
        x0 = int(max(margin, min(w - box_w - margin, x0)))
        y0 = int(max(margin, min(h - box_h - margin, y0)))
        score = _edge_density(gray, x0, y0, box_w, box_h)
        placement = OverlayPlacement(x=x0, y=y0, w=box_w, h=box_h, score=score)
        if best is None or placement.score < best.score:
            best = placement

    if best is None:
        return OverlayPlacement(x=margin, y=h - box_h - margin, w=box_w, h=box_h, score=1e9)
    return best


def apply_poster_overlay(
    input_path: str | Path,
    *,
    text: str,
    out_dir: str | Path,
    search_roots: list[str | Path] | None = None,
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

    max_box_w = int(w * 0.90)
    pad_x = int(max(42, min(84, base * 0.055)))
    pad_y = int(max(30, min(72, base * 0.045)))
    max_text_w = max_box_w - pad_x * 2

    # Title can wrap to 2 lines; details 1 line each; total <= 4 lines
    lines: list[tuple[str, ImageFont.FreeTypeFont]] = []
    for part in _wrap_line(draw, raw[0], font_title, max_text_w, max_lines=2):
        lines.append((part, font_title))
    for extra in raw[1:]:
        for part in _wrap_line(draw, extra, font_body, max_text_w, max_lines=1):
            lines.append((part, font_body))
        if len(lines) >= 4:
            break

    stroke_w = 2
    line_boxes = [draw.textbbox((0, 0), t, font=f, stroke_width=stroke_w) for t, f in lines]
    line_heights = [b[3] - b[1] for b in line_boxes]
    line_widths = [b[2] - b[0] for b in line_boxes]
    gap = int(max(10, min(22, base * 0.014)))
    text_h = sum(line_heights) + gap * (len(lines) - 1)
    box_w = min(max_box_w, max(line_widths) + pad_x * 2)
    box_h = min(int(h * 0.42), text_h + pad_y * 2)

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

