from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class OverlayPlacement:
    x: int
    y: int
    w: int
    h: int
    score: float


def _wrap_text(
    text: str,
    *,
    max_width_px: int,
    font,
    font_scale: float,
    thickness: int,
    max_lines: int = 3,
) -> list[str]:
    parts: list[str] = []
    for block in (text or "").splitlines():
        block = " ".join(block.split()).strip()
        if block:
            parts.append(block)

    if not parts:
        return []

    out: list[str] = []
    for block in parts:
        words = block.split(" ")
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            (tw, _th), _baseline = cv2.getTextSize(
                cand, font, font_scale, thickness
            )
            if tw <= max_width_px:
                cur = cand
                continue
            if cur:
                out.append(cur)
                if len(out) >= max_lines:
                    return out
            cur = w
        if cur:
            out.append(cur)
            if len(out) >= max_lines:
                return out
    return out[:max_lines]


def _edge_map(gray: np.ndarray) -> np.ndarray:
    # Robust and fast proxy for "text density": text typically has lots of edges.
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return edges


def _score_region(gray: np.ndarray, edges: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    roi_e = edges[y : y + h, x : x + w]
    roi_g = gray[y : y + h, x : x + w]
    edge_density = float(np.mean(roi_e > 0))
    variance = float(np.var(roi_g)) / (255.0**2)
    # Lower is better: prefer flat / low-edge areas.
    return edge_density + 0.25 * variance


def _find_best_placement(img_bgr: np.ndarray, box_w: int, box_h: int) -> OverlayPlacement:
    h, w = img_bgr.shape[:2]
    margin = max(12, int(min(w, h) * 0.02))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = _edge_map(gray)

    candidates: list[OverlayPlacement] = []

    xs = [
        margin,
        max(margin, (w - box_w) // 2),
        max(margin, w - box_w - margin),
    ]
    ys = [
        margin,
        max(margin, int(h * 0.20)),
        max(margin, int(h * 0.55)),
        max(margin, h - box_h - margin),
    ]

    for y0 in ys:
        for x0 in xs:
            if x0 + box_w + margin > w or y0 + box_h + margin > h:
                continue
            score = _score_region(gray, edges, x0, y0, box_w, box_h)
            # Gentle preference for lower-third placements (badge feels more natural there).
            center_y = y0 + box_h * 0.5
            prefer = abs(center_y - (h * 0.72)) / max(1.0, h)
            score += 0.08 * prefer
            candidates.append(OverlayPlacement(x=x0, y=y0, w=box_w, h=box_h, score=score))

    if not candidates:
        return OverlayPlacement(x=margin, y=max(margin, h - box_h - margin), w=box_w, h=box_h, score=1e9)

    return min(candidates, key=lambda c: c.score)


def apply_poster_overlay(
    input_path: str | Path,
    *,
    text: str,
    out_dir: str | Path,
) -> Path:
    """Render a readable badge with text into a low-edge area of the poster image."""

    in_path = Path(input_path)
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(in_path)

    text = (text or "").strip()
    if not text:
        return in_path

    h, w = img.shape[:2]
    # Start with a reasonable scale for portrait posters.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.7, min(1.25, float(min(w, h)) / 1200.0 * 1.05))
    thickness = max(2, int(round(font_scale * 2.2)))

    max_box_w = int(w * 0.92)
    lines = _wrap_text(
        text,
        max_width_px=max_box_w - 48,
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        max_lines=3,
    )
    if not lines:
        return in_path

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_h = max(sz[1] for sz in line_sizes)
    box_w = min(max_box_w, max(sz[0] for sz in line_sizes) + 48)
    box_h = int(line_h * len(lines) + (len(lines) - 1) * (line_h * 0.35) + 44)
    box_h = min(box_h, int(h * 0.38))

    placement = _find_best_placement(img, box_w, box_h)
    x0, y0 = placement.x, placement.y

    # Draw translucent fill.
    overlay = img.copy()
    fill = (0, 0, 0)  # black
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), fill, thickness=-1)
    alpha = 0.62
    img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    # Border stroke.
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), thickness=3)

    # Text with outline for readability.
    pad_x = 24
    pad_y = 24
    y = y0 + pad_y + line_h
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = x0 + pad_x
        # Outline (black) then fill (white).
        cv2.putText(img, line, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(line_h * 1.35)
        if y > y0 + box_h - 10:
            break

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}__overlay.png"
    cv2.imwrite(str(out_path), img)
    return out_path

