from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "kaggle"
    / "CrumpleVideo"
    / "poster_overlay.py"
)


def _load_overlay_module():
    spec = importlib.util.spec_from_file_location("crumple_poster_overlay", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bbox_from_diff(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int, int]:
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).sum(axis=2)
    ys, xs = np.where(diff > 30)
    if len(xs) == 0 or len(ys) == 0:
        raise AssertionError("Overlay did not change the image")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def test_apply_poster_overlay_avoids_dense_text_blocks(tmp_path):
    overlay = _load_overlay_module()

    img = np.full((1200, 1200, 3), 255, dtype=np.uint8)
    occupied_blocks = [
        (40, 610, 520, 1170),
        (560, 590, 1160, 1170),
        (660, 80, 1160, 420),
    ]
    for x0, y0, x1, y1 in occupied_blocks:
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    input_path = tmp_path / "poster.png"
    output_dir = tmp_path / "out"
    cv2.imwrite(str(input_path), img)

    out_path = overlay.apply_poster_overlay(
        input_path,
        text="10 марта: 18:30\nКалининград, Дом китобоя",
        out_dir=output_dir,
        highlight_title=False,
    )

    rendered = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    assert rendered is not None
    overlay_bbox = _bbox_from_diff(img, rendered)
    overlay_area = max((overlay_bbox[2] - overlay_bbox[0]) * (overlay_bbox[3] - overlay_bbox[1]), 1)

    overlap_area = 0
    for block in occupied_blocks:
        overlap_area += _rect_overlap(overlay_bbox, block)

    assert overlap_area / overlay_area < 0.05
