#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


_ROW_RE = re.compile(r"timing vk_auto_import_row .*? took_sec=(?P<took>[0-9.]+) stages=(?P<stages>\{.*\})")
_DRAFT_RE = re.compile(r"timing vk_intake_build_drafts .*? stages=(?P<stages>\{.*\})")
_PARSE4O_RE = re.compile(r"timing vk_intake_parse_4o .*? took_sec=(?P<took>[0-9.]+)")


@dataclass(frozen=True)
class Summary:
    count: int
    mean: float
    p50: float
    p90: float
    max_v: float


def _summ(values: list[float]) -> Summary:
    values = [float(v) for v in values if v is not None]
    values.sort()
    if not values:
        return Summary(0, 0.0, 0.0, 0.0, 0.0)
    def _pct(p: float) -> float:
        idx = int(round((len(values) - 1) * p))
        return float(values[max(0, min(idx, len(values) - 1))])
    return Summary(
        count=len(values),
        mean=float(statistics.mean(values)),
        p50=_pct(0.50),
        p90=_pct(0.90),
        max_v=float(max(values)),
    )


def _fmt(s: Summary) -> str:
    return f"n={s.count} mean={s.mean:.2f}s p50={s.p50:.2f}s p90={s.p90:.2f}s max={s.max_v:.2f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="Path to e2e_local_bot_*.log")
    args = ap.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")

    row_took: list[float] = []
    stage_vals: dict[str, list[float]] = defaultdict(list)

    draft_stage_vals: dict[str, list[float]] = defaultdict(list)
    parse4o_vals: list[float] = []

    for line in text.splitlines():
        m = _ROW_RE.search(line)
        if m:
            row_took.append(float(m.group("took")))
            try:
                stages = ast.literal_eval(m.group("stages"))
            except Exception:
                stages = {}
            if isinstance(stages, dict):
                for k, v in stages.items():
                    try:
                        stage_vals[str(k)].append(float(v))
                    except Exception:
                        continue
            continue

        m = _DRAFT_RE.search(line)
        if m:
            try:
                stages = ast.literal_eval(m.group("stages"))
            except Exception:
                stages = {}
            if isinstance(stages, dict):
                for k, v in stages.items():
                    try:
                        draft_stage_vals[str(k)].append(float(v))
                    except Exception:
                        continue
            continue

        m = _PARSE4O_RE.search(line)
        if m:
            parse4o_vals.append(float(m.group("took")))

    print(f"log: {args.log}")
    print("")
    print("vk_auto_import_row total:", _fmt(_summ(row_took)))
    if stage_vals:
        print("")
        print("vk_auto_import_row stages:")
        for k in sorted(stage_vals.keys()):
            print(f"- {k}: {_fmt(_summ(stage_vals[k]))}")

    if draft_stage_vals:
        print("")
        print("vk_intake_build_drafts stages:")
        for k in sorted(draft_stage_vals.keys()):
            print(f"- {k}: {_fmt(_summ(draft_stage_vals[k]))}")

    if parse4o_vals:
        print("")
        print("vk_intake_parse_4o:", _fmt(_summ(parse4o_vals)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

