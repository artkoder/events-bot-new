from __future__ import annotations

import math
import random
from typing import Iterable

from models import Event


class KaggleClient:
    """Lightweight helper that mimics scoring events via Kaggle notebooks.

    In production this can be swapped with a real HTTP call. For now we
    prioritize events with higher ``video_include_count`` and more photos, and
    add a small amount of randomness to avoid ties.
    """

    def __init__(self, seed: int | None = None):
        self._rand = random.Random(seed)

    def score(self, events: Iterable[Event]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for e in events:
            weight = e.video_include_count or 0
            weight += min(e.photo_count, 4) * 0.5
            if e.is_free:
                weight += 0.25
            rarity = 1.0 / (1 + (len(e.topics or []))) if hasattr(e, "topics") else 1.0
            jitter = self._rand.random() * 0.1
            scores[e.id] = round(weight + rarity + jitter, 3)
        return scores

    def rank(self, events: Iterable[Event]) -> list[Event]:
        scored = self.score(events)
        return sorted(
            events,
            key=lambda ev: (-scored.get(ev.id, 0.0), ev.date, ev.time, ev.id),
        )
