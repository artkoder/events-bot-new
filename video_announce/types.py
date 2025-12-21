from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Sequence

from models import Event, VideoAnnounceSession, VideoAnnounceItem


@dataclass
class VideoProfile:
    key: str
    title: str
    description: str
    prompt_name: str = "script"
    kaggle_dataset: str | None = None


@dataclass
class RankedEvent:
    event: Event
    score: float
    position: int


@dataclass
class RenderPayload:
    """Structured payload sent to operators for video generation."""

    session: VideoAnnounceSession
    items: list[VideoAnnounceItem]
    events: list[Event]
    scores: dict[int, float] = field(default_factory=dict)
    prepared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionOverview:
    session: VideoAnnounceSession
    items: Sequence[VideoAnnounceItem]
    events: Sequence[Event]

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def started(self) -> datetime | None:
        return self.session.started_at

    @property
    def finished(self) -> datetime | None:
        return self.session.finished_at

    def render_status_line(self) -> str:
        status = self.session.status.value
        if self.session.video_url:
            status = f"{status} → {self.session.video_url}"
        return status


@dataclass
class SelectionContext:
    tz: timezone
    target_date: date | None = None
    profile: VideoProfile | None = None
    limit: int = 20


@dataclass
class SelectionResult:
    events: list[RankedEvent]
    session: VideoAnnounceSession
