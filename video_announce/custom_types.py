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
    reason: str | None = None
    mandatory: bool = False
    selected: bool | None = None
    selected_reason: str | None = None
    about: str | None = None
    description: str | None = None
    poster_ocr_text: str | None = None
    poster_ocr_title: str | None = None


@dataclass
class RankedChoice:
    event_id: int
    score: float
    reason: str | None = None
    selected: bool | None = None
    selected_reason: str | None = None
    about: str | None = None
    description: str | None = None


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
    candidate_limit: int = 80
    default_selected_min: int = 6
    default_selected_max: int = 8
    primary_window_days: int = 3
    fallback_window_days: int = 10
    promoted_event_ids: set[int] | None = None
    instruction: str | None = None
    random_order: bool = False
    allow_empty_ocr: bool = False


@dataclass
class SelectionBuildResult:
    ranked: list[RankedEvent]
    default_ready_ids: set[int]
    mandatory_ids: set[int]
    candidates: list[Event]
    selected_ids: set[int]
    schedule_map: dict[int, str] = field(default_factory=dict)
    occurrences_map: dict[int, list[dict[str, list[str]]]] = field(default_factory=dict)
    intro_text: str | None = None
    intro_text_valid: bool = True


@dataclass
class SelectionResult:
    events: list[RankedEvent]
    session: VideoAnnounceSession


@dataclass
class PosterEnrichment:
    event_id: int
    title: str | None
    text: str | None
    source: str | None = None


@dataclass
class FinalizedItem:
    event_id: int
    title: str
    about: str
    description: str
    use_ocr: bool = False
    poster_source: str | None = None
