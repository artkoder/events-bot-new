"""Types for 3D preview generation."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Preview3DSessionStatus(str, Enum):
    """Status of a 3D preview generation session."""
    CREATED = "CREATED"
    RENDERING = "RENDERING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class Preview3DSession:
    """Tracks a 3D preview generation session."""
    id: int
    status: Preview3DSessionStatus
    month: str  # e.g., "2026-02"
    mode: str  # "new", "all", "month"
    event_ids: list[int]
    created_at: datetime
    finished_at: Optional[datetime] = None
    kaggle_dataset: Optional[str] = None
    kaggle_kernel_ref: Optional[str] = None
    error: Optional[str] = None
    results_json: Optional[str] = None


@dataclass
class Preview3DResult:
    """Result of processing a single event."""
    event_id: int
    preview_url: Optional[str]
    status: str  # "ok", "error", "skip"
    error: Optional[str] = None
