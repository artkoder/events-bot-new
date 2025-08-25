from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, Index, JSON, text
from sqlalchemy.types import Enum as SAEnum


class User(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    is_superadmin: bool = False
    is_partner: bool = False
    organization: Optional[str] = None
    location: Optional[str] = None
    blocked: bool = False
    last_partner_reminder: Optional[datetime] = None


class PendingUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    requested_at: datetime = Field(default_factory=datetime.utcnow)


class RejectedUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    rejected_at: datetime = Field(default_factory=datetime.utcnow)


class Channel(SQLModel, table=True):
    channel_id: int = Field(primary_key=True)
    title: Optional[str] = None
    username: Optional[str] = None
    is_admin: bool = False
    is_registered: bool = False
    is_asset: bool = False
    daily_time: Optional[str] = None
    last_daily: Optional[str] = None


class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class Event(SQLModel, table=True):
    __table_args__ = (
        Index("idx_event_date", "date"),
        Index("idx_event_end_date", "end_date"),
        Index("idx_event_city", "city"),
        Index("idx_event_type", "event_type"),
        Index("idx_event_is_free", "is_free"),
        Index("ix_event_date_city", "date", "city"),
        Index("ix_event_date_festival", "date", "festival"),
        Index("ix_event_content_hash", "content_hash"),
        Index(
            "ix_event_telegraph_not_null",
            "date",
            sqlite_where=text("telegraph_url IS NOT NULL"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str
    festival: Optional[str] = None
    date: str
    time: str
    location_name: str
    location_address: Optional[str] = None
    city: Optional[str] = None
    ticket_price_min: Optional[int] = None
    ticket_price_max: Optional[int] = None
    ticket_link: Optional[str] = None
    event_type: Optional[str] = None
    emoji: Optional[str] = None
    end_date: Optional[str] = None
    is_free: bool = False
    pushkin_card: bool = False
    silent: bool = False
    telegraph_path: Optional[str] = None
    source_text: str
    telegraph_url: Optional[str] = None
    ics_url: Optional[str] = None
    source_post_url: Optional[str] = None
    source_vk_post_url: Optional[str] = None
    ics_hash: Optional[str] = None
    ics_file_id: Optional[str] = None
    ics_updated_at: Optional[datetime] = None
    ics_post_url: Optional[str] = None
    ics_post_id: Optional[int] = None
    source_chat_id: Optional[int] = None
    source_message_id: Optional[int] = None
    creator_id: Optional[int] = None
    photo_urls: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    photo_count: int = 0
    added_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = None


class MonthPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    month: str = Field(primary_key=True)
    url: str
    path: str
    url2: Optional[str] = None
    path2: Optional[str] = None
    content_hash: Optional[str] = None
    content_hash2: Optional[str] = None


class WeekendPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    start: str = Field(primary_key=True)
    url: str
    path: str
    vk_post_url: Optional[str] = None
    content_hash: Optional[str] = None


class WeekPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    start: str = Field(primary_key=True)
    vk_post_url: Optional[str] = None
    content_hash: Optional[str] = None


class Festival(SQLModel, table=True):
    __table_args__ = (Index("idx_festival_name", "name"), {"extend_existing": True})
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    full_name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    telegraph_url: Optional[str] = None
    telegraph_path: Optional[str] = None
    vk_post_url: Optional[str] = None
    vk_poll_url: Optional[str] = None
    photo_url: Optional[str] = None
    website_url: Optional[str] = None
    vk_url: Optional[str] = None
    tg_url: Optional[str] = None
    ticket_url: Optional[str] = None
    location_name: Optional[str] = None
    location_address: Optional[str] = None
    city: Optional[str] = None
    source_text: Optional[str] = None


class JobTask(str, Enum):
    telegraph_build = "telegraph_build"
    vk_sync = "vk_sync"
    ics_publish = "ics_publish"
    month_pages = "month_pages"
    weekend_pages = "weekend_pages"
    week_pages = "week_pages"
    festival_pages = "festival_pages"
    fest_nav_update_all = "fest_nav:update_all"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobOutbox(SQLModel, table=True):
    __table_args__ = (
        Index("ix_job_outbox_event_task", "event_id", "task"),
        Index("ix_job_outbox_status_next_run_at", "status", "next_run_at"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int
    task: JobTask = Field(sa_column=Column(SAEnum(JobTask)))
    payload: dict | None = Field(default=None, sa_column=Column(JSON))
    status: JobStatus = Field(
        default=JobStatus.pending, sa_column=Column(SAEnum(JobStatus))
    )
    attempts: int = 0
    last_error: Optional[str] = None
    last_result: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    next_run_at: datetime = Field(default_factory=datetime.utcnow)


def create_all(engine) -> None:
    SQLModel.metadata.create_all(engine)
