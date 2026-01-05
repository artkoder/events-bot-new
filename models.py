from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from sqlmodel import Field, SQLModel
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    JSON,
    Boolean,
    Integer,
    SmallInteger,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import Enum as SAEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


TOPIC_LABELS: dict[str, str] = {
    "STANDUP": "Стендап и комедия",
    "QUIZ_GAMES": "Квизы и игры",
    "OPEN_AIR": "Фестивали и open-air",
    "PARTIES": "Вечеринки",
    "CONCERTS": "Концерты",
    "MOVIES": "Кино",
    "EXHIBITIONS": "Выставки и арт",
    "THEATRE": "Театр",
    "THEATRE_CLASSIC": "Классический театр и драма",
    "THEATRE_MODERN": "Современный и экспериментальный театр",
    "LECTURES": "Лекции и встречи",
    "MASTERCLASS": "Мастер-классы",
    "PSYCHOLOGY": "Психология",
    "SCIENCE_POP": "Научпоп",
    "HANDMADE": "Хендмейд/маркеты/ярмарки/МК",
    "FASHION": "Мода и стиль",
    "NETWORKING": "Нетворкинг и карьера",
    "ACTIVE": "Активный отдых и спорт",
    "PERSONALITIES": "Личности и встречи",
    "HISTORICAL_IMMERSION": "Исторические реконструкции и погружение",
    "KIDS_SCHOOL": "Дети и школа",
    "FAMILY": "Семейные события",
    "URBANISM": "Урбанистика",
    "KRAEVEDENIE_KALININGRAD_OBLAST": "Краеведение Калининградской области",
}

TOPIC_IDENTIFIERS: set[str] = set(TOPIC_LABELS.keys())

_TOPIC_LEGACY_ALIASES: dict[str, str] = {
    "art": "EXHIBITIONS",
    "искусство": "EXHIBITIONS",
    "культура": "EXHIBITIONS",
    "выставка": "EXHIBITIONS",
    "выставки": "EXHIBITIONS",
    "gallery": "EXHIBITIONS",
    "галерея": "EXHIBITIONS",
    "ART": "EXHIBITIONS",
    "history_ru": "LECTURES",
    "HISTORY_RU": "LECTURES",
    "history": "LECTURES",
    "история": "LECTURES",
    "история россии": "LECTURES",
    "лекция": "LECTURES",
    "лекции": "LECTURES",
    "встреча": "LECTURES",
    "встречи": "LECTURES",
    "дискуссия": "LECTURES",
    "BUSINESS": "LECTURES",
    "business": "LECTURES",
    "предпринимательство": "LECTURES",
    "urbanism": "URBANISM",
    "урбанистика": "URBANISM",
    "урбанистический": "URBANISM",
    "краеведение": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "краевед": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "краеведческий": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "краеведческие": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "калининград": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "kaliningrad": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "калининградская область": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "калининградской области": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "кёнигсберг": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "кенигсберг": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "kenigsberg": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "koenigsberg": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "konigsberg": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "königsberg": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "kenig": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "янтарный край": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "янтарного края": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "39 регион": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "39-й регион": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "39й регион": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "39йрегион": "KRAEVEDENIE_KALININGRAD_OBLAST",
    "LITERATURE": "LECTURES",
    "literature": "LECTURES",
    "книги": "LECTURES",
    "TECH": "SCIENCE_POP",
    "tech": "SCIENCE_POP",
    "технологии": "SCIENCE_POP",
    "ит": "SCIENCE_POP",
    "психология": "PSYCHOLOGY",
    "psychology": "PSYCHOLOGY",
    "mental health": "PSYCHOLOGY",
    "science": "SCIENCE_POP",
    "science_pop": "SCIENCE_POP",
    "научпоп": "SCIENCE_POP",
    "CINEMA": "MOVIES",
    "cinema": "MOVIES",
    "кино": "MOVIES",
    "фильм": "MOVIES",
    "фильмы": "MOVIES",
    "movie": "MOVIES",
    "movies": "MOVIES",
    "MUSIC": "CONCERTS",
    "music": "CONCERTS",
    "музыка": "CONCERTS",
    "концерт": "CONCERTS",
    "концерты": "CONCERTS",
    "PARTY": "PARTIES",
    "party": "PARTIES",
    "вечеринка": "PARTIES",
    "вечер": "PARTIES",
    "вечеринки": "PARTIES",
    "STANDUP": "STANDUP",
    "standup": "STANDUP",
    "стендап": "STANDUP",
    "стендапы": "STANDUP",
    "комедия": "STANDUP",
    "quiz": "QUIZ_GAMES",
    "quizzes": "QUIZ_GAMES",
    "квиз": "QUIZ_GAMES",
    "квизы": "QUIZ_GAMES",
    "игры": "QUIZ_GAMES",
    "настолки": "QUIZ_GAMES",
    "настольные игры": "QUIZ_GAMES",
    "open_air": "OPEN_AIR",
    "open air": "OPEN_AIR",
    "open-air": "OPEN_AIR",
    "openair": "OPEN_AIR",
    "фестиваль": "OPEN_AIR",
    "фестивали": "OPEN_AIR",
    "мастер-класс": "MASTERCLASS",
    "мастер класс": "MASTERCLASS",
    "мастер-классы": "MASTERCLASS",
    "воркшоп": "MASTERCLASS",
    "workshop": "MASTERCLASS",
    "workshops": "MASTERCLASS",
    "театр": "THEATRE",
    "спектакль": "THEATRE",
    "спектакли": "THEATRE",
    "performance": "THEATRE",
    "performances": "THEATRE",
    "классический спектакль": "THEATRE_CLASSIC",
    "классический театр": "THEATRE_CLASSIC",
    "classic theatre": "THEATRE_CLASSIC",
    "драма": "THEATRE_CLASSIC",
    "драмы": "THEATRE_CLASSIC",
    "драматический театр": "THEATRE_CLASSIC",
    "dramatic theatre": "THEATRE_CLASSIC",
    "классика": "THEATRE_CLASSIC",
    "современный театр": "THEATRE_MODERN",
    "современные спектакли": "THEATRE_MODERN",
    "модерн": "THEATRE_MODERN",
    "экспериментальный театр": "THEATRE_MODERN",
    "experimental theatre": "THEATRE_MODERN",
    "modern theatre": "THEATRE_MODERN",
    "HANDMADE": "HANDMADE",
    "handmade": "HANDMADE",
    "hand-made": "HANDMADE",
    "маркет": "HANDMADE",
    "маркеты": "HANDMADE",
    "маркет-плейс": "HANDMADE",
    "маркетплейс": "HANDMADE",
    "маркетплейсы": "HANDMADE",
    "ярмарка": "HANDMADE",
    "ярмарки": "HANDMADE",
    "ярмарка выходного дня": "HANDMADE",
    "хендмейд": "HANDMADE",
    "HAND-MADE": "HANDMADE",
    "FASHION": "FASHION",
    "fashion": "FASHION",
    "fashion week": "FASHION",
    "показ мод": "FASHION",
    "показы мод": "FASHION",
    "fashion show": "FASHION",
    "fashion shows": "FASHION",
    "styling": "FASHION",
    "stylist": "FASHION",
    "style": "FASHION",
    "стиль": "FASHION",
    "стилист": "FASHION",
    "стилисты": "FASHION",
    "стилизация": "FASHION",
    "мода": "FASHION",
    "модный показ": "FASHION",
    "модные показы": "FASHION",
    "модный дом": "FASHION",
    "NETWORKING": "NETWORKING",
    "networking": "NETWORKING",
    "network": "NETWORKING",
    "нетворкинг": "NETWORKING",
    "нетворк": "NETWORKING",
    "знакомства": "NETWORKING",
    "карьера": "NETWORKING",
    "деловые встречи": "NETWORKING",
    "бизнес-завтрак": "NETWORKING",
    "бизнес завтрак": "NETWORKING",
    "business breakfast": "NETWORKING",
    "карьерный вечер": "NETWORKING",
    "ACTIVE": "ACTIVE",
    "active": "ACTIVE",
    "sport": "ACTIVE",
    "sports": "ACTIVE",
    "спорт": "ACTIVE",
    "спортивные": "ACTIVE",
    "спортзал": "ACTIVE",
    "активности": "ACTIVE",
    "активность": "ACTIVE",
    "активный отдых": "ACTIVE",
    "фитнес": "ACTIVE",
    "йога": "ACTIVE",
    "yoga": "ACTIVE",
    "пробежка": "ACTIVE",
    "PERSONALITIES": "PERSONALITIES",
    "personalities": "PERSONALITIES",
    "personality": "PERSONALITIES",
    "персоны": "PERSONALITIES",
    "личности": "PERSONALITIES",
    "встреча с автором": "PERSONALITIES",
    "встреча с героем": "PERSONALITIES",
    "встреча с артистом": "PERSONALITIES",
    "встреча с персонами": "PERSONALITIES",
    "книжный клуб": "PERSONALITIES",
    "книжные клубы": "PERSONALITIES",
    "book club": "PERSONALITIES",
    "реконструкция": "HISTORICAL_IMMERSION",
    "реконструкции": "HISTORICAL_IMMERSION",
    "историческое погружение": "HISTORICAL_IMMERSION",
    "исторические костюмы": "HISTORICAL_IMMERSION",
    "викинги": "HISTORICAL_IMMERSION",
    "средневековье": "HISTORICAL_IMMERSION",
    "KIDS_SCHOOL": "KIDS_SCHOOL",
    "kids_school": "KIDS_SCHOOL",
    "kids": "KIDS_SCHOOL",
    "дети": "KIDS_SCHOOL",
    "детям": "KIDS_SCHOOL",
    "детский": "KIDS_SCHOOL",
    "детские": "KIDS_SCHOOL",
    "школа": "KIDS_SCHOOL",
    "школьники": "KIDS_SCHOOL",
    "образование": "KIDS_SCHOOL",
    "FAMILY": "FAMILY",
    "family": "FAMILY",
    "семья": "FAMILY",
    "семейные": "FAMILY",
    "семейный": "FAMILY",
    "для всей семьи": "FAMILY",
}

TOPIC_IDENTIFIERS_BY_CASEFOLD: dict[str, str] = {
    key.casefold(): key for key in TOPIC_IDENTIFIERS
}
TOPIC_IDENTIFIERS_BY_CASEFOLD.update(
    {alias.casefold(): canonical for alias, canonical in _TOPIC_LEGACY_ALIASES.items()}
)


def normalize_topic_identifier(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate in TOPIC_IDENTIFIERS:
        return candidate
    return TOPIC_IDENTIFIERS_BY_CASEFOLD.get(candidate.casefold())


class User(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    is_superadmin: bool = False
    is_partner: bool = False
    organization: Optional[str] = None
    location: Optional[str] = None
    blocked: bool = False
    last_partner_reminder: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )


class PendingUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    requested_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


class RejectedUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    rejected_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


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
    search_digest: Optional[str] = None
    festival: Optional[str] = None
    date: str
    time: str
    location_name: str
    location_address: Optional[str] = None
    city: Optional[str] = None
    ticket_price_min: Optional[int] = None
    ticket_price_max: Optional[int] = None
    ticket_link: Optional[str] = None
    vk_ticket_short_url: Optional[str] = None
    vk_ticket_short_key: Optional[str] = None
    vk_ics_short_url: Optional[str] = None
    vk_ics_short_key: Optional[str] = None
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
    vk_repost_url: Optional[str] = None
    ics_hash: Optional[str] = None
    ics_file_id: Optional[str] = None
    ics_updated_at: Optional[datetime] = None
    ics_post_url: Optional[str] = None
    ics_post_id: Optional[int] = None
    source_chat_id: Optional[int] = None
    source_message_id: Optional[int] = None
    creator_id: Optional[int] = None
    tourist_label: Optional[int] = Field(
        default=None, sa_column=Column(SmallInteger)
    )
    tourist_factors: list[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    tourist_note: Optional[str] = None
    tourist_label_by: Optional[int] = None
    tourist_label_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    tourist_label_source: Optional[str] = None
    photo_urls: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    photo_count: int = 0
    video_include_count: int = 0
    topics: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    topics_manual: bool = False
    added_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )
    content_hash: Optional[str] = None
    ticket_status: Optional[str] = None  # 'available', 'sold_out', or None/unknown
    linked_event_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    preview_3d_url: Optional[str] = None  # 3D preview generated by Blender on Kaggle


class VideoAnnounceSessionStatus(str, Enum):
    CREATED = "CREATED"
    SELECTED = "SELECTED"
    RENDERING = "RENDERING"
    DONE = "DONE"
    FAILED = "FAILED"
    PUBLISHED_TEST = "PUBLISHED_TEST"
    PUBLISHED_MAIN = "PUBLISHED_MAIN"


class VideoAnnounceSession(SQLModel, table=True):
    __tablename__ = "videoannounce_session"
    __table_args__ = (
        Index("ix_videoannounce_session_status_created_at", "status", "created_at"),
        Index(
            "ux_videoannounce_session_rendering",
            "status",
            unique=True,
            sqlite_where=text("status = 'RENDERING'"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    status: VideoAnnounceSessionStatus = Field(
        default=VideoAnnounceSessionStatus.CREATED,
        sa_column=Column(SAEnum(VideoAnnounceSessionStatus)),
    )
    profile_key: Optional[str] = None
    selection_params: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    published_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    test_chat_id: Optional[int] = None
    main_chat_id: Optional[int] = None
    kaggle_dataset: Optional[str] = None
    kaggle_kernel_ref: Optional[str] = None
    error: Optional[str] = None
    video_url: Optional[str] = None


class VideoAnnounceItemStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class VideoAnnounceItem(SQLModel, table=True):
    __tablename__ = "videoannounce_item"
    __table_args__ = (
        Index("ix_videoannounce_item_session", "session_id"),
        Index("ix_videoannounce_item_event", "event_id"),
        Index("ix_videoannounce_item_status", "status"),
        Index(
            "ux_videoannounce_item_session_event",
            "session_id",
            "event_id",
            unique=True,
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="videoannounce_session.id")
    event_id: int = Field(foreign_key="event.id")
    status: VideoAnnounceItemStatus = Field(
        default=VideoAnnounceItemStatus.PENDING,
        sa_column=Column(SAEnum(VideoAnnounceItemStatus)),
    )
    position: int = 0
    final_title: Optional[str] = None
    final_about: Optional[str] = None
    final_description: Optional[str] = None
    poster_text: Optional[str] = None
    poster_source: Optional[str] = None
    use_ocr: bool = False
    llm_score: Optional[float] = None
    llm_reason: Optional[str] = None
    is_mandatory: bool = Field(default=False, sa_column=Column(Boolean, default=False))
    include_count: int = Field(default=0, sa_column=Column(Integer, default=0))
    error: Optional[str] = None
    created_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


class VideoAnnounceEventHit(SQLModel, table=True):
    __tablename__ = "videoannounce_eventhit"
    __table_args__ = (
        Index("ix_videoannounce_eventhit_event", "event_id"),
        Index("ix_videoannounce_eventhit_session", "session_id"),
        Index(
            "ux_videoannounce_eventhit_session_event",
            "session_id",
            "event_id",
            unique=True,
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="videoannounce_session.id")
    event_id: int = Field(foreign_key="event.id")
    created_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


class VideoAnnounceLLMTrace(SQLModel, table=True):
    __tablename__ = "videoannounce_llm_trace"
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: Optional[int] = Field(default=None, foreign_key="videoannounce_session.id")
    stage: str
    model: str
    request_json: str
    response_json: str
    created_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


class EventPoster(SQLModel, table=True):
    __table_args__ = (
        Index("ix_eventposter_event", "event_id"),
        UniqueConstraint("event_id", "poster_hash", name="ux_eventposter_event_hash"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="event.id")
    catbox_url: Optional[str] = None
    poster_hash: str
    ocr_text: Optional[str] = None
    ocr_title: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    updated_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


class TomorrowPage(SQLModel, table=True):
    date: str = Field(primary_key=True)  # YYYY-MM-DD
    url: str
    created_at: datetime = Field(default_factory=utc_now)


class MonthPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    month: str = Field(primary_key=True)
    url: str
    path: str
    url2: Optional[str] = None  # Deprecated: use MonthPagePart
    path2: Optional[str] = None  # Deprecated: use MonthPagePart
    content_hash: Optional[str] = None
    content_hash2: Optional[str] = None  # Deprecated: use MonthPagePart


class MonthPagePart(SQLModel, table=True):
    """Stores individual parts of a month page when split into multiple pages."""
    __table_args__ = (
        Index("ix_monthpagepart_month", "month"),
        UniqueConstraint("month", "part_number", name="ux_monthpagepart_month_part"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    month: str  # e.g., "2025-01"
    part_number: int  # 1, 2, 3, ...
    url: str
    path: str
    content_hash: Optional[str] = None
    first_date: Optional[str] = None  # First event date on this page (YYYY-MM-DD)
    last_date: Optional[str] = None   # Last event date on this page (YYYY-MM-DD)


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
    photo_urls: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    aliases: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    website_url: Optional[str] = None
    program_url: Optional[str] = None
    vk_url: Optional[str] = None
    tg_url: Optional[str] = None
    ticket_url: Optional[str] = None
    location_name: Optional[str] = None
    location_address: Optional[str] = None
    city: Optional[str] = None
    activities_json: list[dict] = Field(
        default_factory=list,
        sa_column=Column(
            JSON().with_variant(JSONB, "postgresql"),
            nullable=False,
            server_default=text("'[]'"),
        ),
    )
    source_text: Optional[str] = None
    source_post_url: Optional[str] = None
    source_chat_id: Optional[int] = None
    source_message_id: Optional[int] = None
    nav_hash: Optional[str] = None
    # Parser-related fields (Universal Festival Parser)
    source_url: Optional[str] = None  # Original URL of the festival site
    source_type: Optional[str] = None  # "canonical" | "official" | "external"
    parser_run_id: Optional[str] = None  # Last parser run ID
    parser_version: Optional[str] = None  # Parser version used
    last_parsed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    uds_storage_path: Optional[str] = None  # Path in Supabase Storage to UDS JSON
    contacts_phone: Optional[str] = None  # Phone contact
    contacts_email: Optional[str] = None  # Email contact
    is_annual: Optional[bool] = None  # Is this an annual festival?
    audience: Optional[str] = None  # Target audience description
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )



class JobTask(str, Enum):
    telegraph_build = "telegraph_build"
    vk_sync = "vk_sync"
    ics_publish = "ics_publish"
    tg_ics_post = "tg_ics_post"
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
    paused = "paused"


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
    updated_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )
    next_run_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )
    coalesce_key: Optional[str] = None
    depends_on: Optional[str] = None


class PosterOcrCache(SQLModel, table=True):
    hash: str = Field(primary_key=True)
    detail: str = Field(primary_key=True)
    model: str = Field(primary_key=True)
    text: str
    title: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    created_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


class OcrUsage(SQLModel, table=True):
    date: str = Field(primary_key=True)
    spent_tokens: int = 0


def create_all(engine) -> None:
    SQLModel.metadata.create_all(engine)


class VKInbox(SQLModel, table=True):
    __tablename__ = "vk_inbox"
    id: Optional[int] = Field(default=None, primary_key=True)
    group_id: int
    post_id: int
    date: int
    text: str
    matched_kw: Optional[str] = None
    has_date: int
    event_ts_hint: Optional[int] = None
    status: str = Field(default="pending")
    locked_by: Optional[int] = None
    locked_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    imported_event_id: Optional[int] = None
    review_batch: Optional[str] = None
    created_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )

@dataclass
class VkMissRecord:
    id: str
    url: str
    reason: str | None
    matched_kw: str | None
    timestamp: datetime

@dataclass
class VkMissReviewSession:
    queue: list[VkMissRecord]
    index: int = 0
    last_text: str | None = None
    last_published_at: datetime | None = None
