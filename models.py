from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    JSON,
    SmallInteger,
    UniqueConstraint,
    text,
)
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
    "NETWORKING": "Нетворкинг и карьера",
    "ACTIVE": "Активный отдых и спорт",
    "PERSONALITIES": "Личности и встречи",
    "HISTORICAL_IMMERSION": "Исторические реконструкции и погружение",
    "KIDS_SCHOOL": "Дети и школа",
    "FAMILY": "Семейные события",
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
    "URBANISM": "LECTURES",
    "urbanism": "LECTURES",
    "урбанистика": "LECTURES",
    "город": "LECTURES",
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
    topics: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    topics_manual: bool = False
    added_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )
    content_hash: Optional[str] = None


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    updated_at: datetime = Field(
        default_factory=utc_now, sa_column=Column(DateTime(timezone=True))
    )


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
    source_text: Optional[str] = None
    source_post_url: Optional[str] = None
    source_chat_id: Optional[int] = None
    source_message_id: Optional[int] = None
    nav_hash: Optional[str] = None
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
