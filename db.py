from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager

import aiosqlite
from sqlalchemy.ext.asyncio import AsyncConnection

_KNOWN_DATABASES: set["Database"] = set()

_VALID_JOURNAL_MODES = {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}


async def _add_column(conn, table: str, col_def: str) -> None:
    try:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except Exception as e:
        msg = str(e).lower()
        if "duplicate column name" in msg:
            return
        # SQLite restriction: ALTER TABLE ... ADD COLUMN only supports constant defaults.
        # Some prod snapshots may have older schema, and migrations here may attempt to add
        # columns like "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP", which would crash
        # startup with "Cannot add a column with non-constant default".
        if "non-constant default" in msg:
            sanitized = re.sub(
                r"\s+default\s+\(?current_timestamp\)?\b",
                "",
                col_def,
                flags=re.IGNORECASE,
            ).strip()
            if sanitized and sanitized != col_def:
                await conn.execute(f"ALTER TABLE {table} ADD COLUMN {sanitized}")
                return
        raise


class Database:
    def __init__(self, path: str):
        self.path = path
        # Ensure the directory exists for file-backed sqlite DBs.
        # This avoids failures in local/test environments when DB_PATH points to /data/db.sqlite.
        if path and not path.startswith((":memory:", "file:")):
            parent = os.path.dirname(path)
            if parent and parent not in (".", ""):
                try:
                    os.makedirs(parent, exist_ok=True)
                except PermissionError:
                    fallback = os.path.join(tempfile.gettempdir(), os.path.basename(path))
                    logging.warning(
                        "Database directory is not writable: %s. Falling back to %s",
                        parent,
                        fallback,
                    )
                    self.path = fallback
        self._conn: aiosqlite.Connection | None = None
        self._orm_engine = None
        self._sessionmaker = None
        _KNOWN_DATABASES.add(self)

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    @classmethod
    def _sqlite_timeout_sec(cls) -> float:
        # sqlite3 "timeout" is busy_timeout (seconds). Keep reasonably high
        # to avoid flaky "database is locked" under concurrent async workers.
        return max(0.1, min(cls._read_float_env("DB_TIMEOUT_SEC", 30.0), 120.0))

    @classmethod
    def _sqlite_busy_timeout_ms(cls) -> int:
        raw = (os.getenv("DB_BUSY_TIMEOUT_MS") or "").strip()
        if raw:
            try:
                return int(raw)
            except Exception:
                pass
        return int(cls._sqlite_timeout_sec() * 1000)

    @staticmethod
    def _sqlite_journal_mode() -> str:
        journal_mode = (os.getenv("DB_JOURNAL_MODE") or "WAL").strip().upper()
        if journal_mode not in _VALID_JOURNAL_MODES:
            journal_mode = "WAL"
        return journal_mode

    async def _apply_sqlite_pragmas(self, conn: aiosqlite.Connection) -> None:
        journal_mode = self._sqlite_journal_mode()
        await conn.execute(f"PRAGMA journal_mode={journal_mode}")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA temp_store=MEMORY")
        await conn.execute("PRAGMA cache_size=-40000")
        await conn.execute(f"PRAGMA busy_timeout={self._sqlite_busy_timeout_ms()}")
        await conn.execute("PRAGMA mmap_size=134217728")

    def _create_orm_engine(self):
        from sqlalchemy import event
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.path}",
            future=True,
            poolclass=NullPool,
            connect_args={"timeout": self._sqlite_timeout_sec()},
        )

        journal_mode = self._sqlite_journal_mode()
        busy_timeout_ms = self._sqlite_busy_timeout_ms()

        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = None
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute(f"PRAGMA journal_mode={journal_mode}")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA cache_size=-40000")
                cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
                cursor.execute("PRAGMA mmap_size=134217728")
            except Exception:
                logging.debug("Failed to apply sqlite PRAGMAs on ORM connection", exc_info=True)
            finally:
                try:
                    if cursor is not None:
                        cursor.close()
                except Exception:
                    pass

        return engine

    async def close(self) -> None:
        if self._sessionmaker is not None:
            self._sessionmaker = None
        if self._orm_engine is not None:
            await self._orm_engine.dispose()
            self._orm_engine = None
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        _KNOWN_DATABASES.discard(self)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as conn:
            debug = (os.getenv("DB_INIT_DEBUG") or "").strip().lower() in {"1", "true", "yes"}
            minimal_mode = (os.getenv("DB_INIT_MINIMAL") or "").strip().lower() in {"1", "true", "yes"}
            skip_posterocr_migration = minimal_mode or (
                (os.getenv("DB_INIT_SKIP_POSTER_OCR_MIGRATION") or "").strip().lower() in {"1", "true", "yes"}
            )

            def dbg(msg: str) -> None:
                if debug:
                    logging.info("db.init %s", msg)

            dbg(f"start path={self.path}")
            # WAL is fast but can be problematic on some filesystems (e.g. network/virtual mounts).
            # Allow overriding for local dev snapshots.
            journal_mode = (os.getenv("DB_JOURNAL_MODE") or "WAL").strip().upper()
            if journal_mode not in {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}:
                journal_mode = "WAL"
            if journal_mode != "WAL" and self.path and not self.path.startswith((":memory:", "file:")):
                # Best-effort cleanup of leftover WAL artifacts from previous runs.
                for suffix in ("-wal", "-shm"):
                    try:
                        os.remove(self.path + suffix)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        logging.debug("Failed to remove sqlite artifact %s", self.path + suffix)
            await conn.execute(f"PRAGMA journal_mode={journal_mode}")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA cache_size=-40000")
            await conn.execute(f"PRAGMA busy_timeout={self._sqlite_busy_timeout_ms()}")
            await conn.execute("PRAGMA mmap_size=134217728")
            dbg(f"pragmas journal_mode={journal_mode}")

            pragma_cursor = await conn.execute("PRAGMA table_info('posterocrcache')")
            poster_ocr_columns = await pragma_cursor.fetchall()
            await pragma_cursor.close()
            dbg(f"posterocrcache columns={len(poster_ocr_columns)}")

            detail_exists = any(col[1] == "detail" for col in poster_ocr_columns)
            model_exists = any(col[1] == "model" for col in poster_ocr_columns)
            created_at_exists = any(col[1] == "created_at" for col in poster_ocr_columns)
            pk_columns: list[str] = []
            if poster_ocr_columns:
                pk_info = sorted(
                    ((col[5], col[1]) for col in poster_ocr_columns if col[5]),
                    key=lambda item: item[0],
                )
                pk_columns = [name for _, name in pk_info]

            expected_pk = ["hash", "detail", "model"]
            needs_posterocr_migration = False
            if poster_ocr_columns:
                if not detail_exists or not model_exists:
                    needs_posterocr_migration = True
                elif pk_columns != expected_pk:
                    needs_posterocr_migration = True

            if needs_posterocr_migration and not skip_posterocr_migration:
                await conn.execute("DROP TABLE IF EXISTS posterocrcache_new")
                await conn.execute(
                    """
                    CREATE TABLE posterocrcache_new(
                        hash TEXT NOT NULL,
                        detail TEXT NOT NULL,
                        model TEXT NOT NULL,
                        text TEXT NOT NULL,
                        title TEXT,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (hash, detail, model)
                    )
                    """
                )

                detail_default = "auto"
                model_default = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")

                detail_expr = "COALESCE(detail, ?)" if detail_exists else "?"
                model_expr = "COALESCE(model, ?)" if model_exists else "?"
                created_at_expr = "created_at" if created_at_exists else "CURRENT_TIMESTAMP"

                # Check if 'title' column exists in the old table to copy it
                title_exists = any(col[1] == "title" for col in poster_ocr_columns)
                title_expr = "title" if title_exists else "NULL"

                insert_sql = f"""
                    INSERT INTO posterocrcache_new (
                        hash, detail, model, text, title,
                        prompt_tokens, completion_tokens, total_tokens, created_at
                    )
                    SELECT
                        hash,
                        {detail_expr},
                        {model_expr},
                        text,
                        {title_expr},
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        {created_at_expr}
                    FROM posterocrcache
                """

                params: list[str] = []
                params.append(detail_default)
                params.append(model_default)

                await conn.execute(insert_sql, params)
                await conn.execute("DROP TABLE posterocrcache")
                await conn.execute("ALTER TABLE posterocrcache_new RENAME TO posterocrcache")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user(
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    is_superadmin BOOLEAN DEFAULT 0,
                    is_partner BOOLEAN DEFAULT 0,
                    organization TEXT,
                    location TEXT,
                    blocked BOOLEAN DEFAULT 0,
                    last_partner_reminder TIMESTAMP WITH TIME ZONE
                        -- Existing deployments should backfill naive values as UTC.
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pendinguser(
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    requested_at TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rejecteduser(
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    rejected_at TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel(
                    channel_id INTEGER PRIMARY KEY,
                    title TEXT,
                    username TEXT,
                    is_admin BOOLEAN DEFAULT 0,
                    is_registered BOOLEAN DEFAULT 0,
                    is_asset BOOLEAN DEFAULT 0,
                    daily_time TEXT,
                    last_daily TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS setting(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supabase_delete_queue(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bucket TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_attempt_at TIMESTAMP,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    UNIQUE(bucket, path)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_supabase_delete_queue_created_at ON supabase_delete_queue(created_at)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    festival TEXT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    time_is_default BOOLEAN NOT NULL DEFAULT 0,
                    location_name TEXT NOT NULL,
                    location_address TEXT,
                    city TEXT,
                    ticket_price_min INTEGER,
                    ticket_price_max INTEGER,
                    ticket_link TEXT,
                    event_type TEXT,
                    emoji TEXT,
                    end_date TEXT,
                    end_date_is_inferred BOOLEAN NOT NULL DEFAULT 0,
                    is_free BOOLEAN DEFAULT 0,
                    pushkin_card BOOLEAN DEFAULT 0,
                    silent BOOLEAN DEFAULT 0,
                    lifecycle_status TEXT NOT NULL DEFAULT 'active',
                    telegraph_path TEXT,
                    source_text TEXT NOT NULL,
                    source_texts JSON,
                    telegraph_url TEXT,
                    ics_url TEXT,
                    source_post_url TEXT,
                    source_vk_post_url TEXT,
                    vk_repost_url TEXT,
                    ics_hash TEXT,
                    ics_file_id TEXT,
                    ics_updated_at TIMESTAMP,
                    ics_post_url TEXT,
                    ics_post_id INTEGER,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    creator_id INTEGER,
                    photo_urls JSON,
                    photo_count INTEGER DEFAULT 0,
                    video_include_count INTEGER NOT NULL DEFAULT 0,
                    topics TEXT DEFAULT '[]',
                    topics_manual BOOLEAN DEFAULT 0,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT
                )
                """
            )
            dbg("event core columns")
            await _add_column(conn, "event", "photo_urls JSON")
            await _add_column(conn, "event", "source_texts JSON")
            await _add_column(conn, "event", "ics_hash TEXT")
            await _add_column(conn, "event", "ics_file_id TEXT")
            await _add_column(conn, "event", "ics_updated_at TIMESTAMP")
            await _add_column(conn, "event", "ics_post_url TEXT")
            await _add_column(conn, "event", "ics_post_id INTEGER")
            await _add_column(conn, "event", "vk_repost_url TEXT")
            await _add_column(conn, "event", "vk_source_hash TEXT")
            await _add_column(conn, "event", "vk_ticket_short_url TEXT")
            await _add_column(conn, "event", "vk_ticket_short_key TEXT")
            await _add_column(conn, "event", "vk_ics_short_url TEXT")
            await _add_column(conn, "event", "vk_ics_short_key TEXT")
            await _add_column(conn, "event", "topics TEXT DEFAULT '[]'")
            await _add_column(conn, "event", "topics_manual BOOLEAN DEFAULT 0")
            await _add_column(conn, "event", "tourist_label SMALLINT")
            await _add_column(conn, "event", "tourist_factors TEXT")
            await _add_column(conn, "event", "tourist_note TEXT")
            await _add_column(conn, "event", "tourist_label_by INTEGER")
            await _add_column(conn, "event", "tourist_label_at TIMESTAMP")
            await _add_column(conn, "event", "tourist_label_source TEXT")
            await _add_column(
                conn, "event", "video_include_count INTEGER NOT NULL DEFAULT 0"
            )
            await _add_column(
                conn, "event", "lifecycle_status TEXT NOT NULL DEFAULT 'active'"
            )
            await _add_column(conn, "event", "short_description TEXT")
            await _add_column(conn, "event", "search_digest TEXT")
            await _add_column(conn, "event", "ticket_status TEXT")
            await _add_column(conn, "event", "ticket_trust_level TEXT")
            await _add_column(conn, "event", "linked_event_ids TEXT")
            await _add_column(conn, "event", "preview_3d_url TEXT")
            await _add_column(conn, "event", "time_is_default BOOLEAN NOT NULL DEFAULT 0")
            await _add_column(conn, "event", "end_date_is_inferred BOOLEAN NOT NULL DEFAULT 0")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_tourist_label ON event(tourist_label)"
            )
            dbg("eventposter")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eventposter(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    catbox_url TEXT,
                    supabase_url TEXT,
                    supabase_path TEXT,
                    poster_hash TEXT NOT NULL,
                    phash TEXT,
                    ocr_text TEXT,
                    ocr_title TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
                    UNIQUE(event_id, poster_hash)
                )
                """
            )
            await _add_column(conn, "eventposter", "ocr_title TEXT")
            await _add_column(conn, "eventposter", "phash TEXT")
            await _add_column(conn, "eventposter", "supabase_url TEXT")
            await _add_column(conn, "eventposter", "supabase_path TEXT")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_eventposter_event ON eventposter(event_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_eventposter_phash ON eventposter(phash)"
            )

            dbg("event_media_asset")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_media_asset(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'video',
                    supabase_url TEXT,
                    supabase_path TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    mime_type TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_media_asset_event ON event_media_asset(event_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_media_asset_kind ON event_media_asset(kind)"
            )

            dbg("event_source")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_source(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_chat_username TEXT,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    source_text TEXT,
                    imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    trust_level TEXT,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
                    UNIQUE(event_id, source_url)
                )
                """
            )
            await _add_column(conn, "event_source", "source_text TEXT")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_source_event ON event_source(event_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_source_type_url ON event_source(source_type, source_url)"
            )
            # Smart Update часто проверяет идемпотентность по `source_url` без знания `event_id`.
            # Индексы (event_id, source_url) и (source_type, source_url) не ускоряют такой lookup,
            # поэтому держим отдельный индекс по source_url.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_source_url ON event_source(source_url)"
            )

            dbg("event_source_fact")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_source_fact(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    source_id INTEGER NOT NULL,
                    fact TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'added',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
                    FOREIGN KEY(source_id) REFERENCES event_source(id) ON DELETE CASCADE
                )
                """
            )
            # Schema evolution (older snapshots may lack the status column).
            await _add_column(conn, "event_source_fact", "status TEXT NOT NULL DEFAULT 'added'")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_source_fact_event ON event_source_fact(event_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_source_fact_source ON event_source_fact(source_id)"
            )

            # Backfill `event_source` for legacy events created before Smart Update started
            # recording sources (idempotent). This improves Smart Update idempotency and reduces
            # duplicate event creation when the same post is reprocessed.
            skip_event_source_backfill = (
                (os.getenv("DB_INIT_SKIP_EVENT_SOURCE_BACKFILL") or "").strip().lower()
                in {"1", "true", "yes"}
            )
            if not skip_event_source_backfill:
                dbg("seed event_source backfill")
                try:
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO event_source(
                            event_id,
                            source_type,
                            source_url,
                            source_chat_id,
                            source_message_id
                        )
                        SELECT
                            e.id,
                            CASE
                                WHEN e.source_post_url LIKE '%t.me/%' THEN 'telegram'
                                WHEN e.source_post_url LIKE '%vk.com/%' THEN 'vk'
                                ELSE 'legacy'
                            END,
                            e.source_post_url,
                            e.source_chat_id,
                            e.source_message_id
                        FROM event e
                        WHERE e.source_post_url IS NOT NULL AND TRIM(e.source_post_url) != ''
                        """
                    )
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO event_source(
                            event_id,
                            source_type,
                            source_url,
                            source_chat_id,
                            source_message_id
                        )
                        SELECT
                            e.id,
                            'vk',
                            e.source_vk_post_url,
                            e.source_chat_id,
                            e.source_message_id
                        FROM event e
                        WHERE e.source_vk_post_url IS NOT NULL AND TRIM(e.source_vk_post_url) != ''
                        """
                    )
                except Exception:
                    logging.warning(
                        "db.init: event_source backfill failed (non-fatal)",
                        exc_info=True,
                    )

            dbg("telegram_source")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_source(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    title TEXT,
                    enabled BOOLEAN NOT NULL DEFAULT 1,
                    default_location TEXT,
                    default_ticket_link TEXT,
                    trust_level TEXT,
                    filters_json TEXT,
                    festival_source BOOLEAN DEFAULT 0,
                    festival_series TEXT,
                    about TEXT,
                    about_links_json JSON,
                    meta_hash TEXT,
                    meta_fetched_at TIMESTAMP,
                    suggested_festival_series TEXT,
                    suggested_website_url TEXT,
                    suggestion_confidence REAL,
                    suggestion_rationale TEXT,
                    last_scanned_message_id INTEGER,
                    last_scan_at TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_telegram_source_enabled ON telegram_source(enabled)"
            )
            await _add_column(conn, "telegram_source", "festival_source BOOLEAN DEFAULT 0")
            await _add_column(conn, "telegram_source", "festival_series TEXT")
            await _add_column(conn, "telegram_source", "filters_json TEXT")
            await _add_column(conn, "telegram_source", "title TEXT")
            await _add_column(conn, "telegram_source", "about TEXT")
            await _add_column(conn, "telegram_source", "about_links_json JSON")
            await _add_column(conn, "telegram_source", "meta_hash TEXT")
            await _add_column(conn, "telegram_source", "meta_fetched_at TIMESTAMP")
            await _add_column(conn, "telegram_source", "suggested_festival_series TEXT")
            await _add_column(conn, "telegram_source", "suggested_website_url TEXT")
            await _add_column(conn, "telegram_source", "suggestion_confidence REAL")
            await _add_column(conn, "telegram_source", "suggestion_rationale TEXT")

            dbg("telegram_scanned_message")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_scanned_message(
                    source_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    message_date TIMESTAMP,
                    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    events_extracted INTEGER NOT NULL DEFAULT 0,
                    events_imported INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    PRIMARY KEY (source_id, message_id),
                    FOREIGN KEY(source_id) REFERENCES telegram_source(id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_tg_scanned_source ON telegram_scanned_message(source_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_tg_scanned_processed_at ON telegram_scanned_message(processed_at)"
            )

            dbg("telegram_source_force_message")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_source_force_message(
                    source_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, message_id),
                    FOREIGN KEY(source_id) REFERENCES telegram_source(id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_tg_force_source ON telegram_source_force_message(source_id)"
            )

            dbg("telegram_post_metric")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_post_metric(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    age_day INTEGER NOT NULL,
                    source_url TEXT,
                    message_ts INTEGER,
                    collected_ts INTEGER NOT NULL,
                    views INTEGER,
                    likes INTEGER,
                    reactions_json JSON,
                    UNIQUE(source_id, message_id, age_day),
                    FOREIGN KEY(source_id) REFERENCES telegram_source(id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_tg_metric_source_age ON telegram_post_metric(source_id, age_day)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_tg_metric_source_message ON telegram_post_metric(source_id, message_id)"
            )
            await _add_column(conn, "telegram_post_metric", "reactions_json JSON")

            # Canonical Telegram sources (safe seed).
            skip_tg_seed = (os.getenv("DB_INIT_SKIP_TG_SOURCES_SEED") or "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if not skip_tg_seed:
                try:
                    from telegram_sources_seed import seed_telegram_sources

                    await seed_telegram_sources(conn)
                except Exception:
                    logging.exception("telegram_source seed failed (non-fatal)")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ops_run(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    trigger TEXT NOT NULL DEFAULT 'manual',
                    chat_id INTEGER,
                    operator_id INTEGER,
                    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'running',
                    metrics_json JSON NOT NULL DEFAULT '{}',
                    details_json JSON NOT NULL DEFAULT '{}'
                )
                """
            )
            await _add_column(conn, "ops_run", "trigger TEXT NOT NULL DEFAULT 'manual'")
            await _add_column(conn, "ops_run", "chat_id INTEGER")
            await _add_column(conn, "ops_run", "operator_id INTEGER")
            await _add_column(conn, "ops_run", "finished_at TIMESTAMP")
            await _add_column(conn, "ops_run", "status TEXT NOT NULL DEFAULT 'running'")
            await _add_column(conn, "ops_run", "metrics_json JSON NOT NULL DEFAULT '{}'")
            await _add_column(conn, "ops_run", "details_json JSON NOT NULL DEFAULT '{}'")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_ops_run_kind_started_at ON ops_run(kind, started_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_ops_run_status_started_at ON ops_run(status, started_at)"
            )

            # Telegram web preview (Instant View) probe results for Telegraph pages.
            # Used by the Telegraph cache sanitizer to track pages missing `cached_page`/photo
            # (often leads to “black screen” in Telegram clients).
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegraph_preview_probe(
                    url TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    ref_id INTEGER,
                    ref_key TEXT,
                    last_checked_at TIMESTAMP,
                    last_ok INTEGER NOT NULL DEFAULT 0,
                    last_has_cached_page INTEGER NOT NULL DEFAULT 0,
                    last_has_photo INTEGER NOT NULL DEFAULT 0,
                    last_title TEXT,
                    last_site_name TEXT,
                    last_error TEXT,
                    total_checks INTEGER NOT NULL DEFAULT 0,
                    total_ok INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_ok_at TIMESTAMP,
                    last_fail_at TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_telegraph_preview_probe_kind ON telegraph_preview_probe(kind)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_telegraph_preview_probe_last_checked ON telegraph_preview_probe(last_checked_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_telegraph_preview_probe_failures ON telegraph_preview_probe(consecutive_failures)"
            )

            # For local/offline regression runs we sometimes only need the core tables
            # (event + Smart Update + Telegram monitoring metadata). Building the full
            # schema and optional indexes on a prod snapshot can be slow.
            if (os.getenv("DB_INIT_MINIMAL") or "").strip().lower() in {"1", "true", "yes"}:
                dbg("minimal mode: returning after core tables")
                await conn.commit()
                return

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monthpage(
                    month TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    path TEXT NOT NULL,
                    url2 TEXT,
                    path2 TEXT,
                    content_hash TEXT,
                    content_hash2 TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monthpagepart(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    month TEXT NOT NULL,
                    part_number INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT,
                    first_date TEXT,
                    last_date TEXT,
                    UNIQUE(month, part_number)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_monthpagepart_month ON monthpagepart(month)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weekendpage(
                    start TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    path TEXT NOT NULL,
                    vk_post_url TEXT,
                    content_hash TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tomorrowpage(
                    date TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weekpage(
                    start TEXT PRIMARY KEY,
                    vk_post_url TEXT,
                    content_hash TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS festival(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    full_name TEXT,
                    description TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    telegraph_url TEXT,
                    telegraph_path TEXT,
                    vk_post_url TEXT,
                    vk_poll_url TEXT,
                    photo_url TEXT,
                    photo_urls JSON,
                    aliases JSON,
                    website_url TEXT,
                    program_url TEXT,
                    vk_url TEXT,
                    tg_url TEXT,
                    ticket_url TEXT,
                    location_name TEXT,
                    location_address TEXT,
                    city TEXT,
                    source_text TEXT,
                    source_post_url TEXT,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await _add_column(conn, "festival", "location_name TEXT")
            await _add_column(conn, "festival", "location_address TEXT")
            await _add_column(conn, "festival", "city TEXT")
            await _add_column(conn, "festival", "program_url TEXT")
            await _add_column(conn, "festival", "ticket_url TEXT")
            await _add_column(conn, "festival", "nav_hash TEXT")
            await _add_column(conn, "festival", "photo_urls JSON")
            await _add_column(conn, "festival", "aliases JSON")
            await _add_column(conn, "festival", "source_post_url TEXT")
            await _add_column(conn, "festival", "source_chat_id INTEGER")
            await _add_column(conn, "festival", "source_message_id INTEGER")
            await _add_column(conn, "festival", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            # Parser-related fields (Universal Festival Parser)
            await _add_column(conn, "festival", "source_url TEXT")
            await _add_column(conn, "festival", "source_type TEXT")
            await _add_column(conn, "festival", "parser_run_id TEXT")
            await _add_column(conn, "festival", "parser_version TEXT")
            await _add_column(conn, "festival", "last_parsed_at TIMESTAMP")
            await _add_column(conn, "festival", "uds_storage_path TEXT")
            await _add_column(conn, "festival", "contacts_phone TEXT")
            await _add_column(conn, "festival", "contacts_email TEXT")
            await _add_column(conn, "festival", "is_annual BOOLEAN")
            await _add_column(conn, "festival", "audience TEXT")
            await _add_column(
                conn,
                "festival",
                "activities_json JSON NOT NULL DEFAULT '[]'",
            )
            await conn.execute(
                "UPDATE festival SET activities_json = '[]' WHERE activities_json IS NULL"
            )

            festival_cursor = await conn.execute("PRAGMA table_info('festival')")
            festival_columns = await festival_cursor.fetchall()
            await festival_cursor.close()
            festival_column_names = {column[1] for column in festival_columns}
            if "created_at" not in festival_column_names:
                await conn.execute("ALTER TABLE festival ADD COLUMN created_at TIMESTAMP")
                await conn.execute(
                    "UPDATE festival SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                )
            if "updated_at" not in festival_column_names:
                await conn.execute("ALTER TABLE festival ADD COLUMN updated_at TIMESTAMP")
            await conn.execute(
                "UPDATE festival SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP) "
                "WHERE updated_at IS NULL"
            )

            await conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS festival_set_created_at
                AFTER INSERT ON festival
                FOR EACH ROW
                WHEN NEW.created_at IS NULL
                BEGIN
                    UPDATE festival
                    SET created_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END;
                """
            )
            await conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS festival_set_updated_at
                AFTER UPDATE ON festival
                FOR EACH ROW
                WHEN NEW.updated_at IS NULL OR NEW.updated_at = OLD.updated_at
                BEGIN
                    UPDATE festival
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END;
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS festival_queue(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    source_kind TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_text TEXT,
                    source_chat_username TEXT,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    source_group_id INTEGER,
                    source_post_id INTEGER,
                    festival_context TEXT,
                    festival_name TEXT,
                    festival_full TEXT,
                    festival_series TEXT,
                    dedup_links_json JSON NOT NULL DEFAULT '[]',
                    signals_json JSON NOT NULL DEFAULT '{}',
                    result_json JSON NOT NULL DEFAULT '{}',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await _add_column(conn, "festival_queue", "status TEXT NOT NULL DEFAULT 'pending'")
            await _add_column(conn, "festival_queue", "source_kind TEXT")
            await _add_column(conn, "festival_queue", "source_url TEXT")
            await _add_column(conn, "festival_queue", "source_text TEXT")
            await _add_column(conn, "festival_queue", "source_chat_username TEXT")
            await _add_column(conn, "festival_queue", "source_chat_id INTEGER")
            await _add_column(conn, "festival_queue", "source_message_id INTEGER")
            await _add_column(conn, "festival_queue", "source_group_id INTEGER")
            await _add_column(conn, "festival_queue", "source_post_id INTEGER")
            await _add_column(conn, "festival_queue", "festival_context TEXT")
            await _add_column(conn, "festival_queue", "festival_name TEXT")
            await _add_column(conn, "festival_queue", "festival_full TEXT")
            await _add_column(conn, "festival_queue", "festival_series TEXT")
            await _add_column(conn, "festival_queue", "dedup_links_json JSON NOT NULL DEFAULT '[]'")
            await _add_column(conn, "festival_queue", "signals_json JSON NOT NULL DEFAULT '{}'")
            await _add_column(conn, "festival_queue", "result_json JSON NOT NULL DEFAULT '{}'")
            await _add_column(conn, "festival_queue", "attempts INTEGER NOT NULL DEFAULT 0")
            await _add_column(conn, "festival_queue", "last_error TEXT")
            await _add_column(conn, "festival_queue", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            await _add_column(conn, "festival_queue", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            await _add_column(conn, "festival_queue", "next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_festival_queue_status_next_run ON festival_queue(status, next_run_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_festival_queue_source_kind ON festival_queue(source_kind)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_festival_queue_source_url ON festival_queue(source_url)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ticket_site_queue(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL DEFAULT 'active',
                    site_kind TEXT NOT NULL,
                    url TEXT NOT NULL,
                    event_id INTEGER,
                    source_post_url TEXT,
                    source_chat_username TEXT,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_result_json JSON NOT NULL DEFAULT '{}',
                    last_run_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await _add_column(conn, "ticket_site_queue", "status TEXT NOT NULL DEFAULT 'active'")
            await _add_column(conn, "ticket_site_queue", "site_kind TEXT")
            await _add_column(conn, "ticket_site_queue", "url TEXT")
            await _add_column(conn, "ticket_site_queue", "event_id INTEGER")
            await _add_column(conn, "ticket_site_queue", "source_post_url TEXT")
            await _add_column(conn, "ticket_site_queue", "source_chat_username TEXT")
            await _add_column(conn, "ticket_site_queue", "source_chat_id INTEGER")
            await _add_column(conn, "ticket_site_queue", "source_message_id INTEGER")
            await _add_column(conn, "ticket_site_queue", "attempts INTEGER NOT NULL DEFAULT 0")
            await _add_column(conn, "ticket_site_queue", "last_error TEXT")
            await _add_column(conn, "ticket_site_queue", "last_result_json JSON NOT NULL DEFAULT '{}'")
            await _add_column(conn, "ticket_site_queue", "last_run_at TIMESTAMP")
            await _add_column(conn, "ticket_site_queue", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            await _add_column(conn, "ticket_site_queue", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            await _add_column(conn, "ticket_site_queue", "next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_ticket_site_queue_status_next_run ON ticket_site_queue(status, next_run_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_ticket_site_queue_site_kind ON ticket_site_queue(site_kind)"
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_ticket_site_queue_url ON ticket_site_queue(url)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS joboutbox(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    task TEXT NOT NULL,
                    payload TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    last_result TEXT,
                    coalesce_key TEXT,
                    depends_on TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await _add_column(conn, "joboutbox", "last_result TEXT")
            await _add_column(conn, "joboutbox", "coalesce_key TEXT")
            await _add_column(conn, "joboutbox", "depends_on TEXT")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_source(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    screen_name TEXT,
                    name TEXT,
                    location TEXT,
                    default_time TEXT,
                    default_ticket_link TEXT,
                    festival_source BOOLEAN DEFAULT 0,
                    festival_series TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_vk_source_group ON vk_source(group_id)"
            )

            await _add_column(conn, "vk_source", "default_ticket_link TEXT")
            await _add_column(conn, "vk_source", "festival_source BOOLEAN DEFAULT 0")
            await _add_column(conn, "vk_source", "festival_series TEXT")

            # Seed well-known VK sources with stable defaults so live E2E / fresh prod
            # snapshots don't lose operator UX improvements after DB refresh.
            try:
                await conn.execute(
                    """
                    UPDATE vk_source
                    SET location = ?
                    WHERE group_id = ?
                      AND (
                        location IS NULL
                        OR TRIM(location) = ''
                        OR location IN (
                            'Гаражка, Калининград',
                            'Гаражка Калининград',
                            'Garazhka Kaliningrad'
                        )
                      )
                    """,
                    ("Понарт, Судостроительная 6/2, Калининград", 226847232),
                )
            except Exception:
                logging.warning("db.init: failed to seed vk_source defaults", exc_info=True)

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_tmp_post(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    group_id INTEGER NOT NULL,
                    post_id INTEGER NOT NULL,
                    date INTEGER NOT NULL,
                    text TEXT,
                    photos JSON,
                    url TEXT
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_vk_tmp_post_batch ON vk_tmp_post(batch, id)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_post_metric(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    post_id INTEGER NOT NULL,
                    age_day INTEGER NOT NULL,
                    source_url TEXT,
                    post_ts INTEGER,
                    collected_ts INTEGER NOT NULL,
                    views INTEGER,
                    likes INTEGER,
                    UNIQUE(group_id, post_id, age_day)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_vk_metric_group_age ON vk_post_metric(group_id, age_day)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_vk_metric_group_post ON vk_post_metric(group_id, post_id)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS page_section_cache(
                    page_key TEXT NOT NULL,
                    section_key TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(page_key, section_key)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_psc_page ON page_section_cache(page_key)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_crawl_cursor (
                    group_id     INTEGER PRIMARY KEY,
                    last_seen_ts INTEGER DEFAULT 0,
                    last_post_id INTEGER DEFAULT 0,
                    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checked_at   INTEGER
                )
                """
            )

            await _add_column(conn, "vk_crawl_cursor", "checked_at INTEGER")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_inbox (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id     INTEGER NOT NULL,
                    post_id      INTEGER NOT NULL,
                    date         INTEGER NOT NULL,
                    text         TEXT NOT NULL,
                    matched_kw   TEXT,
                    has_date     INTEGER NOT NULL,
                    event_ts_hint INTEGER,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    locked_by    INTEGER,
                    locked_at    TIMESTAMP,
                    imported_event_id INTEGER,
                    review_batch TEXT,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_vk_inbox_unique ON vk_inbox(group_id, post_id)"
            )

            await _add_column(conn, "vk_inbox", "event_ts_hint INTEGER")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_review_batch (
                    batch_id     TEXT PRIMARY KEY,
                    operator_id  INTEGER NOT NULL,
                    months_csv   TEXT NOT NULL,
                    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at  TIMESTAMP
                )
                """
            )

            # VK inbox -> imported events mapping (VK posts may yield multiple events).
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_inbox_import_event (
                    inbox_id   INTEGER NOT NULL,
                    event_id   INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (inbox_id, event_id)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_vk_inbox_import_event_event ON vk_inbox_import_event(event_id)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posterocrcache(
                    hash TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text TEXT NOT NULL,
                    title TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (hash, detail, model)
                )
                """
            )
            await _add_column(conn, "posterocrcache", "title TEXT")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocrusage(
                    date TEXT PRIMARY KEY,
                    spent_tokens INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            # Cache: resolve city/settlement -> (is in Kaliningrad oblast?) via Wikidata/LLM.
            # Used to deterministically filter out-of-region events without repeatedly
            # querying external sources for the same city names.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS geo_city_region_cache(
                    city_norm TEXT PRIMARY KEY,
                    is_kaliningrad_oblast BOOLEAN,
                    region_code TEXT,
                    region_name TEXT,
                    source TEXT,
                    wikidata_qid TEXT,
                    details JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await _add_column(
                conn,
                "geo_city_region_cache",
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            )
            await conn.execute(
                "UPDATE geo_city_region_cache "
                "SET created_at = COALESCE(created_at, updated_at, CURRENT_TIMESTAMP) "
                "WHERE created_at IS NULL"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_geo_city_region_cache_created_at "
                "ON geo_city_region_cache(created_at)"
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_festival_name ON festival(name)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_date ON event(date)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_end_date ON event(end_date)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_city ON event(city)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON event(event_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_is_free ON event(is_free)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_date_city ON event(date, city)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_date_festival ON event(date, festival)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_content_hash ON event(content_hash)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_date_time ON event(date, time)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_festival_date_time ON event(festival, date, time)"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videoannounce_session(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL DEFAULT 'CREATED',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    published_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    kaggle_dataset TEXT,
                    kaggle_kernel_ref TEXT,
                    error TEXT,
                    video_url TEXT
                )
                """
            )
            await _add_column(conn, "videoannounce_session", "profile_key TEXT")
            await _add_column(conn, "videoannounce_session", "selection_params JSON")
            await _add_column(conn, "videoannounce_session", "test_chat_id BIGINT")
            await _add_column(conn, "videoannounce_session", "main_chat_id BIGINT")
            await _add_column(conn, "videoannounce_session", "published_at TIMESTAMP")
            await _add_column(conn, "videoannounce_session", "kaggle_dataset TEXT")
            await _add_column(conn, "videoannounce_session", "kaggle_kernel_ref TEXT")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videoannounce_item(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    event_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    position INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES videoannounce_session(id) ON DELETE CASCADE,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
                    UNIQUE(session_id, event_id)
                )
                """
            )
            await _add_column(conn, "videoannounce_item", "final_title TEXT")
            await _add_column(conn, "videoannounce_item", "final_about TEXT")
            await _add_column(conn, "videoannounce_item", "final_description TEXT")
            await _add_column(conn, "videoannounce_item", "poster_text TEXT")
            await _add_column(conn, "videoannounce_item", "poster_source TEXT")
            await _add_column(
                conn, "videoannounce_item", "use_ocr INTEGER NOT NULL DEFAULT 0"
            )
            await _add_column(conn, "videoannounce_item", "llm_score REAL")
            await _add_column(conn, "videoannounce_item", "llm_reason TEXT")
            await _add_column(
                conn,
                "videoannounce_item",
                "is_mandatory BOOLEAN NOT NULL DEFAULT 0",
            )
            await _add_column(
                conn, "videoannounce_item", "include_count INTEGER NOT NULL DEFAULT 0"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videoannounce_eventhit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    event_id INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES videoannounce_session(id) ON DELETE CASCADE,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
                    UNIQUE(session_id, event_id)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_session_status_created_at ON videoannounce_session(status, created_at)"
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_videoannounce_session_rendering ON videoannounce_session(status) WHERE status = 'RENDERING'"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_item_session ON videoannounce_item(session_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_item_event ON videoannounce_item(event_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_item_status ON videoannounce_item(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_eventhit_event ON videoannounce_eventhit(event_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_eventhit_session ON videoannounce_eventhit(session_id)"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videoannounce_llm_trace(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    stage TEXT NOT NULL,
                    model TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES videoannounce_session(id) ON DELETE SET NULL
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_videoannounce_llm_trace_session ON videoannounce_llm_trace(session_id)"
            )

            await conn.commit()

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.path, timeout=self._sqlite_timeout_sec())
            await self._apply_sqlite_pragmas(self._conn)
        return self._conn

    @asynccontextmanager
    async def raw_conn(self):
        conn = await self._ensure_conn()
        yield conn

    @asynccontextmanager
    async def get_session(self):
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        if self._orm_engine is None:
            self._orm_engine = self._create_orm_engine()
        if self._sessionmaker is None:
            self._sessionmaker = sessionmaker(
                self._orm_engine, expire_on_commit=False, class_=AsyncSession
            )
        async with self._sessionmaker() as session:
            yield session

    @property
    def engine(self):
        if self._orm_engine is None:
            self._orm_engine = self._create_orm_engine()
        return self._orm_engine

    async def exec_driver_sql(
        self, sql: str, params: tuple | dict | None = None
    ):
        async with self.engine.begin() as conn:  # type: AsyncConnection
            result = await conn.exec_driver_sql(sql, params or ())
            try:
                return result.fetchall()
            except Exception:
                return []


async def close_known_databases() -> None:
    for db in list(_KNOWN_DATABASES):
        try:
            await db.close()
        except Exception:
            logging.exception("db.close failed for %s", getattr(db, "path", None))
    _KNOWN_DATABASES.clear()


async def wal_checkpoint_truncate(engine):
    async with engine.begin() as conn:
        result = await conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        rows = result.fetchall()
    logging.info("db_checkpoint result=%s", rows)
    return rows


async def optimize(engine):
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA optimize")


async def vacuum(engine):
    async with engine.begin() as conn:
        await conn.exec_driver_sql("VACUUM")
