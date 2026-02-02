"""
Step definitions for Telegram bot BDD scenarios.

Maps Russian Gherkin steps to HumanUserClient actions.
"""

import json
import logging
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from behave import given, when, then

logger = logging.getLogger("e2e.steps")


# =============================================================================
# Helper Functions
# =============================================================================

def run_async(context, awaitable):
    """Run async coroutine in the behave sync context."""
    return context.loop.run_until_complete(awaitable)


def get_all_buttons(message):
    """Extract all button texts from message (inline + reply keyboard)."""
    buttons = []
    
    if message and message.buttons:
        for row in message.buttons:
            for btn in row:
                buttons.append(btn.text)
    
    return buttons


def find_button(message, text):
    """Find button by text (partial match)."""
    if message and message.buttons:
        for row in message.buttons:
            for btn in row:
                if text in btn.text:
                    return btn
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _find_event_id_in_text(text: str, title: str) -> int | None:
    if not text or not title:
        return None
    title_norm = _normalize_text(title)
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if not match:
            continue
        event_id = int(match.group(1))
        rest = _normalize_text(match.group(2))
        if title_norm in rest:
            return event_id
    return None


def _extract_report_stat(text: str, label: str) -> int | None:
    if not text:
        return None
    pattern = rf"{re.escape(label)}\s*:\s*(\d+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_run_id(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"run_id:\s*([a-f0-9]{8,})", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _load_tg_results(run_id: str) -> dict | None:
    if not run_id:
        return None
    results_path = Path(tempfile.gettempdir()) / f"tg-monitor-{run_id}" / "telegram_results.json"
    if not results_path.exists():
        return None
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load telegram_results.json: %s", exc)
        return None


def _extract_catbox_urls(html: str) -> set[str]:
    if not html:
        return set()
    return set(
        re.findall(r"https?://files\\.catbox\\.moe/[^\"'\\s<>]+", html)
    )


async def _fetch_telegraph_pages(links: list[str]) -> list[str]:
    import aiohttp

    html_pages: list[str] = []
    async with aiohttp.ClientSession() as session:
        for link in links:
            async with session.get(link, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                html_pages.append(await resp.text())
    return html_pages


def _db_path() -> str:
    env = os.getenv("DB_PATH")
    if env:
        return env
    fresh = "db_prod_snapshot_2026-01-28_154329.sqlite"
    if Path(fresh).exists():
        return fresh
    return "db_prod_snapshot.sqlite"


def _ensure_test_context(context) -> None:
    if not hasattr(context, "test_event_ids"):
        context.test_event_ids = []
    if not hasattr(context, "test_events_by_title"):
        context.test_events_by_title = {}
    if not hasattr(context, "event_ticket_backup"):
        context.event_ticket_backup = {}


def _ensure_event_source_fact_table(conn: sqlite3.Connection) -> None:
    _ensure_event_source_table(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS event_source_fact(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            source_id INTEGER NOT NULL,
            fact TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES event_source(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_event_source_fact_event ON event_source_fact(event_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_event_source_fact_source ON event_source_fact(source_id)"
    )


def _ensure_event_source_table(conn: sqlite3.Connection) -> None:
    conn.execute(
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_event_source_event ON event_source(event_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_event_source_type_url ON event_source(source_type, source_url)"
    )


def _insert_event(conn: sqlite3.Connection, data: dict) -> int:
    title = data.get("title") or "TEST EVENT"
    date = data.get("date") or "2026-01-01"
    time = data.get("time") or "19:00"
    location_name = data.get("location_name") or "Тестовая площадка"
    source_text = data.get("source_text") or title
    description = data.get("description") or source_text
    city = data.get("city") or "Калининград"
    fields = {
        "title": title,
        "description": description,
        "date": date,
        "time": time,
        "location_name": location_name,
        "source_text": source_text,
        "city": city,
    }
    optional_fields = [
        "location_address",
        "ticket_link",
        "ticket_price_min",
        "ticket_price_max",
        "ticket_status",
        "ticket_trust_level",
        "event_type",
        "emoji",
        "end_date",
        "is_free",
        "pushkin_card",
        "search_digest",
    ]
    for key in optional_fields:
        if key in data and data[key] not in (None, ""):
            fields[key] = data[key]

    columns = ", ".join(fields.keys())
    placeholders = ", ".join(["?"] * len(fields))
    values = list(fields.values())
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO event ({columns}) VALUES ({placeholders})",
        values,
    )
    return int(cur.lastrowid)


def _insert_event_source(
    conn: sqlite3.Connection,
    event_id: int,
    source_url: str,
    source_type: str = "site",
    source_text: str | None = None,
) -> int:
    _ensure_event_source_table(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO event_source(event_id, source_type, source_url, source_text, imported_at)
        VALUES(?,?,?,?,?)
        """,
        (
            event_id,
            source_type,
            source_url,
            source_text,
            datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        ),
    )
    return int(cur.lastrowid)


def _fetch_event_by_title(conn: sqlite3.Connection, title: str) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM event WHERE title = ? ORDER BY id DESC LIMIT 1",
        (title,),
    )
    return cur.fetchone()


def _fetch_event_id(context, title: str) -> int | None:
    if hasattr(context, "test_events_by_title"):
        event_id = context.test_events_by_title.get(title)
        if event_id:
            return event_id
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        row = _fetch_event_by_title(conn, title)
        return int(row["id"]) if row else None
    finally:
        conn.close()


def _cleanup_test_events(context) -> None:
    _ensure_test_context(context)
    ids = list({int(i) for i in context.test_event_ids if i})
    if not ids:
        return
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        _ensure_event_source_table(conn)
        _ensure_event_source_fact_table(conn)
        placeholders = ",".join("?" for _ in ids)
        cur.execute(f"DELETE FROM event_source_fact WHERE event_id IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM event_source WHERE event_id IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM eventposter WHERE event_id IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM event WHERE id IN ({placeholders})", ids)
        conn.commit()
    finally:
        conn.close()


def _table_to_dict(table) -> dict:
    if not table:
        return {}
    headings = list(table.headings)
    if "field" in headings and "value" in headings:
        return {row["field"]: row["value"] for row in table}
    if len(table) == 1:
        row = table[0]
        return {heading: row[heading] for heading in headings}
    return {row[headings[0]]: row[headings[1]] for row in table}


INT_FIELDS = {
    "ticket_price_min",
    "ticket_price_max",
    "source_chat_id",
    "source_message_id",
    "creator_id",
}
BOOL_FIELDS = {"is_free", "pushkin_card"}


def _coerce_field_value(field: str, raw: str | None):
    if raw is None:
        return None
    value = str(raw).strip()
    if value.lower() in {"", "null", "none", "—"}:
        return None
    if field in INT_FIELDS:
        return int(value)
    if field in BOOL_FIELDS:
        return value.lower() in {"1", "true", "yes", "да", "y"}
    return value


def _get_smart_db(context):
    if hasattr(context, "smart_db") and context.smart_db:
        return context.smart_db
    from db import Database

    db = Database(_db_path())
    run_async(context, db.init())
    context.smart_db = db
    return db


def _load_event_row_by_title(title: str) -> sqlite3.Row | None:
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM event WHERE title = ? ORDER BY id DESC LIMIT 1",
            (title,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def _ensure_sources(rows: list[dict]) -> None:
    """Replace telegram_source set with provided rows (upsert), remove the rest."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        usernames = [str(r.get("username") or "").lstrip("@").strip() for r in rows]
        usernames = [u for u in usernames if u]
        if usernames:
            placeholders = ",".join("?" for _ in usernames)
            cur.execute(
                f"SELECT id FROM telegram_source WHERE username NOT IN ({placeholders})",
                usernames,
            )
            other_ids = [row[0] for row in cur.fetchall()]
            if other_ids:
                other_ph = ",".join("?" for _ in other_ids)
                cur.execute(
                    f"DELETE FROM telegram_scanned_message WHERE source_id IN ({other_ph})",
                    other_ids,
                )
                cur.execute(
                    f"DELETE FROM telegram_source WHERE id IN ({other_ph})",
                    other_ids,
                )

        for row in rows:
            username = str(row.get("username") or "").lstrip("@").strip()
            if not username:
                continue
            has_trust = "trust_level" in row
            has_default_location = "default_location" in row
            has_default_ticket = "default_ticket_link" in row

            trust = (row.get("trust_level") or None) if has_trust else None
            default_location = (row.get("default_location") or None) if has_default_location else None
            default_ticket_link = (row.get("default_ticket_link") or None) if has_default_ticket else None

            cur.execute("SELECT id FROM telegram_source WHERE username = ?", (username,))
            existing = cur.fetchone()
            if existing:
                set_parts = ["enabled=1"]
                params: list = []
                if has_trust:
                    set_parts.append("trust_level=?")
                    params.append(trust)
                if has_default_location:
                    set_parts.append("default_location=?")
                    params.append(default_location)
                if has_default_ticket:
                    set_parts.append("default_ticket_link=?")
                    params.append(default_ticket_link)
                params.append(username)
                cur.execute(
                    f"UPDATE telegram_source SET {', '.join(set_parts)} WHERE username=?",
                    params,
                )
            else:
                cur.execute(
                    """
                    INSERT INTO telegram_source(username, enabled, trust_level, default_location, default_ticket_link)
                    VALUES(?,?,?,?,?)
                    """,
                    (username, 1, trust, default_location, default_ticket_link),
                )

        conn.commit()
    finally:
        conn.close()


def _ensure_only_source(username: str) -> None:
    username = username.lstrip("@").strip()
    _ensure_sources([{"username": username}])


def _run_tg_monitor(context) -> None:
    step_send_command(context, "/tg")
    step_click_inline_button(context, "🚀 Запустить мониторинг")
    step_wait_for_message_text(context, "Starting Telegram Monitor")
    step_wait_long_operation(context, "Telegram Monitor")


# =============================================================================
# Предыстория (Background)
# =============================================================================

@given("я авторизован в клиенте Telethon")
def step_authorized(context):
    """Verify client is connected and authorized."""
    assert context.client is not None, "Client not initialized"
    assert context.client._connected, "Client not connected"
    logger.info("✓ Клиент авторизован")


@given("я открыл чат с ботом")
def step_open_bot_chat(context):
    """Open chat with target bot, store entity."""
    async def _open():
        entity = await context.client.client.get_entity(context.bot_username)
        context.bot_entity = entity
        logger.info(f"✓ Открыт чат с @{context.bot_username}")
        return entity
    
    run_async(context, _open())


@given("я нахожусь в главном меню")
def step_in_main_menu(context):
    """Ensure we're in main menu (send /start if needed)."""
    if not hasattr(context, "bot_entity"):
        step_open_bot_chat(context)
    
    # Send /start to reset state
    step_send_command(context, "/start")
    logger.info("✓ Находимся в главном меню")


@given("в списке источников нет других каналов кроме @{username}")
def step_only_source(context, username):
    """Ensure only one Telegram source exists in DB."""
    _ensure_only_source(username)
    context.only_source_username = username
    logger.info(f"✓ Оставлен только источник @{username}")


@given("в списке источников настроен только канал @{username}")
def step_only_source_alias(context, username):
    """Alias for clarity in newer scenarios."""
    step_only_source(context, username)


@given("в списке источников Telegram настроены:")
def step_configure_sources_table(context):
    """Upsert telegram_source rows from a table and remove all other sources."""
    if not context.table:
        raise AssertionError("Ожидалась таблица с колонками username/trust_level/default_location")
    rows: list[dict] = []
    for row in context.table:
        rows.append(
            {
                "username": row["username"],
                "trust_level": (row["trust_level"] or "").strip() or None,
                "default_location": (row["default_location"] or "").strip() or None,
            }
        )
    _ensure_sources(rows)
    logger.info("✓ Источники Telegram настроены: %s", [r.get("username") for r in rows if r.get("username")])


@given('я выбираю контрольный пост с постером в канале "{channel}"')
def step_pick_control_post(context, channel):
    """Pick a recent Telegram message with a poster to make monitoring assertions deterministic.

    Also updates telegram_source.last_scanned_message_id and clears scanned marks for that source so the
    chosen post is included in the next monitor run.
    """
    username = str(channel or "").lstrip("@").strip()
    if not username:
        raise AssertionError("Пустой канал для контрольного поста")

    async def _pick():
        entity = await context.client.client.get_entity(username)
        scan_limit = int(os.getenv("E2E_CONTROL_POST_SCAN_LIMIT", "80"))
        try:
            tg_limit = int(os.getenv("TG_MONITORING_LIMIT", str(scan_limit)))
            scan_limit = min(scan_limit, max(tg_limit, 1))
        except Exception:
            pass
        messages = await context.client.client.get_messages(entity, limit=scan_limit)
        best = None
        best_score = -1
        for msg in messages:
            has_photo = bool(getattr(msg, "photo", None))
            has_text = bool((msg.text or "").strip())
            if not has_photo or not has_text:
                continue
            # Avoid trivial/empty captions; prefer something that looks like an announcement.
            if len((msg.text or "").strip()) < 20:
                continue
            text = (msg.text or "").strip()
            score = 0
            if re.search(r"\\b\\d{1,2}[:\\.]\\d{2}\\b", text):
                score += 3
            if re.search(r"\\b\\d{1,2}\\s+[а-яА-Я]{3,}\\b", text):
                score += 2
            if len(text) > 120:
                score += 1
            if score > best_score:
                best_score = score
                best = msg
            if best_score >= 4:
                break
        return best

    picked = run_async(context, _pick())
    if not picked:
        raise AssertionError(f"Не удалось найти пост с постером в @{username}")

    message_id = int(picked.id)
    context.control_post_username = username
    context.control_post_message_id = message_id
    context.control_post_url = f"https://t.me/{username}/{message_id}"
    logger.info("✓ Контрольный пост выбран: %s", context.control_post_url)

    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM telegram_source WHERE username=?", (username,))
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO telegram_source(username, enabled) VALUES(?,1)",
                (username,),
            )
            source_id = int(cur.lastrowid)
        else:
            source_id = int(row[0])
        cur.execute("DELETE FROM telegram_scanned_message WHERE source_id=?", (source_id,))
        last_id = max(message_id - 1, 0)
        cur.execute(
            "UPDATE telegram_source SET enabled=1, last_scanned_message_id=? WHERE id=?",
            (last_id, source_id),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("✓ Подготовлен источник @%s для сканирования с last_scanned_message_id=%s", username, last_id)

@given('я знаю контрольное событие "{title}" на дату "{date}" из канала "{channel}"')
def step_control_event(context, title, date, channel):
    """Store control event data, try to resolve actual title from DB."""
    channel_name = channel.lstrip("@")
    context.control_event_date = date
    context.control_event_channel = channel_name
    context.control_event_title = title
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.title
            FROM event e
            JOIN event_source es ON es.event_id = e.id
            WHERE es.source_type = 'telegram'
              AND (es.source_chat_username = ? OR es.source_url LIKE ?)
              AND e.date LIKE ?
            ORDER BY e.id DESC
            LIMIT 1
            """,
            (channel_name, f"%t.me/{channel_name}/%", f"{date}%"),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row:
        event_id, event_title = row
        context.control_event_id = event_id
        title_norm = _normalize_text(title)
        actual_norm = _normalize_text(event_title)
        if title_norm in {"", "название события"} or title_norm not in actual_norm:
            context.control_event_title = event_title
            logger.info(
                "✓ Контрольное событие уточнено: %s (id=%s)",
                event_title,
                event_id,
            )
    else:
        logger.info("⚠️ Контрольное событие не найдено в БД для %s %s", channel, date)


@given('мониторинг уже обработал сообщение "{url}"')
def step_monitoring_already_processed(context, url):
    """Ensure monitoring has been run at least once and store baseline stats."""
    if getattr(context, "baseline_report_stats", None):
        logger.info("✓ Базовый отчёт уже сохранён")
        return
    _run_tg_monitor(context)
    report_stats = getattr(context, "last_report_stats", None) or {}
    if not report_stats:
        report_text = getattr(context, "last_report_text", None) or ""
        report_stats = {
            "Сообщений пропущено": _extract_report_stat(report_text, "Сообщений пропущено"),
            "Создано": _extract_report_stat(report_text, "Создано"),
        }
    context.baseline_report_stats = report_stats
    context.processed_message_url = url
    logger.info("✓ Базовый отчёт сохранён: %s", report_stats)


@given("в источниках есть сообщение с валидным событием")
def step_sources_have_valid_event(context):
    """Ensure at least one enabled Telegram source exists."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM telegram_source WHERE enabled = 1")
        count = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()
    if count == 0:
        raise AssertionError("Нет активных источников Telegram для мониторинга")
    logger.info("✓ Активных источников: %s", count)


@given('события драмтеатра уже загружены через "/parse"')
def step_dramteatr_events_loaded(context):
    """Verify dramteatr events exist in DB."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM event WHERE title LIKE ? LIMIT 1",
            ("%Вишнёвый сад%",),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError("В БД нет событий драмтеатра (например, Вишнёвый сад)")
    context.dramteatr_event_id = row[0]
    logger.info("✓ Найдено событие драмтеатра (id=%s)", row[0])


@given('в базе есть событие "{title}" на дату "{date}" и время "{time}"')
def step_event_exists_in_db(context, title, date, time):
    """Ensure event exists in DB by title/date/time."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id FROM event
            WHERE title LIKE ? AND date LIKE ? AND time = ?
            LIMIT 1
            """,
            (f"%{title}%", f"{date}%", time),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(
            f"Событие '{title}' {date} {time} не найдено в БД"
        )
    context.last_event_id = row[0]
    logger.info("✓ Событие найдено в БД (id=%s)", row[0])


@given('очищены отметки мониторинга для "{username}"')
def step_reset_monitor_marks(context, username):
    """Reset telegram_scanned_message and last_scanned_message_id for a source."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM telegram_source WHERE username=?",
            (username.lstrip("@"),),
        )
        row = cur.fetchone()
        if not row:
            raise AssertionError(f"Источник @{username} не найден в БД")
        source_id = row[0]
        cur.execute(
            "DELETE FROM telegram_scanned_message WHERE source_id=?",
            (source_id,),
        )
        cur.execute(
            "UPDATE telegram_source SET last_scanned_message_id=NULL WHERE id=?",
            (source_id,),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("✓ Сброшены отметки мониторинга для @%s", username)


# =============================================================================
# Когда (When) - Actions
# =============================================================================

@when('я отправляю команду "{command}"')
@then('я отправляю команду "{command}"')
def step_send_command(context, command):
    """Send command to bot using human-like behavior."""
    async def _send():
        response = await context.client.human_send_and_wait(
            context.bot_entity,
            command,
            timeout=30
        )
        context.last_response = response
        logger.info(f"→ Отправлено: {command}")
        if response and response.text:
            preview = response.text[:100].replace('\n', ' ')
            logger.info(f"← Ответ: {preview}...")
        return response
    
    run_async(context, _send())


@when('я отправляю сообщение "{text}"')
@then('я отправляю сообщение "{text}"')
def step_send_message(context, text):
    """Send arbitrary text message."""
    async def _send():
        response = await context.client.human_send_and_wait(
            context.bot_entity,
            text,
            timeout=120  # Increased timeout for long operations
        )
        context.last_response = response
        logger.info(f"→ Отправлено сообщение: {text}")
        if response and response.text:
            preview = response.text[:100].replace('\n', ' ')
            logger.info(f"← Ответ: {preview}...")
        return response
    
    run_async(context, _send())


@when('я нажимаю инлайн-кнопку "{btn_text}"')
@then('я нажимаю инлайн-кнопку "{btn_text}"')
def step_click_inline_button(context, btn_text):
    """Click inline button by text."""
    async def _click():
        msg = context.last_response
        btn = find_button(msg, btn_text)
        
        if not btn:
            available = get_all_buttons(msg)
            raise AssertionError(
                f"Кнопка '{btn_text}' не найдена. Доступные: {available}"
            )
        
        # Human-like delay before click
        await context.client._gaussian_delay(0.5, 1.5)
        
        # Click the button
        await btn.click()
        logger.info(f"→ Нажата кнопка: {btn_text}")
        
        # Wait for response/edit
        import asyncio
        await asyncio.sleep(2)  # Wait for bot to respond
        
        # Get updated message
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]
            logger.info("← Получен обновлённый ответ")
    
    run_async(context, _click())


@when("я запускаю мониторинг повторно")
def step_run_monitor_repeat(context):
    """Run Telegram monitoring flow again."""
    _run_tg_monitor(context)


@when('если событие "{title}" есть в списке, я удаляю его')
def step_delete_event_if_present(context, title):
    """Delete event by title if it exists in current /events list."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    target_title = title
    if title == "Название события" and hasattr(context, "control_event_title"):
        target_title = context.control_event_title
    event_id = _find_event_id_in_text(text, target_title)
    if not event_id:
        logger.info("✓ Событие не найдено, удалять нечего: %s", target_title)
        return
    btn = find_button(msg, f"❌ {event_id}")
    if not btn:
        raise AssertionError(f"Кнопка удаления ❌ {event_id} не найдена")

    async def _click():
        await context.client._gaussian_delay(0.5, 1.5)
        await btn.click()
        import asyncio
        await asyncio.sleep(2)
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]

    run_async(context, _click())
    logger.info("✓ Событие удалено: id=%s title=%s", event_id, target_title)


@when('я открываю карточку события "{title}"')
def step_open_event_card(context, title):
    """Open event edit card by title from /events list."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    target_title = title
    if title == "Название события" and hasattr(context, "control_event_title"):
        target_title = context.control_event_title
    event_id = _find_event_id_in_text(text, target_title)
    if not event_id:
        raise AssertionError(f"Событие '{target_title}' не найдено в списке")
    btn = find_button(msg, f"✎ {event_id}")
    if not btn:
        raise AssertionError(f"Кнопка редактирования ✎ {event_id} не найдена")

    async def _click():
        await context.client._gaussian_delay(0.5, 1.5)
        await btn.click()
        import asyncio
        await asyncio.sleep(2)
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]

    run_async(context, _click())
    logger.info("✓ Открыта карточка события: id=%s title=%s", event_id, target_title)


@when("я открываю карточку события из выбранного поста")
def step_open_event_card_from_selected_post(context):
    """Resolve event_id from event_source by the previously selected control post URL and open it via /events UI."""
    post_url = getattr(context, "control_post_url", None)
    username = getattr(context, "control_post_username", None)
    message_id = getattr(context, "control_post_message_id", None)
    if not post_url or not username or not message_id:
        raise AssertionError("Контрольный пост не выбран (ожидались control_post_url/username/message_id в context)")

    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        _ensure_event_source_table(conn)
        cur.execute(
            "SELECT event_id FROM event_source WHERE source_url = ? ORDER BY imported_at DESC LIMIT 1",
            (post_url,),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                """
                SELECT event_id FROM event_source
                WHERE source_chat_username = ? AND source_message_id = ?
                ORDER BY imported_at DESC
                LIMIT 1
                """,
                (username, int(message_id)),
            )
            row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT event_id FROM event_source WHERE source_url LIKE ? ORDER BY imported_at DESC LIMIT 1",
                (f"%t.me/{username}/{int(message_id)}%",),
            )
            row = cur.fetchone()
        if not row:
            raise AssertionError(f"Не найден event_source для контрольного поста: {post_url}")
        event_id = int(row["event_id"]) if isinstance(row, sqlite3.Row) else int(row[0])
        cur.execute("SELECT date FROM event WHERE id = ?", (event_id,))
        date_row = cur.fetchone()
        if not date_row:
            raise AssertionError(f"Событие id={event_id} не найдено в таблице event")
        raw_date = str(date_row[0] or "")
        event_date = raw_date.split("..", 1)[0].strip()
    finally:
        conn.close()

    context.control_event_id = event_id
    context.control_event_date = event_date
    logger.info("✓ Контрольное событие: id=%s date=%s url=%s", event_id, event_date, post_url)

    # Open via /events and click edit button by id (UI flow)
    step_send_command(context, f"/events {event_date}")
    msg = context.last_response
    btn = find_button(msg, f"✎ {event_id}")
    if not btn:
        available = get_all_buttons(msg)
        raise AssertionError(f"Кнопка ✎ {event_id} не найдена в /events {event_date}. Кнопки: {available}")

    async def _click():
        await context.client._gaussian_delay(0.5, 1.5)
        await btn.click()
        import asyncio
        await asyncio.sleep(2)
        messages = await context.client.client.get_messages(context.bot_entity, limit=1)
        if messages:
            context.last_response = messages[0]

    run_async(context, _click())
    logger.info("✓ Открыта карточка контрольного события: id=%s", event_id)


@when("я закрываю карточку события")
@then("я закрываю карточку события")
def step_close_event_card(context):
    """Close event edit card by clicking Done button."""
    msg = context.last_response
    btn = find_button(msg, "Done") if msg else None

    async def _click():
        nonlocal btn
        if not btn:
            messages = await context.client.client.get_messages(
                context.bot_entity, limit=20
            )
            for m in messages:
                candidate = find_button(m, "Done")
                if candidate:
                    btn = candidate
                    break
        if not btn:
            raise AssertionError("Кнопка Done не найдена для закрытия карточки")
        await context.client._gaussian_delay(0.5, 1.5)
        await btn.click()
        import asyncio
        await asyncio.sleep(2)
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]

    run_async(context, _click())
    logger.info("✓ Карточка события закрыта")


@when("я сохраняю исходную телеграф страницу события")
def step_save_telegraph_snapshot(context):
    """Fetch and store current Telegraph HTML + catbox urls."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    links = re.findall(r"https://telegra\\.ph/[a-zA-Z0-9_-]+", text)
    if not links:
        raise AssertionError("Не найдено ссылок telegra.ph для сохранения")

    async def _save():
        html_pages = await _fetch_telegraph_pages(links)
        catbox_urls = set()
        for html in html_pages:
            catbox_urls.update(_extract_catbox_urls(html))
        context.telegraph_snapshot = {
            "links": links,
            "html": html_pages,
            "catbox_urls": catbox_urls,
        }

    run_async(context, _save())
    logger.info("✓ Сохранён снимок Telegraph (%s ссылок, %s catbox)", len(links), len(context.telegraph_snapshot["catbox_urls"]))


# =============================================================================
# Тогда (Then) - Assertions
# =============================================================================

@then('я должен увидеть сообщение, содержащее текст "{text}"')
def step_see_message_with_text(context, text):
    """Assert last response contains text."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    assert msg.text is not None, "Ответ бота пустой"

    # Case-insensitive search in last response
    if text.lower() in msg.text.lower():
        if text.lower() == "starting telegram monitor":
            context.monitor_started_message_id = msg.id
        logger.info(f"✓ Найден текст: '{text}'")
        return

    # Fallback: search in recent messages (handles concurrent bot updates)
    async def _search():
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=5
        )
        for candidate in messages:
            if candidate.text and text.lower() in candidate.text.lower():
                context.last_response = candidate
                return True
        return False

    found = run_async(context, _search())
    if not found:
        raise AssertionError(
            f"Текст '{text}' не найден в последних сообщениях. "
            f"Последний ответ: {msg.text[:200]}"
        )
    if text.lower() == "starting telegram monitor":
        context.monitor_started_message_id = context.last_response.id
    logger.info(f"✓ Найден текст: '{text}' (в последних сообщениях)")


@then('я не должен увидеть сообщение, содержащее текст "{text}"')
def step_not_see_message_with_text(context, text):
    """Assert last response does not contain the given substring (case-insensitive)."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    body = msg.text or ""
    if text.lower() in body.lower():
        raise AssertionError(f"Неожиданный текст '{text}' найден в ответе:\n{body}")
    logger.info("✓ Сообщение не содержит: %s", text)


@then("я должен увидеть клавиатуру с кнопками:")
def step_see_keyboard_buttons(context):
    """Assert keyboard has expected buttons from table."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    
    actual_buttons = get_all_buttons(msg)
    expected_buttons = [row["name"] for row in context.table]
    
    missing = []
    for expected in expected_buttons:
        found = any(expected in actual for actual in actual_buttons)
        if not found:
            missing.append(expected)
    
    if missing:
        raise AssertionError(
            f"Не найдены кнопки: {missing}. Доступные: {actual_buttons}"
        )
    
    logger.info(f"✓ Все ожидаемые кнопки найдены: {expected_buttons}")


@then("я логирую в консоль список всех кнопок, которые вижу")
@when("я логирую в консоль список всех кнопок, которые вижу")
def step_log_all_buttons(context):
    """Log all visible buttons to console."""
    msg = context.last_response
    buttons = get_all_buttons(msg)
    
    print("\n" + "=" * 50)
    print(f"[REPORT] Текст сообщения: {msg.text if msg else 'None'}")
    print("[REPORT] Видимые кнопки:")
    for i, btn in enumerate(buttons, 1):
        print(f"  {i}. {btn}")
    print("=" * 50 + "\n")
    
    logger.info(f"[REPORT] Всего кнопок: {len(buttons)}")


@then("бот должен прислать сообщение с блоком событий")
def step_see_events_block(context):
    """Assert response contains events block."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    assert msg.text is not None, "Ответ бота пустой"
    
    # Check for typical events indicators (dates, times, emojis)
    text = msg.text
    has_events = (
        len(text) > 50 or  # Non-trivial content
        any(char in text for char in ["📅", "🎭", "🎵", "🎪", "📍"]) or
        re.search(r'\d{1,2}[:\.]\d{2}', text)  # Time pattern
    )
    
    assert has_events, f"Не похоже на блок событий: {text[:100]}"
    logger.info("✓ Получен блок событий")


@then('под сообщением должна быть кнопка "{btn_text}"')
def step_should_have_button(context, btn_text):
    """Assert message has specific button."""
    msg = context.last_response
    btn = find_button(msg, btn_text)
    
    if not btn:
        available = get_all_buttons(msg)
        raise AssertionError(
            f"Кнопка '{btn_text}' не найдена. Доступные: {available}"
        )
    
    logger.info(f"✓ Найдена кнопка: '{btn_text}'")


@then('событие "{title}" присутствует в списке')
def step_event_present_in_list(context, title):
    """Assert event title is present in the /events list."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    target_title = title
    if title == "Название события" and hasattr(context, "control_event_title"):
        target_title = context.control_event_title
    event_id = _find_event_id_in_text(text, target_title)
    assert event_id is not None, f"Событие '{target_title}' не найдено в списке"
    context.last_event_id = event_id
    logger.info("✓ Событие найдено: id=%s title=%s", event_id, target_title)


@then('событие "{title}" отсутствует в списке')
def step_event_absent_in_list(context, title):
    """Assert event title is absent in the /events list."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    target_title = title
    if title == "Название события" and hasattr(context, "control_event_title"):
        target_title = context.control_event_title
    event_id = _find_event_id_in_text(text, target_title)
    assert event_id is None, f"Событие '{target_title}' найдено, но должно отсутствовать"
    logger.info("✓ Событие отсутствует: %s", target_title)


@then("я жду обновления сообщения")
def step_wait_for_update(context):
    """Wait for message to be edited/updated."""
    import asyncio
    
    async def _wait():
        await asyncio.sleep(3)  # Give bot time to update
        
        # Refresh last message
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]
    
    run_async(context, _wait())
    logger.info("✓ Дождались обновления")


@when('я жду сообщения с текстом "{text}"')
@then('я жду сообщения с текстом "{text}"')
def step_wait_for_message_text(context, text):
    """Wait for a new message containing specific text."""
    async def _wait():
        import asyncio
        # Try for 5 seconds
        for _ in range(10):
            messages = await context.client.client.get_messages(
                context.bot_entity, limit=5
            )
            for msg in messages:
                if msg.text and text.lower() in msg.text.lower():
                    context.last_response = msg
                    logger.info(f"✓ Найдено ожидаемое сообщение: '{text}'")
                    return
            await asyncio.sleep(0.5)
        
        raise AssertionError(f"Сообщение с текстом '{text}' не получено за 5 секунд. Последние: {[m.text for m in messages]}")

    run_async(context, _wait())


@then("я пишу в лог количество отображенных событий")
def step_log_events_count(context):
    """Log estimated number of events in the message."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    
    # Count events by looking for patterns (dates, times, or bullets)
    date_pattern = r'\d{1,2}\s+[а-яА-Я]+(?:\s+\d{4})?'
    time_pattern = r'\d{1,2}[:\.]\d{2}'
    
    dates = len(re.findall(date_pattern, text))
    times = len(re.findall(time_pattern, text))
    
    # Rough estimate: each event typically has a date or time
    estimated_events = max(dates, times, 1)
    
    print("\n" + "=" * 50)
    print(f"[REPORT] Примерное количество событий: {estimated_events}")
    print(f"[REPORT] Найдено дат: {dates}, времён: {times}")
    print(f"[REPORT] Длина текста: {len(text)} символов")
    print("=" * 50 + "\n")
    
    logger.info(f"[REPORT] Событий: ~{estimated_events}")


@then("я логирую полный текст сообщения")
def step_log_full_message(context):
    """Log the full text of the last response."""
    msg = context.last_response
    text = msg.text if msg and msg.text else "[No text]"
    
    print("\n" + "=" * 50)
    print("[REPORT] Полный текст ответа:")
    print(text)
    print("=" * 50 + "\n")
    
    logger.info(f"[REPORT] Текст сообщения ({len(text)} chars)")


@then("я должен найти в ответе действующую ссылку на телеграф")
def step_check_telegraph_link(context):
    """Assert response contains valid and accessible Telegraph links."""
    import aiohttp
    
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    
    # Regex for Telegraph links
    link_pattern = r"https://telegra\.ph/[a-zA-Z0-9_-]+"
    links = re.findall(link_pattern, text)
    
    assert len(links) > 0, f"Не найдено ни одной ссылки на telegra.ph в тексте:\n{text}"
    
    print("\n" + "=" * 50)
    print(f"[REPORT] Найдены ссылки Telegraph ({len(links)}):")
    for link in links:
        print(f"  - {link}")
    print("=" * 50 + "\n")
    
    # Verify each link is accessible via HTTP
    async def _verify():
        async with aiohttp.ClientSession() as session:
            for link in links:
                try:
                    async with session.head(link, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            raise AssertionError(f"Telegraph ссылка {link} вернула статус {resp.status}")
                        logger.info(f"✓ Ссылка работает: {link}")
                except Exception as e:
                    raise AssertionError(f"Не удалось проверить ссылку {link}: {e}")
    
    run_async(context, _verify())
    context.telegraph_links = links
    logger.info(f"✓ Все {len(links)} Telegraph ссылок валидны")


@then('каждая Telegraph страница должна содержать "{required_text}"')
def step_verify_telegraph_content(context, required_text):
    """Verify each Telegraph page contains required content."""
    import aiohttp
    
    links = getattr(context, 'telegraph_links', [])
    if not links:
        raise AssertionError("Нет сохранённых Telegraph ссылок для проверки")
    
    required_items = [item.strip() for item in required_text.split(",")]
    
    async def _verify_content():
        async with aiohttp.ClientSession() as session:
            failed_pages = []
            
            for link in links:
                try:
                    async with session.get(link, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            failed_pages.append(f"{link}: HTTP {resp.status}")
                            continue
                        
                        html = await resp.text()
                        
                        missing = []
                        for item in required_items:
                            if item.lower() not in html.lower():
                                missing.append(item)
                        
                        if missing:
                            failed_pages.append(f"{link}: отсутствует [{', '.join(missing)}]")
                        else:
                            logger.info(f"✓ Страница {link} содержит все элементы: {required_items}")
                
                except Exception as e:
                    failed_pages.append(f"{link}: ошибка {e}")
            
            if failed_pages:
                print("\n" + "=" * 60)
                print("[ERROR] Проверка контента Telegraph страниц:")
                for fail in failed_pages:
                    print(f"  ✗ {fail}")
                print("=" * 60 + "\n")
                raise AssertionError(f"Не все страницы содержат требуемый контент: {failed_pages}")
    
    run_async(context, _verify_content())
    logger.info(f"✓ Все {len(links)} страниц содержат: {required_items}")


@then("я жду медиа-сообщения")
def step_check_media_message(context):
    """Wait for a message with media."""
    import asyncio
    async def _wait():
        for i in range(10): # 5 seconds
            messages = await context.client.client.get_messages(
                 context.bot_entity, limit=5
            )
            for msg in messages:
                if msg.media:
                    context.last_response = msg
                    logger.info("✓ Медиа-сообщение получено")
                    return
            await asyncio.sleep(0.5)
        raise AssertionError("Медиа-сообщение не получено")
    run_async(context, _wait())

@then('под сообщением должны быть кнопки: "{buttons}"')
def step_check_inline_buttons_custom(context, buttons):
    """Verify specific buttons are present (partial match)."""
    expected = [b.strip() for b in buttons.split(",")]
    msg = context.last_response
    visible = get_all_buttons(msg)
    
    missing = []
    for exp in expected:
        found = False
        for v in visible:
            if exp.strip('"').strip("'") in v:
                found = True
                break
        if not found:
            missing.append(exp)
    
    if missing:
        print(f"[ERROR] Expected: {expected}")
        print(f"[ERROR] Visible: {visible}")
        raise AssertionError(f"Не найдены кнопки: {missing}")
    logger.info(f"✓ Найдены все кнопки: {expected}")


@then('я жду долгой операции с текстом "{text}"')
def step_wait_long_operation(context, text):
    """Wait for a long operation for message containing text.

    Kaggle jobs can take >5 minutes even in normal conditions, so keep the
    timeout generous and configurable.
    """
    async def _wait():
        import asyncio
        import os

        text_norm = (text or "").strip().lower()
        if text_norm == "telegram monitor":
            timeout_sec = int(os.getenv("E2E_TG_MONITOR_TIMEOUT_SEC", str(20 * 60)))
        else:
            timeout_sec = int(os.getenv("E2E_LONG_OPERATION_TIMEOUT_SEC", str(5 * 60)))

        reconnect_attempts = 0
        min_id = getattr(context, "monitor_started_message_id", None)
        for i in range(int(timeout_sec / 0.5)):
            try:
                messages = await context.client.client.get_messages(
                    context.bot_entity, limit=10
                )
            except Exception as exc:
                msg = str(exc)
                if "AuthKeyDuplicatedError" in msg or "disconnected" in msg:
                    reconnect_attempts += 1
                    if reconnect_attempts > 5:
                        raise
                    logger.warning(
                        "Reconnecting Telethon after disconnect (%s/%s): %s",
                        reconnect_attempts,
                        5,
                        msg,
                    )
                    await asyncio.sleep(5)
                    try:
                        await context.client.connect()
                    except Exception as reconnect_exc:
                        logger.warning("Reconnect failed: %s", reconnect_exc)
                    continue
                raise
            for msg in messages:
                if min_id and msg.id <= min_id:
                    continue
                if msg.text and text.lower() in msg.text.lower():
                    if text.lower() == "telegram monitor" and "run_id:" not in msg.text.lower():
                        continue
                    context.last_response = msg
                    context.last_report_text = msg.text
                    run_id = _extract_run_id(msg.text)
                    if run_id:
                        context.last_monitor_run_id = run_id
                    context.last_report_stats = {
                        "Сообщений пропущено": _extract_report_stat(msg.text, "Сообщений пропущено"),
                        "Создано": _extract_report_stat(msg.text, "Создано"),
                    }
                    logger.info(f"✓ Найден результат долгой операции: '{text}' (за {i*0.5:.1f}с)")
                    return
            await asyncio.sleep(0.5)
        
        last_texts = [m.text[:100] if m.text else "(no text)" for m in messages[:3]]
        raise AssertionError(
            f"Сообщение с текстом '{text}' не получено за {timeout_sec}с. Последние: {last_texts}"
        )

    run_async(context, _wait())


@then('в отчёте мониторинга значение "{label}" равно "{expected}"')
def step_report_stat_equals(context, label, expected):
    report_text = getattr(context, "last_report_text", None) or (
        context.last_response.text if context.last_response else ""
    )
    value = _extract_report_stat(report_text, label)
    if value is None:
        raise AssertionError(f"Не найден счётчик '{label}' в отчёте:\n{report_text}")
    if value != int(expected):
        raise AssertionError(f"Ожидали '{label}: {expected}', получили '{label}: {value}'")
    logger.info("✓ Отчёт: %s=%s", label, value)


@then('в отчёте увеличивается счётчик "{label}"')
def step_report_counter_increases(context, label):
    """Assert report counter increased compared to baseline."""
    report_text = getattr(context, "last_report_text", None) or (context.last_response.text if context.last_response else "")
    current_value = _extract_report_stat(report_text, label)
    if current_value is None:
        raise AssertionError(f"Не найден счётчик '{label}' в отчёте")
    baseline = getattr(context, "baseline_report_stats", {}).get(label)
    if baseline is None:
        raise AssertionError(f"Нет базового значения для '{label}'")
    assert current_value > baseline, f"Счётчик '{label}' не увеличился: {baseline} -> {current_value}"
    logger.info("✓ Счётчик увеличился: %s %s->%s", label, baseline, current_value)


@then("новые события не создаются")
def step_no_new_events_created(context):
    """Assert report shows no newly created events."""
    report_text = getattr(context, "last_report_text", None) or (context.last_response.text if context.last_response else "")
    created = _extract_report_stat(report_text, "Создано")
    if created is None:
        raise AssertionError("Не найден счётчик 'Создано' в отчёте")
    assert created == 0, f"Ожидалось Создано: 0, получено: {created}"
    logger.info("✓ Новые события не создавались")


@then('событие содержит новые изображения из поста "{post_url}"')
def step_event_has_images_from_post(context, post_url):
    """Verify telegraph pages include poster images from the Telegram post."""
    run_id = getattr(context, "last_monitor_run_id", None)
    if not run_id:
        raise AssertionError("Не найден run_id последнего мониторинга")
    data = _load_tg_results(run_id)
    if not data:
        raise AssertionError(f"Не удалось загрузить telegram_results.json для run_id={run_id}")
    messages = data.get("messages") or []
    target = None
    for msg in messages:
        if msg.get("source_link") == post_url:
            target = msg
            break
    if not target:
        match = re.search(r"t\\.me/([^/]+)/([0-9]+)", post_url)
        if match:
            username = match.group(1)
            message_id = int(match.group(2))
            for msg in messages:
                if (
                    msg.get("source_username") == username
                    and int(msg.get("message_id") or 0) == message_id
                ):
                    target = msg
                    break
    if not target:
        raise AssertionError(f"Пост {post_url} не найден в telegram_results.json")
    posters = target.get("posters") or []
    catbox_urls = [p.get("catbox_url") for p in posters if p.get("catbox_url")]
    if not catbox_urls:
        raise AssertionError(f"Нет catbox_url для поста {post_url}")
    links = getattr(context, "telegraph_links", None)
    if not links:
        text = context.last_response.text if context.last_response else ""
        links = re.findall(r"https://telegra\\.ph/[a-zA-Z0-9_-]+", text)
    if not links:
        raise AssertionError("Нет Telegraph ссылок для проверки изображений")

    async def _verify():
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for link in links:
                async with session.get(link, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    html = await resp.text()
                    if any(url in html for url in catbox_urls):
                        logger.info("✓ Найдена картинка из поста на странице %s", link)
                        return
        raise AssertionError("Catbox URL из поста не найден на Telegraph страницах")

    run_async(context, _verify())


@then("телеграф сохраняет исходные изображения")
def step_telegraph_preserves_images(context):
    """Ensure Telegraph still contains all previously seen catbox URLs."""
    snapshot = getattr(context, "telegraph_snapshot", None)
    if not snapshot:
        raise AssertionError("Нет сохранённого снимка Telegraph")
    prev_urls = set(snapshot.get("catbox_urls") or [])
    if not prev_urls:
        raise AssertionError("Снимок Telegraph не содержит catbox URL")

    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    links = re.findall(r"https://telegra\\.ph/[a-zA-Z0-9_-]+", text)
    if not links:
        links = snapshot.get("links") or []
    if not links:
        raise AssertionError("Не найдено ссылок telegra.ph для проверки")

    async def _verify():
        html_pages = await _fetch_telegraph_pages(links)
        now_urls = set()
        for html in html_pages:
            now_urls.update(_extract_catbox_urls(html))
        missing = sorted(prev_urls - now_urls)
        if missing:
            raise AssertionError(f"Исходные изображения пропали: {missing[:5]}")

    run_async(context, _verify())
    logger.info("✓ Исходные изображения сохранены")


@then("я вижу лог источников с датой, временем, источником и фактами")
def step_source_facts_log_present(context):
    """Verify source facts log contains timestamp, source reference, and fact bullets."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""

    # The "🧾 Лог источников" button sends a NEW message; depending on timing
    # context.last_response may still point to the event card. Re-fetch and
    # pick the actual log message by its header.
    if not text or "Лог источников" not in text:
        async def _fetch():
            messages = await context.client.client.get_messages(context.bot_entity, limit=20)
            for candidate in messages:
                if candidate.text and "Лог источников" in candidate.text:
                    return candidate.text
            return ""

        text = run_async(context, _fetch())

    if not text:
        raise AssertionError("Лог источников пуст или не получен")

    # Date/time pattern: 2026-01-28 19:00
    if not re.search(r"\b20\d{2}-\d{2}-\d{2} \d{2}:\d{2}\b", text):
        raise AssertionError("В логе нет даты и времени (формат YYYY-MM-DD HH:MM)")

    # Source marker: URL or source type keyword
    if not re.search(r"(https?://|t\\.me/|telegram|vk|site|parser|manual|bot)", text, re.IGNORECASE):
        raise AssertionError("В логе нет указания источника")

    # Thesis marker: bullet list
    if not re.search(r"[•*\-]\s+\S+", text):
        raise AssertionError("В логе нет фактов (ожидаются bullets)")

    logger.info("✓ Лог источников содержит дату/время, источник и факты")


@then("в карточке события отображается блок OCR")
def step_event_card_has_ocr_block(context):
    """Ensure event edit card shows OCR block."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    if "Poster OCR:" not in text:
        raise AssertionError("Блок Poster OCR не найден в карточке события")
    logger.info("✓ Блок Poster OCR отображается")


@then("в карточке события есть catbox_url")
def step_event_card_has_catbox_url(context):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    if "catbox_url:" not in text and "catbox.moe" not in text:
        raise AssertionError("В карточке события не найден catbox_url (ожидали catbox_url: ...)")
    logger.info("✓ В карточке события есть catbox_url")


@then('в OCR есть текст "{needle}"')
def step_event_card_ocr_contains(context, needle):
    """Ensure OCR block includes specific text fragment."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    if "Poster OCR:" not in text:
        raise AssertionError("Блок Poster OCR не найден в карточке события")
    if needle.lower() not in text.lower():
        raise AssertionError(f"Не найден OCR фрагмент: {needle}")
    logger.info("✓ OCR содержит фрагмент: %s", needle)


@then('в карточке события location_name равен "{expected}"')
def step_event_card_location_equals(context, expected):
    """Ensure location_name line matches expected value."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    match = re.search(r"^location_name:\\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise AssertionError("Строка location_name не найдена в карточке события")
    value = match.group(1).strip()
    if value != expected:
        raise AssertionError(f"location_name отличается: '{value}' != '{expected}'")
    logger.info("✓ location_name совпадает: %s", expected)


@then('в карточке события location_name не содержит "{unexpected}"')
def step_event_card_location_not_contains(context, unexpected):
    """Ensure location_name line does not include unexpected fragment."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    match = re.search(r"^location_name:\\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise AssertionError("Строка location_name не найдена в карточке события")
    value = match.group(1).strip()
    if unexpected.lower() in value.lower():
        raise AssertionError(f"location_name содержит нежелательный фрагмент: {unexpected}")
    logger.info("✓ location_name не содержит: %s", unexpected)


@given("в базе создано тестовое событие:")
def step_create_test_event(context):
    """Insert test events into the DB."""
    _ensure_test_context(context)
    table_rows = []
    for row in context.table:
        table_rows.append({key: row[key] for key in context.table.headings})

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        for row in table_rows:
            data = {}
            for key, value in row.items():
                data[key] = _coerce_field_value(key, value)
            event_id = _insert_event(conn, data)
            title = data.get("title") or ""
            context.test_event_ids.append(event_id)
            if title:
                context.test_events_by_title[title] = event_id
        conn.commit()
    finally:
        conn.close()


@given('для события "{title}" добавлен источник "{source_url}" типа "{source_type}"')
def step_add_event_source(context, title, source_url, source_type):
    _ensure_test_context(context)
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для добавления источника")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _insert_event_source(conn, event_id, source_url, source_type)
        conn.commit()
    finally:
        conn.close()


@when('я запускаю Smart Update на основе события "{title}" с правками:')
def step_run_smart_update_from_event(context, title):
    _ensure_test_context(context)
    overrides = _table_to_dict(context.table) if context.table else {}
    overrides = {k: _coerce_field_value(k, v) for k, v in overrides.items()}
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        row = _fetch_event_by_title(conn, title)
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие '{title}' не найдено для Smart Update")

    from smart_event_update import EventCandidate, smart_event_update

    def _pick(key: str, fallback):
        return overrides[key] if key in overrides else fallback

    candidate_data = {
        "source_type": _pick("source_type", "manual"),
        "source_url": _pick("source_url", None),
        "source_text": _pick("source_text", row["source_text"] or row["description"]),
        "title": _pick("title", row["title"]),
        "date": _pick("date", row["date"]),
        "time": _pick("time", row["time"]),
        "end_date": _pick("end_date", row["end_date"]),
        "festival": _pick("festival", row["festival"]),
        "location_name": _pick("location_name", row["location_name"]),
        "location_address": _pick("location_address", row["location_address"]),
        "city": _pick("city", row["city"]),
        "ticket_link": _pick("ticket_link", row["ticket_link"]),
        "ticket_price_min": _pick("ticket_price_min", row["ticket_price_min"]),
        "ticket_price_max": _pick("ticket_price_max", row["ticket_price_max"]),
        "ticket_status": _pick("ticket_status", row["ticket_status"]),
        "event_type": _pick("event_type", row["event_type"]),
        "emoji": _pick("emoji", row["emoji"]),
        "is_free": _pick("is_free", row["is_free"]),
        "pushkin_card": _pick("pushkin_card", row["pushkin_card"]),
        "search_digest": _pick("search_digest", row["search_digest"]),
        "raw_excerpt": _pick("raw_excerpt", row["description"]),
        "source_chat_username": _pick("source_chat_username", None),
        "source_chat_id": _pick("source_chat_id", None),
        "source_message_id": _pick("source_message_id", None),
        "creator_id": _pick("creator_id", None),
        "trust_level": _pick("trust_level", None),
    }
    candidate = EventCandidate(**candidate_data)
    db = _get_smart_db(context)
    result = run_async(
        context,
        smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False),
    )
    context.last_smart_update = result
    context.last_smart_update_event_id = result.event_id
    context.last_smart_update_candidate_title = candidate.title
    if result.created and result.event_id:
        context.test_event_ids.append(result.event_id)
        if candidate.title:
            context.test_events_by_title[candidate.title] = result.event_id


@when("я запускаю Smart Update с кандидатом:")
def step_run_smart_update_candidate(context):
    _ensure_test_context(context)
    payload = _table_to_dict(context.table) if context.table else {}
    payload = {k: _coerce_field_value(k, v) for k, v in payload.items()}
    from smart_event_update import EventCandidate, smart_event_update

    candidate = EventCandidate(**payload)
    db = _get_smart_db(context)
    result = run_async(
        context,
        smart_event_update(db, candidate, check_source_url=True, schedule_tasks=False),
    )
    context.last_smart_update = result
    context.last_smart_update_event_id = result.event_id
    context.last_smart_update_candidate_title = candidate.title
    if result.created and result.event_id:
        context.test_event_ids.append(result.event_id)
        if candidate.title:
            context.test_events_by_title[candidate.title] = result.event_id


@then('результат Smart Update имеет статус "{status}"')
def step_smart_update_status(context, status):
    result = getattr(context, "last_smart_update", None)
    if not result:
        raise AssertionError("Нет результата Smart Update")
    actual = getattr(result, "status", None)
    if actual != status:
        raise AssertionError(f"Ожидался статус '{status}', получили '{actual}'")


@then('создано новое событие с заголовком "{title}"')
def step_new_event_created(context, title):
    result = getattr(context, "last_smart_update", None)
    if not result or not result.created or not result.event_id:
        raise AssertionError("Smart Update не создал новое событие")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        row = _fetch_event_by_title(conn, title)
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Созданное событие '{title}' не найдено в БД")


@then('для события "{title}" количество источников равно "{count}"')
def step_event_sources_count(context, title, count):
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для проверки источников")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM event_source WHERE event_id = ?",
            (event_id,),
        )
        actual = cur.fetchone()[0]
    finally:
        conn.close()
    if actual != int(count):
        raise AssertionError(f"Ожидалось источников {count}, получили {actual}")


@then('событие "{title}" имеет поля:')
def step_event_fields(context, title):
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для проверки полей")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM event WHERE id = ?", (event_id,))
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие '{title}' не найдено в БД")
    checks = _table_to_dict(context.table)
    for field, expected_raw in checks.items():
        expected = _coerce_field_value(field, expected_raw)
        actual = row[field]
        if field in BOOL_FIELDS:
            actual = bool(actual)
        if expected is None:
            if actual not in (None, "", 0):
                raise AssertionError(f"{field}: ожидали пусто, получили {actual}")
            continue
        if str(actual) != str(expected):
            raise AssertionError(f"{field}: ожидали '{expected}', получили '{actual}'")


@then('для события "{title}" лог фактов содержит "{text}"')
def step_event_facts_contains(context, title, text):
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для проверки лога фактов")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_fact_table(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT fact FROM event_source_fact WHERE event_id = ?",
            (event_id,),
        )
        facts = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
    if not any(text in (fact or "") for fact in facts):
        raise AssertionError(f"В логе фактов нет строки, содержащей '{text}'")


@then("я очищаю тестовые события")
def step_cleanup_test_events(context):
    _cleanup_test_events(context)


@given('в базе есть событие "{title}"')
def step_event_exists(context, title):
    row = _load_event_row_by_title(title)
    if not row:
        raise AssertionError(f"Событие '{title}' не найдено в слепке БД")
    _ensure_test_context(context)
    context.test_events_by_title[title] = int(row["id"])


@given('в базе есть минимум "{count}" событий "{location}" на дату "{date}" и время "{time}" с hall-hint')
def step_parallel_events_exist(context, count, location, date, time):
    from smart_event_update import _extract_hall_hint

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM event
            WHERE location_name = ? AND date = ? AND time = ?
            """,
            (location, date, time),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    with_hall = []
    for row in rows:
        text = (row["source_text"] or "") + "\n" + (row["description"] or "")
        hall = _extract_hall_hint(text)
        if hall:
            with_hall.append((row["id"], row["title"], hall))
    if len(with_hall) < int(count):
        raise AssertionError(
            f"Ожидалось >= {count} событий с hall-hint, найдено {len(with_hall)}"
        )
    context.parallel_events = with_hall


@then('Smart Update вернул event_id как у события "{title}"')
def step_smart_update_event_id_matches(context, title):
    expected_id = _fetch_event_id(context, title)
    if not expected_id:
        raise AssertionError(f"Событие '{title}' не найдено для сравнения")
    result_id = getattr(context, "last_smart_update_event_id", None)
    if result_id != expected_id:
        raise AssertionError(f"Ожидался event_id {expected_id}, получили {result_id}")


@then('я удаляю тестовый источник "{source_url}" у события "{title}"')
def step_remove_event_source(context, source_url, title):
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для удаления источника")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM event_source WHERE event_id = ? AND source_url = ?",
            (event_id, source_url),
        )
        conn.commit()
    finally:
        conn.close()


@given('у события "{title}" временно установлен ticket_link "{url}"')
def step_set_temp_ticket_link(context, title, url):
    _ensure_test_context(context)
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для обновления ticket_link")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT ticket_link FROM event WHERE id = ?", (event_id,))
        old = cur.fetchone()[0]
        context.event_ticket_backup[title] = old
        cur.execute(
            "UPDATE event SET ticket_link = ? WHERE id = ?",
            (url, event_id),
        )
        conn.commit()
    finally:
        conn.close()


@then('я восстанавливаю ticket_link у события "{title}"')
def step_restore_ticket_link(context, title):
    _ensure_test_context(context)
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для восстановления ticket_link")
    if title not in context.event_ticket_backup:
        return
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        old = context.event_ticket_backup.get(title)
        cur.execute(
            "UPDATE event SET ticket_link = ? WHERE id = ?",
            (old, event_id),
        )
        conn.commit()
    finally:
        conn.close()
