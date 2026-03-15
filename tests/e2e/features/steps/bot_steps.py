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
import time
from html import escape, unescape
from datetime import date, datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from behave import given, step, then, when

logger = logging.getLogger("e2e.steps")

_RUS_STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко", "у", "о", "об", "от", "за",
    "для", "из", "или", "а", "но", "что", "как", "это", "этот", "эта", "эти",
    "бы", "не", "ни", "же", "ли", "мы", "вы", "он", "она", "они", "его", "ее",
    "их", "при", "над", "под", "до", "после", "без", "через", "между", "также",
}

_TECH_FACT_PREFIXES = (
    "добавлена афиша:",
    "добавлено изображение:",
    "добавлено изображение из",
    "telegraph:",
    "источник:",
)

_NON_TEXT_FACT_PREFIXES = (
    "дата:",
    "время:",
    "локация:",
    "location:",
    "добавлена афиша:",
    "добавлены афиши:",
    "добавлено изображение:",
    "добавлено изображение из",
    "афиша в источнике:",
    "источник:",
    "telegraph:",
    "текст дополнен:",
)

_RU_MONTH_GENITIVE = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


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


def _raise_on_bot_ui_errors(messages, *, baseline_id: int) -> None:
    """Fail fast on operator-visible errors in Telegram UI during live E2E.

    We treat bot status messages like:
      - "Результат: ошибка …"
      - "Результат: ошибка извлечения событий …"
    as hard failures, because otherwise the E2E run may appear to "hang" while
    silently skipping posts.
    """
    for msg in messages or []:
        mid = int(getattr(msg, "id", 0) or 0)
        if mid <= int(baseline_id or 0):
            continue
        text = (msg.text or "").strip()
        if not text:
            continue
        if "Результат:" not in text:
            continue
        # Non-fatal outcomes.
        if re.search(r"^\s*Результат:\s*(лимит|limit|событий не найдено)", text, flags=re.IGNORECASE | re.MULTILINE):
            continue
        if re.search(r"^\s*Результат:\s*ошибка\b", text, flags=re.IGNORECASE | re.MULTILINE):
            preview = text.replace("\n", " ")[:500]
            raise AssertionError(
                "В Telegram UI обнаружена ошибка выполнения операции. "
                f"message_id={mid} text={preview}"
            )


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


async def _probe_image_webp(url: str) -> tuple[bool, str]:
    """Return (is_webp, debug) for an image URL (best-effort, low bandwidth)."""
    import aiohttp

    headers = {
        "User-Agent": "events-bot-e2e/1.0",
        "Range": "bytes=0-63",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                status = int(getattr(resp, "status", 0) or 0)
                ct = str(resp.headers.get("Content-Type") or "").lower()
                body = await resp.content.read(64)
    except Exception as exc:
        return False, f"fetch_error={type(exc).__name__}:{exc}"

    riff = (len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP")
    ct_webp = ("image/webp" in ct)
    ok = (status in {200, 206}) and (riff or ct_webp)
    return ok, f"status={status} ct={ct!r} riff_webp={riff}"


def _normalize_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value or "").strip().lower())


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</?(p|div|h1|h2|h3|h4|h5|h6|li|ul|ol|blockquote)[^>]*>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _split_fact_status_marker(fact: str) -> tuple[str | None, str]:
    """Parse status marker prefix from source-log fact line.

    Source log bullets look like:
      • ✅ <fact>        -> added
      • ↩️ <fact>        -> duplicate (ignored)
      • ⚠️ <fact>        -> conflict (ignored)
      • ℹ️ <fact>        -> note (service)
    Older logs may have no marker -> treat as added.
    """
    raw = (fact or "").strip()
    if not raw:
        return None, ""
    # NOTE: use a single backslash in regex escapes. Double-backslashes here would
    # make the pattern match a literal "\".
    m = re.match(r"^(?P<icon>✅|\+|↩️|⚠️|ℹ️)\s+(?P<body>.+)$", raw)
    if not m:
        return None, raw
    icon = (m.group("icon") or "").strip()
    body = (m.group("body") or "").strip()
    mapping = {
        "✅": "added",
        "+": "added",
        "↩️": "duplicate",
        "⚠️": "conflict",
        "ℹ️": "note",
    }
    return mapping.get(icon), body


def _extract_semantic_facts_from_log(log_text: str) -> list[str]:
    facts: list[str] = []
    for line in (log_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^[•*\-]\s+(.+)$", s)
        if not m:
            continue
        raw_fact = re.sub(r"\s+", " ", m.group(1)).strip()
        status, fact = _split_fact_status_marker(raw_fact)
        if not fact:
            continue
        # Only "added" facts are expected to be reflected on Telegraph.
        if status in {"duplicate", "conflict", "note"}:
            continue
        fact_l = fact.lower()
        if any(fact_l.startswith(prefix) for prefix in _TECH_FACT_PREFIXES):
            continue
        if fact_l.startswith("текст дополнен:"):
            # Keep only the meaningful payload, the prefix is an internal log label.
            fact = fact.split(":", 1)[1].strip() if ":" in fact else fact
            fact_l = fact.lower()
            if not fact:
                continue
        if any(fact_l.startswith(prefix) for prefix in _NON_TEXT_FACT_PREFIXES):
            # Non-textual facts are still useful in the source log, but the event page
            # may render them in a different format (date/time/location blocks, images),
            # so we don't assert them by substring search in Telegraph HTML.
            continue
        if "http://" in fact_l or "https://" in fact_l:
            # URL-only or URL-heavy bullets are technical artifacts; body text check is not meaningful.
            bare = re.sub(r"https?://\S+", "", fact_l).strip(" :;,.")
            if len(bare) < 8:
                continue
        facts.append(fact)
    # Preserve order, remove exact duplicates.
    out: list[str] = []
    seen: set[str] = set()
    for f in facts:
        key = _normalize_search_text(f)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _is_textual_fact(fact: str) -> bool:
    fact_norm = _normalize_search_text(fact)
    if not fact_norm:
        return False
    if any(fact_norm.startswith(prefix) for prefix in _NON_TEXT_FACT_PREFIXES):
        return False
    if len(re.findall(r"[a-zа-яё0-9]{4,}", fact_norm, flags=re.IGNORECASE)) < 3:
        return False
    return True


def _parse_source_log_sections(log_text: str) -> list[dict]:
    """Parse operator source-log message into per-source sections."""
    sections: list[dict] = []
    current: dict | None = None
    header_re = re.compile(
        r"^\s*(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}[^—]*)\s+—\s+(?P<source>[^|]+?)\s*\|\s*(?P<url>\S+)\s*$"
    )
    bullet_re = re.compile(r"^[•*\-]\s+(.+)$")
    for line in (log_text or "").splitlines():
        raw = (line or "").strip()
        if not raw:
            continue
        m = header_re.match(raw)
        if m:
            current = {
                "timestamp": (m.group("ts") or "").strip(),
                "source": (m.group("source") or "").strip(),
                "url": (m.group("url") or "").strip(),
                "facts": [],
            }
            sections.append(current)
            continue
        b = bullet_re.match(raw)
        if b and current is not None:
            fact = re.sub(r"\s+", " ", (b.group(1) or "").strip())
            if fact:
                current["facts"].append(fact)
    for section in sections:
        facts = section.get("facts") or []
        semantic = []
        text_facts = []
        facts_by_status: dict[str, list[str]] = {"added": [], "duplicate": [], "conflict": [], "note": []}
        for fact in facts:
            status, clean = _split_fact_status_marker(fact)
            status_key = status or "added"
            if status_key not in facts_by_status:
                status_key = "added"
            if clean:
                facts_by_status[status_key].append(clean)
            fact_norm = _normalize_search_text(clean)
            if not fact_norm:
                continue
            if status in {"duplicate", "conflict", "note"}:
                continue
            if not any(fact_norm.startswith(prefix) for prefix in _TECH_FACT_PREFIXES):
                semantic.append(clean)
            if _is_textual_fact(clean):
                text_facts.append(clean)
        section["semantic_facts"] = semantic
        section["text_facts"] = text_facts
        section["facts_by_status"] = facts_by_status
    return sections


def _ensure_stage_snapshot_state(context) -> tuple[dict, Path]:
    if not hasattr(context, "stage_snapshots") or not isinstance(context.stage_snapshots, dict):
        context.stage_snapshots = {}
    if not hasattr(context, "stage_snapshot_dir"):
        scenario_name = "scenario"
        if getattr(context, "scenario", None) and getattr(context.scenario, "name", None):
            scenario_name = str(context.scenario.name)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", scenario_name).strip("_").lower() or "scenario"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = Path("artifacts/e2e/stage_snapshots") / f"{ts}_{slug}"
        path.mkdir(parents=True, exist_ok=True)
        context.stage_snapshot_dir = path
    return context.stage_snapshots, Path(context.stage_snapshot_dir)


def _resolve_telegraph_token_for_e2e() -> str | None:
    candidates: list[Path] = []
    env_path = (os.getenv("TELEGRAPH_TOKEN_FILE") or "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("/data/telegraph_token.txt"))
    candidates.append(Path("artifacts/run/telegraph_token.txt"))

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not path.exists():
                continue
            token = (path.read_text(encoding="utf-8") or "").strip()
            if token:
                return token
        except Exception:
            continue
    return None


def _text_to_telegraph_paragraphs(text: str, max_chars: int) -> str:
    trimmed = (text or "").strip()
    if not trimmed:
        return "<p>(пустой текст)</p>"
    trimmed = trimmed[:max_chars].strip()
    # Telegraph text extraction often contains invisible separators (ZWSP).
    # Normalize them so paragraph splitting works deterministically.
    trimmed = trimmed.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
    chunks = [c.strip() for c in re.split(r"\n{2,}", trimmed) if c.strip()]
    if not chunks:
        chunks = [trimmed]
    html_chunks: list[str] = []
    for chunk in chunks[:180]:
        raw = (chunk or "").strip()
        if not raw:
            continue
        # Preserve Markdown-style quotes as true Telegraph quotes in snapshot archive pages.
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        is_quote = bool(lines) and all(ln.lstrip().startswith(">") for ln in lines)
        if is_quote:
            quote_lines: list[str] = []
            for ln in lines:
                s = ln.lstrip()
                s = s[1:].lstrip()  # drop leading ">"
                if s:
                    quote_lines.append(escape(s))
            if quote_lines:
                html_chunks.append(f"<blockquote>{'<br/>'.join(quote_lines)}</blockquote>")
                continue
        html_chunks.append(f"<p>{escape(raw).replace(chr(10), '<br/>')}</p>")
    return "".join(html_chunks)


def _create_stage_snapshot_archive_page(
    *,
    label: str,
    event_title: str,
    source_telegraph_url: str,
    telegraph_text: str,
) -> str | None:
    token = _resolve_telegraph_token_for_e2e()
    if not token:
        logger.warning("stage snapshot archive: telegraph token not found")
        return None
    try:
        from telegraph import Telegraph
        from telegraph.utils import html_to_nodes
    except Exception as exc:
        logger.warning("stage snapshot archive: telegraph package unavailable: %s", exc)
        return None

    tg = Telegraph(access_token=token)
    title = f"E2E snapshot {label} - {(event_title or 'event')[:80]}".strip()
    for limit in (24000, 18000, 12000, 8000):
        try:
            body = (
                f"<p><strong>Snapshot:</strong> {escape(label)}</p>"
                f"<p><strong>Event title:</strong> {escape(event_title or '')}</p>"
                f'<p><strong>Source page:</strong> <a href="{escape(source_telegraph_url)}">'
                f"{escape(source_telegraph_url)}</a></p>"
                f"<hr/>"
                f"{_text_to_telegraph_paragraphs(telegraph_text, max_chars=limit)}"
            )
            nodes = html_to_nodes(body)
            page = tg.create_page(
                title=title,
                author_name="E2E Stage Snapshot",
                content=nodes,
                return_content=False,
            )
            url = (page.get("url") or "").strip() if isinstance(page, dict) else ""
            if url:
                return url
        except Exception:
            continue
    logger.warning("stage snapshot archive: failed to create page for label=%s", label)
    return None


def _capture_stage_snapshot(context, *, label: str, event_id: int) -> dict:
    snapshots, out_dir = _ensure_stage_snapshot_state(context)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label or "").strip()).strip("_").lower() or "stage"
    telegraph_url, telegraph_html = _fetch_telegraph_html_for_event(context, int(event_id))
    telegraph_text = _html_to_text(telegraph_html)
    log_text = _fetch_source_log_text(context, int(event_id))
    sections = _parse_source_log_sections(log_text)
    all_semantic_facts = _extract_semantic_facts_from_log(log_text)
    all_text_facts = [f for f in all_semantic_facts if _is_textual_fact(f)]

    with sqlite3.connect(_db_path(), timeout=30) as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT title, date, time, location_name, telegraph_url, telegraph_path FROM event WHERE id = ?",
            (int(event_id),),
        ).fetchone()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено для snapshot")

    snapshot = {
        "label": safe_label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": {
            "id": int(event_id),
            "title": row[0],
            "date": row[1],
            "time": row[2],
            "location_name": row[3],
            "telegraph_url": row[4],
            "telegraph_path": row[5],
        },
        "telegraph": {
            "url": telegraph_url,
            "text_len": len(_normalize_search_text(telegraph_text)),
            "html_len": len(telegraph_html or ""),
            "catbox_urls": sorted(_extract_catbox_urls(telegraph_html)),
        },
        "source_log": {
            "sections": sections,
            "semantic_facts_total": len(all_semantic_facts),
            "text_facts_total": len(all_text_facts),
        },
    }
    snapshot_archive_url = _create_stage_snapshot_archive_page(
        label=safe_label,
        event_title=str(row[0] or ""),
        source_telegraph_url=telegraph_url,
        telegraph_text=telegraph_text,
    )
    snapshot["telegraph"]["snapshot_url"] = snapshot_archive_url
    snapshots[safe_label] = snapshot

    json_path = out_dir / f"{safe_label}.json"
    log_path = out_dir / f"{safe_label}.source_log.txt"
    telegraph_path = out_dir / f"{safe_label}.telegraph.txt"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(log_text or "", encoding="utf-8")
    telegraph_path.write_text(telegraph_text or "", encoding="utf-8")
    snapshot["artifact_json"] = str(json_path)
    snapshot["artifact_log"] = str(log_path)
    snapshot["artifact_telegraph_text"] = str(telegraph_path)
    return snapshot


def _fact_matches_telegraph_text(fact: str, telegraph_text_norm: str) -> bool:
    # Time facts: allow matching by the time token alone.
    # Many pages display time in the summary block (🗓 ... 19:00), while the fact
    # may be phrased as "Начало спектакля в 19:00." in the source log.
    if fact:
        times = re.findall(r"\b\d{1,2}[:.]\d{2}\b", fact)
        if times:
            for t in times:
                cand = t.replace(".", ":")
                if cand and cand in (telegraph_text_norm or ""):
                    return True

    def _normalize_token_stem(token: str) -> str:
        t = (token or "").strip().lower().replace("ё", "е")
        if len(t) <= 4:
            return t
        # Lightweight RU suffix stripping for paraphrase-tolerant fact checks.
        for suf in (
            "иями",
            "ями",
            "ами",
            "иями",
            "ости",
            "ость",
            "ения",
            "ению",
            "ением",
            "ение",
            "овых",
            "евых",
            "ого",
            "ему",
            "ому",
            "ыми",
            "ими",
            "ым",
            "им",
            "иях",
            "ах",
            "ях",
            "ов",
            "ев",
            "ей",
            "ой",
            "ий",
            "ый",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ом",
            "ем",
            "ам",
            "ям",
            "а",
            "я",
            "ы",
            "и",
            "е",
            "о",
            "у",
            "ю",
        ):
            # Allow shorter stems for short tokens (e.g. "хитом" -> "хит").
            min_keep = 5
            if len(t) <= 6:
                min_keep = 3
            if t.endswith(suf) and len(t) - len(suf) >= min_keep:
                return t[: -len(suf)]
        return t

    def _token_present(token: str, haystack: str) -> bool:
        if not token:
            return False
        if token in haystack:
            return True
        stem = _normalize_token_stem(token)
        if not stem:
            return False
        if len(stem) >= 5 and stem in haystack:
            return True
        # For short stems derived from a longer inflected token, allow a smaller threshold.
        return len(token) >= 5 and len(stem) >= 3 and stem in haystack

    fact_norm = _normalize_search_text(fact)
    if not fact_norm:
        return True
    if fact_norm in telegraph_text_norm:
        return True
    # Date facts are often normalized differently in log vs Telegraph body
    # (e.g. "2026-02-12" vs "12 февраля").
    if fact_norm.startswith("дата:"):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", fact_norm)
        if m:
            y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            variants = [
                f"{y:04d}-{mm:02d}-{dd:02d}",
                f"{dd:02d}.{mm:02d}.{y:04d}",
                f"{dd:02d}.{mm:02d}",
                f"{dd} {_RU_MONTH_GENITIVE.get(mm, '')} {y}".strip(),
                f"{dd} {_RU_MONTH_GENITIVE.get(mm, '')}".strip(),
            ]
            for v in variants:
                if v and _normalize_search_text(v) in telegraph_text_norm:
                    return True
    # Time facts may be rendered inline with date ("14 февраля в 17:00")
    # without explicit "Время:" label.
    if fact_norm.startswith("время:") or fact_norm.startswith("time:"):
        m = re.search(r"([01]?\d|2[0-3])[:.\s]([0-5]\d)", fact_norm)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            variants = [
                f"{hh:02d}:{mm:02d}",
                f"{hh}:{mm:02d}",
                f"{hh:02d}.{mm:02d}",
                f"{hh}.{mm:02d}",
            ]
            for v in variants:
                if _normalize_search_text(v) in telegraph_text_norm:
                    return True

    # Fallback: keyword overlap for lightly paraphrased rewrite.
    tokens = re.findall(r"[a-zа-яё0-9]{4,}", fact_norm, flags=re.IGNORECASE)
    tokens = [t for t in tokens if t not in _RUS_STOPWORDS]
    if not tokens:
        return fact_norm in telegraph_text_norm

    unique = list(dict.fromkeys(tokens))
    present = sum(1 for t in unique if _token_present(t, telegraph_text_norm))
    required = max(2, int(len(unique) * 0.4 + 0.5))
    return present >= required


def _resolve_event_telegraph_url(event_id: int) -> str:
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT telegraph_url, telegraph_path FROM event WHERE id = ?", (int(event_id),))
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено в таблице event")
    telegraph_url = (row[0] or "").strip()
    telegraph_path = (row[1] or "").strip().lstrip("/")
    if telegraph_url:
        return telegraph_url
    if telegraph_path:
        return f"https://telegra.ph/{telegraph_path}"
    raise AssertionError(f"У события id={event_id} не заполнены telegraph_url/telegraph_path")


def _fetch_source_log_text(context, event_id: int | None = None, limit: int = 40) -> str:
    msg = context.last_response
    direct = msg.text if msg and msg.text else ""
    if direct and "Лог источников" in direct:
        return direct

    target_url = None
    if event_id:
        with sqlite3.connect(_db_path(), timeout=30) as conn:
            cur = conn.cursor()
            cur.execute("SELECT telegraph_url, telegraph_path FROM event WHERE id = ?", (int(event_id),))
            row = cur.fetchone()
            if row:
                target_url = (row[0] or "").strip() or (
                    f"https://telegra.ph/{(row[1] or '').strip().lstrip('/')}" if (row[1] or "").strip() else None
                )

    async def _fetch():
        messages = await context.client.client.get_messages(context.bot_entity, limit=limit)
        if target_url:
            for candidate in messages:
                t = candidate.text or ""
                if "Лог источников" in t and target_url in t:
                    return t
        for candidate in messages:
            t = candidate.text or ""
            if "Лог источников" in t:
                return t
        return ""

    return run_async(context, _fetch())


def _fetch_telegraph_html_for_event(context, event_id: int) -> tuple[str, str]:
    url = _resolve_event_telegraph_url(event_id)

    async def _fetch():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    raise AssertionError(f"Telegraph {url} вернул статус {resp.status}")
                return await resp.text()

    html = run_async(context, _fetch())
    return url, html


def _db_path() -> str:
    env = os.getenv("DB_PATH")
    if env:
        return env
    preferred = "db_prod_snapshot.sqlite"
    if Path(preferred).exists():
        return preferred
    fallback = "db_prod_snapshot_2026-01-28_154329.sqlite"
    if Path(fallback).exists():
        return fallback
    return preferred


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
    if (os.getenv("E2E_SKIP_DB_INIT") or "").strip().lower() not in {"1", "true", "yes"}:
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
    """Upsert telegram_source rows from provided list (non-destructive by default).

    Safety:
    - In shared test/prod-like DBs we must not delete carefully curated sources.
    - When E2E runs against an isolated DB copy (`E2E_DB_BASE_PATH` is set), we
      may disable other sources for determinism, but still avoid deleting rows.
    """
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        usernames = [str(r.get("username") or "").lstrip("@").strip() for r in rows]
        usernames = [u for u in usernames if u]
        # Deterministic mode: disable all other sources on isolated DB copies.
        # Do not delete anything (operators may want to re-run with the same seed).
        safe_isolated = bool((os.getenv("E2E_DB_BASE_PATH") or "").strip())
        if safe_isolated and usernames:
            placeholders = ",".join("?" for _ in usernames)
            cur.execute(
                f"UPDATE telegram_source SET enabled=0 WHERE username NOT IN ({placeholders})",
                usernames,
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


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _purge_events_by_patterns(conn: sqlite3.Connection, *, like_patterns: list[str], location_prefixes: list[str] | None = None) -> int:
    """Delete events (and related rows) that match given source URL patterns.

    Best-effort: older prod snapshots may miss some tables, so we guard by sqlite_master checks.
    """
    cur = conn.cursor()
    event_ids: set[int] = set()

    if _table_exists(conn, "event_source"):
        _ensure_event_source_table(conn)
        for pat in like_patterns:
            cur.execute(
                "SELECT DISTINCT event_id FROM event_source WHERE source_url LIKE ?",
                (pat,),
            )
            for (eid,) in cur.fetchall():
                try:
                    event_ids.add(int(eid))
                except Exception:
                    pass

    if like_patterns:
        for pat in like_patterns:
            # event.source_post_url / source_vk_post_url / ticket_link are commonly populated
            for col in ["source_post_url", "source_vk_post_url", "ticket_link"]:
                try:
                    cur.execute(
                        f"SELECT id FROM event WHERE {col} LIKE ?",
                        (pat,),
                    )
                except Exception:
                    continue
                for (eid,) in cur.fetchall():
                    try:
                        event_ids.add(int(eid))
                    except Exception:
                        pass

    if location_prefixes:
        for pref in location_prefixes:
            try:
                cur.execute(
                    "SELECT id FROM event WHERE location_name LIKE ?",
                    (f"{pref}%",),
                )
            except Exception:
                continue
            for (eid,) in cur.fetchall():
                try:
                    event_ids.add(int(eid))
                except Exception:
                    pass

    ids = sorted(event_ids)
    if not ids:
        return 0

    placeholders = ",".join("?" for _ in ids)
    for table, col in [
        ("event_source_fact", "event_id"),
        ("event_source", "event_id"),
        ("eventposter", "event_id"),
        ("job_outbox", "event_id"),
    ]:
        if not _table_exists(conn, table):
            continue
        try:
            cur.execute(f"DELETE FROM {table} WHERE {col} IN ({placeholders})", ids)
        except Exception:
            continue
    cur.execute(f"DELETE FROM event WHERE id IN ({placeholders})", ids)
    conn.commit()
    return len(ids)


@given('база очищена от событий источника "{source}"')
def step_purge_events_for_source(context, source):
    key = (source or "").strip().lower()
    if not key:
        raise AssertionError("Пустой source для очистки")
    patterns: list[str] = []
    location_prefixes: list[str] | None = None
    if key in {"dramteatr", "dramteatr39", "драмтеатр"}:
        patterns = ["%t.me/dramteatr39/%", "%dramteatr39.ru/%"]
        location_prefixes = ["Драматический театр"]
    elif key in {"tretyakov", "tretyakovka", "третьяков", "третьяковка"}:
        patterns = ["%t.me/tretyakovka_kaliningrad/%", "%kaliningrad.tretyakovgallery.ru/%", "%tretyakovgallery.ru/%"]
        location_prefixes = ["Филиал Третьяковской галереи"]
    elif key in {"vorotagallery", "ворота", "ворота_галерея"}:
        patterns = ["%t.me/vorotagallery/%"]
    elif key in {"domkitoboya", "домкитобоя", "дом_китобоя"}:
        patterns = ["%t.me/domkitoboya/%"]
    elif key in {"kaliningradlibrary", "научнаябиблиотека", "научная_библиотека", "научная библиотека"}:
        patterns = ["%t.me/kaliningradlibrary/%", "%vk.ru/wall-30777579_%"]
        location_prefixes = ["Научная библиотека", "Калининградская областная научная библиотека"]
    else:
        # Generic fallback: treat source as Telegram username.
        safe = re.sub(r"[^a-z0-9_]+", "", key)
        if not safe:
            raise AssertionError(f"Неизвестный источник для очистки: {source}")
        patterns = [f"%t.me/{safe}/%"]

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        removed = _purge_events_by_patterns(conn, like_patterns=patterns, location_prefixes=location_prefixes)
    finally:
        conn.close()
    logger.info("✓ Очистка БД по источнику=%s: удалено событий=%d", key, removed)


def _run_tg_monitor(context) -> None:
    step_send_command(context, "/tg")
    baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
    step_click_inline_button(context, "🚀 Запустить мониторинг")

    async def _wait_start():
        import asyncio

        for _ in range(30):
            messages = await context.client.client.get_messages(context.bot_entity, limit=10)
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                txt = str(getattr(msg, "text", "") or "")
                if "starting telegram monitor" in txt.lower():
                    context.last_response = msg
                    context.monitor_started_message_id = msg.id
                    return
            await asyncio.sleep(0.5)
        raise AssertionError("Не найдено новое сообщение 'Starting Telegram Monitor'")

    run_async(context, _wait_start())
    step_wait_long_operation(context, "Telegram Monitor")

def _parse_tg_post_url(post_url: str) -> tuple[str, int]:
    """Return (username, message_id) from https://t.me/<username>/<id>[?...]."""
    url = (post_url or "").strip()
    m = re.search(r"t\.me/([^/]+)/([0-9]+)", url)
    if not m:
        raise AssertionError(f"Некорректная ссылка на пост: {post_url}")
    return m.group(1), int(m.group(2))


def _extract_tg_post_urls_from_message(msg) -> list[str]:
    """Extract canonical t.me/<channel>/<id> links from message text + URL entities."""
    urls: list[str] = []
    text = (getattr(msg, "raw_text", None) or getattr(msg, "text", None) or "").strip()
    if text:
        for m in re.finditer(r"(https?://)?t\.me/[^/\s]+/\d+(?:\?single)?", text):
            raw = m.group(0)
            if not raw.startswith(("http://", "https://")):
                raw = f"https://{raw}"
            urls.append(raw)

    entities = list(getattr(msg, "entities", None) or [])
    if entities and text:
        try:
            from telethon.tl.types import MessageEntityTextUrl, MessageEntityUrl
        except Exception:
            MessageEntityTextUrl = MessageEntityUrl = tuple()  # type: ignore[assignment]
        for ent in entities:
            url = None
            if MessageEntityTextUrl and isinstance(ent, MessageEntityTextUrl):
                url = (getattr(ent, "url", None) or "").strip()
            elif MessageEntityUrl and isinstance(ent, MessageEntityUrl):
                offset = int(getattr(ent, "offset", 0) or 0)
                length = int(getattr(ent, "length", 0) or 0)
                if length > 0:
                    url = text[offset : offset + length].strip()
            if not url:
                continue
            if "t.me/" not in url:
                continue
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            urls.append(url)

    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        try:
            username, message_id = _parse_tg_post_url(raw)
        except Exception:
            continue
        canonical = f"https://t.me/{username}/{int(message_id)}"
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _norm_url_for_compare(url: str | None) -> str:
    """Normalize URLs to make E2E comparisons robust (strip query/signatures)."""
    if not url:
        return ""
    raw = str(url).strip()
    if not raw:
        return ""
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(raw)
        scheme = (parts.scheme or "https").lower()
        netloc = (parts.netloc or "").lower()
        path = parts.path or ""
        if path and path != "/":
            path = path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return raw.split("?", 1)[0].split("#", 1)[0].rstrip("/")


def _score_linked_post_candidate(msg, source_username: str, links: list[str]) -> int:
    text = (getattr(msg, "raw_text", None) or getattr(msg, "text", None) or "").strip()
    text_l = text.lower()
    score = 0

    same_channel_links = [u for u in links if f"/{source_username.lower()}/" in u.lower()]
    if same_channel_links:
        score += 5
    elif links:
        score += 1

    if re.search(r"\b\d{1,2}[:.]\d{2}\b", text_l):
        score += 3
    if re.search(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b", text_l):
        score += 2
    if re.search(r"\b\d{1,2}\s+(январ|феврал|март|апрел|мая|июн|июл|август|сентябр|октябр|ноябр|декабр)", text_l):
        score += 2
    if re.search(r"\b(спектакл|концерт|лекц|выставк|показ|фестиваль|экскурси|читк|мастер-класс|вечеринк)\b", text_l):
        score += 2
    if re.search(r"\b(билет|вход|начало|адрес|ул\.|проспект|набережн|театр|музей|клуб|бар)\b", text_l):
        score += 1
    if re.search(r"\b\d{1,2}[./-]\d{1,2}\s*\|", text_l):
        score += 2

    if len(text) > 120:
        score += 1
    if getattr(msg, "photo", None):
        score += 1

    if re.search(r"\b(розыгрыш|конкурс|giveaway|подписк|реклама|промокод|репост)\b", text_l):
        score -= 7

    msg_dt = getattr(msg, "date", None)
    if msg_dt is not None:
        try:
            if msg_dt.tzinfo is None:
                msg_dt = msg_dt.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - msg_dt).days
            if age_days > 14:
                score -= 5
            elif age_days > 7:
                score -= 2
        except Exception:
            pass

    return score


def _extract_posters_from_tg_results(data: dict, post_url: str) -> list[str]:
    """Return poster URLs (catbox/supabase) for a given post_url from telegram_results.json."""
    if not data:
        return []
    messages = data.get("messages") or []
    target = None
    for msg in messages:
        if msg.get("source_link") == post_url:
            target = msg
            break
    if not target:
        username, message_id = _parse_tg_post_url(post_url)
        for msg in messages:
            if (
                msg.get("source_username") == username
                and int(msg.get("message_id") or 0) == int(message_id)
            ):
                target = msg
                break
    if not target:
        return []
    posters = list(target.get("posters") or [])
    for ev in target.get("events") or []:
        if isinstance(ev, dict) and ev.get("posters"):
            posters.extend(list(ev.get("posters") or []))
    poster_urls: list[str] = []
    for p in posters:
        # Prefer Catbox for rendering checks (Telegraph pages typically use Catbox,
        # Supabase is a fallback and may be disabled in local/E2E environments).
        for url in [p.get("catbox_url"), p.get("supabase_url")]:
            if url and url not in poster_urls:
                poster_urls.append(url)
    return poster_urls


def _find_message_in_tg_results(data: dict, post_url: str) -> dict | None:
    if not data:
        return None
    messages = data.get("messages") or []
    target_norm = _norm_url_for_compare(post_url)
    for msg in messages:
        src = _norm_url_for_compare(msg.get("source_link"))
        if src and src == target_norm:
            return msg
    try:
        username, message_id = _parse_tg_post_url(post_url)
    except Exception:
        return None
    for msg in messages:
        if (
            str(msg.get("source_username") or "").strip() == username
            and int(msg.get("message_id") or 0) == int(message_id)
        ):
            return msg
    return None


def _event_id_from_card_text(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"^id:\\s*(\\d+)\\s*$", text, flags=re.MULTILINE)
    if not m:
        return None
    return int(m.group(1))


def _norm_event_title_key(value: str | None) -> str:
    raw = (value or "").strip().lower().replace("ё", "е")
    raw = re.sub(r"[^\w\s]+", " ", raw, flags=re.U)
    return re.sub(r"\s+", " ", raw).strip()


def _norm_location_key(value: str | None) -> str:
    raw = (value or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", raw).strip()


def _find_row_buttons_for_username(message, username: str):
    """Return (toggle_btn, trust_btn, loc_btn, ticket_btn, festival_btn, reset_btn, delete_btn) for a username on /tg list message."""
    uname = username.lstrip("@").strip()
    if not message or not getattr(message, "buttons", None):
        return None, None, None, None, None, None, None
    toggle_btn = trust_btn = loc_btn = ticket_btn = festival_btn = reset_btn = delete_btn = None
    rows = list(message.buttons or [])
    for idx, row in enumerate(rows):
        texts = [getattr(b, "text", "") for b in row]
        has_toggle_row = any(
            (("Disable" in t or "Enable" in t) and f"@{uname}" in t) for t in texts
        )
        if not has_toggle_row:
            continue
        for b in row:
            t = getattr(b, "text", "")
            if f"@{uname}" in t and ("Disable" in t or "Enable" in t):
                toggle_btn = b
            if "Trust" in t:
                trust_btn = b
        # Next rows are in a fixed order in the UI.
        if idx + 1 < len(rows):
            for b in rows[idx + 1]:
                t = getattr(b, "text", "")
                if t.startswith("📍 "):
                    loc_btn = b
                if t.startswith("🎟 "):
                    ticket_btn = b
        if idx + 2 < len(rows):
            for b in rows[idx + 2]:
                t = getattr(b, "text", "")
                if t.startswith("🎪 "):
                    festival_btn = b
        if idx + 3 < len(rows):
            for b in rows[idx + 3]:
                t = getattr(b, "text", "")
                if t.startswith("♻️") and f"@{uname}" in t:
                    reset_btn = b
        if idx + 4 < len(rows):
            for b in rows[idx + 4]:
                t = getattr(b, "text", "")
                if t.startswith("🗑️") and f"@{uname}" in t:
                    delete_btn = b
        break
    return toggle_btn, trust_btn, loc_btn, ticket_btn, festival_btn, reset_btn, delete_btn


def _extract_tg_sources_pagination(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    m = re.search(r"\\bpage\\s+(\\d+)\\s*/\\s*(\\d+)\\b", text, flags=re.IGNORECASE)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


async def _collect_tg_sources_pages(context, *, max_pages: int = 30) -> list[str]:
    texts: list[str] = []
    msg = context.last_response
    if not msg:
        return texts
    last_text = None
    for _ in range(max_pages):
        text = (msg.text or "")
        texts.append(text)
        cur, total = _extract_tg_sources_pagination(text)
        if cur is not None and total is not None and cur >= total:
            break
        next_btn = find_button(msg, "➡️ Далее")
        if not next_btn:
            break
        await context.client._gaussian_delay(0.4, 1.0)
        await next_btn.click()
        import asyncio
        await asyncio.sleep(2.2)
        try:
            updated = await context.client.client.get_messages(context.bot_entity, ids=[msg.id])
            if updated:
                msg = updated[0]
        except Exception:
            messages = await context.client.client.get_messages(context.bot_entity, limit=1)
            if messages:
                msg = messages[0]
        context.last_response = msg
        if last_text is not None and (msg.text or "") == last_text:
            break
        last_text = (msg.text or "")
    return texts


@then("список источников Telegram в UI использует пагинацию")
def step_tg_sources_has_pagination(context):
    msg = context.last_response
    if not msg:
        raise AssertionError("Нет сообщения со списком источников Telegram")
    cur, total = _extract_tg_sources_pagination(msg.text or "")
    if not total or total < 2:
        raise AssertionError(f"Ожидалась пагинация (>=2 страниц), но найдено: page={cur}/{total}")
    buttons = get_all_buttons(msg)
    if not any("➡️ Далее" in b for b in buttons):
        raise AssertionError(f"Ожидалась кнопка '➡️ Далее' на первой странице. Кнопки: {buttons}")
    logger.info("✓ Пагинация списка источников включена: page=%s/%s", cur, total)


@then("в списке источников Telegram через UI есть источники:")
def step_tg_sources_contains_table(context):
    if not context.table:
        raise AssertionError("Ожидалась таблица | username |")
    expected = []
    for row in context.table:
        expected.append(str(row["username"] or "").strip().lstrip("@"))
    if not expected:
        raise AssertionError("Пустой список ожидаемых источников")

    async def _collect():
        pages = await _collect_tg_sources_pages(context)
        joined = "\n".join(pages)
        found = set(re.findall(r"@([A-Za-z0-9_]{4,64})", joined))
        missing = [u for u in expected if u not in found]
        if missing:
            preview = joined[:1500].replace("\n", " ")
            raise AssertionError(f"Не найдены источники: {missing}. Preview: {preview}")
        logger.info("✓ Найдены все источники (%d) через пагинацию", len(expected))

    run_async(context, _collect())


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
@then('я выбираю контрольный пост с постером в канале "{channel}"')
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
        # Ensure the "force message" table exists even on older prod snapshots.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_source_force_message(
                source_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, message_id)
            )
            """
        )
        # Force-scan a specific post by message_id without expanding the days-back window.
        cur.execute("DELETE FROM telegram_source_force_message WHERE source_id=?", (source_id,))
        cur.execute(
            "INSERT OR REPLACE INTO telegram_source_force_message(source_id, message_id) VALUES(?,?)",
            (source_id, int(message_id)),
        )
        # Ensure re-import is possible even if the post was scanned earlier.
        cur.execute(
            "DELETE FROM telegram_scanned_message WHERE source_id=? AND message_id=?",
            (source_id, int(message_id)),
        )
        # Skip regular history scan for this source in E2E: forced post is enough.
        last_id = 10**12
        cur.execute(
            "UPDATE telegram_source SET enabled=1, last_scanned_message_id=? WHERE id=?",
            (last_id, source_id),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("✓ Подготовлен источник @%s для сканирования с last_scanned_message_id=%s", username, last_id)

@given('я выбираю конкретный пост "{post_url}"')
@when('я выбираю конкретный пост "{post_url}"')
@then('я выбираю конкретный пост "{post_url}"')
def step_pick_specific_post(context, post_url):
    """Pick an exact Telegram post by URL (message_id) without scanning older history."""
    username, message_id = _parse_tg_post_url(post_url)
    context.control_post_username = username
    context.control_post_message_id = int(message_id)
    context.control_post_url = f"https://t.me/{username}/{int(message_id)}"
    logger.info("✓ Контрольный пост задан явно: %s", context.control_post_url)

    conn = sqlite3.connect(_db_path(), timeout=30)
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_source_force_message(
                source_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, message_id)
            )
            """
        )
        cur.execute("DELETE FROM telegram_source_force_message WHERE source_id=?", (source_id,))
        cur.execute(
            "INSERT OR REPLACE INTO telegram_source_force_message(source_id, message_id) VALUES(?,?)",
            (source_id, int(message_id)),
        )
        cur.execute(
            "DELETE FROM telegram_scanned_message WHERE source_id=? AND message_id=?",
            (source_id, int(message_id)),
        )
        last_id = 10**12
        cur.execute(
            "UPDATE telegram_source SET enabled=1, last_scanned_message_id=? WHERE id=?",
            (last_id, source_id),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("✓ Подготовлен источник @%s для сканирования конкретного поста: last_scanned_message_id=%s", username, last_id)


@given('я выбираю конкретные посты "{post_urls_csv}"')
@when('я выбираю конкретные посты "{post_urls_csv}"')
@then('я выбираю конкретные посты "{post_urls_csv}"')
def step_pick_specific_posts(context, post_urls_csv):
    """Pick exact Telegram posts (comma-separated URLs) for one monitor run."""
    raw_urls = [u.strip() for u in str(post_urls_csv or "").split(",") if u.strip()]
    if not raw_urls:
        raise AssertionError("Не переданы ссылки на посты")

    targets_by_username: dict[str, list[int]] = {}
    canonical_urls: list[str] = []
    first_username = ""
    first_message_id = 0
    for idx, post_url in enumerate(raw_urls):
        username, message_id = _parse_tg_post_url(post_url)
        canonical = f"https://t.me/{username}/{int(message_id)}"
        canonical_urls.append(canonical)
        if idx == 0:
            first_username = username
            first_message_id = int(message_id)
        bucket = targets_by_username.setdefault(username, [])
        if int(message_id) not in bucket:
            bucket.append(int(message_id))

    if not first_username or first_message_id <= 0:
        raise AssertionError("Не удалось распарсить ссылки на посты")

    context.control_post_username = first_username
    context.control_post_message_id = int(first_message_id)
    context.control_post_url = canonical_urls[0]
    context.control_post_urls = canonical_urls
    logger.info("✓ Контрольные посты заданы явно: %s", canonical_urls)

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_source_force_message(
                source_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, message_id)
            )
            """
        )

        for username, message_ids in targets_by_username.items():
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

            cur.execute("DELETE FROM telegram_source_force_message WHERE source_id=?", (source_id,))
            for message_id in message_ids:
                cur.execute(
                    "INSERT OR REPLACE INTO telegram_source_force_message(source_id, message_id) VALUES(?,?)",
                    (source_id, int(message_id)),
                )
                cur.execute(
                    "DELETE FROM telegram_scanned_message WHERE source_id=? AND message_id=?",
                    (source_id, int(message_id)),
                )
            last_id = 10**12
            cur.execute(
                "UPDATE telegram_source SET enabled=1, last_scanned_message_id=? WHERE id=?",
                (last_id, source_id),
            )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "✓ Подготовлены источники для массового сканирования постов: %s",
        {k: v for k, v in targets_by_username.items()},
    )


@given('в VK inbox для E2E приоритетные посты "{post_ids_csv}"')
def step_prepare_vk_inbox_priority_posts(context, post_ids_csv):
    """Make VK queue deterministic for E2E by prioritizing a short post list."""
    raw = [p.strip() for p in str(post_ids_csv or "").split(",")]
    default_group_id = int((os.getenv("E2E_VK_GROUP_ID", "30777579") or "30777579").strip() or "30777579")
    # Use a known sentinel so vk_review.pick_next() skips re-parsing event_ts_hint from text.
    # Otherwise scenarios become time-sensitive (posts mentioning past dates get auto-rejected).
    from vk_intake import OCR_PENDING_SENTINEL

    targets: list[tuple[int, int]] = []
    for token in raw:
        if not token:
            continue
        group_id: int
        post_id: int
        m_pair = re.match(r"^\s*(-?\d+)\s*[:_]\s*(\d+)\s*$", token)
        if m_pair:
            group_id = abs(int(m_pair.group(1)))
            post_id = int(m_pair.group(2))
        elif token.isdigit():
            group_id = default_group_id
            post_id = int(token)
        else:
            raise AssertionError(f"Некорректный post_id: '{token}'")
        targets.append((group_id, post_id))
    if not targets:
        raise AssertionError("Не переданы post_id для подготовки VK inbox")

    # Deduplicate while preserving order.
    targets = list(dict.fromkeys(targets))

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        missing: list[tuple[int, int]] = []
        for group_id, post_id in targets:
            cur.execute(
                "SELECT 1 FROM vk_inbox WHERE group_id=? AND post_id=? LIMIT 1",
                (int(group_id), int(post_id)),
            )
            if not cur.fetchone():
                missing.append((int(group_id), int(post_id)))
    finally:
        conn.close()

    async def _fetch_missing_rows(
        wanted: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[int, str]]:
        import main as main_mod

        out: dict[tuple[int, int], tuple[int, str]] = {}
        for group_id, post_id in wanted:
            text = ""
            ts = int(time.time())
            try:
                resp = await main_mod.vk_api("wall.getById", posts=f"-{int(group_id)}_{int(post_id)}")
                raw_resp = resp
                if isinstance(resp, dict) and "response" in resp:
                    raw_resp = resp.get("response")
                items = []
                if isinstance(raw_resp, dict):
                    maybe_items = raw_resp.get("items")
                    if isinstance(maybe_items, list):
                        items = [it for it in maybe_items if isinstance(it, dict)]
                    elif any(k in raw_resp for k in ("text", "attachments", "date")):
                        items = [raw_resp]
                elif isinstance(raw_resp, list):
                    items = [it for it in raw_resp if isinstance(it, dict)]
                if items:
                    item = items[0]
                    text = str(item.get("text") or "").strip()
                    post_ts = item.get("date")
                    if isinstance(post_ts, (int, float)):
                        ts = int(post_ts)
            except Exception as exc:
                logger.warning(
                    "VK E2E seed fallback for -%s_%s (wall.getById failed): %s",
                    group_id,
                    post_id,
                    exc,
                )
            out[(int(group_id), int(post_id))] = (int(ts), text)
        return out

    fetched_missing: dict[tuple[int, int], tuple[int, str]] = {}
    if missing:
        fetched_missing = run_async(context, _fetch_missing_rows(missing))

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        for group_id, post_id in missing:
            ts, text = fetched_missing.get((group_id, post_id), (int(time.time()), ""))
            fallback = f"VK post -{int(group_id)}_{int(post_id)}"
            body = (text or "").strip() or fallback
            has_date = 1 if re.search(r"\b\d{1,2}[./-]\d{1,2}\b", body) else 0
            cur.execute(
                """
                INSERT OR IGNORE INTO vk_inbox(
                    group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    int(group_id),
                    int(post_id),
                    int(ts),
                    body,
                    OCR_PENDING_SENTINEL,
                    0,
                    int(ts),
                    "pending",
                ),
            )

        # Keep only explicit targets in queue to avoid flaky ordering.
        cur.execute(
            """
            UPDATE vk_inbox
            SET status='skipped', locked_by=NULL, locked_at=NULL, review_batch=NULL
            WHERE status IN ('pending', 'importing', 'locked')
            """
        )
        # `vk_review.pick_next()` rejects rows with `event_ts_hint` earlier than
        # "now + VK_REVIEW_REJECT_H hours" (defaults to 2h). For deterministic
        # E2E we pin targets slightly in the future so they are always pickable,
        # even if the underlying snapshot contains past `event_ts_hint`.
        base_ts = int(time.time()) + 3 * 3600
        for offset, (group_id, post_id) in enumerate(targets):
            ts = base_ts - offset
            cur.execute(
                """
                UPDATE vk_inbox
                SET status='pending',
                    locked_by=NULL,
                    locked_at=NULL,
                    review_batch=NULL,
                    imported_event_id=NULL,
                    date=?,
                    matched_kw=?,
                    has_date=0,
                    event_ts_hint=?
                WHERE group_id=? AND post_id=?
                """,
                (int(ts), OCR_PENDING_SENTINEL, int(ts), int(group_id), int(post_id)),
            )

        where = " OR ".join(["(group_id=? AND post_id=?)" for _ in targets])
        params: list[int] = []
        for group_id, post_id in targets:
            params.extend([int(group_id), int(post_id)])
        cur.execute(
            f"SELECT group_id, post_id, status FROM vk_inbox WHERE {where}",
            tuple(params),
        )
        rows = cur.fetchall()
        conn.commit()
    finally:
        conn.close()

    found_pairs = {(int(r[0]), int(r[1])) for r in rows}
    missing_after = [f"-{gid}_{pid}" for gid, pid in targets if (gid, pid) not in found_pairs]
    if missing_after:
        raise AssertionError(f"Не удалось подготовить посты в vk_inbox: {missing_after}")

    context.vk_priority_targets = [(int(g), int(p)) for g, p in targets]
    logger.info(
        "✓ VK inbox подготовлен для E2E: targets=%s rows=%s",
        context.vk_priority_targets,
        len(rows),
    )


@given("в VK inbox для E2E выбраны все активные посты очереди")
def step_prepare_vk_inbox_all_active_targets(context):
    """Prepare full VK queue targets for E2E, including skipped rows."""
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        # Full queue run should include previously skipped/locked rows too.
        cur.execute(
            """
            UPDATE vk_inbox
            SET status='pending',
                locked_by=NULL,
                locked_at=NULL,
                review_batch=NULL
            WHERE status IN ('skipped', 'locked', 'importing')
            """
        )
        cur.execute(
            """
            SELECT id, group_id, post_id
            FROM vk_inbox
            WHERE status IN ('pending', 'locked')
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        cap_raw = (os.getenv("E2E_VK_ALL_ACTIVE_TARGETS_LIMIT") or "").strip()
        cap = int(cap_raw) if cap_raw.isdigit() else 0
        if cap > 0:
            rows = list(rows)[:cap]
            selected_ids = [int(r[0]) for r in rows]
            placeholders = ",".join("?" for _ in selected_ids) or "NULL"
            # Exclude the rest from this run to keep `vk_auto_import --limit=N`
            # deterministic (otherwise remaining pending rows would prevent
            # the "targets processed" wait condition from ever becoming true).
            cur.execute(
                f"""
                UPDATE vk_inbox
                SET status='skipped',
                    locked_by=NULL,
                    locked_at=NULL,
                    review_batch=NULL
                WHERE status IN ('pending', 'locked', 'importing')
                  AND id NOT IN ({placeholders})
                """,
                tuple(selected_ids),
            )
            # Keep selected rows pending/unlocked.
            cur.execute(
                f"""
                UPDATE vk_inbox
                SET status='pending',
                    locked_by=NULL,
                    locked_at=NULL,
                    review_batch=NULL
                WHERE id IN ({placeholders})
                """,
                tuple(selected_ids),
            )
        conn.commit()
    finally:
        conn.close()

    targets = [(int(r[1]), int(r[2])) for r in rows]
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise AssertionError("В vk_inbox нет активных постов (pending/locked) после подготовки очереди")

    context.vk_priority_targets = targets
    logger.info(
        "✓ VK inbox targets=all-active count=%s",
        len(context.vk_priority_targets),
    )


@given('в VK inbox для E2E выбраны первые "{n}" активных постов очереди')
def step_prepare_vk_inbox_first_n_active_targets(context, n):
    """Prepare VK queue targets for E2E but cap to first N pending/locked rows."""
    raw_n = str(n or "").strip()
    if not raw_n.isdigit():
        raise AssertionError(f"n должно быть числом, получено: {n!r}")
    cap = int(raw_n)
    if cap <= 0:
        raise AssertionError(f"n должно быть >0, получено: {cap}")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE vk_inbox
            SET status='pending',
                locked_by=NULL,
                locked_at=NULL,
                review_batch=NULL
            WHERE status IN ('skipped', 'locked', 'importing')
            """
        )
        cur.execute(
            """
            SELECT id, group_id, post_id
            FROM vk_inbox
            WHERE status IN ('pending', 'locked')
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        if not rows:
            raise AssertionError("В vk_inbox нет активных постов (pending/locked) после подготовки очереди")

        rows = list(rows)[:cap]
        selected_ids = [int(r[0]) for r in rows]
        placeholders = ",".join("?" for _ in selected_ids) or "NULL"
        # Exclude the rest from this run to keep /vk_auto_import deterministic.
        cur.execute(
            f"""
            UPDATE vk_inbox
            SET status='skipped',
                locked_by=NULL,
                locked_at=NULL,
                review_batch=NULL
            WHERE status IN ('pending', 'locked', 'importing')
              AND id NOT IN ({placeholders})
            """,
            tuple(selected_ids),
        )
        cur.execute(
            f"""
            UPDATE vk_inbox
            SET status='pending',
                locked_by=NULL,
                locked_at=NULL,
                review_batch=NULL
            WHERE id IN ({placeholders})
            """,
            tuple(selected_ids),
        )
        conn.commit()
    finally:
        conn.close()

    targets = [(int(r[1]), int(r[2])) for r in rows]
    targets = list(dict.fromkeys(targets))
    context.vk_priority_targets = targets
    logger.info("✓ VK inbox targets=first-%s count=%s", cap, len(targets))


@then("в сводке VK очереди показано количество постов")
def step_vk_queue_summary_has_counts(context):
    """Assert /vk queue summary contains status counters and compute total."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    text = msg.text or ""
    if not text.strip():
        raise AssertionError("Сводка VK очереди пустая")

    counts: dict[str, int] = {}
    for key, value in re.findall(
        r"(?im)^\s*(pending|locked|skipped|imported|rejected)\s*:\s*(\d+)\s*$",
        text,
    ):
        counts[str(key).lower()] = int(value)

    required_keys = ("pending", "locked", "skipped", "imported", "rejected")
    missing = [k for k in required_keys if k not in counts]
    if missing:
        raise AssertionError(
            "Не удалось распознать полную сводку VK очереди. "
            f"Отсутствуют: {missing}. Текст:\n{text}"
        )

    total = sum(counts[k] for k in required_keys)
    context.vk_queue_summary_counts = counts
    context.vk_queue_summary_total = int(total)
    logger.info("✓ VK queue summary: total=%s counts=%s", total, counts)


@given('я выбираю в канале "{channel}" пост со ссылкой на другой telegram пост')
def step_pick_linked_post_from_channel(context, channel):
    username = str(channel or "").lstrip("@").strip()
    if not username:
        raise AssertionError("Пустой channel для linked-post сценария")

    forced_primary = (os.getenv("E2E_LINKED_PRIMARY_POST_URL") or "").strip()
    forced_linked = (os.getenv("E2E_LINKED_POST_URL") or "").strip()
    if forced_primary:
        logger.info(
            "linked-post: using forced URLs primary=%s linked=%s",
            forced_primary,
            forced_linked or "auto",
        )
        primary_url = forced_primary
        if forced_linked:
            context.linked_post_url = forced_linked
            step_pick_specific_post(context, primary_url)
            return

    async def _pick():
        entity = await context.client.client.get_entity(username)
        scan_limit = int(os.getenv("E2E_LINKED_POST_SCAN_LIMIT", "220"))
        messages = await context.client.client.get_messages(entity, limit=scan_limit)
        best_pair: tuple[str, str] | None = None
        best_score = -10**9
        for msg in messages:
            primary_url = f"https://t.me/{username}/{int(msg.id)}"
            links = _extract_tg_post_urls_from_message(msg)
            links = [u for u in links if _norm_url_for_compare(u) != _norm_url_for_compare(primary_url)]
            if not links:
                continue
            same_channel: list[str] = []
            for u in links:
                try:
                    u_name, _u_id = _parse_tg_post_url(u)
                except Exception:
                    continue
                if u_name == username:
                    same_channel.append(u)
            linked_url = same_channel[0] if same_channel else links[0]
            score = _score_linked_post_candidate(msg, username, links)
            if score > best_score:
                best_score = score
                best_pair = (primary_url, linked_url)
            if best_score >= 8:
                break
        if best_pair:
            logger.info("linked-post: selected score=%s", best_score)
        return best_pair

    picked = run_async(context, _pick())
    if not picked:
        raise AssertionError(
            f"В @{username} не найден недавний пост со ссылкой на другой telegram пост "
            "(увеличьте E2E_LINKED_POST_SCAN_LIMIT или выберите пост вручную)."
        )
    primary_url, linked_url = picked
    context.linked_post_url = linked_url
    logger.info("✓ Найден linked-post кейс: primary=%s linked=%s", primary_url, linked_url)
    step_pick_specific_post(context, primary_url)

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
@then('события драмтеатра уже загружены через "/parse"')
def step_dramteatr_events_loaded(context):
    """Verify dramteatr events (site parser) exist in DB."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title
            FROM event
            WHERE location_name = ?
              AND source_post_url LIKE ?
            ORDER BY date DESC
            LIMIT 1
            """,
            ("Драматический театр", "https://dramteatr39.ru/%"),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError("В БД нет событий драмтеатра с источником dramteatr39.ru")
    context.dramteatr_event_id = int(row[0])
    logger.info("✓ Найдено событие драмтеатра (id=%s title=%s)", row[0], row[1])


@then("существует смерженное событие драмтеатра с источниками Telegram и site")
def step_dramteatr_has_merged_sources(context):
    """Find at least one event that has both dramteatr39 Telegram source and dramteatr site source."""
    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        _ensure_event_source_table(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.title, e.telegraph_url,
                   (SELECT COUNT(*) FROM event_source es WHERE es.event_id = e.id) AS sources_total
            FROM event e
            WHERE EXISTS (
                SELECT 1 FROM event_source es
                WHERE es.event_id = e.id AND es.source_url LIKE '%t.me/dramteatr39/%'
            )
              AND EXISTS (
                SELECT 1 FROM event_source es
                WHERE es.event_id = e.id AND es.source_url LIKE '%dramteatr39%ru%'
            )
            ORDER BY e.id DESC
            LIMIT 5
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        raise AssertionError(
            "Не найдено смерженных событий драмтеатра (ожидали источники t.me/dramteatr39 + dramteatr39.ru)"
        )
    context.merged_dramteatr_events = [
        {"id": int(r["id"]), "title": r["title"], "telegraph_url": r["telegraph_url"], "sources_total": int(r["sources_total"] or 0)}
        for r in rows
    ]
    logger.info(
        "✓ Найдено смерженных событий драмтеатра: %s",
        [(r["id"], (r["telegraph_url"] or "")) for r in context.merged_dramteatr_events],
    )


@then("страница Telegraph смерженного события драмтеатра содержит счётчик источников")
def step_dramteatr_merged_event_has_sources_footer(context):
    import time
    import aiohttp

    items = getattr(context, "merged_dramteatr_events", None) or []
    if not items:
        raise AssertionError("Нет списка смерженных событий драмтеатра (сначала выполните поиск)")
    item = items[0]
    event_id = int(item.get("id") or 0)
    sources_total = int(item.get("sources_total") or 0)
    if sources_total < 2:
        raise AssertionError(f"Ожидали >=2 источника у смерженного события, получили {sources_total} (event_id={event_id})")

    url = (item.get("telegraph_url") or "").strip()
    if not url:
        wait_sec = int(os.getenv("E2E_TELEGRAPH_WAIT_SEC", "240"))
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            conn = sqlite3.connect(_db_path(), timeout=30)
            try:
                cur = conn.cursor()
                cur.execute("SELECT telegraph_url FROM event WHERE id = ?", (event_id,))
                row = cur.fetchone()
            finally:
                conn.close()
            candidate = (row[0] if row else "") if row is not None else ""
            candidate = (candidate or "").strip()
            if candidate.startswith("https://telegra.ph/"):
                url = candidate
                break
            time.sleep(3)
    if not url:
        raise AssertionError(f"У смерженного события нет telegraph_url (event_id={event_id})")

    async def _fetch():
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                return await resp.text()

    # Telegraph rebuild runs via the bot worker queue; give it a bit of time to catch up.
    wait_sec = int(os.getenv("E2E_TELEGRAPH_FOOTER_WAIT_SEC", "480"))
    deadline = time.time() + wait_sec
    last_html = ""
    while time.time() < deadline:
        last_html = run_async(context, _fetch())
        m = re.search(r"Источников:\s*(\d+)", last_html)
        if m:
            n = int(m.group(1))
            if n >= 2:
                logger.info("✓ Telegraph footer: Источников=%s (%s)", n, url)
                return
        time.sleep(12)
    if not re.search(r"Источников:\s*(\d+)", last_html or ""):
        raise AssertionError(f"На странице {url} не найден счётчик 'Источников: N'")
    raise AssertionError(f"Счётчик 'Источников' не достиг 2+ за {wait_sec}s: {url}")


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
@then('очищены отметки мониторинга для "{username}"')
def step_reset_monitor_marks(context, username):
    """Reset telegram monitoring marks via the real UI (/tg -> list -> reset)."""
    uname = username.lstrip("@").strip()
    if not uname:
        raise AssertionError("Пустой username для сброса отметок")
    step_send_command(context, "/tg")
    step_click_inline_button(context, "📋 Список источников")
    step_click_inline_button(context, f"♻️ Сбросить отметки @{uname}")
    step_wait_for_message_text(context, "сброшены")
    logger.info("✓ Сброшены отметки мониторинга для @%s (UI)", uname)


@given('очищены отметки мониторинга для Telegram источников "{usernames_csv}"')
def step_reset_monitor_marks_bulk_db(context, usernames_csv):
    """Reset telegram monitoring marks in DB (fast path, without UI)."""
    raw = [u.strip() for u in str(usernames_csv or "").split(",") if u.strip()]
    usernames = [u.lstrip("@").strip() for u in raw if u.lstrip("@").strip()]
    if not usernames:
        raise AssertionError("Не переданы usernames для сброса отметок мониторинга")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(usernames))
        cur.execute(
            f"SELECT id, username FROM telegram_source WHERE username IN ({placeholders})",
            tuple(usernames),
        )
        existing = {str(r[1] or ""): int(r[0]) for r in cur.fetchall() or []}
        for uname in usernames:
            if uname not in existing:
                cur.execute(
                    "INSERT INTO telegram_source(username, enabled, last_scanned_message_id) VALUES(?,1,NULL)",
                    (uname,),
                )
                existing[uname] = int(cur.lastrowid)
        source_ids = [int(existing[u]) for u in usernames if u in existing]
        if source_ids:
            ph = ",".join(["?"] * len(source_ids))
            # Delete per-message marks so Kaggle run can re-process history.
            cur.execute(
                f"DELETE FROM telegram_scanned_message WHERE source_id IN ({ph})",
                tuple(source_ids),
            )
            cur.execute(
                f"UPDATE telegram_source SET enabled=1, last_scanned_message_id=NULL WHERE id IN ({ph})",
                tuple(source_ids),
            )
        conn.commit()
    finally:
        conn.close()
    logger.info("✓ Сброшены отметки мониторинга (DB) для: %s", usernames)


@then("существует событие с источниками VK и Telegram")
def step_event_has_vk_and_telegram_sources(context):
    """Find at least one event that has both vk and telegram sources."""
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # event_source can be missing in very old snapshots; ensure it exists for queries.
        try:
            _ensure_event_source_table(conn)
        except Exception:
            pass
        cur.execute(
            """
            SELECT
                event_id,
                SUM(CASE WHEN source_type='vk' THEN 1 ELSE 0 END) AS vk_cnt,
                SUM(CASE WHEN source_type='telegram' THEN 1 ELSE 0 END) AS tg_cnt
            FROM event_source
            GROUP BY event_id
            HAVING vk_cnt > 0 AND tg_cnt > 0
            ORDER BY event_id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError("Не найдено ни одного события с источниками VK и Telegram")
    event_id = int(row["event_id"])
    context.vk_tg_merged_event_id = event_id
    logger.info("✓ Найдено событие с источниками VK+Telegram: event_id=%s", event_id)


@then("существует событие с источниками VK, Telegram и парсером")
def step_event_has_vk_tg_and_parser_sources(context):
    """Find at least one event that has VK + Telegram + parser:<...> sources."""
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            _ensure_event_source_table(conn)
        except Exception:
            pass
        cur.execute(
            """
            SELECT
                event_id,
                SUM(CASE WHEN (source_url LIKE '%vk.com/%' OR source_url LIKE '%vk.ru/%') THEN 1 ELSE 0 END) AS vk_cnt,
                SUM(CASE WHEN source_url LIKE '%t.me/%' THEN 1 ELSE 0 END) AS tg_cnt,
                SUM(CASE WHEN source_type LIKE 'parser:%' THEN 1 ELSE 0 END) AS parser_cnt
            FROM event_source
            GROUP BY event_id
            HAVING vk_cnt > 0 AND tg_cnt > 0 AND parser_cnt > 0
            ORDER BY event_id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError("Не найдено ни одного события с источниками VK + Telegram + parser")
    event_id = int(row["event_id"])
    context.vk_tg_parser_event_id = event_id
    logger.info("✓ Найдено событие с источниками VK+Telegram+parser: event_id=%s", event_id)


@when("я запрашиваю /log для события с источниками VK и Telegram")
def step_request_log_for_vk_tg_merged_event(context):
    eid = int(getattr(context, "vk_tg_merged_event_id", 0) or 0)
    if eid <= 0:
        raise AssertionError("Нет event_id для VK+Telegram события (сначала найдите его в БД)")
    step_send_command(context, f"/log {eid}")


@when("я запрашиваю /log для события с источниками VK, Telegram и парсером")
def step_request_log_for_vk_tg_parser_merged_event(context):
    eid = int(getattr(context, "vk_tg_parser_event_id", 0) or 0)
    if eid <= 0:
        raise AssertionError("Нет event_id для VK+Telegram+parser события (сначала найдите его в БД)")
    step_send_command(context, f"/log {eid}")


@then("в логе источников есть источники VK и Telegram")
def step_source_log_has_vk_and_tg(context):
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    text = msg.text or ""
    if not text.strip():
        raise AssertionError("Лог источников пуст")
    low = text.lower()
    has_vk = ("vk.com/" in low) or ("vk.ru/" in low)
    has_tg = ("t.me/" in low)
    if not has_vk or not has_tg:
        raise AssertionError(
            "В логе источников не найдены оба типа источников (VK и Telegram).\n"
            f"has_vk={has_vk} has_tg={has_tg}\n"
            f"Текст:\n{text}"
        )
    logger.info("✓ Лог источников содержит VK и Telegram ссылки")

@given("я очищаю список источников Telegram через UI")
def step_clear_all_sources_ui(context):
    """Delete all Telegram sources using the real UI buttons."""
    if (os.getenv("E2E_ALLOW_TG_SOURCES_CLEAR") or "").strip() != "1":
        raise AssertionError(
            "Шаг 'очистить список источников Telegram' отключён по умолчанию, "
            "чтобы не сносить вручную настроенные источники. "
            "Для явного разрешения установите E2E_ALLOW_TG_SOURCES_CLEAR=1 "
            "(и запускайте на изолированной копии DB_PATH)."
        )
    if not (os.getenv("E2E_DB_BASE_PATH") or "").strip():
        raise AssertionError(
            "Отказываюсь очищать Telegram источники: DB не выглядит изолированной копией "
            "(нет E2E_DB_BASE_PATH). Запускайте behave на snapshot с E2E_DB_ISOLATE=1."
        )
    step_send_command(context, "/tg")
    step_click_inline_button(context, "📋 Список источников")

    async def _clear():
        import asyncio

        # Keep clicking delete buttons until none remain.
        for _ in range(20):
            msg = context.last_response
            buttons = get_all_buttons(msg)
            delete_buttons = [b for b in buttons if b.startswith("🗑️ Удалить @")]
            if not delete_buttons:
                return
            # Click first delete button
            btn_text = delete_buttons[0]
            btn = find_button(msg, btn_text)
            if not btn:
                return
            await context.client._gaussian_delay(0.5, 1.2)
            await btn.click()
            # Allow bot to send confirmation + refreshed list
            await asyncio.sleep(2.5)
            messages = await context.client.client.get_messages(context.bot_entity, limit=2)
            if messages:
                context.last_response = messages[0]
        raise AssertionError("Не удалось очистить список источников Telegram за 20 попыток")

    run_async(context, _clear())
    logger.info("✓ Список Telegram источников очищен (UI)")


@given('trust для источника "{username}" установлен в "{level}"')
@then('trust для источника "{username}" установлен в "{level}"')
def step_set_source_trust_ui(context, username, level):
    """Cycle trust button until the desired trust_level is shown in the list."""
    uname = username.lstrip("@").strip()
    desired = (level or "").strip().lower()
    if desired not in {"low", "medium", "high"}:
        raise AssertionError("trust должен быть low|medium|high")

    for _ in range(6):
        step_send_command(context, "/tg")
        step_click_inline_button(context, "📋 Список источников")
        msg = context.last_response
        text = (msg.text or "")
        if f"@{uname} (trust={desired})" in text:
            logger.info("✓ trust уже установлен: @%s trust=%s", uname, desired)
            return
        _toggle, trust_btn, _loc, _ticket, _festival, _reset, _delete = _find_row_buttons_for_username(msg, uname)
        if not trust_btn:
            raise AssertionError(
                f"Не найдена кнопка Trust для @{uname}. Кнопки: {get_all_buttons(msg)}"
            )
        async def _click():
            import asyncio
            await context.client._gaussian_delay(0.4, 1.0)
            await trust_btn.click()
            await asyncio.sleep(2.0)
            messages = await context.client.client.get_messages(context.bot_entity, limit=1)
            if messages:
                context.last_response = messages[0]
        run_async(context, _click())
    raise AssertionError(f"Не удалось выставить trust={desired} для @{uname} через UI")
    logger.info("✓ trust=%s установлен для @%s (UI)", desired, uname)


@given('default_location для источника "{username}" установлен в "{location}"')
@then('default_location для источника "{username}" установлен в "{location}"')
def step_set_source_location_ui(context, username, location):
    uname = username.lstrip("@").strip()
    loc = (location or "").strip()
    if not uname:
        raise AssertionError("Пустой username для default_location")
    if not loc:
        raise AssertionError("Пустой default_location")

    step_send_command(context, "/tg")
    step_click_inline_button(context, "📋 Список источников")
    msg = context.last_response
    _toggle, _trust, loc_btn, _ticket, _festival, _reset, _delete = _find_row_buttons_for_username(msg, uname)
    if not loc_btn:
        raise AssertionError(f"Не найдена кнопка '📍 Локация' для @{uname}. Кнопки: {get_all_buttons(msg)}")

    async def _set():
        import asyncio
        await context.client._gaussian_delay(0.4, 1.0)
        await loc_btn.click()
        await asyncio.sleep(1.5)
        await context.client.human_send_and_wait(context.bot_entity, loc, timeout=30)
        await asyncio.sleep(2.0)
        messages = await context.client.client.get_messages(context.bot_entity, limit=1)
        if messages:
            context.last_response = messages[0]

    run_async(context, _set())
    logger.info("✓ default_location установлен для @%s: %s (UI)", uname, loc)


@given('festival_series для источника "{username}" установлен в "{series}"')
@then('festival_series для источника "{username}" установлен в "{series}"')
def step_set_source_festival_series_ui(context, username, series):
    uname = username.lstrip("@").strip()
    value = (series or "").strip()
    if not uname:
        raise AssertionError("Пустой username для festival_series")
    if not value:
        raise AssertionError("Пустой festival_series (используйте '-' для очистки)")

    step_send_command(context, "/tg")
    step_click_inline_button(context, "📋 Список источников")
    msg = context.last_response
    _toggle, _trust, _loc, _ticket, festival_btn, _reset, _delete = _find_row_buttons_for_username(msg, uname)
    if not festival_btn:
        raise AssertionError(f"Не найдена кнопка '🎪 Фестиваль' для @{uname}. Кнопки: {get_all_buttons(msg)}")

    async def _set():
        import asyncio
        await context.client._gaussian_delay(0.4, 1.0)
        await festival_btn.click()
        await asyncio.sleep(1.5)
        await context.client.human_send_and_wait(context.bot_entity, value, timeout=30)
        await asyncio.sleep(2.0)
        messages = await context.client.client.get_messages(context.bot_entity, limit=1)
        if messages:
            context.last_response = messages[0]

    run_async(context, _set())
    logger.info("✓ festival_series установлен для @%s: %s (UI)", uname, value)


@given("в UI /tg настроены источники Telegram:")
def step_configure_tg_sources_via_ui(context):
    """Ensure a set of Telegram sources exists and has desired trust/defaults/festival settings (UI-only)."""
    if not context.table:
        raise AssertionError("Ожидалась таблица источников (username/trust_level/default_location/festival_series)")

    def _cell(row, key: str) -> str:
        try:
            if hasattr(row, "get"):
                return (row.get(key) or "").strip()
            return (row[key] or "").strip()
        except Exception:
            return ""

    def _ensure_exists(username: str) -> None:
        step_send_command(context, "/tg")
        step_click_inline_button(context, "📋 Список источников")
        msg = context.last_response
        text = (msg.text or "")
        if f"@{username}" in text:
            return
        step_send_command(context, "/tg")
        step_click_inline_button(context, "➕ Добавить источник")
        step_send_message(context, f"@{username}")
        added_text = (context.last_response.text or "")
        if f"@{username}" not in added_text:
            raise AssertionError(f"Не удалось добавить источник @{username} через UI. Ответ:\n{added_text}")

    sources: list[dict] = []
    for row in context.table:
        username = _cell(row, "username").lstrip("@")
        if not username:
            raise AssertionError("Пустой username в таблице источников")
        trust = (_cell(row, "trust_level") or _cell(row, "trust") or "").lower()
        default_location = _cell(row, "default_location") or _cell(row, "location")
        festival_series = _cell(row, "festival_series") or _cell(row, "festival") or _cell(row, "series")
        sources.append(
            {
                "username": username,
                "trust_level": trust,
                "default_location": default_location,
                "festival_series": festival_series,
            }
        )

    for src in sources:
        username = src["username"]
        _ensure_exists(username)
        if src["trust_level"]:
            step_set_source_trust_ui(context, username, src["trust_level"])
        if src["default_location"]:
            step_set_source_location_ui(context, username, src["default_location"])
        if src["festival_series"]:
            step_set_source_festival_series_ui(context, username, src["festival_series"])

    logger.info("✓ /tg sources configured via UI: %s", [s["username"] for s in sources])


@then("в UI /tg список источников содержит:")
def step_assert_tg_sources_in_ui_list(context):
    if not context.table:
        raise AssertionError("Ожидалась таблица источников для проверки списка /tg")
    step_send_command(context, "/tg")
    step_click_inline_button(context, "📋 Список источников")
    msg = context.last_response
    text = (msg.text or "")
    if not text:
        raise AssertionError("Пустой ответ списка источников /tg")

    def _cell(row, key: str) -> str:
        try:
            if hasattr(row, "get"):
                return (row.get(key) or "").strip()
            return (row[key] or "").strip()
        except Exception:
            return ""

    missing: list[str] = []
    for row in context.table:
        username = _cell(row, "username").lstrip("@")
        if not username:
            continue
        trust = (_cell(row, "trust_level") or _cell(row, "trust")).lower()
        series = _cell(row, "festival_series") or _cell(row, "festival") or _cell(row, "series")

        if trust and f"@{username} (trust={trust})" not in text:
            missing.append(f"@{username} trust={trust}")
            continue
        if not trust and f"@{username}" not in text:
            missing.append(f"@{username}")
            continue
        if series:
            pattern = rf"@{re.escape(username)}\\s+\\(trust=\\w+\\)[\\s\\S]*?\\n\\s*↳\\s+festival:\\s+{re.escape(series)}\\b"
            if not re.search(pattern, text):
                missing.append(f"@{username} festival={series}")

    if missing:
        raise AssertionError("В /tg списке источников отсутствуют/не совпадают настройки: " + ", ".join(missing))
    logger.info("✓ /tg UI list содержит ожидаемые источники и настройки (count=%s)", len(context.table))


# =============================================================================
# Когда (When) - Actions
# =============================================================================

@when('я отправляю команду "{command}"')
@then('я отправляю команду "{command}"')
def step_send_command(context, command):
    """Send command to bot using human-like behavior."""
    async def _send():
        cmd = (command or "").strip()
        timeout = float(os.getenv("E2E_COMMAND_TIMEOUT_SEC", "60"))
        if cmd.lower() == "/start":
            timeout = float(os.getenv("E2E_START_TIMEOUT_SEC", str(timeout)))
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = await context.client.human_send_and_wait(
                    context.bot_entity,
                    command,
                    timeout=timeout,
                )
                context.last_response = response
                logger.info(f"→ Отправлено: {command}")
                if response and response.text:
                    preview = response.text[:100].replace('\n', ' ')
                    logger.info(f"← Ответ: {preview}...")
                return response
            except TimeoutError as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning("Команда '%s' не получила ответ вовремя, ретрай...", command)
                    await context.client._gaussian_delay(0.4, 1.0)
                    continue
                raise
        if last_exc:
            raise last_exc

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

@when('я отправляю сообщение с текстом "{text}"')
@then('я отправляю сообщение с текстом "{text}"')
def step_send_message_with_text_alias(context, text):
    """Alias for older/newer feature phrasing."""
    step_send_message(context, text)


@when('я отправляю фото "{rel_path}" с подписью:')
@then('я отправляю фото "{rel_path}" с подписью:')
def step_send_photo_with_caption(context, rel_path):
    """Send a local image file to the bot with a caption (used for add-event UI flows)."""
    caption = (getattr(context, "text", None) or "").strip()
    if not caption:
        raise AssertionError("Пустая подпись для фото (ожидали multiline текст после шага)")

    path = Path(rel_path)
    if not path.is_absolute():
        # Behave is usually started from repo root, but keep this robust.
        if not path.exists():
            repo_root = Path(__file__).resolve().parents[4]
            path = (repo_root / path).resolve()
    if not path.exists() or not path.is_file():
        raise AssertionError(f"Файл изображения не найден: {rel_path} (resolved={path})")

    async def _send():
        # Human-like pause before upload.
        await context.client._gaussian_delay(0.7, 1.8)
        entity = await context.client.client.get_entity(context.bot_entity)
        sent = await context.client.client.send_file(entity, file=str(path), caption=caption)
        # Store our outgoing message id for better diagnostics (bot uses it as source marker).
        context.last_user_message_id = int(getattr(sent, "id", 0) or 0)
        logger.info("→ Отправлено фото: %s (bytes=%s) msg_id=%s", str(path), path.stat().st_size, context.last_user_message_id)

    run_async(context, _send())


@when('я нажимаю инлайн-кнопку "{btn_text}"')
@then('я нажимаю инлайн-кнопку "{btn_text}"')
def step_click_inline_button(context, btn_text):
    """Click inline button by text."""
    async def _click():
        msg = context.last_response
        btn = find_button(msg, btn_text)

        if not btn:
            messages = await context.client.client.get_messages(context.bot_entity, limit=12)
            for candidate in messages:
                btn = find_button(candidate, btn_text)
                if btn:
                    msg = candidate
                    context.last_response = candidate
                    logger.info(
                        "↺ Кнопка %r найдена в другом сообщении бота id=%s",
                        btn_text,
                        getattr(candidate, "id", None),
                    )
                    break

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

@when('я нажимаю кнопку "{btn_text}"')
@then('я нажимаю кнопку "{btn_text}"')
def step_click_any_button(context, btn_text):
    """Click a UI button (inline preferred; reply keyboard falls back to sending text)."""
    msg = context.last_response
    available = get_all_buttons(msg)
    if available and btn_text not in available:
        raise AssertionError(f"Кнопка '{btn_text}' не найдена. Доступные: {available}")
    btn = find_button(msg, btn_text)
    # Inline buttons can be clicked; reply-keyboard buttons should be sent as text.
    if btn and hasattr(btn, "click"):
        step_click_inline_button(context, btn_text)
        return
    step_send_message(context, btn_text)


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
    context.last_event_id = int(event_id)
    btn = find_button(msg, f"✎ {event_id}")
    if not btn:
        # /events can fall back to compact mode (no per-event buttons) when the message
        # is too long for Telegram. Use /edit <id> as an admin shortcut.
        step_send_command(context, f"/edit {event_id}")
        card = context.last_response
        card_text = (card.text or "") if card else ""
        if f"id: {event_id}" not in card_text:
            raise AssertionError(
                f"Кнопка редактирования ✎ {event_id} не найдена и /edit {event_id} не открыл карточку"
            )
        logger.info("✓ Открыта карточка события через /edit: id=%s title=%s", event_id, target_title)
        return

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
        event_ids: list[int] = []

        cur.execute(
            "SELECT event_id FROM event_source WHERE source_url = ? ORDER BY imported_at DESC",
            (post_url,),
        )
        event_ids = [int(r["event_id"]) if isinstance(r, sqlite3.Row) else int(r[0]) for r in cur.fetchall()]
        if not event_ids:
            cur.execute(
                """
                SELECT event_id FROM event_source
                WHERE source_chat_username = ? AND source_message_id = ?
                ORDER BY imported_at DESC
                """,
                (username, int(message_id)),
            )
            event_ids = [int(r["event_id"]) if isinstance(r, sqlite3.Row) else int(r[0]) for r in cur.fetchall()]
        if not event_ids:
            cur.execute(
                "SELECT event_id FROM event_source WHERE source_url LIKE ? ORDER BY imported_at DESC",
                (f"%t.me/{username}/{int(message_id)}%",),
            )
            event_ids = [int(r["event_id"]) if isinstance(r, sqlite3.Row) else int(r[0]) for r in cur.fetchall()]
        if not event_ids:
            raise AssertionError(f"Не найден event_source для контрольного поста: {post_url}")

        # For schedule-like posts with multiple events, pick the nearest future event.
        # This keeps telegram-first checks stable and avoids coupling to insert order.
        from datetime import date as _date

        today = _date.today()
        picked = None
        for eid in event_ids:
            cur.execute("SELECT date FROM event WHERE id = ?", (eid,))
            date_row = cur.fetchone()
            if not date_row:
                continue
            raw = str((date_row["date"] if isinstance(date_row, sqlite3.Row) else date_row[0]) or "")
            iso = raw.split("..", 1)[0].strip()
            parsed = None
            try:
                parsed = _date.fromisoformat(iso)
            except Exception:
                parsed = None

            if parsed is not None and parsed >= today:
                key = (0, (parsed - today).days, eid)
            elif parsed is not None:
                key = (1, abs((today - parsed).days), eid)
            else:
                key = (2, 10**9, eid)

            if picked is None or key < picked[0]:
                picked = (key, eid, raw)

        if picked is None:
            # Fallback to the most recently imported event id.
            event_id = int(event_ids[0])
            cur.execute("SELECT date FROM event WHERE id = ?", (event_id,))
            date_row = cur.fetchone()
            if not date_row:
                raise AssertionError(f"Событие id={event_id} не найдено в таблице event")
            raw_date = str((date_row["date"] if isinstance(date_row, sqlite3.Row) else date_row[0]) or "")
        else:
            _, event_id, raw_date = picked

        event_date = raw_date.split("..", 1)[0].strip()
    finally:
        conn.close()

    context.control_event_id = event_id
    context.control_event_date = event_date
    # Keep generic fallback used by many checks when card text is truncated
    # and no longer contains explicit "id: N" marker.
    context.last_event_id = event_id
    logger.info("✓ Контрольное событие: id=%s date=%s url=%s", event_id, event_date, post_url)

    # Open via /events and click edit button by id (UI flow)
    step_send_command(context, f"/events {event_date}")
    msg = context.last_response
    btn = find_button(msg, f"✎ {event_id}")
    if not btn:
        # /events is paginated/compact for dense dates; use direct admin shortcut.
        step_send_command(context, f"/edit {event_id}")
        card = context.last_response
        card_text = (card.text or "") if card else ""
        if f"id: {event_id}" not in card_text:
            available = get_all_buttons(msg)
            raise AssertionError(
                f"Кнопка ✎ {event_id} не найдена в /events {event_date}, "
                f"и /edit {event_id} не открыл карточку. Кнопки: {available}"
            )
        logger.info("✓ Открыта карточка контрольного события через /edit: id=%s", event_id)
        return

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


@when('я открываю карточку события из выбранного поста с заголовком содержащим "{title_hint}"')
def step_open_event_card_from_selected_post_title_hint(context, title_hint):
    post_url = getattr(context, "control_post_url", None)
    username = getattr(context, "control_post_username", None)
    message_id = getattr(context, "control_post_message_id", None)
    if not post_url or not username or not message_id:
        raise AssertionError("Контрольный пост не выбран (ожидались control_post_url/username/message_id)")

    hint = _norm_event_title_key(title_hint)
    if not hint:
        raise AssertionError("Пустой title_hint для выбора события")

    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        _ensure_event_source_table(conn)

        cur.execute(
            """
            SELECT DISTINCT e.id, e.title, e.date
            FROM event_source es
            JOIN event e ON e.id = es.event_id
            WHERE es.source_url = ?
               OR (es.source_chat_username = ? AND es.source_message_id = ?)
               OR es.source_url LIKE ?
            ORDER BY e.id DESC
            """,
            (post_url, username, int(message_id), f"%t.me/{username}/{int(message_id)}%"),
        )
        rows = list(cur.fetchall())
    finally:
        conn.close()

    if not rows:
        raise AssertionError(f"Не найдено событий из выбранного поста: {post_url}")

    filtered = [r for r in rows if hint in _norm_event_title_key(r["title"])]
    if not filtered:
        titles = [str(r["title"] or "") for r in rows[:10]]
        raise AssertionError(
            f"Событие с title_hint='{title_hint}' не найдено среди событий поста {post_url}. "
            f"Кандидаты: {titles}"
        )

    today = date.today()
    picked: tuple[tuple[int, int, int], int, str] | None = None
    for row in filtered:
        event_id = int(row["id"])
        raw_date = str(row["date"] or "")
        iso = raw_date.split("..", 1)[0].strip()
        parsed = None
        try:
            parsed = date.fromisoformat(iso)
        except Exception:
            parsed = None
        if parsed is not None and parsed >= today:
            key = (0, (parsed - today).days, event_id)
        elif parsed is not None:
            key = (1, abs((today - parsed).days), event_id)
        else:
            key = (2, 10**9, event_id)
        if picked is None or key < picked[0]:
            picked = (key, event_id, iso)

    if picked is None:
        raise AssertionError("Не удалось выбрать событие по title_hint")

    _, event_id, event_date = picked
    context.control_event_id = int(event_id)
    context.control_event_date = event_date
    context.last_event_id = int(event_id)

    def _is_target_card(msg) -> bool:
        text = str(getattr(msg, "text", "") or "")
        return f"id: {event_id}" in text

    step_send_command(context, f"/edit {event_id}")
    if not _is_target_card(getattr(context, "last_response", None)):
        deadline = time.monotonic() + 35.0
        found = False
        while time.monotonic() < deadline:
            async def _fetch():
                return await context.client.client.get_messages(context.bot_entity, limit=15)

            messages = run_async(context, _fetch())
            for msg in messages:
                if _is_target_card(msg):
                    context.last_response = msg
                    found = True
                    break
            if found:
                break
            time.sleep(1.5)
        if not found:
            # Last retry as explicit command in case command response got buried
            step_send_command(context, f"/edit {event_id}")
            if not _is_target_card(getattr(context, "last_response", None)):
                raise AssertionError(f"/edit {event_id} не открыл карточку события")
    logger.info(
        "✓ Открыта карточка события из выбранного поста по title_hint: id=%s hint=%s post=%s",
        event_id,
        title_hint,
        post_url,
    )


@then("описание события отрерайчено (не дословно)")
def step_description_is_rewritten(context):
    """Basic guard: description must not equal the latest telegram source text verbatim."""
    event_id = getattr(context, "control_event_id", None)
    if not event_id:
        # Fallback to any last event id if a scenario opened a card by title.
        event_id = getattr(context, "last_event_id", None)
    if not event_id:
        raise AssertionError("Нет event_id для проверки рерайта")

    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        row = cur.execute(
            """
            SELECT e.description, es.source_text
            FROM event e
            JOIN event_source es ON es.event_id = e.id
            WHERE e.id = ?
              AND es.source_type LIKE 'telegram%'
            ORDER BY es.imported_at DESC
            LIMIT 1
            """,
            (int(event_id),),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Для события id={event_id} не найден telegram source_text")
    desc, src_text = (row[0] or "").strip(), (row[1] or "").strip()
    if not desc:
        raise AssertionError("Пустое описание события")
    if not src_text:
        raise AssertionError("Пустой source_text у telegram источника")
    if _normalize_text(desc) == _normalize_text(src_text):
        raise AssertionError("Описание совпадает дословно с текстом источника (ожидался рерайт)")
    logger.info("✓ Описание не совпадает дословно с source_text (рерайт присутствует)")


@then('описание открытого события содержит "{needle}"')
def step_open_event_description_contains(context, needle):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки description")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT description FROM event WHERE id = ?", (int(event_id),)).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено")
    desc = (row[0] or "").strip()
    if needle.lower() not in desc.lower():
        raise AssertionError(f"В description события id={event_id} не найдено: {needle}")
    logger.info("✓ Description содержит ожидаемый фрагмент: %s", needle)


@then('у открытого события тип "{expected_type}"')
def step_open_event_type_equals(context, expected_type):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки event_type")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT event_type FROM event WHERE id = ?", (int(event_id),)).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено")
    got = str(row[0] or "").strip().casefold()
    want = str(expected_type or "").strip().casefold()
    if got != want:
        raise AssertionError(f"event_type mismatch для id={event_id}: expected={want!r}, got={got!r}")
    logger.info("✓ event_type=%s для event_id=%s", got, event_id)


@then("у открытой выставки заполнены date и end_date")
def step_open_exhibition_has_date_range(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки диапазона выставки")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT event_type, date, end_date FROM event WHERE id = ?",
            (int(event_id),),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено")
    event_type, date_raw, end_raw = row
    if str(event_type or "").strip().casefold() != "выставка":
        raise AssertionError(f"Событие id={event_id} не выставка: event_type={event_type!r}")
    date_iso = str(date_raw or "").split("..", 1)[0].strip()
    end_iso = str(end_raw or "").split("..", 1)[0].strip()
    if not date_iso:
        raise AssertionError(f"У выставки id={event_id} не заполнено поле date")
    if not end_iso:
        raise AssertionError(f"У выставки id={event_id} не заполнено поле end_date")
    try:
        d = date.fromisoformat(date_iso)
        e = date.fromisoformat(end_iso)
    except Exception as exc:
        raise AssertionError(
            f"Некорректный date/end_date у выставки id={event_id}: date={date_iso!r}, end_date={end_iso!r}"
        ) from exc
    if e < d:
        raise AssertionError(
            f"Некорректный период у выставки id={event_id}: end_date({end_iso}) < date({date_iso})"
        )
    logger.info("✓ Выставка id=%s имеет валидный период: %s..%s", event_id, date_iso, end_iso)


@then("я сохраняю открытую выставку в набор проверки /exhibitions")
def step_save_open_exhibition_for_exhibitions_check(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для сохранения выставки")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT title, event_type, date, end_date FROM event WHERE id = ?",
            (int(event_id),),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено")
    title, event_type, date_iso, end_iso = row
    if str(event_type or "").strip().casefold() != "выставка":
        raise AssertionError(f"Событие id={event_id} не выставка: event_type={event_type!r}")
    saved = getattr(context, "saved_exhibitions_for_check", None)
    if not isinstance(saved, list):
        saved = []
        context.saved_exhibitions_for_check = saved
    if any(int(item.get("id", 0)) == int(event_id) for item in saved):
        return
    saved.append(
        {
            "id": int(event_id),
            "title": str(title or "").strip(),
            "date": str(date_iso or "").strip(),
            "end_date": str(end_iso or "").strip(),
        }
    )
    logger.info("✓ Сохранена выставка для /exhibitions: id=%s title=%s", event_id, title)


@then("в /exhibitions отображаются все сохраненные выставки")
def step_exhibitions_contains_saved_exhibitions(context):
    saved = list(getattr(context, "saved_exhibitions_for_check", []) or [])
    if not saved:
        raise AssertionError("Нет сохранённых выставок для проверки /exhibitions")

    async def _collect_recent_text() -> str:
        messages = await context.client.client.get_messages(context.bot_entity, limit=20)
        if not messages:
            return ""
        top_id = int(getattr(context.last_response, "id", 0) or messages[0].id or 0)
        chunks: list[str] = []
        for m in messages:
            mid = int(getattr(m, "id", 0) or 0)
            if top_id and mid > top_id:
                continue
            if top_id and mid < max(1, top_id - 12):
                continue
            text = (m.text or "").strip()
            if text:
                chunks.append(text)
        return "\n\n".join(chunks)

    combined_text = run_async(context, _collect_recent_text())
    if not combined_text:
        combined_text = (context.last_response.text if context.last_response else "") or ""
    hay = _normalize_text(combined_text).replace("ё", "е")
    missing: list[str] = []
    for item in saved:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        title_norm = _normalize_text(title).replace("ё", "е")
        if title_norm and title_norm in hay:
            continue
        # Fallback for wrapped/truncated lines: require at least 2 significant title tokens.
        tokens = [
            t
            for t in re.findall(r"[a-zа-яё0-9]{4,}", title_norm, flags=re.IGNORECASE)
            if t not in {"выставка", "калининград", "галерея"}
        ]
        token_hits = sum(1 for t in tokens if t in hay)
        if token_hits >= min(2, len(tokens)) and len(tokens) > 0:
            continue
        missing.append(title)
    if missing:
        raise AssertionError(
            "В /exhibitions не найдены сохранённые выставки:\n"
            + "\n".join(f"- {m}" for m in missing)
            + f"\n\nТекст /exhibitions:\n{combined_text}"
        )
    logger.info("✓ /exhibitions содержит все сохранённые выставки (%s)", len(saved))


@then("описание открытого события не раздуто относительно Telegram source_text")
def step_open_event_description_not_overexpanded(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки объёма описания")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        row = cur.execute(
            """
            SELECT
                e.description,
                es.source_text,
                COALESCE(e.source_texts, ''),
                (
                    SELECT COUNT(*)
                    FROM event_source es2
                    WHERE es2.event_id = e.id
                      AND es2.source_type LIKE 'telegram%'
                ) AS tg_sources_count
            FROM event e
            JOIN event_source es ON es.event_id = e.id
            WHERE e.id = ?
              AND es.source_type LIKE 'telegram%'
            ORDER BY es.imported_at DESC
            LIMIT 1
            """,
            (int(event_id),),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Для события id={event_id} не найден Telegram source_text")
    desc = (row[0] or "").strip()
    latest_src = (row[1] or "").strip()
    aggregated_src = (row[2] or "").strip()
    tg_sources_count = int(row[3] or 0)
    if not desc or (not latest_src and not aggregated_src):
        raise AssertionError("Пустой description/source_text, проверить раздувание нельзя")

    # For multi-source events compare against accumulated telegram corpus.
    # Otherwise we get false positives when the latest source text is short.
    src_for_cap = latest_src
    if tg_sources_count > 1 and len(aggregated_src) > len(latest_src):
        src_for_cap = aggregated_src

    src_len = len(src_for_cap)
    max_allowed = max(260, int(src_len * 1.9) + 120)
    if len(desc) > max_allowed:
        raise AssertionError(
            f"Description слишком раздуто относительно Telegram source_text: "
            f"desc_len={len(desc)} src_len={src_len} cap={max_allowed} tg_sources={tg_sources_count}"
        )
    logger.info(
        "✓ Description объём в норме: desc_len=%s src_len=%s cap=%s tg_sources=%s",
        len(desc),
        src_len,
        max_allowed,
        tg_sources_count,
    )


@then('в карточке события есть источник "{needle}"')
def step_event_card_has_source_reference(context, needle):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    if not text:
        raise AssertionError("Пустая карточка события")
    if needle.lower() not in text.lower():
        raise AssertionError(f"В карточке события не найден источник: {needle}")
    logger.info("✓ В карточке события найден источник: %s", needle)


@then('у открытого события источников минимум "{count}"')
def step_open_event_sources_min(context, count):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(context, "last_event_id", None)
    if not event_id:
        raise AssertionError("Не удалось определить event_id из карточки события")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM event_source WHERE event_id=?", (int(event_id),))
        actual = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()
    expected = int(count)
    if actual < expected:
        raise AssertionError(f"Ожидали источников >= {expected}, получили {actual} (event_id={event_id})")
    logger.info("✓ Источников у события достаточно: %s (>= %s)", actual, expected)


@then('у открытого события источников ровно "{count}"')
def step_open_event_sources_exact(context, count):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id из карточки события")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM event_source WHERE event_id=?", (int(event_id),))
        actual = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()
    expected = int(count)
    if actual != expected:
        raise AssertionError(f"Ожидали источников ровно {expected}, получили {actual} (event_id={event_id})")
    logger.info("✓ Источников у события ровно: %s", actual)


@then("в карточке события есть источник выбранного связанного поста")
def step_event_card_has_selected_linked_source(context):
    linked_url = (getattr(context, "linked_post_url", None) or "").strip()
    if not linked_url:
        raise AssertionError("В context нет linked_post_url (сначала выберите linked-post кейс)")
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id из карточки события")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM event_source WHERE event_id=? AND source_url=?",
            (int(event_id), linked_url),
        )
        actual = int(cur.fetchone()[0] or 0)
    finally:
        conn.close()
    if actual < 1:
        raise AssertionError(f"У события id={event_id} не найден источник связанного поста: {linked_url}")
    logger.info("✓ У события id=%s найден linked source: %s", event_id, linked_url)


@then("primary и linked посты из @meowafisha привязаны к одному событию (без дубля)")
def step_primary_and_linked_bound_to_single_event(context):
    primary_url = (getattr(context, "control_post_url", None) or "").strip()
    linked_url = (getattr(context, "linked_post_url", None) or "").strip()
    if not primary_url or not linked_url:
        raise AssertionError("Нет primary/linked URL в context")

    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id из карточки события")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        _ensure_event_source_table(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT event_id FROM event_source WHERE source_url IN (?, ?)",
            (primary_url, linked_url),
        )
        event_ids = [int(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()
    if len(event_ids) != 1:
        raise AssertionError(
            f"Ожидали один event_id для primary/linked постов, получили {event_ids}. "
            f"primary={primary_url} linked={linked_url}"
        )
    if int(event_id) != int(event_ids[0]):
        raise AssertionError(
            f"Открытая карточка event_id={event_id} не совпадает с event_id связки={event_ids[0]}"
        )
    logger.info("✓ Primary и linked посты привязаны к одному событию: event_id=%s", event_id)


@then("после linked-post LLM сверки у события заполнены title, location_name, date и time")
def step_linked_post_llm_fields_are_filled(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id из карточки события")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT title, location_name, date, time FROM event WHERE id=?",
            (int(event_id),),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено в БД")

    title = str(row["title"] or "").strip()
    location_name = str(row["location_name"] or "").strip()
    date_value = str(row["date"] or "").strip()
    time_value = str(row["time"] or "").strip()
    if not title:
        raise AssertionError("Поле title пустое")
    if not location_name:
        raise AssertionError("Поле location_name пустое")
    if not re.match(r"^\d{4}-\d{2}-\d{2}", date_value):
        raise AssertionError(f"Поле date некорректно: {date_value!r}")
    if time_value in {"", "00:00", "00:00:00"}:
        raise AssertionError(f"Поле time не заполнено/плейсхолдер: {time_value!r}")
    logger.info(
        "✓ Linked-post якоря заполнены: title=%r location=%r date=%r time=%r",
        title,
        location_name,
        date_value,
        time_value,
    )


@then("не создан дубль события из-за отсутствующего времени в Telegram")
def step_no_duplicate_due_to_missing_telegram_time(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки дублей")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        current = cur.execute(
            "SELECT id, title, date, time, location_name FROM event WHERE id = ?",
            (int(event_id),),
        ).fetchone()
        if not current:
            raise AssertionError(f"Событие id={event_id} не найдено в БД")

        event_date = str(current["date"] or "").split("..", 1)[0].strip()
        title_key = _norm_event_title_key(current["title"])
        location_key = _norm_location_key(current["location_name"])

        cur.execute("SELECT id, title, date, time, location_name FROM event WHERE date LIKE ?", (f"{event_date}%",))
        candidates = []
        for row in cur.fetchall():
            if _norm_location_key(row["location_name"]) != location_key:
                continue
            if _norm_event_title_key(row["title"]) != title_key:
                continue
            candidates.append(row)
    finally:
        conn.close()

    if len(candidates) <= 1:
        logger.info("✓ Дубликатов по title/date/location нет (event_id=%s)", event_id)
        return

    placeholder_times = {"", "00:00", "00:00:00"}
    times = [str(r["time"] or "").strip() for r in candidates]
    has_placeholder = any(t in placeholder_times for t in times)
    has_real = any(t not in placeholder_times for t in times)
    if has_placeholder and has_real:
        details = ", ".join(f"id={int(r['id'])}:time={str(r['time'] or '').strip() or '<empty>'}" for r in candidates)
        raise AssertionError(
            "Найден дубль события из-за различия времени (Telegram без времени vs parser с временем): "
            f"{details}"
        )

    unique_times = {t for t in times if t}
    if len(candidates) > 1 and len(unique_times) <= 1:
        details = ", ".join(f"id={int(r['id'])}:time={str(r['time'] or '').strip() or '<empty>'}" for r in candidates)
        raise AssertionError(f"Найдены дубли одного события: {details}")

    logger.info("✓ Проверка дублей пройдена: matched=%s unique_times=%s", len(candidates), sorted(unique_times))


@then('открытое событие содержит изображения из поста "{post_url}"')
def step_open_event_has_images_from_post(context, post_url):
    import time
    import aiohttp

    run_id = getattr(context, "last_monitor_run_id", None)
    if not run_id:
        raise AssertionError("Не найден run_id последнего мониторинга")
    data = _load_tg_results(run_id)
    if not data:
        raise AssertionError(f"Не удалось загрузить telegram_results.json для run_id={run_id}")
    poster_urls = _extract_posters_from_tg_results(data, post_url)
    if not poster_urls:
        raise AssertionError(f"Нет URL афиш (catbox_url/supabase_url) для поста {post_url}")

    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(context, "last_event_id", None)
    if not event_id:
        raise AssertionError("Не удалось определить event_id из карточки события")

    wait_sec = int(os.getenv("E2E_TELEGRAPH_WAIT_SEC", "240"))
    deadline = time.time() + wait_sec
    url = ""
    while time.time() < deadline:
        conn = sqlite3.connect(_db_path(), timeout=30)
        try:
            cur = conn.cursor()
            cur.execute("SELECT telegraph_url FROM event WHERE id=?", (int(event_id),))
            row = cur.fetchone()
        finally:
            conn.close()
        candidate = (row[0] if row else "") if row is not None else ""
        candidate = (candidate or "").strip()
        if candidate.startswith("https://telegra.ph/"):
            url = candidate
            break
        time.sleep(3)
    if not url:
        raise AssertionError(f"У события нет telegraph_url за {wait_sec}s (event_id={event_id})")

    async def _fetch():
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                return await resp.text()

    # Telegraph rebuild is async; allow it to catch up.
    wait_images = int(os.getenv("E2E_TELEGRAPH_IMAGES_WAIT_SEC", "480"))
    deadline = time.time() + wait_images
    last_html = ""
    while time.time() < deadline:
        last_html = run_async(context, _fetch())
        if any(p in last_html for p in poster_urls):
            logger.info("✓ Telegraph страница содержит изображение из поста: %s", url)
            return
        time.sleep(12)
    raise AssertionError(f"URL афиш из поста не найден на Telegraph странице за {wait_images}s: {url}")


@then('в логе источников есть источник "{needle}"')
def step_source_log_contains_source(context, needle):
    def _event_id() -> int | None:
        return _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
            context, "last_event_id", None
        )

    def _telegraph_url_for_event(event_id: int) -> str | None:
        import sqlite3

        conn = sqlite3.connect(_db_path(), timeout=30)
        try:
            cur = conn.cursor()
            cur.execute("SELECT telegraph_url, telegraph_path FROM event WHERE id=?", (int(event_id),))
            row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        url, path = (row[0] or "").strip(), (row[1] or "").strip()
        if url:
            return url
        if path:
            return f"https://telegra.ph/{path.lstrip('/')}"
        return None

    async def _fetch_log_text(event_id: int | None) -> str:
        messages = await context.client.client.get_messages(context.bot_entity, limit=40)
        target_url = _telegraph_url_for_event(event_id) if event_id else None
        for candidate in messages:
            t = candidate.text or ""
            if "Лог источников" not in t:
                continue
            if target_url and target_url not in t:
                continue
            return t
        # Fallback: most recent log message (best-effort).
        for candidate in messages:
            t = candidate.text or ""
            if "Лог источников" in t:
                return t
        return ""

    text = getattr(context.last_response, "text", "") or ""
    if not text or "Лог источников" not in text:
        text = run_async(context, _fetch_log_text(_event_id()))
    if not text:
        raise AssertionError("Лог источников пуст или не получен")
    if needle.lower() not in text.lower():
        raise AssertionError(f"В логе источников не найдено: {needle}")
    logger.info("✓ Лог источников содержит: %s", needle)


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


@when('я жду новое сообщение с текстом "{text}"')
@then('я жду новое сообщение с текстом "{text}"')
def step_wait_for_new_message_text(context, text):
    """Wait for a NEW message (after the current last_response) containing specific text."""
    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_WAIT_NEW_MESSAGE_TIMEOUT_SEC", "180"))
        poll_sec = float(os.getenv("E2E_WAIT_NEW_MESSAGE_POLL_SEC", "1.0"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_messages = []
        while time.monotonic() < deadline:
            last_messages = await context.client.client.get_messages(context.bot_entity, limit=10)
            _raise_on_bot_ui_errors(last_messages, baseline_id=baseline_id)
            for msg in last_messages:
                mid = int(getattr(msg, "id", 0) or 0)
                if mid <= baseline_id:
                    continue
                if msg.text and text.lower() in msg.text.lower():
                    context.last_response = msg
                    logger.info("✓ Найдено новое сообщение: %r (mid=%s baseline=%s)", text, mid, baseline_id)
                    return
            await asyncio.sleep(poll_sec)
        previews = [(m.id, (m.text or "")[:80].replace("\n", " ")) for m in last_messages[:5]]
        raise AssertionError(
            f"Новое сообщение с текстом '{text}' не получено за {timeout_sec} секунд. "
            f"baseline_id={baseline_id}, последние={previews}"
        )

    run_async(context, _wait())


@when("я запоминаю message_id как baseline")
@then("я запоминаю message_id как baseline")
def step_remember_message_id_baseline(context):
    """Remember the current last_response message id as a baseline for later checks."""
    baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
    context.baseline_message_id = baseline_id
    logger.info("✓ baseline_message_id=%s", baseline_id)


@when('я жду новое сообщение после baseline с текстом "{text}"')
@then('я жду новое сообщение после baseline с текстом "{text}"')
def step_wait_for_new_message_after_baseline(context, text):
    """Wait for a NEW message (after remembered baseline_message_id) containing specific text."""

    async def _wait():
        import asyncio

        baseline_id = int(getattr(context, "baseline_message_id", 0) or 0)
        timeout_sec = int(os.getenv("E2E_WAIT_NEW_MESSAGE_TIMEOUT_SEC", "180"))
        poll_sec = float(os.getenv("E2E_WAIT_NEW_MESSAGE_POLL_SEC", "1.0"))
        deadline = time.monotonic() + float(timeout_sec)
        last_messages = []
        while time.monotonic() < deadline:
            last_messages = await context.client.client.get_messages(context.bot_entity, limit=10)
            _raise_on_bot_ui_errors(last_messages, baseline_id=baseline_id)
            for msg in last_messages:
                mid = int(getattr(msg, "id", 0) or 0)
                if mid <= baseline_id:
                    continue
                if msg.text and text.lower() in msg.text.lower():
                    context.last_response = msg
                    logger.info(
                        "✓ Найдено новое сообщение после baseline: %r (mid=%s baseline=%s)",
                        text,
                        mid,
                        baseline_id,
                    )
                    return
            await asyncio.sleep(poll_sec)
        previews = [(m.id, (m.text or "")[:80].replace("\n", " ")) for m in last_messages[:5]]
        raise AssertionError(
            f"Новое сообщение после baseline с текстом '{text}' не получено за {timeout_sec} секунд. "
            f"baseline_id={baseline_id}, последние={previews}"
        )

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
    links = list(getattr(context, "telegraph_links", None) or [])
    if not links:
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


@then('каждая Telegraph страница не должна содержать "{forbidden_text}"')
def step_verify_telegraph_content_absence(context, forbidden_text):
    """Verify each Telegraph page does not contain forbidden content."""
    import aiohttp

    links = getattr(context, "telegraph_links", [])
    if not links:
        raise AssertionError("Нет сохранённых Telegraph ссылок для проверки")

    forbidden_items = [item.strip() for item in forbidden_text.split(",") if item.strip()]
    if not forbidden_items:
        return

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
                        html_lower = html.lower()

                        present = []
                        for item in forbidden_items:
                            if item.lower() in html_lower:
                                present.append(item)

                        if present:
                            failed_pages.append(f"{link}: найдено [{', '.join(present)}]")
                        else:
                            logger.info(f"✓ Страница {link} не содержит запрещённых элементов: {forbidden_items}")

                except Exception as e:
                    failed_pages.append(f"{link}: ошибка {e}")

            if failed_pages:
                print("\n" + "=" * 60)
                print("[ERROR] Проверка отсутствия контента Telegraph страниц:")
                for fail in failed_pages:
                    print(f"  ✗ {fail}")
                print("=" * 60 + "\n")
                raise AssertionError(f"На некоторых страницах найден запрещённый контент: {failed_pages}")

    run_async(context, _verify_content())
    logger.info(f"✓ Все {len(links)} страниц не содержат: {forbidden_items}")


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
            # Kaggle can be slow (kernel cold start + OCR/vision). Keep a generous default,
            # still overridable via env for faster local runs.
            timeout_sec = int(os.getenv("E2E_TG_MONITOR_TIMEOUT_SEC", str(35 * 60)))
            poll_sec = float(os.getenv("E2E_TG_MONITOR_POLL_SEC", "4"))
        elif any(tok in text_norm for tok in ["source parsing", "парсинг", "parse"]):
            timeout_sec = int(os.getenv("E2E_PARSE_TIMEOUT_SEC", str(35 * 60)))
            poll_sec = float(os.getenv("E2E_PARSE_POLL_SEC", "4"))
        elif "фестиваль" in text_norm:
            timeout_sec = int(os.getenv("E2E_FESTIVAL_PARSE_TIMEOUT_SEC", str(35 * 60)))
            poll_sec = float(os.getenv("E2E_FESTIVAL_PARSE_POLL_SEC", "4"))
        else:
            timeout_sec = int(os.getenv("E2E_LONG_OPERATION_TIMEOUT_SEC", str(5 * 60)))
            poll_sec = float(os.getenv("E2E_LONG_OPERATION_POLL_SEC", "2"))
        poll_max = float(os.getenv("E2E_LONG_OPERATION_MAX_POLL_SEC", "10"))

        reconnect_attempts = 0
        min_id = getattr(context, "monitor_started_message_id", None)
        if not min_id:
            min_id = getattr(getattr(context, "last_response", None), "id", None)
        elapsed = 0.0
        messages = []
        while elapsed < timeout_sec:
            try:
                messages = await context.client.client.get_messages(
                    context.bot_entity, limit=8
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
                    if text.lower() == "telegram monitor":
                        msg_l = msg.text.lower()
                        has_run_id = "run_id:" in msg_l
                        has_completion_markers = any(
                            marker in msg_l
                            for marker in (
                                "созданные события",
                                "обновлённые события",
                                "сообщений с событиями",
                                "событий извлечено",
                                "мониторинг заверш",
                            )
                        )
                        if not (has_run_id or has_completion_markers):
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
                    logger.info(f"✓ Найден результат долгой операции: '{text}' (за {elapsed:.1f}с)")
                    return
            await asyncio.sleep(poll_sec)
            elapsed += poll_sec
            if poll_sec < poll_max:
                poll_sec = min(poll_max, poll_sec * 1.15)
        
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

@then('в отчёте мониторинга значение "{label}" минимум "{expected_min}"')
def step_report_stat_min(context, label, expected_min):
    report_text = getattr(context, "last_report_text", None) or (
        context.last_response.text if context.last_response else ""
    )
    value = _extract_report_stat(report_text, label)
    if value is None:
        raise AssertionError(f"Не найден счётчик '{label}' в отчёте:\n{report_text}")
    if value < int(expected_min):
        raise AssertionError(f"Ожидали '{label} >= {expected_min}', получили '{label}: {value}'")
    logger.info("✓ Отчёт: %s>=%s (value=%s)", label, expected_min, value)


@then('в отчёте мониторинга сумма "{label_a}" и "{label_b}" минимум "{expected_min}"')
def step_report_stat_sum_min(context, label_a, label_b, expected_min):
    report_text = getattr(context, "last_report_text", None) or (
        context.last_response.text if context.last_response else ""
    )
    a = _extract_report_stat(report_text, label_a)
    b = _extract_report_stat(report_text, label_b)
    if a is None:
        raise AssertionError(f"Не найден счётчик '{label_a}' в отчёте:\n{report_text}")
    if b is None:
        raise AssertionError(f"Не найден счётчик '{label_b}' в отчёте:\n{report_text}")
    total = int(a) + int(b)
    if total < int(expected_min):
        raise AssertionError(
            f"Ожидали '{label_a}+{label_b} >= {expected_min}', получили {label_a}={a} {label_b}={b} total={total}"
        )
    logger.info("✓ Отчёт: %s+%s>=%s (total=%s)", label_a, label_b, expected_min, total)


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


@when("новые события не создаются")
@then("новые события не создаются")
def step_no_new_events_created(context):
    """Assert report shows no newly created events."""
    report_text = getattr(context, "last_report_text", None) or (context.last_response.text if context.last_response else "")
    created = _extract_report_stat(report_text, "Создано")
    if created is None:
        raise AssertionError("Не найден счётчик 'Создано' в отчёте")
    assert created == 0, f"Ожидалось Создано: 0, получено: {created}"
    logger.info("✓ Новые события не создавались")


@then("в отчёте мониторинга для каждого созданного события есть Telegraph, /log и ICS")
def step_monitoring_report_has_unified_event_blocks(context):
    """Verify operator-facing monitoring report contains actionable per-event links."""
    async def _fetch():
        messages = await context.client.client.get_messages(context.bot_entity, limit=60)
        for msg in messages:
            t = msg.text or ""
            # Telegraph link is embedded into the event title, so the report may not contain
            # a dedicated "Telegraph:" line when telegraph_url exists.
            if "Созданные события" in t and "Лог:" in t and "ICS:" in t:
                return msg
        return None

    msg = run_async(context, _fetch())
    if not msg:
        raise AssertionError("Не найдено сообщение с блоком 'Созданные события' и ссылками 'Лог:'/'ICS:'")
    text = msg.text or ""

    ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
    if not ids:
        raise AssertionError(f"В отчёте не найдено созданных событий (id=...). Текст:\n{text}")

    # Telethon renders entities as Markdown in msg.text (e.g. "Лог: [/log 1](https://...)").
    log_cnt = len(re.findall(r"^Лог:\s+.*?/log\s+\d+.*$", text, flags=re.MULTILINE))
    ics_cnt = len(re.findall(r"^ICS:\s+\S+", text, flags=re.MULTILINE))
    if "Telegraph: ⏳" in text:
        raise AssertionError(f"В отчёте найден 'Telegraph: ⏳ в очереди'. Текст:\n{text}")

    # Prefer URL entities (reliable for embedded title links). Fall back to text search.
    urls = []
    for ent in (getattr(msg, "entities", None) or []):
        u = getattr(ent, "url", None)
        if u:
            urls.append(str(u))
    tele_links = {u for u in urls if "telegra.ph/" in u}
    if not tele_links:
        tele_links = set(re.findall(r"https?://telegra\\.ph/\\S+", text))
    tele_cnt = len(tele_links)
    if tele_cnt < len(ids):
        raise AssertionError(
            f"Ожидали ссылок на Telegraph >= {len(ids)}, получили {tele_cnt}. Текст:\n{text}"
        )
    if log_cnt < len(ids):
        raise AssertionError(
            f"Ожидали Лог строк >= {len(ids)}, получили {log_cnt}. Текст:\n{text}"
        )
    if ics_cnt < len(ids):
        raise AssertionError(
            f"Ожидали ICS строк >= {len(ids)}, получили {ics_cnt}. Текст:\n{text}"
        )
    for eid in ids[:10]:
        if f"/log {eid}" not in text:
            raise AssertionError(f"В отчёте нет команды '/log {eid}'. Текст:\n{text}")

    # Ensure /log is a clickable deep-link (Telegram won't include args in command click targets).
    for eid in ids[:10]:
        want = f"start=log_{eid}"
        if not any(want in u for u in urls):
            raise AssertionError(
                "В отчёте нет кликабельной ссылки на лог (ожидали deep-link через start=log_<id>).\n"
                f"event_id={eid}\n"
                f"Текст:\n{text}"
            )
    logger.info(
        "✓ Unified monitoring report: events=%s telegraph=%s log=%s ics=%s",
        len(ids),
        tele_cnt,
        log_cnt,
        ics_cnt,
    )


@then("я жду отчёт VK auto import с событиями и ссылками")
def step_wait_vk_auto_report_has_unified_event_blocks(context):
    """Wait for a per-post VK auto report block with actionable links."""

    async def _wait():
        import asyncio

        timeout_env = os.getenv("E2E_VK_AUTO_REPORT_TIMEOUT_SEC")
        timeout_sec = int(timeout_env) if timeout_env else int(15 * 60)
        poll_sec = float(os.getenv("E2E_VK_AUTO_REPORT_POLL_SEC", "3"))
        progress_log_sec = float(os.getenv("E2E_VK_AUTO_PROGRESS_LOG_SEC", "60"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        next_progress_log = time.monotonic() + progress_log_sec
        last_preview: list[str] = []
        targets: list[tuple[int, int]] = list(getattr(context, "vk_priority_targets", []) or [])
        if (not timeout_env) and targets:
            per_post_sec = int(os.getenv("E2E_VK_AUTO_PER_POST_TIMEOUT_SEC", str(4 * 60)))
            buffer_sec = int(os.getenv("E2E_VK_AUTO_REPORT_BUFFER_SEC", str(10 * 60)))
            timeout_sec = max(timeout_sec, (len(targets) * per_post_sec) + buffer_sec)
            logger.info(
                "VK auto wait: computed timeout=%ss targets=%s per_post=%ss buffer=%ss",
                timeout_sec,
                len(targets),
                per_post_sec,
                buffer_sec,
            )
        deadline = time.monotonic() + float(timeout_sec)
        last_target_state = ""
        seen_report_msg_ids: set[int] = set()
        aggregated_event_ids: list[int] = []
        aggregated_event_ids_seen: set[int] = set()
        aggregated_tele_links: list[str] = []
        aggregated_tele_seen: set[str] = set()
        log_links_seen_for_eid: set[int] = set()
        latest_report_msg = None
        latest_report_text = ""

        def _targets_processed() -> tuple[bool, str]:
            if not targets:
                return True, "no-targets"
            db_uri = f"file:{_db_path()}?mode=ro"
            try:
                conn = sqlite3.connect(db_uri, uri=True, timeout=1)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower():
                    return False, "db-locked(connect)"
                raise
            try:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                where = " OR ".join(["(group_id=? AND post_id=?)" for _ in targets])
                params: list[int] = []
                for group_id, post_id in targets:
                    params.extend([int(group_id), int(post_id)])
                try:
                    cur.execute(
                        f"SELECT group_id, post_id, status FROM vk_inbox WHERE {where}",
                        tuple(params),
                    )
                    rows = list(cur.fetchall())
                except sqlite3.OperationalError as exc:
                    # Writers can briefly lock DB during long VK import; keep polling.
                    if "locked" in str(exc).lower():
                        return False, "db-locked"
                    raise
            finally:
                conn.close()
            status_map = {(int(r["group_id"]), int(r["post_id"])): str(r["status"] or "") for r in rows}
            states = [f"-{g}_{p}:{status_map.get((g, p), 'missing')}" for g, p in targets]
            active = {"pending", "locked", "importing"}
            done = all(status_map.get((g, p), "missing") not in active for g, p in targets)
            return done, ", ".join(states)

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=80)
            last_preview = [
                (m.text or "").replace("\n", " ")[:140]
                for m in messages[:8]
            ]
            _raise_on_bot_ui_errors(messages, baseline_id=baseline_id)
            done, last_target_state = _targets_processed()
            now = time.monotonic()
            if now >= next_progress_log:
                logger.info(
                    "VK auto wait: reports=%s events=%s log_links=%s state=%s",
                    len(seen_report_msg_ids),
                    len(aggregated_event_ids),
                    len(log_links_seen_for_eid),
                    last_target_state,
                )
                next_progress_log = now + progress_log_sec

            # Process from older -> newer to preserve deterministic event order.
            for msg in reversed(messages):
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                text = msg.text or ""
                if not text:
                    continue
                if "Лог:" not in text or "ICS:" not in text:
                    continue
                if "Smart Update (детали событий)" not in text:
                    continue
                if ("Созданные события" not in text) and ("Обновлённые события" not in text):
                    continue

                ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
                if not ids:
                    continue
                tele_links = re.findall(r"https://telegra\.ph/[a-zA-Z0-9_-]+", text)
                if not tele_links:
                    continue
                if not re.search(r"^\s*ICS:\s+\S+", text, flags=re.MULTILINE):
                    continue
                if not re.search(r"^\s*Лог:\s+.*?/log\s+\d+.*$", text, flags=re.MULTILINE):
                    continue
                if not re.search(r"^\s*Факты:\s+.*Иллюстрации:\s+", text, flags=re.MULTILINE):
                    continue
                mid = int(getattr(msg, "id", 0) or 0)
                latest_report_msg = msg
                latest_report_text = text
                if mid not in seen_report_msg_ids:
                    seen_report_msg_ids.add(mid)
                    for eid in ids:
                        if eid not in aggregated_event_ids_seen:
                            aggregated_event_ids_seen.add(eid)
                            aggregated_event_ids.append(eid)
                    for link in tele_links:
                        if link not in aggregated_tele_seen:
                            aggregated_tele_seen.add(link)
                            aggregated_tele_links.append(link)
                    urls = []
                    for ent in (getattr(msg, "entities", None) or []):
                        u = getattr(ent, "url", None)
                        if u:
                            urls.append(str(u))
                    for eid in ids:
                        want = f"start=log_{eid}"
                        if any(want in u for u in urls):
                            log_links_seen_for_eid.add(eid)

            if done and latest_report_msg and aggregated_event_ids:
                context.last_response = latest_report_msg
                context.telegraph_links = aggregated_tele_links
                context.last_vk_auto_report_event_ids = list(aggregated_event_ids)
                context.last_vk_auto_report_text = latest_report_text
                logger.info(
                    "✓ VK auto report ready: report_messages=%s events=%s telegraph_links=%s",
                    len(seen_report_msg_ids),
                    len(aggregated_event_ids),
                    len(aggregated_tele_links),
                )
                for eid in aggregated_event_ids[:10]:
                    if eid not in log_links_seen_for_eid:
                        raise AssertionError(
                            "В отчётах VK auto нет кликабельной ссылки на лог "
                            "(ожидали deep-link через start=log_<id>).\n"
                            f"event_id={eid}"
                        )
                return

            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не найден валидный отчёт VK auto import с событиями и ссылками "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}. "
            f"Состояние целевых постов: {last_target_state}"
        )

    run_async(context, _wait())


@then("я жду первый отчёт VK auto import с событиями и ссылками")
def step_wait_vk_auto_first_report_has_unified_event_blocks(context):
    """Wait for the first VK auto-import report message with actionable links.

    Unlike the full-step, this does NOT require the whole VK queue to finish.
    It's intended for mass E2E scenarios where draining the queue can exceed
    timeout limits, but we still want a "live" signal with valid report blocks.
    """

    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_VK_AUTO_FIRST_REPORT_TIMEOUT_SEC", str(12 * 60)))
        poll_sec = float(os.getenv("E2E_VK_AUTO_REPORT_POLL_SEC", "3"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=80)
            last_preview = [(m.text or "").replace("\n", " ")[:140] for m in messages[:8]]
            _raise_on_bot_ui_errors(messages, baseline_id=baseline_id)

            for msg in reversed(messages):
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                text = msg.text or ""
                if not text:
                    continue
                if "Лог:" not in text or "ICS:" not in text:
                    continue
                if "Smart Update (детали событий)" not in text:
                    continue
                if ("Созданные события" not in text) and ("Обновлённые события" not in text):
                    continue

                ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
                if not ids:
                    continue
                tele_links = re.findall(r"https://telegra\.ph/[a-zA-Z0-9_-]+", text)
                if not tele_links:
                    continue
                if not re.search(r"^\s*ICS:\s+\S+", text, flags=re.MULTILINE):
                    continue
                if not re.search(r"^\s*Лог:\s+.*?/log\s+\d+.*$", text, flags=re.MULTILINE):
                    continue
                if not re.search(r"^\s*Факты:\s+.*Иллюстрации:\s+", text, flags=re.MULTILINE):
                    continue

                # Ensure /log is a clickable deep-link (Telegram won't include args in command click targets).
                urls = []
                for ent in (getattr(msg, "entities", None) or []):
                    u = getattr(ent, "url", None)
                    if u:
                        urls.append(str(u))
                for eid in ids[:10]:
                    want = f"start=log_{eid}"
                    if not any(want in u for u in urls):
                        raise AssertionError(
                            "В отчёте нет кликабельной ссылки на лог "
                            "(ожидали deep-link через start=log_<id>).\n"
                            f"event_id={eid}"
                        )

                context.last_response = msg
                context.last_vk_auto_report_event_ids = list(dict.fromkeys(ids))
                context.last_vk_auto_report_text = text
                context.telegraph_links = list(dict.fromkeys(tele_links))
                logger.info(
                    "✓ VK auto first report ready: events=%s telegraph_links=%s",
                    len(context.last_vk_auto_report_event_ids),
                    len(context.telegraph_links),
                )
                return

            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не найден первый валидный отчёт VK auto import с событиями и ссылками "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@then("я жду сообщение прогресса VK auto import")
def step_wait_vk_auto_progress_message(context):
    """Wait for at least one VK auto-import progress message (e.g. '13/87')."""

    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_VK_AUTO_PROGRESS_TIMEOUT_SEC", "180"))
        poll_sec = float(os.getenv("E2E_VK_AUTO_PROGRESS_POLL_SEC", "2"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        progress_re = re.compile(r"Разбираю VK пост\s+(\d+)\s*/\s*(\d+|\?)", re.IGNORECASE)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=50)
            last_preview = [(m.text or "").replace("\n", " ")[:140] for m in messages[:8]]
            _raise_on_bot_ui_errors(messages, baseline_id=baseline_id)
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                text = (msg.text or "").strip()
                if not text:
                    continue
                m = progress_re.search(text)
                if not m:
                    continue
                cur = int(m.group(1))
                total_raw = m.group(2)
                total = None if total_raw == "?" else int(total_raw)
                if cur <= 0:
                    continue
                if total is not None and total <= 0:
                    continue
                logger.info("✓ VK auto progress: %s/%s", cur, total_raw)
                return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            f"Не найдено сообщение прогресса VK auto import за {timeout_sec}с. "
            f"Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@then("я жду завершение VK auto import")
def step_wait_vk_auto_finished_message(context):
    """Wait for final completion summary of VK auto import."""

    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_VK_AUTO_FINISH_TIMEOUT_SEC", str(90 * 60)))
        poll_sec = float(os.getenv("E2E_VK_AUTO_FINISH_POLL_SEC", "2.5"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=60)
            last_preview = [(m.text or "").replace("\n", " ")[:160] for m in messages[:10]]
            _raise_on_bot_ui_errors(messages, baseline_id=baseline_id)
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                text = (msg.text or "").strip()
                if not text:
                    continue
                if text.startswith("🏁 VK auto import завершён"):
                    context.last_response = msg
                    logger.info("✓ VK auto finished summary received")
                    return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не найдено сообщение завершения VK auto import "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@then("я жду унифицированный отчёт Smart Update для Telegram Monitoring")
def step_wait_tg_smart_update_report(context):
    """Wait for Smart Update per-event report generated by Telegram monitoring."""

    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_TG_SMART_UPDATE_REPORT_TIMEOUT_SEC", str(15 * 60)))
        poll_sec = float(os.getenv("E2E_TG_SMART_UPDATE_REPORT_POLL_SEC", "2.0"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=80)
            last_preview = [(m.text or "").replace("\n", " ")[:140] for m in messages[:8]]
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                text = msg.text or ""
                if not text:
                    continue
                if "Smart Update (детали событий)" not in text:
                    continue
                if ("Созданные события" not in text) and ("Обновлённые события" not in text):
                    continue
                if "Лог:" not in text or "ICS:" not in text:
                    continue
                if not re.search(r"^\s*Факты:\s+", text, flags=re.MULTILINE):
                    continue

                ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
                if not ids:
                    continue
                context.last_response = msg
                context.last_tg_su_report_text = text
                context.last_tg_su_report_event_ids = list(dict.fromkeys(ids))
                context.last_tg_su_report_telegraph_links = list(
                    dict.fromkeys(re.findall(r"https://telegra\\.ph/[a-zA-Z0-9_-]+", text))
                )
                logger.info(
                    "✓ TG Smart Update report ready: events=%s telegraph_links=%s",
                    len(context.last_tg_su_report_event_ids),
                    len(context.last_tg_su_report_telegraph_links),
                )
                return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не найден валидный отчёт Smart Update для Telegram Monitoring "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


def _request_source_logs_for_event_ids(context, event_ids: list[int], *, label: str) -> None:
    unique_ids = [int(x) for x in dict.fromkeys(int(i) for i in event_ids if int(i) > 0)]
    if not unique_ids:
        raise AssertionError(f"{label}: нет event_id для запроса /log")

    timeout_sec = int(os.getenv("E2E_LOG_TIMEOUT_SEC", "90"))
    poll_sec = float(os.getenv("E2E_LOG_POLL_SEC", "1.5"))
    opened: list[int] = []

    for event_id in unique_ids:
        try:
            target_url = _resolve_event_telegraph_url(event_id)
        except Exception:
            target_url = ""

        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)

        async def _send():
            await context.client.client.send_message(context.bot_entity, f"/log {event_id}")

        run_async(context, _send())

        async def _wait():
            import asyncio

            deadline = time.monotonic() + float(timeout_sec)
            while time.monotonic() < deadline:
                messages = await context.client.client.get_messages(context.bot_entity, limit=40)
                for msg in messages:
                    if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                        continue
                    text = msg.text or ""
                    if "Лог источников" not in text:
                        continue
                    if target_url and target_url not in text:
                        continue
                    context.last_response = msg
                    return
                await asyncio.sleep(poll_sec)
            raise AssertionError(
                f"{label}: не получили лог источников для event_id={event_id} "
                f"за {timeout_sec}с"
            )

        run_async(context, _wait())
        opened.append(event_id)

    context.last_opened_log_event_ids = opened
    logger.info("✓ %s: /log открыт для всех событий: %s", label, opened)


@when("я запрашиваю /log для последнего события")
@then("я запрашиваю /log для последнего события")
def step_request_log_for_last_event(context):
    """Open /log for the last event card (used for menu add-event smoke scenarios)."""
    text = getattr(getattr(context, "last_response", None), "text", None)
    event_id = _event_id_from_card_text(text) or getattr(context, "last_event_id", None)
    if not event_id:
        raise AssertionError("Не удалось определить event_id из последнего сообщения (ожидали карточку события с 'id: N')")
    context.last_event_id = int(event_id)
    _request_source_logs_for_event_ids(context, [int(event_id)], label="last event")


@when("я запрашиваю /log для первого события из последнего VK отчёта")
def step_request_log_for_first_vk_report_event(context):
    report_text = getattr(context, "last_vk_auto_report_text", None) or (
        context.last_response.text if context.last_response else ""
    )
    if not report_text:
        raise AssertionError("Нет текста последнего VK отчёта")
    ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", report_text)]
    if not ids:
        raise AssertionError(f"В VK отчёте не найден event_id. Текст:\n{report_text}")
    event_id = int(ids[0])
    context.last_event_id = event_id
    baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)

    async def _send():
        await context.client.client.send_message(context.bot_entity, f"/log {event_id}")

    run_async(context, _send())

    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_LOG_TIMEOUT_SEC", "60"))
        poll_sec = float(os.getenv("E2E_LOG_POLL_SEC", "1.5"))
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=30)
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                text = msg.text or ""
                if "Лог источников" in text and f"/log {event_id}" not in text:
                    context.last_response = msg
                    return
            await asyncio.sleep(poll_sec)
        raise AssertionError(f"Не получили лог источников для event_id={event_id} за {timeout_sec}с")

    run_async(context, _wait())


@when("я запрашиваю /log для каждого события из последнего VK отчёта")
def step_request_log_for_each_vk_report_event(context):
    ids = list(getattr(context, "last_vk_auto_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_vk_auto_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
    _request_source_logs_for_event_ids(context, ids, label="VK auto import")


@when("я запрашиваю /log для каждого события из последнего Telegram отчёта")
def step_request_log_for_each_tg_report_event(context):
    ids = list(getattr(context, "last_tg_su_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_tg_su_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
    _request_source_logs_for_event_ids(context, ids, label="Telegram monitoring")


@when("я запрашиваю /log для первого события из последнего Telegram отчёта")
def step_request_log_for_first_tg_report_event(context):
    ids = list(getattr(context, "last_tg_su_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_tg_su_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text)]
    if not ids:
        raise AssertionError("В Telegram Smart Update отчёте не найден event_id для /log")
    _request_source_logs_for_event_ids(context, [int(ids[0])], label="Telegram monitoring")


@then("в отчёте /parse есть Telegraph и /log для событий")
def step_parse_report_has_unified_event_blocks(context):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    if not text or "Парсинг источников завершен" not in text:
        raise AssertionError("Ожидали сообщение '/parse' с текстом 'Парсинг источников завершен'")
    if "Smart Update (детали событий)" not in text:
        raise AssertionError(f"В отчёте /parse нет секции Smart Update (детали событий). Текст:\n{text}")
    if not re.search(r"^\s*Источник:\s+https?://\S+", text, flags=re.MULTILINE):
        raise AssertionError(f"В отчёте /parse нет строки Источник: https://... Текст:\n{text}")
    if not re.search(r"^\s*Лог:\s+.*?/log\s+\d+.*$", text, flags=re.MULTILINE):
        raise AssertionError(f"В отчёте /parse нет строки Лог: /log <id>. Текст:\n{text}")
    if not re.search(r"^\s*ICS:\s+", text, flags=re.MULTILINE):
        raise AssertionError(f"В отчёте /parse нет строки ICS:. Текст:\n{text}")
    if not re.search(r"^\s*Факты:\s+", text, flags=re.MULTILINE):
        raise AssertionError(f"В отчёте /parse нет строки Факты:. Текст:\n{text}")

    urls = []
    for ent in (getattr(msg, "entities", None) or []):
        u = getattr(ent, "url", None)
        if u:
            urls.append(str(u))
    if not any("telegra.ph/" in u for u in urls) and ("telegra.ph/" not in text):
        raise AssertionError(
            "В отчёте /parse не найдена ссылка на Telegraph (ожидали telegra.ph в заголовке события).\n"
            f"Текст:\n{text}"
        )
    if not any("start=log_" in u for u in urls):
        raise AssertionError(
            "В отчёте /parse нет кликабельной ссылки на лог (ожидали deep-link через start=log_<id>).\n"
            f"Текст:\n{text}"
        )
    logger.info("✓ /parse report содержит Telegraph + /log + ICS")


@then('событие содержит новые изображения из поста "{post_url}"')
def step_event_has_images_from_post(context, post_url):
    """Verify telegraph pages include poster images from the Telegram post."""
    import time

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
        match = re.search(r"t\.me/([^/]+)/([0-9]+)", post_url)
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
    poster_urls: list[str] = []
    for p in posters:
        url = p.get("supabase_url") or p.get("catbox_url")
        if url and url not in poster_urls:
            poster_urls.append(url)
    if not poster_urls:
        raise AssertionError(f"Нет URL афиш (catbox_url/supabase_url) для поста {post_url}")
    links = getattr(context, "telegraph_links", None)
    if not links:
        text = context.last_response.text if context.last_response else ""
        links = re.findall(r"https://telegra\\.ph/[a-zA-Z0-9_-]+", text)
    if not links:
        # Telegraph build is async; resolve the event via DB and wait a bit for telegraph_url to appear.
        wait_sec = int(os.getenv("E2E_TELEGRAPH_WAIT_SEC", "240"))
        event_id = getattr(context, "control_event_id", None)
        if not event_id:
            # Best-effort: resolve by post URL from event_source (same logic as open card step).
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
                if row:
                    event_id = int(row["event_id"])
            finally:
                conn.close()

        if event_id:
            deadline = time.time() + wait_sec
            while time.time() < deadline:
                db_path = _db_path()
                conn = sqlite3.connect(db_path, timeout=30)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT telegraph_url FROM event WHERE id = ?", (int(event_id),))
                    row = cur.fetchone()
                finally:
                    conn.close()
                url = (row[0] if row else None) if row is not None else None
                url = (url or "").strip()
                if url.startswith("https://telegra.ph/"):
                    links = [url]
                    break
                time.sleep(3)
    if not links:
        raise AssertionError("Нет Telegraph ссылок для проверки изображений")

    async def _verify():
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for link in links:
                async with session.get(link, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    html = await resp.text()
                    if any(url in html for url in poster_urls):
                        logger.info("✓ Найдена картинка из поста на странице %s", link)
                        return
        raise AssertionError("URL афиш из поста не найден на Telegraph страницах")

    run_async(context, _verify())


@then("событие содержит изображения из контрольного поста")
def step_event_has_images_from_control_post(context):
    url = getattr(context, "control_post_url", None)
    if not url:
        raise AssertionError("Не задан control_post_url (сначала выберите контрольный пост)")
    step_event_has_images_from_post(context, url)


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
    if not re.search(r"(https?://|t\.me/|telegram|vk|site|parser|manual|bot)", text, re.IGNORECASE):
        raise AssertionError("В логе нет указания источника")

    # Thesis marker: bullet list
    if not re.search(r"[•*\-]\s+\S+", text):
        raise AssertionError("В логе нет фактов (ожидаются bullets)")

    logger.info("✓ Лог источников содержит дату/время, источник и факты")


@then("в логе источников есть факт с URL афиши")
def step_source_log_has_poster_url_fact(context):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""

    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if (not text) or ("Лог источников" not in text):
        async def _fetch():
            messages = await context.client.client.get_messages(context.bot_entity, limit=40)
            target_url = None
            if event_id:
                # Match the log to the current event to avoid picking an older log message.
                import sqlite3

                conn = sqlite3.connect(_db_path(), timeout=30)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT telegraph_url, telegraph_path FROM event WHERE id=?", (int(event_id),))
                    row = cur.fetchone()
                finally:
                    conn.close()
                if row:
                    target_url = (row[0] or "").strip() or (
                        f"https://telegra.ph/{(row[1] or '').lstrip('/')}" if (row[1] or "").strip() else None
                    )
            for candidate in messages:
                t = candidate.text or ""
                if "Лог источников" not in t:
                    continue
                if target_url and target_url not in t:
                    continue
                return t
            for candidate in messages:
                if candidate.text and "Лог источников" in candidate.text:
                    return candidate.text
            return ""

        text = run_async(context, _fetch())

    if not re.search(r"(?:Добавлена афиша):\s+https?://\S+", text or ""):
        raise AssertionError("В логе источников не найден факт 'Добавлена афиша: <URL>'")
    logger.info("✓ Лог источников содержит факт с URL афиши")


@then('в логе источников количество фактов "{fact_prefix}" равно "{count}"')
def step_source_log_fact_count_equals(context, fact_prefix, count):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )

    if (not text) or ("Лог источников" not in text):
        async def _fetch():
            messages = await context.client.client.get_messages(context.bot_entity, limit=40)
            target_url = None
            if event_id:
                import sqlite3

                conn = sqlite3.connect(_db_path(), timeout=30)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT telegraph_url, telegraph_path FROM event WHERE id=?", (int(event_id),))
                    row = cur.fetchone()
                finally:
                    conn.close()
                if row:
                    target_url = (row[0] or "").strip() or (
                        f"https://telegra.ph/{(row[1] or '').lstrip('/')}" if (row[1] or "").strip() else None
                    )
            for candidate in messages:
                t = candidate.text or ""
                if "Лог источников" not in t:
                    continue
                if target_url and target_url not in t:
                    continue
                return t
            for candidate in messages:
                if candidate.text and "Лог источников" in candidate.text:
                    return candidate.text
            return ""

        text = run_async(context, _fetch())

    if not text:
        raise AssertionError("Лог источников пуст или не получен")
    prefix = (fact_prefix or "").strip()
    if not prefix:
        raise AssertionError("Пустой fact_prefix")

    actual = len(re.findall(re.escape(prefix), text, flags=re.IGNORECASE))
    expected = int(count)
    if actual != expected:
        raise AssertionError(
            f"Ожидали фактов '{prefix}' = {expected}, получили {actual}. Текст лога:\n{text}"
        )
    logger.info("✓ Количество фактов '%s' = %s", prefix, expected)


@then('в логе источников для источника "{source_url}" количество фактов "{fact_prefix}" минимум "{count}"')
def step_source_log_fact_count_min_for_source(context, source_url, fact_prefix, count):
    want = int(count)
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    text = _fetch_source_log_text(context, int(event_id) if event_id else None, limit=60)
    if not text or "Лог источников" not in text:
        raise AssertionError("Лог источников пуст или не получен")

    url = (source_url or "").strip()
    if not url:
        raise AssertionError("source_url пуст")
    prefix = (fact_prefix or "").strip()
    if not prefix:
        raise AssertionError("fact_prefix пуст")

    lines = (text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if url in line:
            start = i
            break
    if start is None:
        raise AssertionError(f"В логе источников не найден блок для источника: {url}")

    block = []
    for line in lines[start + 1 :]:
        if re.match(r"^\\d{4}-\\d{2}-\\d{2}\\s+\\d{2}:\\d{2}\\b", line.strip()):
            break
        block.append(line)

    got = 0
    for l in block:
        ll = l.strip()
        if not ll.startswith("•"):
            continue
        if prefix.lower() in ll.lower():
            got += 1
    if got < want:
        sample = "\n".join(block[:120])
        raise AssertionError(
            f"Ожидали минимум {want} фактов '{prefix}' для источника {url}, получили {got}.\n"
            f"Блок:\n{sample}"
        )
    logger.info("✓ В логе источников для источника %s фактов '%s' >= %s (got=%s)", url, prefix, want, got)


@then('в базе у открытого события количество афиш равно "{count}"')
def step_db_open_event_poster_count_equals(context, count):
    event_id = getattr(context, "last_event_id", None) or _event_id_from_card_text(
        getattr(getattr(context, "last_response", None), "text", None)
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки eventposter")
    db_path = _db_path()
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM eventposter WHERE event_id = ?", (int(event_id),))
        actual = int(cur.fetchone()[0])
    finally:
        conn.close()
    expected = int(count)
    if actual != expected:
        raise AssertionError(f"Ожидали eventposter для event_id={event_id}: {expected}, получили {actual}")
    logger.info("✓ eventposter count=%s для event_id=%s", expected, event_id)


@then('в базе у открытого события количество афиш из поста "{post_url}" равно "{count}"')
def step_db_open_event_poster_count_from_post_equals(context, post_url, count):
    run_id = getattr(context, "last_monitor_run_id", None)
    if not run_id:
        raise AssertionError("Не найден run_id последнего мониторинга (нужен для сопоставления афиш)")
    data = _load_tg_results(run_id)
    if not data:
        raise AssertionError(f"Не удалось загрузить telegram_results.json для run_id={run_id}")
    poster_urls = {_norm_url_for_compare(u) for u in _extract_posters_from_tg_results(data, post_url) if _norm_url_for_compare(u)}
    if not poster_urls:
        raise AssertionError(f"Не найдены афиши в telegram_results.json для поста {post_url}")

    event_id = getattr(context, "last_event_id", None) or _event_id_from_card_text(
        getattr(getattr(context, "last_response", None), "text", None)
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки eventposter")

    db_path = _db_path()
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT catbox_url, supabase_url FROM eventposter WHERE event_id = ?",
            (int(event_id),),
        )
        urls: set[str] = set()
        for catbox_url, supabase_url in cur.fetchall():
            if catbox_url:
                urls.add(_norm_url_for_compare(str(catbox_url)))
            if supabase_url:
                urls.add(_norm_url_for_compare(str(supabase_url)))
        actual = len(urls & poster_urls)
    finally:
        conn.close()

    expected = int(count)
    if actual != expected:
        raise AssertionError(
            f"Ожидали eventposter(match post={post_url}) для event_id={event_id}: {expected}, получили {actual}"
        )
    logger.info("✓ eventposter from post count=%s для event_id=%s", expected, event_id)


@then("в логе источников есть ссылка на Telegraph")
def step_source_log_has_telegraph_link(context):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""

    if not text or "Лог источников" not in text:
        async def _fetch():
            messages = await context.client.client.get_messages(context.bot_entity, limit=20)
            for candidate in messages:
                if candidate.text and "Лог источников" in candidate.text:
                    return candidate.text
            return ""

        text = run_async(context, _fetch())

    if not re.search(r"^\s*(?:📄\s*)?Telegraph:\s+https?://\S+", text, flags=re.MULTILINE):
        raise AssertionError("В логе источников не найдена ссылка на Telegraph (ожидали '📄 Telegraph: https://...')")
    logger.info("✓ Лог источников содержит ссылку на Telegraph")


@then("я дожидаюсь выполнения задач обновления Telegraph для событий из последнего VK отчёта")
def step_drain_telegraph_jobs_for_last_vk_report(context):
    ids = list(getattr(context, "last_vk_auto_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_vk_auto_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text or "")]
    if not ids:
        raise AssertionError("Не удалось определить event_id из последнего VK отчёта")

    async def _drain():
        import main as main_mod
        from db import Database
        from main import JobTask

        class _NoopBot:
            async def send_message(self, *_args, **_kwargs):
                return None

        db = Database(_db_path())
        await db.init()
        allowed = {JobTask.ics_publish, JobTask.telegraph_build}
        # A few drain rounds are enough to execute dependent ics_publish -> telegraph_build.
        for _ in range(4):
            progressed = False
            for eid in ids:
                before_url = _resolve_event_telegraph_url(int(eid))
                await main_mod.run_event_update_jobs(
                    db,
                    _NoopBot(),
                    event_id=int(eid),
                    allowed_tasks=allowed,
                )
                after_url = _resolve_event_telegraph_url(int(eid))
                if after_url != before_url:
                    progressed = True
            if not progressed:
                break

    run_async(context, _drain())

    refreshed_links: list[str] = []
    for eid in ids:
        try:
            url = _resolve_event_telegraph_url(int(eid))
        except Exception:
            continue
        if url and url not in refreshed_links:
            refreshed_links.append(url)
    if refreshed_links:
        context.telegraph_links = refreshed_links
    logger.info(
        "✓ Telegraph jobs drained for VK report events=%s links=%s",
        ids,
        len(getattr(context, "telegraph_links", []) or []),
    )


@then("первая Telegraph страница содержит хотя бы одно изображение")
def step_first_telegraph_page_contains_image(context):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    links = list(getattr(context, "telegraph_links", None) or [])
    if not links:
        links = re.findall(r"https://telegra\.ph/[a-zA-Z0-9_-]+", text or "")
    if not links:
        raise AssertionError("Не нашли ссылку на Telegraph для проверки изображений")
    first = links[0]

    async def _fetch():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(first, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    raise AssertionError(f"Telegraph {first} вернул статус {resp.status}")
                return await resp.text()

    html = run_async(context, _fetch())
    if not re.search(r"<img\b", html or "", flags=re.IGNORECASE):
        raise AssertionError(f"На странице {first} не найдено изображений (<img>)")
    logger.info("✓ Telegraph содержит изображение: %s", first)


@then('первая Telegraph страница содержит блок "{marker}"')
def step_first_telegraph_page_contains_marker(context, marker):
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    links = list(getattr(context, "telegraph_links", None) or [])
    if not links:
        links = re.findall(r"https://telegra\.ph/[a-zA-Z0-9_-]+", text or "")
    if not links:
        raise AssertionError("Не нашли ссылку на Telegraph для проверки блока")
    first = links[0]

    async def _fetch():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(first, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    raise AssertionError(f"Telegraph {first} вернул статус {resp.status}")
                return await resp.text()

    html = run_async(context, _fetch())
    marker_norm = _normalize_search_text(marker)
    body_norm = _normalize_search_text(_html_to_text(html))
    if marker_norm not in body_norm:
        raise AssertionError(f"На странице {first} не найден блок '{marker}'")
    logger.info("✓ Telegraph содержит блок '%s': %s", marker, first)


@then('каждая Telegraph страница из последнего VK отчёта содержит блок "{marker}"')
def step_each_vk_report_telegraph_page_contains_marker(context, marker):
    """Assert every Telegraph page referenced by last VK report has marker."""
    ids = list(getattr(context, "last_vk_auto_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_vk_auto_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text or "")]
    if not ids:
        raise AssertionError("Не удалось определить event_id из последнего VK отчёта")

    urls: list[str] = []
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        for eid in ids:
            row = cur.execute(
                "SELECT telegraph_url, telegraph_path FROM event WHERE id = ?",
                (int(eid),),
            ).fetchone()
            if not row:
                raise AssertionError(f"Событие id={eid} не найдено в БД")
            url = (row[0] or "").strip()
            if not url:
                path = (row[1] or "").strip().lstrip("/")
                if not path:
                    raise AssertionError(f"У события id={eid} не заполнены telegraph_url/telegraph_path")
                url = f"https://telegra.ph/{path}"
            if url not in urls:
                urls.append(url)
    finally:
        conn.close()

    async def _verify():
        html_pages = await _fetch_telegraph_pages(urls)
        missing: list[str] = []
        needle = _normalize_search_text(marker)
        for url, html in zip(urls, html_pages):
            body_norm = _normalize_search_text(_html_to_text(html))
            if needle not in body_norm:
                missing.append(url)
        if missing:
            sample = "\n".join(missing[:20])
            raise AssertionError(
                f"Маркер '{marker}' не найден на Telegraph страницах из VK отчёта ({len(missing)} шт):\n{sample}"
            )

    run_async(context, _verify())
    logger.info("✓ Все Telegraph страницы VK отчёта содержат блок '%s' (count=%s)", marker, len(urls))


@then("на каждой Telegraph странице из последнего VK отчёта присутствует основной текст события")
def step_vk_report_telegraph_pages_contain_event_description_text(context):
    """Ensure Telegraph pages contain the stored event.description (not only the summary/footer)."""
    ids = list(getattr(context, "last_vk_auto_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_vk_auto_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text or "")]
    if not ids:
        raise AssertionError("Не удалось определить event_id из последнего VK отчёта")

    db_path = _db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        rows = []
        for eid in ids:
            row = cur.execute(
                "SELECT id, description, telegraph_url, telegraph_path FROM event WHERE id = ?",
                (int(eid),),
            ).fetchone()
            if not row:
                raise AssertionError(f"Событие id={eid} не найдено в БД")
            rows.append(row)
    finally:
        conn.close()

    urls: list[str] = []
    expectations: dict[str, str] = {}
    for (eid, desc, tele_url, tele_path) in rows:
        url = (tele_url or "").strip()
        if not url:
            path = (tele_path or "").strip().lstrip("/")
            if not path:
                raise AssertionError(f"У события id={eid} не заполнены telegraph_url/telegraph_path")
            url = f"https://telegra.ph/{path}"
        if url not in urls:
            urls.append(url)
        expectations[url] = str(desc or "")

    # Keep this deterministic: we don't need perfect NLP, only to catch cases where the body was
    # truncated after an internal <hr> and only the summary/footer remained.
    stop = {
        "который", "которая", "которые", "чтобы", "когда", "где", "это", "этот", "эта",
        "эти", "его", "ее", "их", "для", "про", "при", "или", "а", "но", "и", "в", "на",
        "по", "из", "от", "до", "как", "что", "также",
    }

    def _pick_keywords(text: str) -> list[str]:
        norm = _normalize_search_text(text)
        words = re.findall(r"[a-zа-яё]{5,}", norm, flags=re.IGNORECASE)
        uniq: list[str] = []
        seen: set[str] = set()
        for w in words:
            wl = w.lower()
            if wl in stop:
                continue
            if wl in seen:
                continue
            seen.add(wl)
            uniq.append(wl)
            if len(uniq) >= 10:
                break
        return uniq

    async def _verify():
        html_pages = await _fetch_telegraph_pages(urls)
        missing: list[str] = []
        for url, html in zip(urls, html_pages):
            expected_desc = expectations.get(url, "")
            keywords = _pick_keywords(expected_desc)
            if not keywords:
                # If description is empty/too short in DB, skip: this step is about missing body on Telegraph.
                continue
            body_norm = _normalize_search_text(_html_to_text(html))
            present = sum(1 for w in keywords if w in body_norm)
            # Need at least a few keywords to be present to consider the main body rendered.
            if present < min(3, len(keywords)):
                missing.append(url)
        if missing:
            sample = "\n".join(missing[:20])
            raise AssertionError(
                "На некоторых Telegraph страницах не найден основной текст события "
                f"(нет ключевых слов из event.description):\n{sample}"
            )

    run_async(context, _verify())
    logger.info("✓ Telegraph страницы VK отчёта содержат основной текст события (count=%s)", len(urls))


@then("для событий из последнего VK отчёта с афишами Telegraph содержит изображения")
def step_vk_report_events_with_posters_have_images(context):
    """If event has posters in DB, corresponding Telegraph page must contain <img>."""
    ids = list(getattr(context, "last_vk_auto_report_event_ids", []) or [])
    if not ids:
        text = getattr(context, "last_vk_auto_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        ids = [int(x) for x in re.findall(r"\(id=(\d+)\)", text or "")]
    if not ids:
        raise AssertionError("Не удалось определить event_id из последнего VK отчёта")

    check_items: list[tuple[int, str, int]] = []
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        for eid in ids:
            poster_count = int(
                cur.execute(
                    "SELECT COUNT(*) FROM eventposter WHERE event_id = ?",
                    (int(eid),),
                ).fetchone()[0]
            )
            if poster_count <= 0:
                continue
            row = cur.execute(
                "SELECT telegraph_url, telegraph_path FROM event WHERE id = ?",
                (int(eid),),
            ).fetchone()
            if not row:
                raise AssertionError(f"Событие id={eid} не найдено в БД")
            url = (row[0] or "").strip()
            if not url:
                path = (row[1] or "").strip().lstrip("/")
                if not path:
                    raise AssertionError(f"У события id={eid} не заполнены telegraph_url/telegraph_path")
                url = f"https://telegra.ph/{path}"
            check_items.append((int(eid), url, poster_count))
    finally:
        conn.close()

    if not check_items:
        logger.info("✓ В VK отчёте нет событий с афишами, проверка <img> пропущена")
        return


def _extract_telegraph_links_from_text(text: str) -> list[str]:
    links = re.findall(r"https://telegra\\.ph/[a-zA-Z0-9_-]+", text or "")
    return list(dict.fromkeys(links))


@then("на страницах Telegraph из последнего VK отчёта есть изображения Catbox в формате WEBP")
def step_vk_report_telegraph_catbox_webp(context):
    links = list(getattr(context, "telegraph_links", []) or [])
    if not links:
        report_text = getattr(context, "last_vk_auto_report_text", None) or (
            context.last_response.text if context.last_response else ""
        )
        links = _extract_telegraph_links_from_text(report_text)
    if not links:
        raise AssertionError("Не найдены Telegraph ссылки для проверки (после VK отчёта)")

    max_pages = int(os.getenv("E2E_WEBP_CHECK_MAX_PAGES", "8"))
    max_images = int(os.getenv("E2E_WEBP_CHECK_MAX_IMAGES", "12"))
    links = links[:max_pages]

    async def _verify():
        html_pages = await _fetch_telegraph_pages(links)
        catbox_urls: list[str] = []
        seen: set[str] = set()
        for html in html_pages:
            for u in sorted(_extract_catbox_urls(html)):
                if u in seen:
                    continue
                seen.add(u)
                catbox_urls.append(u)
        if not catbox_urls:
            raise AssertionError(
                "Не найдены изображения Catbox на Telegraph страницах из VK отчёта "
                f"(pages={len(links)})."
            )
        catbox_urls = catbox_urls[:max_images]
        for url in catbox_urls:
            ok, dbg = await _probe_image_webp(url)
            if not ok:
                raise AssertionError(f"Catbox изображение не WEBP или недоступно: url={url} {dbg}")

    run_async(context, _verify())
    logger.info("✓ VK report: catbox изображения доступны и WEBP (pages=%s)", len(links))


@then("на страницах Telegraph из последнего Telegram отчёта есть изображения Catbox в формате WEBP")
def step_tg_report_telegraph_catbox_webp(context):
    links = list(getattr(context, "last_tg_su_report_telegraph_links", []) or [])
    if not links:
        report_text = getattr(context, "last_tg_su_report_text", None) or ""
        links = _extract_telegraph_links_from_text(report_text)
    if not links:
        raise AssertionError("Не найдены Telegraph ссылки для проверки (после Telegram Smart Update отчёта)")

    max_pages = int(os.getenv("E2E_WEBP_CHECK_MAX_PAGES", "8"))
    max_images = int(os.getenv("E2E_WEBP_CHECK_MAX_IMAGES", "12"))
    links = links[:max_pages]

    async def _verify():
        html_pages = await _fetch_telegraph_pages(links)
        catbox_urls: list[str] = []
        seen: set[str] = set()
        for html in html_pages:
            for u in sorted(_extract_catbox_urls(html)):
                if u in seen:
                    continue
                seen.add(u)
                catbox_urls.append(u)
        if not catbox_urls:
            raise AssertionError(
                "Не найдены изображения Catbox на Telegraph страницах из Telegram отчёта "
                f"(pages={len(links)})."
            )
        catbox_urls = catbox_urls[:max_images]
        for url in catbox_urls:
            ok, dbg = await _probe_image_webp(url)
            if not ok:
                raise AssertionError(f"Catbox изображение не WEBP или недоступно: url={url} {dbg}")

    run_async(context, _verify())
    logger.info("✓ Telegram report: catbox изображения доступны и WEBP (pages=%s)", len(links))

    async def _verify():
        import aiohttp

        missing: list[str] = []
        async with aiohttp.ClientSession() as session:
            for eid, url, poster_count in check_items:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        missing.append(f"id={eid} status={resp.status} url={url}")
                        continue
                    html = await resp.text()
                if not re.search(r"<img\b", html or "", flags=re.IGNORECASE):
                    missing.append(f"id={eid} posters={poster_count} url={url}")
        if missing:
            sample = "\n".join(missing[:20])
            raise AssertionError(
                "События с афишами опубликованы без изображений на Telegraph:\n"
                f"{sample}"
            )

    run_async(context, _verify())
    logger.info("✓ Для событий VK отчёта с афишами Telegraph содержит изображения (count=%s)", len(check_items))


@when('я фиксирую снимок состояния события как "{label}"')
@then('я фиксирую снимок состояния события как "{label}"')
def step_capture_event_stage_snapshot(context, label):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для фиксации snapshot")
    snap = _capture_stage_snapshot(context, label=label, event_id=int(event_id))
    logger.info(
        "✓ Снимок этапа '%s' сохранён: sources=%s semantic_facts=%s text_facts=%s file=%s",
        snap.get("label"),
        len((snap.get("source_log") or {}).get("sections") or []),
        int((snap.get("source_log") or {}).get("semantic_facts_total") or 0),
        int((snap.get("source_log") or {}).get("text_facts_total") or 0),
        snap.get("artifact_json"),
    )


@then('в снимке "{label}" у источника "{source_hint}" минимум "{count}" текстовых тезисов')
def step_snapshot_source_has_min_text_facts(context, label, source_hint, count):
    snapshots = getattr(context, "stage_snapshots", None) or {}
    key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label or "").strip()).strip("_").lower() or "stage"
    snap = snapshots.get(key)
    if not snap:
        raise AssertionError(f"Snapshot '{label}' не найден. Сначала выполните шаг фиксации snapshot.")
    hint = _normalize_search_text(source_hint)
    sections = (snap.get("source_log") or {}).get("sections") or []
    matched = []
    for sec in sections:
        src = _normalize_search_text((sec.get("source") or "") + " " + (sec.get("url") or ""))
        if hint and hint not in src:
            continue
        matched.append(sec)
    if not matched:
        raise AssertionError(
            f"В snapshot '{label}' не найден источник '{source_hint}'. "
            f"Секции: {[s.get('source') for s in sections]}"
        )
    actual = sum(len(sec.get("text_facts") or []) for sec in matched)
    expected = int(count)
    if actual < expected:
        raise AssertionError(
            f"В snapshot '{label}' у источника '{source_hint}' текстовых тезисов {actual}, ожидали >= {expected}. "
            f"artifact={snap.get('artifact_json')}"
        )
    logger.info("✓ Snapshot '%s': у источника '%s' текстовых тезисов=%s (>= %s)", label, source_hint, actual, expected)


@then('в снимке "{label}" есть архивная Telegraph ссылка')
def step_snapshot_has_archive_link(context, label):
    snapshots = getattr(context, "stage_snapshots", None) or {}
    key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label or "").strip()).strip("_").lower() or "stage"
    snap = snapshots.get(key)
    if not snap:
        raise AssertionError(f"Snapshot '{label}' не найден")
    url = str((((snap.get("telegraph") or {}).get("snapshot_url")) or "")).strip()
    if not url.startswith("https://telegra.ph/"):
        raise AssertionError(
            f"В snapshot '{label}' нет архивной telegra.ph ссылки. artifact={snap.get('artifact_json')}"
        )
    logger.info("✓ Snapshot '%s' содержит архивную Telegraph ссылку: %s", label, url)


@then("из выбранного поста созданы все будущие события как отдельные события без дублей")
def step_selected_post_future_events_created_without_duplicates(context):
    post_url = (getattr(context, "control_post_url", None) or "").strip()
    username = (getattr(context, "control_post_username", None) or "").strip()
    message_id = int(getattr(context, "control_post_message_id", 0) or 0)
    if not post_url or not username or not message_id:
        raise AssertionError("Контрольный пост не выбран")

    run_id = getattr(context, "last_monitor_run_id", None)
    if not run_id:
        raise AssertionError("Не найден run_id последнего мониторинга")
    data = _load_tg_results(run_id)
    if not data:
        raise AssertionError(f"Не удалось загрузить telegram_results.json для run_id={run_id}")
    msg = _find_message_in_tg_results(data, post_url)
    if not msg:
        raise AssertionError(f"Пост {post_url} не найден в telegram_results.json")

    all_events = [ev for ev in (msg.get("events") or []) if isinstance(ev, dict)]
    if not all_events:
        raise AssertionError(f"В сообщении {post_url} нет events в telegram_results.json")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT default_location FROM telegram_source WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        default_location = str((row["default_location"] if row else "") or "").strip()

        cur.execute(
            """
            SELECT DISTINCT e.id, e.title, e.date, e.time, e.location_name
            FROM event_source es
            JOIN event e ON e.id = es.event_id
            WHERE es.source_url = ?
               OR (es.source_chat_username = ? AND es.source_message_id = ?)
               OR es.source_url LIKE ?
            ORDER BY e.id
            """,
            (post_url, username, int(message_id), f"%t.me/{username}/{int(message_id)}%"),
        )
        imported_rows = list(cur.fetchall())
    finally:
        conn.close()

    if not imported_rows:
        raise AssertionError(f"Не найдено импортированных событий из поста {post_url}")

    today = date.today()
    future_candidates: list[dict] = []
    past_candidates: list[dict] = []
    for ev in all_events:
        raw_date = str(ev.get("date") or "").split("..", 1)[0].strip()
        if not raw_date:
            continue
        try:
            parsed_date = date.fromisoformat(raw_date)
        except Exception:
            continue
        title = str(ev.get("title") or "").strip()
        location_name = str(ev.get("location_name") or "").strip() or default_location
        item = {
            "title": title,
            "date": raw_date,
            "location_name": location_name,
        }
        if parsed_date >= today:
            future_candidates.append(item)
        else:
            past_candidates.append(item)

    if not future_candidates:
        raise AssertionError(
            f"В посте {post_url} не найдено будущих событий для проверки (today={today.isoformat()})"
        )

    def _row_matches_candidate(row: sqlite3.Row, cand: dict) -> bool:
        row_date = str(row["date"] or "").split("..", 1)[0].strip()
        if row_date != str(cand["date"]):
            return False
        row_title = _norm_event_title_key(str(row["title"] or ""))
        cand_title = _norm_event_title_key(str(cand["title"] or ""))
        if not (row_title == cand_title or row_title in cand_title or cand_title in row_title):
            return False
        row_loc = _norm_location_key(str(row["location_name"] or ""))
        cand_loc = _norm_location_key(str(cand.get("location_name") or ""))
        if cand_loc and row_loc and cand_loc != row_loc and cand_loc not in row_loc and row_loc not in cand_loc:
            return False
        return True

    missing: list[str] = []
    duplicate_matches: list[str] = []
    matched_event_ids: set[int] = set()
    for cand in future_candidates:
        matches = [row for row in imported_rows if _row_matches_candidate(row, cand)]
        if not matches:
            missing.append(f"{cand['date']} | {cand['title']}")
            continue
        ids = {int(m["id"]) for m in matches}
        if len(ids) > 1:
            duplicate_matches.append(
                f"{cand['date']} | {cand['title']} -> ids={sorted(ids)}"
            )
        matched_event_ids.update(ids)

    if missing:
        raise AssertionError(
            "Не все будущие события из multi-event поста созданы отдельно:\n"
            + "\n".join(f"- {line}" for line in missing[:12])
        )
    if duplicate_matches:
        raise AssertionError(
            "Обнаружены дубли по будущим событиям из multi-event поста:\n"
            + "\n".join(f"- {line}" for line in duplicate_matches[:12])
        )

    imported_past: list[str] = []
    for cand in past_candidates:
        if any(_row_matches_candidate(row, cand) for row in imported_rows):
            imported_past.append(f"{cand['date']} | {cand['title']}")
    if imported_past:
        raise AssertionError(
            "Из multi-event поста импортированы события с прошедшими датами:\n"
            + "\n".join(f"- {line}" for line in imported_past[:12])
        )

    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row in imported_rows:
        row_date = str(row["date"] or "").split("..", 1)[0].strip()
        key = (
            _norm_event_title_key(str(row["title"] or "")),
            row_date,
            _norm_location_key(str(row["location_name"] or "")),
        )
        grouped.setdefault(key, []).append(int(row["id"]))
    structural_duplicates = {k: v for k, v in grouped.items() if len(set(v)) > 1}
    if structural_duplicates:
        lines = []
        for (title_k, row_date, loc_k), ids in list(structural_duplicates.items())[:12]:
            lines.append(f"{row_date} | {title_k} | {loc_k} -> ids={sorted(set(ids))}")
        raise AssertionError("Найдены структурные дубли событий:\n" + "\n".join(f"- {ln}" for ln in lines))

    logger.info(
        "✓ Multi-event post проверен: future_created=%s unique_events=%s past_skipped=%s post=%s",
        len(future_candidates),
        len(matched_event_ids),
        len(past_candidates),
        post_url,
    )


def _vk_wall_token_from_url(post_url: str) -> str:
    raw = (post_url or "").strip()
    match = re.search(r"(wall-?\d+_\d+)", raw)
    if not match:
        raise AssertionError(f"Некорректная VK ссылка поста: {post_url}")
    return match.group(1)


def _select_events_by_vk_post(
    post_url: str,
    *,
    date_value: str | None = None,
    time_value: str | None = None,
    location_hint: str | None = None,
) -> list[sqlite3.Row]:
    token = _vk_wall_token_from_url(post_url)
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.title, e.date, e.time, e.location_name, es.source_url
            FROM event_source es
            JOIN event e ON e.id = es.event_id
            WHERE es.source_url LIKE ?
            ORDER BY es.imported_at DESC, e.id DESC
            """,
            (f"%{token}%",),
        )
        rows = list(cur.fetchall())
    finally:
        conn.close()

    out: list[sqlite3.Row] = []
    want_date = str(date_value or "").split("..", 1)[0].strip() or None
    want_time = (time_value or "").strip() or None
    want_loc = _norm_location_key(location_hint) if location_hint else ""
    for row in rows:
        row_date = str((row["date"] or "")).split("..", 1)[0].strip()
        row_time = str(row["time"] or "").strip()
        row_loc = _norm_location_key(str(row["location_name"] or ""))
        if want_date and row_date != want_date:
            continue
        if want_time and row_time != want_time:
            continue
        if want_loc and want_loc not in row_loc:
            continue
        out.append(row)
    return out


@then('из VK поста "{post_url}" создано событие с заголовком содержащим "{title_hint}" на дату "{date}" и время "{time}"')
def step_vk_post_created_expected_event(context, post_url, title_hint, date, time):
    rows = _select_events_by_vk_post(post_url, date_value=date, time_value=time)
    if not rows:
        raise AssertionError(
            f"Не найдено событий из VK поста {post_url} на {date} {time}"
        )
    hint_norm = _normalize_search_text(title_hint)
    matched = []
    for row in rows:
        title_norm = _normalize_search_text(str(row["title"] or ""))
        if hint_norm and hint_norm not in title_norm:
            continue
        matched.append(row)
    if not matched:
        preview = [
            f"id={int(r['id'])} title={r['title']!r} date={r['date']} time={r['time']}"
            for r in rows[:8]
        ]
        raise AssertionError(
            f"Для VK поста {post_url} не найдено событие с title_hint='{title_hint}'.\n"
            f"Кандидаты: {preview}"
        )
    logger.info(
        "✓ VK post %s -> event id=%s title=%s",
        post_url,
        int(matched[0]["id"]),
        matched[0]["title"],
    )


@then('из VK постов "{post_a}" и "{post_b}" созданы разные события в "{location}" на дату "{date}" и время "{time}"')
def step_vk_posts_create_distinct_parallel_events(context, post_a, post_b, location, date, time):
    a_rows = _select_events_by_vk_post(
        post_a, date_value=date, time_value=time, location_hint=location
    )
    b_rows = _select_events_by_vk_post(
        post_b, date_value=date, time_value=time, location_hint=location
    )
    if not a_rows:
        raise AssertionError(
            f"Не найдено событий из поста {post_a} для {location} {date} {time}"
        )
    if not b_rows:
        raise AssertionError(
            f"Не найдено событий из поста {post_b} для {location} {date} {time}"
        )
    ids_a = {int(r["id"]) for r in a_rows}
    ids_b = {int(r["id"]) for r in b_rows}
    if ids_a & ids_b:
        raise AssertionError(
            "Посты библиотеки склеились в одно событие, ожидались разные события.\n"
            f"post_a={post_a} ids={sorted(ids_a)}\n"
            f"post_b={post_b} ids={sorted(ids_b)}"
        )
    logger.info(
        "✓ VK parallel posts kept separate: post_a_ids=%s post_b_ids=%s",
        sorted(ids_a),
        sorted(ids_b),
    )


@then('в снимке "{after_label}" источников больше чем в "{before_label}"')
def step_snapshot_sources_grew(context, after_label, before_label):
    snapshots = getattr(context, "stage_snapshots", None) or {}
    after_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(after_label or "").strip()).strip("_").lower() or "after"
    before_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(before_label or "").strip()).strip("_").lower() or "before"
    after = snapshots.get(after_key)
    before = snapshots.get(before_key)
    if not after or not before:
        raise AssertionError("Не найдены snapshot для сравнения источников")
    after_n = len((after.get("source_log") or {}).get("sections") or [])
    before_n = len((before.get("source_log") or {}).get("sections") or [])
    if after_n <= before_n:
        raise AssertionError(
            f"Ожидали рост числа источников: before={before_n}, after={after_n}. "
            f"after_artifact={after.get('artifact_json')}"
        )
    logger.info("✓ Источники выросли: %s -> %s", before_n, after_n)


@then('в снимке "{after_label}" текст Telegraph не короче чем в "{before_label}"')
def step_snapshot_telegraph_not_shorter(context, after_label, before_label):
    snapshots = getattr(context, "stage_snapshots", None) or {}
    after_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(after_label or "").strip()).strip("_").lower() or "after"
    before_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(before_label or "").strip()).strip("_").lower() or "before"
    after = snapshots.get(after_key)
    before = snapshots.get(before_key)
    if not after or not before:
        raise AssertionError("Не найдены snapshot для сравнения длины Telegraph текста")
    after_len = int(((after.get("telegraph") or {}).get("text_len")) or 0)
    before_len = int(((before.get("telegraph") or {}).get("text_len")) or 0)
    if after_len < before_len:
        raise AssertionError(
            f"После /parse текст Telegraph стал короче: before={before_len}, after={after_len}. "
            f"before={before.get('artifact_telegraph_text')} after={after.get('artifact_telegraph_text')}"
        )
    logger.info("✓ Telegraph текст не сократился: %s -> %s", before_len, after_len)


@then("все смысловые факты из лога источников есть на странице Telegraph события")
@then("на странице Telegraph события отражены все смысловые факты из лога источников")
@then("количество уникальных добавленных фактов из лога источников совпадает с количеством найденных на странице Telegraph")
def step_source_facts_are_present_on_telegraph(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки фактов на Telegraph")

    log_text = _fetch_source_log_text(context, int(event_id))
    if not log_text or "Лог источников" not in log_text:
        raise AssertionError("Не удалось получить сообщение 'Лог источников' для проверки фактов")

    facts = _extract_semantic_facts_from_log(log_text)
    if not facts:
        raise AssertionError("В логе источников не найдено смысловых фактов для сверки с Telegraph")

    telegraph_url, html = _fetch_telegraph_html_for_event(context, int(event_id))
    telegraph_text_norm = _normalize_search_text(_html_to_text(html))
    if not telegraph_text_norm:
        raise AssertionError(f"Telegraph страница пуста: {telegraph_url}")

    missing: list[str] = []
    for fact in facts:
        if not _fact_matches_telegraph_text(fact, telegraph_text_norm):
            missing.append(fact)

    if missing:
        sample = "\n".join(f"- {m}" for m in missing[:8])
        raise AssertionError(
            "Несовпадение факт-покрытия между логом источников и Telegraph.\n"
            f"Фактов (✅, уникальных) в логе: {len(facts)}\n"
            f"Найдено фактов на Telegraph (по поиску): {len(facts) - len(missing)}\n"
            f"Отсутствуют на Telegraph:\n{sample}\n"
            f"Telegraph: {telegraph_url}"
        )

    context.last_telegraph_url = telegraph_url
    context.last_telegraph_html = html
    context.last_telegraph_text = _html_to_text(html)
    logger.info(
        "✓ Fact coverage: log_unique_added=%s telegraph_matched=%s",
        len(facts),
        len(facts) - len(missing),
    )


def _get_telegraph_text_quality_context(context) -> tuple[int, str, str, str, list[str], int]:
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки качества текста Telegraph")

    html = getattr(context, "last_telegraph_html", None)
    url = getattr(context, "last_telegraph_url", None)
    if not html:
        url, html = _fetch_telegraph_html_for_event(context, int(event_id))
    text = _html_to_text(html)
    text_norm = _normalize_search_text(text)
    if len(text_norm) < 300:
        raise AssertionError(f"Текст Telegraph слишком короткий (<300 символов): {url}")

    paragraph_chunks = re.findall(
        r"(?is)<(?:p|h1|h2|h3|h4|li|blockquote)[^>]*>(.*?)</(?:p|h1|h2|h3|h4|li|blockquote)>",
        html,
    )
    paragraphs = []
    for chunk in paragraph_chunks:
        cleaned = _html_to_text(chunk).strip()
        if cleaned:
            paragraphs.append(cleaned)
    if len(paragraphs) < 3:
        # Fallback for malformed HTML: split by blank lines in plain text.
        fallback = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        paragraphs = fallback

    if len(paragraphs) < 3:
        raise AssertionError(f"Текст не разбит на небольшие абзацы (нашли {len(paragraphs)}): {url}")

    longest = max(len(p) for p in paragraphs)
    if longest > 900:
        raise AssertionError(f"Есть слишком длинный абзац ({longest} символов), текст тяжело читать: {url}")
    return int(event_id), (url or ""), html, text, paragraphs, longest


@then("текст Telegraph страницы события читабелен и форматирован")
@then("текст Telegraph структурирован в короткие абзацы и содержит форматирование")
def step_telegraph_text_is_readable_and_formatted(context):
    _event_id, url, html, _text, paragraphs, longest = _get_telegraph_text_quality_context(context)

    formatting_patterns = [
        r"(?is)<strong\b",
        r"(?is)<b\b",
        r"(?is)<em\b",
        r"(?is)<i\b",
        r"(?is)<h1\b",
        r"(?is)<h2\b",
        r"(?is)<h3\b",
        r"(?is)<h4\b",
        r"(?is)<ul\b",
        r"(?is)<ol\b",
        r"(?is)<li\b",
        r"(?is)<blockquote\b",
    ]
    if not any(re.search(p, html) for p in formatting_patterns):
        raise AssertionError(f"На странице нет выразимого форматирования (заголовки/выделения/списки): {url}")

    logger.info(
        "✓ Telegraph текст структурирован: абзацев=%s, max_paragraph=%s, есть форматирование",
        len(paragraphs),
        longest,
    )


@then("в тексте Telegraph нет нейросетевых клише")
def step_telegraph_text_no_neural_cliches(context):
    _event_id, url, _html, text, _paragraphs, _longest = _get_telegraph_text_quality_context(context)
    text_norm = _normalize_search_text(text)
    patterns = [
        r"\\bобеща\\w+\\s+стать\\b",
        r"\\bярк\\w+\\s+событ\\w+\\b",
        r"\\bзаметн\\w+\\s+событ\\w+\\b",
        r"\\bкультурн\\w+\\s+жизн\\w+\\b",
        r"\\bне\\s+остав\\w+\\s+равнодуш\\w+\\b",
        r"\\bнезабываем\\w+\\b",
        r"\\bуникальн\\w+\\s+возможн\\w+\\b",
    ]
    hits = [p for p in patterns if re.search(p, text_norm)]
    if hits:
        raise AssertionError(f"В Telegraph тексте найдены нейросетевые клише (patterns={hits}): {url}")
    logger.info("✓ В Telegraph тексте нет нейросетевых клише")


@then('в тексте Telegraph не разрывается "{needle}" на разные абзацы')
def step_telegraph_text_no_broken_phrase_split(context, needle):
    _event_id, url, html, _text, _paragraphs, _longest = _get_telegraph_text_quality_context(context)
    n = (needle or "").strip()
    if not n:
        raise AssertionError("needle пуст")
    parts = n.split()
    if len(parts) >= 2:
        a = re.escape(" ".join(parts[:-1]))
        b = re.escape(parts[-1])
        if re.search(rf"(?is){a}\\s*</p>\\s*<p[^>]*>\\s*{b}", html):
            raise AssertionError(f"Фраза разорвана между абзацами: '{n}' ({url})")
    logger.info("✓ В Telegraph тексте нет разрыва фразы '%s' между абзацами", n)


@then("стиль Telegraph текста нейтрально-профессиональный и без агрессивной рекламы")
def step_telegraph_text_style_neutral_and_professional(context):
    _event_id, url, _html, text, _paragraphs, _longest = _get_telegraph_text_quality_context(context)
    text_norm = _normalize_search_text(text)
    aggressive_patterns = [
        r"\bкупи\b",
        r"\bпокупай\b",
        r"\bуспей\b",
        r"\bжми\b",
        r"\bподписывайся\b",
        r"\bрозыгрыш\b",
        r"\bскидк",
        r"\bтолько сегодня\b",
    ]
    promo_hits = sum(1 for p in aggressive_patterns if re.search(p, text_norm))
    exclamation_count = text.count("!")
    if promo_hits >= 2 or exclamation_count > 6:
        raise AssertionError(
            f"Текст выглядит избыточно рекламным/агрессивным (promo_hits={promo_hits}, '!': {exclamation_count}): {url}"
        )
    logger.info(
        "✓ Telegraph стиль нейтрально-профессиональный: promo_hits=%s, exclamations=%s",
        promo_hits,
        exclamation_count,
    )


@then("в тексте Telegraph нет строк расписания других событий")
def step_telegraph_text_has_no_foreign_schedule_lines(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки чужих строк расписания")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT title, date, end_date FROM event WHERE id = ?", (int(event_id),)).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие id={event_id} не найдено")
    title = str(row[0] or "")
    event_date = str((row[1] or "")).split("..", 1)[0].strip()
    end_date = str((row[2] or "")).split("..", 1)[0].strip()

    text = getattr(context, "last_telegraph_text", None)
    if not text:
        _url, html = _fetch_telegraph_html_for_event(context, int(event_id))
        text = _html_to_text(html)

    allowed_tokens = set()
    for raw in [event_date, end_date]:
        if not raw:
            continue
        m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", raw)
        if not m:
            continue
        dd = int(m.group(3))
        mm = int(m.group(2))
        allowed_tokens.add(f"{dd:02d}.{mm:02d}")
        allowed_tokens.add(f"{dd}.{mm}")

    title_tokens = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", title.lower().replace("ё", "е")) if len(w) >= 4]
    bad_lines: list[str] = []
    sched_re = re.compile(r"^\s*(\d{1,2})[./](\d{1,2})\s*\|\s*(.+)$")
    for line in (text or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue
        m = sched_re.match(s)
        if not m:
            continue
        token = f"{int(m.group(1)):02d}.{int(m.group(2)):02d}"
        if allowed_tokens and token not in allowed_tokens:
            bad_lines.append(s)
            continue
        rhs = (m.group(3) or "").lower().replace("ё", "е")
        if title_tokens and not any(t in rhs for t in title_tokens[:5]):
            bad_lines.append(s)

    if bad_lines:
        sample = "\n".join(f"- {ln}" for ln in bad_lines[:8])
        raise AssertionError(
            "В Telegraph тексте найдены строки расписания, похожие на чужие события:\n"
            f"{sample}"
        )
    logger.info("✓ В Telegraph тексте нет строк расписания других событий")


@then('в логе источников для источника "{source_url}" факт про режиссёра помечен как дубль')
def step_source_log_director_fact_marked_duplicate(context, source_url):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки дубля факта про режиссёра")

    log_text = _fetch_source_log_text(context, int(event_id))
    sections = _parse_source_log_sections(log_text)
    url = (source_url or "").strip()
    if not url:
        raise AssertionError("source_url пустой")

    section = None
    for s in sections:
        if str(s.get("url") or "").strip() == url:
            section = s
            break
    if section is None:
        # Fallback: substring match (some sources may normalize trailing slashes).
        for s in sections:
            if url in str(s.get("url") or ""):
                section = s
                break
    if section is None:
        raise AssertionError(f"Не найден раздел лога для источника: {url}")

    facts_by_status = section.get("facts_by_status") or {}
    added = [str(x or "") for x in (facts_by_status.get("added") or [])]
    dup = [str(x or "") for x in (facts_by_status.get("duplicate") or [])]

    def _norm(v: str) -> str:
        return _normalize_search_text(v).replace("ё", "е")

    added_norm = "\n".join(_norm(x) for x in added)
    dup_norm = "\n".join(_norm(x) for x in dup)

    needle = "равинск"
    if needle not in dup_norm:
        raise AssertionError(
            "Ожидали, что факт про режиссёра (Равинский) помечен как дубль (↩️) в секции источника.\n"
            f"source={url}\n"
            f"duplicate_facts={dup}\n"
            f"added_facts={added}"
        )
    if needle in added_norm:
        raise AssertionError(
            "Факт про режиссёра (Равинский) ошибочно помечен как добавленный (✅), ожидали ↩️.\n"
            f"source={url}\n"
            f"added_facts={added}\n"
            f"duplicate_facts={dup}"
        )
    logger.info("✓ Факт про режиссёра помечен как дубль: source=%s", url)


@then("в тексте Telegraph нет утверждений о премьере")
def step_telegraph_text_has_no_premiere_claim(context):
    html = getattr(context, "last_telegraph_html", None)
    url = getattr(context, "last_telegraph_url", None)
    if not html:
        event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
            context, "last_event_id", None
        )
        if not event_id:
            raise AssertionError("Не удалось определить event_id для проверки Telegraph текста")
        url, html = _fetch_telegraph_html_for_event(context, int(event_id))
    text_norm = _normalize_search_text(_html_to_text(html)).replace("ё", "е")
    if re.search(r"\bпремьер", text_norm):
        raise AssertionError(f"В тексте Telegraph найдено утверждение о премьере: {url}")
    logger.info("✓ В Telegraph тексте нет утверждений о премьере")


@then("в тексте Telegraph есть цитата режиссёра (blockquote)")
def step_telegraph_text_has_director_quote_blockquote(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки цитаты в Telegraph")

    # Telegraph edits can be eventually consistent; keep this check strict but retry briefly.
    timeout_sec = int(os.getenv("E2E_TELEGRAPH_BLOCKQUOTE_TIMEOUT_SEC", "90"))
    pause_sec = float(os.getenv("E2E_TELEGRAPH_BLOCKQUOTE_RETRY_PAUSE_SEC", "3"))
    deadline = time.monotonic() + max(1, timeout_sec)
    last_url = None
    last_html = None

    while True:
        url, html = _fetch_telegraph_html_for_event(context, int(event_id))
        last_url, last_html = url, html
        context.last_telegraph_url = url
        context.last_telegraph_html = html

        blocks = re.findall(r"(?is)<blockquote\b[^>]*>(.*?)</blockquote>", html or "")
        if blocks:
            for b in blocks:
                txt = _normalize_search_text(_html_to_text(b)).replace("ё", "е")
                if "равинск" in txt or "егор" in txt:
                    logger.info("✓ В Telegraph тексте есть цитата режиссёра (blockquote)")
                    return

        if time.monotonic() >= deadline:
            if not blocks:
                raise AssertionError(f"На странице нет blockquote (ожидали цитату): {last_url}")
            sample = "\\n".join(_html_to_text(b).strip() for b in blocks[:3])
            raise AssertionError(
                "На странице есть blockquote, но не нашли атрибуцию режиссёра (Равинский/Егор).\n"
                f"url={last_url}\n"
                f"sample_blockquotes:\\n{sample}"
            )
        time.sleep(max(0.1, pause_sec))


@then("для страницы события Telegraph доступен telegram web preview (cached_page + photo)")
def step_event_telegraph_preview_ready(context):
    event_id = _event_id_from_card_text(getattr(context.last_response, "text", None)) or getattr(
        context, "last_event_id", None
    )
    if not event_id:
        raise AssertionError("Не удалось определить event_id для проверки Telegram preview")
    url = _resolve_event_telegraph_url(int(event_id))
    timeout_sec = int(os.getenv("E2E_WEB_PREVIEW_READY_TIMEOUT_SEC", "180"))
    retry_pause_sec = float(os.getenv("E2E_WEB_PREVIEW_RETRY_PAUSE_SEC", "5"))
    deadline = time.monotonic() + max(1, timeout_sec)
    last_state = "preview not requested yet"
    while True:
        step_check_telegram_web_preview(context, url)
        webpage = getattr(context, "last_webpage_preview", None)
        has_cached = bool(getattr(webpage, "cached_page", None)) if webpage else False
        has_photo = bool(getattr(webpage, "photo", None)) if webpage else False
        if has_cached and has_photo:
            break
        title = getattr(webpage, "title", None) if webpage else None
        last_state = f"title={title!r} photo={has_photo} cached_page={has_cached}"
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Telegram preview не стал полным за {timeout_sec}s: {last_state}"
            )
        logger.info("⏳ Ожидание полного Telegram preview: %s", last_state)
        time.sleep(max(0.5, retry_pause_sec))
    step_web_preview_has_cached_page(context)
    step_web_preview_has_photo(context)
    logger.info("✓ Telegram preview готов для страницы события: %s", url)


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


@when('я проверяю telegram web preview для ссылки "{url}"')
def step_check_telegram_web_preview(context, url):
    """Fetch Telegram web preview (MessageMediaWebPage) for a given URL.

    Note: cached_page/photo availability is decided by Telegram servers and can be impacted by
    preview image reachability (e.g. external hosts).
    """
    import asyncio
    from telethon.tl.types import MessageMediaWebPage

    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise AssertionError(f"Invalid URL: {url}")

    async def _check():
        sent = await context.client.client.send_message("me", url)
        # web preview metadata may arrive slightly later
        attach_wait_sec = int(os.getenv("E2E_WEB_PREVIEW_ATTACH_TIMEOUT_SEC", "20"))
        for _ in range(max(1, attach_wait_sec)):
            msg = await context.client.client.get_messages("me", ids=sent.id)
            media = getattr(msg, "media", None)
            if isinstance(media, MessageMediaWebPage):
                return media.webpage
            await asyncio.sleep(1.0)
        return None

    webpage = run_async(context, _check())
    if not webpage:
        raise AssertionError("Telegram did not attach web preview (MessageMediaWebPage)")
    context.last_webpage_preview = webpage
    logger.info(
        "✓ Telegram web preview: title=%r site=%r photo=%s cached_page=%s",
        getattr(webpage, "title", None),
        getattr(webpage, "site_name", None),
        bool(getattr(webpage, "photo", None)),
        bool(getattr(webpage, "cached_page", None)),
    )


@then("telegram web preview содержит cached_page")
def step_web_preview_has_cached_page(context):
    webpage = getattr(context, "last_webpage_preview", None)
    if not webpage:
        raise AssertionError("Нет last_webpage_preview (сначала вызовите шаг проверки web preview)")
    assert getattr(webpage, "cached_page", None), "cached_page отсутствует (Instant View не сформирован)"
    logger.info("✓ cached_page присутствует")


@then("telegram web preview содержит фото")
def step_web_preview_has_photo(context):
    webpage = getattr(context, "last_webpage_preview", None)
    if not webpage:
        raise AssertionError("Нет last_webpage_preview (сначала вызовите шаг проверки web preview)")
    assert getattr(webpage, "photo", None), "photo отсутствует (Telegram не смог скачать preview image)"
    logger.info("✓ photo присутствует")


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


@then('описание события "{title}" содержит "{needle}"')
def step_event_description_contains(context, title, needle):
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для проверки описания")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        row = conn.execute("SELECT description FROM event WHERE id = ?", (event_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие '{title}' не найдено в БД")
    desc = (row[0] or "").strip()
    if needle.lower() not in desc.lower():
        raise AssertionError(f"Фрагмент '{needle}' не найден в описании события '{title}'")


@then('в описании события "{title}" фрагмент "{needle}" встречается ровно "{count}" раз')
def step_event_description_fragment_count(context, title, needle, count):
    event_id = _fetch_event_id(context, title)
    if not event_id:
        raise AssertionError(f"Событие '{title}' не найдено для проверки описания")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        row = conn.execute("SELECT description FROM event WHERE id = ?", (event_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"Событие '{title}' не найдено в БД")
    desc = row[0] or ""
    actual = desc.lower().count((needle or "").lower())
    expected = int(count)
    if actual != expected:
        raise AssertionError(
            f"Ожидалось, что '{needle}' встречается {expected} раз, но встречается {actual} раз"
        )


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


def _festival_queue_autorun_status() -> str:
    raw = (os.getenv("ENABLE_FESTIVAL_QUEUE") or "").strip().lower()
    return "вкл" if raw in {"1", "true", "yes", "on"} else "выкл"


def _recent_bot_texts(context, limit: int = 40) -> list[str]:
    async def _fetch():
        msgs = await context.client.client.get_messages(context.bot_entity, limit=limit)
        out: list[str] = []
        for msg in msgs:
            text = (msg.text or "").strip()
            if text:
                out.append(text)
        return out

    return run_async(context, _fetch())


def _festival_assert_text(context) -> str:
    messages = _recent_bot_texts(context, limit=60)
    return "\n\n".join([*messages]).strip()


def _festival_queue_has_source_url(source_url: str) -> bool:
    wanted_norm = _norm_url_for_compare(source_url)
    if not wanted_norm:
        return False
    wanted_vk = _vk_ids_from_url(source_url)
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_url
            FROM festival_queue
            ORDER BY id DESC
            LIMIT 300
            """
        )
        rows = cur.fetchall() or []
    finally:
        conn.close()
    for row in rows:
        raw = row["source_url"] if isinstance(row, sqlite3.Row) else row[0]
        if wanted_vk:
            cand_vk = _vk_ids_from_url(str(raw or ""))
            if cand_vk and cand_vk == wanted_vk:
                return True
        candidate = _norm_url_for_compare(str(raw or ""))
        if candidate and candidate == wanted_norm:
            return True
    return False


def _vk_ids_from_url(post_url: str) -> tuple[int, int] | None:
    raw = (post_url or "").strip()
    m = re.search(r"wall-?(\d+)_([0-9]+)", raw)
    if not m:
        return None
    return abs(int(m.group(1))), int(m.group(2))


@then('я жду, что источник "{source_url}" появится в фестивальной очереди')
def step_wait_source_in_festival_queue(context, source_url):
    """Wait until Smart Update -> festival queue handoff is observable in DB."""
    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_FEST_QUEUE_ENQUEUE_TIMEOUT_SEC", str(10 * 60)))
        poll_sec = float(os.getenv("E2E_FEST_QUEUE_ENQUEUE_POLL_SEC", "2.0"))
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if _festival_queue_has_source_url(source_url):
                logger.info("✓ Источник появился в festival_queue: %s", source_url)
                return
            await asyncio.sleep(poll_sec)
        raise AssertionError(f"Источник не появился в festival_queue за {timeout_sec}с: {source_url}")

    run_async(context, _wait())


@then('в ответах бота есть сообщение о постановке источника в фестивальную очередь для "{source_url}"')
def step_assert_bot_mentions_festival_queue_for_source(context, source_url):
    text = _festival_assert_text(context)
    low = text.lower()
    if "фестивальн" not in low or "очеред" not in low:
        raise AssertionError("В ответах бота нет сообщений про фестивальную очередь")
    # URL may vary by domain (vk.com/vk.ru), so for VK match by wall-<gid>_<pid>.
    if "vk.com" in source_url.lower() or "vk.ru" in source_url.lower():
        vk_ids = _vk_ids_from_url(source_url)
        if not vk_ids:
            raise AssertionError(f"Некорректная VK ссылка поста: {source_url}")
        gid, pid = vk_ids
        token = f"wall-{gid}_{pid}"
        if token not in low:
            raise AssertionError(f"Не найдено упоминание VK-поста в сообщениях бота: {source_url}")
        return
    # For TG use direct substring match (canonical link is expected).
    if source_url not in text:
        raise AssertionError(f"Не найдено упоминание источника в сообщениях бота: {source_url}")


@then('в VK inbox пост "{post_url}" не импортирован как событие')
def step_assert_vk_inbox_not_imported_as_event(context, post_url):
    ids = _vk_ids_from_url(post_url)
    if not ids:
        raise AssertionError(f"Некорректная VK ссылка поста: {post_url}")
    group_id, post_id = ids
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT imported_event_id, status FROM vk_inbox WHERE group_id=? AND post_id=? LIMIT 1",
            (int(group_id), int(post_id)),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"VK inbox row не найден для поста: {post_url}")
    imported_event_id, status = row[0], row[1]
    if imported_event_id is not None:
        raise AssertionError(
            f"Ожидали, что festival-post не будет импортирован как событие, но imported_event_id={imported_event_id} status={status}"
        )
    logger.info("✓ VK inbox post not imported as event: status=%s", status)


@then('в festival_queue для источника "{source_url}" статус "{status}"')
def step_assert_festival_queue_status_for_source(context, source_url, status):
    wanted = _norm_url_for_compare(source_url)
    want_status = (status or "").strip().lower()
    if want_status not in {"pending", "running", "done", "error"}:
        raise AssertionError("status должен быть pending|running|done|error")

    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_FEST_QUEUE_STATUS_TIMEOUT_SEC", str(10 * 60)))
        poll_sec = float(os.getenv("E2E_FEST_QUEUE_STATUS_POLL_SEC", "2.0"))
        deadline = time.monotonic() + float(timeout_sec)

        while time.monotonic() < deadline:
            conn = sqlite3.connect(_db_path(), timeout=30)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT status, source_url
                    FROM festival_queue
                    ORDER BY id DESC
                    LIMIT 400
                    """
                )
                rows = cur.fetchall() or []
            finally:
                conn.close()

            for st, url in rows:
                url_norm = _norm_url_for_compare(str(url or ""))
                if not url_norm:
                    continue
                if url_norm != wanted:
                    # For VK allow match by wall id to ignore domain differences.
                    want_vk = _vk_ids_from_url(source_url)
                    cand_vk = _vk_ids_from_url(str(url or ""))
                    if want_vk and cand_vk and want_vk == cand_vk:
                        url_norm = wanted
                    else:
                        continue
                got = str(st or "").strip().lower()
                if got == want_status:
                    logger.info("✓ festival_queue status=%s for %s", got, source_url)
                    return
            await asyncio.sleep(poll_sec)

        raise AssertionError(f"Не найден festival_queue статус={want_status} для {source_url} за {timeout_sec}с")

    run_async(context, _wait())


def _extract_latest_festival_report(context) -> tuple[str | None, str | None, str | None]:
    texts = _recent_bot_texts(context, limit=80)
    all_text = "\n\n".join(texts)
    name_match = re.search(r"Фестиваль обновлён:\s*(.+)", all_text)
    url_match = re.search(r"Открыть страницу фестиваля:\s*(https?://\S+)", all_text)
    index_match = re.search(r"Открыть страницу Фестивали:\s*(https?://\S+)", all_text)
    name = name_match.group(1).strip() if name_match else None
    fest_url = url_match.group(1).strip() if url_match else None
    index_url = index_match.group(1).strip() if index_match else None
    if name:
        context.last_festival_name = name
    if fest_url:
        context.last_festival_url = fest_url
    if index_url:
        context.festivals_index_url = index_url
    return name, fest_url, index_url


def _extract_text_url_by_label_from_message(msg, label: str) -> str | None:
    """Extract entity url for a given link label from a Telegram message (best-effort)."""
    if not msg:
        return None
    text = (getattr(msg, "text", None) or "") or ""
    ents = getattr(msg, "entities", None) or []
    want = (label or "").strip().lower()
    if not want or not text or not ents:
        return None
    for ent in ents:
        url = getattr(ent, "url", None)
        if not url:
            continue
        try:
            off = int(getattr(ent, "offset", 0) or 0)
            ln = int(getattr(ent, "length", 0) or 0)
        except Exception:
            continue
        if ln <= 0:
            continue
        chunk = text[off : off + ln].strip().lower()
        if want in chunk:
            return str(url)
    return None


def _extract_recent_named_link(context, label: str, *, limit: int = 60) -> str | None:
    async def _fetch():
        msgs = await context.client.client.get_messages(context.bot_entity, limit=limit)
        for msg in msgs:
            url = _extract_text_url_by_label_from_message(msg, label)
            if url:
                return url
        return None

    return run_async(context, _fetch())


@step('я запоминаю верхнее сообщение в чате "{chat_ref}"')
@given('я запоминаю верхнее сообщение в чате "{chat_ref}"')
def step_remember_top_message_in_chat(context, chat_ref):
    async def _remember():
        try:
            entity = await context.client.client.get_entity(chat_ref)
            messages = await context.client.client.get_messages(entity, limit=1)
            if not hasattr(context, "chat_baselines"):
                context.chat_baselines = {}
            context.chat_baselines[str(chat_ref)] = int(getattr(messages[0], "id", 0) or 0) if messages else 0
        except Exception as exc:
            raise AssertionError(f"Не удалось прочитать baseline для чата {chat_ref}: {type(exc).__name__}: {exc}") from exc
    run_async(context, _remember())
    logger.info("✓ Запомнили baseline для чата %s", chat_ref)


@step('в чате "{chat_ref}" появилось новое сообщение с текстом "{text}"')
@then('в чате "{chat_ref}" появилось новое сообщение с текстом "{text}"')
def step_wait_new_message_in_chat(context, chat_ref, text):
    async def _wait():
        import asyncio

        entity = await context.client.client.get_entity(chat_ref)
        baseline_id = int(getattr(getattr(context, "chat_baselines", {}), "get", lambda _k, _d=0: 0)(str(chat_ref), 0) or 0)
        timeout_sec = int(os.getenv("E2E_WAIT_NEW_MESSAGE_TIMEOUT_SEC", "180"))
        deadline = time.monotonic() + float(timeout_sec)
        last_preview = []
        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(entity, limit=20)
            last_preview = [(m.text or "").replace("\n", " ")[:140] for m in messages[:8]]
            for msg in messages:
                mid = int(getattr(msg, "id", 0) or 0)
                if mid <= baseline_id:
                    continue
                payload = (msg.text or msg.caption or "")
                if payload and text.lower() in payload.lower():
                    if not hasattr(context, "chat_baselines"):
                        context.chat_baselines = {}
                    context.chat_baselines[str(chat_ref)] = mid
                    return
            await asyncio.sleep(2.0)
        raise AssertionError(
            f"В чате {chat_ref} не найдено новое сообщение с текстом {text!r}. "
            f"baseline_id={baseline_id}; recent={last_preview}"
        )
    run_async(context, _wait())
    logger.info("✓ В чате %s появилось новое сообщение с текстом %r", chat_ref, text)


@step('в чате "{chat_ref}" появилось новое медиа-сообщение')
@then('в чате "{chat_ref}" появилось новое медиа-сообщение')
def step_wait_new_media_message_in_chat(context, chat_ref):
    async def _wait():
        import asyncio

        entity = await context.client.client.get_entity(chat_ref)
        baseline_id = int(getattr(getattr(context, "chat_baselines", {}), "get", lambda _k, _d=0: 0)(str(chat_ref), 0) or 0)
        timeout_sec = int(os.getenv("E2E_WAIT_NEW_MESSAGE_TIMEOUT_SEC", "180"))
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(entity, limit=20)
            for msg in messages:
                mid = int(getattr(msg, "id", 0) or 0)
                if mid <= baseline_id:
                    continue
                if getattr(msg, "media", None):
                    if not hasattr(context, "chat_baselines"):
                        context.chat_baselines = {}
                    context.chat_baselines[str(chat_ref)] = mid
                    return
            await asyncio.sleep(2.0)
        raise AssertionError(f"В чате {chat_ref} не найдено новое медиа-сообщение после baseline_id={baseline_id}")
    run_async(context, _wait())
    logger.info("✓ В чате %s появилось новое медиа-сообщение", chat_ref)


def _fetch_festivals_index_from_db() -> str | None:
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT value FROM setting WHERE key IN ('festivals_index_url', 'fest_index_url') ORDER BY CASE key WHEN 'festivals_index_url' THEN 0 ELSE 1 END LIMIT 1"
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    value = str(row[0] or "").strip()
    return value or None


def _fetch_latest_festival_from_db() -> tuple[str | None, str | None]:
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name, telegraph_url, telegraph_path
            FROM festival
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None, None
    name = str(row["name"] or "").strip() or None
    url = str(row["telegraph_url"] or "").strip()
    if not url:
        path = str(row["telegraph_path"] or "").strip().lstrip("/")
        if path:
            url = f"https://telegra.ph/{path}"
    return name, (url or None)


def _fetch_url_html(context, url: str) -> str:
    async def _fetch():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    raise AssertionError(f"Страница {url} вернула статус {resp.status}")
                return await resp.text()

    return run_async(context, _fetch())


@then("я жду сообщения о старте обработки очереди")
def step_wait_festival_queue_started(context):
    async def _wait():
        import asyncio

        last = getattr(context, "last_response", None)
        last_txt = (getattr(last, "text", None) or "").strip()
        # `/fest_queue` immediately sends the "start" message; do not require a second one.
        if "Старт обработки фестивальной очереди" in last_txt:
            return

        timeout_sec = int(os.getenv("E2E_FEST_QUEUE_START_TIMEOUT_SEC", "120"))
        poll_sec = float(os.getenv("E2E_FEST_QUEUE_START_POLL_SEC", "1.5"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=40)
            last_preview = [(m.text or "").replace("\n", " ")[:160] for m in messages[:8]]
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                txt = (msg.text or "").strip()
                if "Старт обработки фестивальной очереди" in txt:
                    context.last_response = msg
                    return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не найдено сообщение о старте фестивальной очереди "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@given("я сбрасываю festival_queue статусы running/error в pending")
def step_reset_festival_queue_running_to_pending(context):
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        # Ensure the queue is runnable in a snapshot: some rows may have next_run_at in the future
        # due to backoff/scheduling. Bump all pending rows to "now".
        cur.execute(
            """
            UPDATE festival_queue
            SET next_run_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE status='pending'
            """
        )
        cur.execute(
            """
            UPDATE festival_queue
            SET status='pending',
                last_error=NULL,
                next_run_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE status IN ('running','error')
            """
        )
        conn.commit()
    finally:
        conn.close()


@given('я выставляю festival_queue статус "{status}" для источников:')
def step_set_festival_queue_status_for_sources(context, status):
    want_status = str(status or "").strip().lower()
    if want_status not in {"pending", "running", "done", "error"}:
        raise AssertionError(f"Неподдерживаемый статус festival_queue: {status}")
    table = getattr(context, "table", None)
    if table is None:
        raise AssertionError("Не передана таблица с url")

    urls: list[str] = []
    for row in table:
        url = str(row.get("url") or "").strip()
        if url:
            urls.append(url)
    if not urls:
        raise AssertionError("Пустой список url для обновления festival_queue")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, source_url FROM festival_queue")
        rows = cur.fetchall() or []

        updates: list[int] = []
        for source_url in urls:
            wanted = _norm_url_for_compare(source_url)
            want_vk = _vk_ids_from_url(source_url)
            matched_id: int | None = None
            for row_id, row_url in rows:
                row_norm = _norm_url_for_compare(str(row_url or ""))
                if row_norm == wanted:
                    matched_id = int(row_id)
                    break
                cand_vk = _vk_ids_from_url(row_norm)
                if want_vk and cand_vk and want_vk == cand_vk:
                    matched_id = int(row_id)
                    break
            if matched_id is None:
                raise AssertionError(f"Источник не найден в festival_queue: {source_url}")
            updates.append(matched_id)

        for row_id in updates:
            if want_status == "pending":
                cur.execute(
                    """
                    UPDATE festival_queue
                    SET status='pending',
                        attempts=0,
                        last_error=NULL,
                        next_run_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (row_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE festival_queue
                    SET status=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (want_status, row_id),
                )
        conn.commit()
    finally:
        conn.close()


@then("я вижу прогресс фестивальной очереди (не молчать)")
def step_wait_festival_queue_progress_visible(context):
    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_FEST_QUEUE_PROGRESS_TIMEOUT_SEC", "90"))
        poll_sec = float(os.getenv("E2E_FEST_QUEUE_PROGRESS_POLL_SEC", "1.5"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=60)
            last_preview = [(m.text or "").replace("\n", " ")[:160] for m in messages[:10]]
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                txt = (msg.text or "").strip()
                if "🎪 Фестивальная очередь:" not in txt:
                    continue
                context.fest_queue_progress_message_id = int(getattr(msg, "id", 0) or 0)
                context.fest_queue_progress_text = txt
                # Ensure it is not a single silent placeholder forever.
                if any(
                    marker in txt
                    for marker in (
                        "найдено элементов",
                        "нет элементов",
                        "Статус: running",
                        "завершено",
                        "✅ done",
                        "❌ error",
                    )
                ):
                    return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не увидел прогресс фестивальной очереди "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@then('в festival_queue для источников статус "{status}":')
def step_assert_festival_queue_status_for_sources_table(context, status):
    want_status = str(status or "").strip().lower()
    table = getattr(context, "table", None)
    if table is None:
        raise AssertionError("Не передана таблица с url")
    urls = []
    for row in table:
        url = str(row.get("url") or "").strip()
        if url:
            urls.append(url)
    if not urls:
        raise AssertionError("Пустой список url для проверки")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT status, source_url FROM festival_queue")
        rows = cur.fetchall() or []
    finally:
        conn.close()

    rows_norm: list[tuple[str, str]] = []
    for st, url in rows:
        rows_norm.append((str(st or "").strip().lower(), _norm_url_for_compare(str(url or ""))))

    missing: list[str] = []
    mismatched: list[tuple[str, str]] = []
    for source_url in urls:
        wanted = _norm_url_for_compare(source_url)
        matched_status = None
        for st_norm, url_norm in rows_norm:
            if url_norm == wanted:
                matched_status = st_norm
                break
            want_vk = _vk_ids_from_url(source_url)
            cand_vk = _vk_ids_from_url(url_norm)
            if want_vk and cand_vk and want_vk == cand_vk:
                matched_status = st_norm
                break
        if matched_status is None:
            missing.append(source_url)
        elif matched_status != want_status:
            mismatched.append((source_url, matched_status))

    if missing or mismatched:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if mismatched:
            parts.append(f"mismatched={mismatched}")
        raise AssertionError(
            f"festival_queue статусы не совпали с ожидаемым status={want_status}: " + "; ".join(parts)
        )


@then("я жду сообщения о завершении обработки очереди")
def step_wait_festival_queue_finished(context):
    async def _wait():
        import asyncio

        timeout_sec = int(os.getenv("E2E_FEST_QUEUE_FINISH_TIMEOUT_SEC", str(35 * 60)))
        poll_sec = float(os.getenv("E2E_FEST_QUEUE_FINISH_POLL_SEC", "2.5"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=60)
            last_preview = [(m.text or "").replace("\n", " ")[:160] for m in messages[:10]]
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                txt = (msg.text or "").strip()
                if "Завершение обработки фестивальной очереди" in txt:
                    context.last_response = msg
                    return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            "Не найдено сообщение о завершении фестивальной очереди "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@then('я жду сообщение отчёта фестивальной очереди с текстом "{text}"')
def step_wait_festival_queue_detail_text(context, text):
    """Wait for a queue detail message like 'Фестиваль обновлён: ...'."""
    async def _wait():
        import asyncio

        needle = (text or "").strip().lower()
        if not needle:
            raise AssertionError("Пустой текст ожидания")
        timeout_sec = int(os.getenv("E2E_FEST_QUEUE_DETAIL_TIMEOUT_SEC", str(10 * 60)))
        poll_sec = float(os.getenv("E2E_FEST_QUEUE_DETAIL_POLL_SEC", "2.0"))
        baseline_id = int(getattr(getattr(context, "last_response", None), "id", 0) or 0)
        deadline = time.monotonic() + float(timeout_sec)
        last_preview: list[str] = []

        while time.monotonic() < deadline:
            messages = await context.client.client.get_messages(context.bot_entity, limit=80)
            last_preview = [(m.text or "").replace("\n", " ")[:160] for m in messages[:10]]
            for msg in messages:
                if int(getattr(msg, "id", 0) or 0) <= baseline_id:
                    continue
                txt = (msg.text or "").strip().lower()
                if needle in txt:
                    context.last_response = msg
                    return
            await asyncio.sleep(poll_sec)

        raise AssertionError(
            f"Не найдено сообщение отчёта фестивальной очереди с текстом '{text}' "
            f"за {timeout_sec}с. Последние сообщения: {last_preview}"
        )

    run_async(context, _wait())


@then('в сообщении должна быть ссылка "{link_label}"')
def step_assert_message_has_named_link(context, link_label):
    if link_label.strip() == "Открыть страницу фестиваля":
        url = _extract_text_url_by_label_from_message(getattr(context, "last_response", None), link_label)
        if not url:
            url = _extract_recent_named_link(context, link_label)
        if url:
            context.last_festival_url = url
            return
        _extract_latest_festival_report(context)
        url = getattr(context, "last_festival_url", None)
        if not url:
            raise AssertionError("Не найдена ссылка 'Открыть страницу фестиваля'")
        return
    text = _festival_assert_text(context)
    if link_label not in text:
        raise AssertionError(f"Не найдено упоминание '{link_label}' в сообщениях")


@given("фестиваль успешно обновлён из очереди")
def step_given_festival_updated_from_queue(context):
    name, fest_url, index_url = _extract_latest_festival_report(context)
    if not name or not fest_url:
        db_name, db_url = _fetch_latest_festival_from_db()
        if db_name:
            context.last_festival_name = db_name
        if db_url:
            context.last_festival_url = db_url
    if not getattr(context, "last_festival_url", None):
        raise AssertionError("Не удалось определить ссылку страницы фестиваля после обработки очереди")
    if not getattr(context, "last_festival_name", None):
        raise AssertionError("Не удалось определить имя фестиваля после обработки очереди")
    if not index_url:
        db_index = _fetch_festivals_index_from_db()
        if db_index:
            context.festivals_index_url = db_index


@when('я открываю страницу "Все фестивали" в Telegraph')
def step_open_all_festivals_page(context):
    url = getattr(context, "festivals_index_url", None) or _fetch_festivals_index_from_db()
    if not url:
        raise AssertionError("Не найдена ссылка на страницу 'Все фестивали'")
    context.festivals_index_url = url
    context.last_opened_html = _fetch_url_html(context, url)


@then('в списке есть фестиваль "{festival_name}"')
def step_assert_named_festival_in_index(context, festival_name):
    name = str(festival_name or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля для проверки")

    timeout_sec = int(os.getenv("E2E_FESTIVALS_INDEX_WAIT_TIMEOUT_SEC", str(6 * 60)))
    poll_sec = float(os.getenv("E2E_FESTIVALS_INDEX_WAIT_POLL_SEC", "4"))
    deadline = time.monotonic() + float(timeout_sec)
    last_preview = ""

    while time.monotonic() < deadline:
        url = getattr(context, "festivals_index_url", None) or _fetch_festivals_index_from_db()
        if not url:
            raise AssertionError("Не найдена ссылка на страницу 'Все фестивали'")
        context.festivals_index_url = url
        html = _fetch_url_html(context, url)
        context.last_opened_html = html
        last_preview = re.sub(r"\\s+", " ", str(html or "")).strip()[:220]
        if name.lower() in str(html or "").lower():
            return
        time.sleep(poll_sec)

    raise AssertionError(
        f"Фестиваль '{name}' не найден в индексе за {timeout_sec}с. Превью HTML: {last_preview}"
    )


@when('я открываю фестиваль "{festival_name}" из страницы "Все фестивали" в Telegraph')
def step_open_named_festival_from_index(context, festival_name):
    name = str(festival_name or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        step_open_all_festivals_page(context)
        html = str(getattr(context, "last_opened_html", "") or "")
    anchors = list(
        re.finditer(
            r'<a\b[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<label>.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    best_href = ""
    for m in anchors:
        href = (m.group("href") or "").strip()
        label_raw = (m.group("label") or "").strip()
        label = re.sub(r"<[^>]+>", " ", label_raw)
        label = re.sub(r"\s+", " ", label).strip()
        if name.lower() in label.lower():
            best_href = href
            break
    if not best_href:
        raise AssertionError(f"Не найдена ссылка на фестиваль '{name}' на странице 'Все фестивали'")
    # Telegraph often uses root-relative links like `/Some-Page-01-01`.
    if best_href.startswith("/"):
        best_href = "https://telegra.ph" + best_href
    elif best_href.startswith("telegra.ph/"):
        best_href = "https://" + best_href
    context.last_festival_name = name
    context.last_festival_url = best_href
    context.last_opened_html = _fetch_url_html(context, best_href)


@then("страница должна содержать список мероприятий фестиваля")
def step_assert_festival_page_has_events_list(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница фестиваля")
    if "Мероприятия фестиваля" not in html:
        if "Расписание скоро обновим" in html:
            raise AssertionError("На странице фестиваля нет списка событий: показан заглушечный текст")
        raise AssertionError("На странице фестиваля не найден блок 'Мероприятия фестиваля'")
    if not re.search(r"<h4\b", html, flags=re.IGNORECASE):
        raise AssertionError("На странице фестиваля не найдено ни одного элемента события (<h4>)")


@then('в БД есть фестиваль "{festival_name}"')
def step_assert_db_has_festival(context, festival_name):
    name = str(festival_name or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM festival WHERE lower(name)=lower(?) LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"В БД не найден фестиваль '{name}'")


@then('в БД есть события фестиваля "{festival_name}" из VK источников:')
def step_assert_db_has_festival_events_for_vk_sources(context, festival_name):
    name = str(festival_name or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля")
    table = getattr(context, "table", None)
    if table is None:
        raise AssertionError("Не передана таблица с url")
    urls = []
    for row in table:
        url = str(row.get("url") or "").strip()
        if url:
            urls.append(url)
    if not urls:
        raise AssertionError("Пустой список url для проверки")

    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        missing: list[str] = []
        found: dict[str, dict[str, str]] = {}
        for url in urls:
            cur.execute(
                """
                SELECT title, telegraph_url, date, festival, source_post_url, source_vk_post_url
                FROM event
                WHERE lower(festival)=lower(?)
                  AND (source_post_url=? OR source_vk_post_url=?)
                ORDER BY date ASC, id ASC
                LIMIT 1
                """,
                (name, url, url),
            )
            ev = cur.fetchone()
            if not ev:
                missing.append(url)
                continue
            found[url] = {
                "title": str(ev["title"] or ""),
                "telegraph_url": str(ev["telegraph_url"] or ""),
                "date": str(ev["date"] or ""),
            }
    finally:
        conn.close()

    if missing:
        raise AssertionError(f"В БД нет событий фестиваля '{name}' для источников: {missing}")

    context.festival_expected_sources = list(urls)
    context.festival_events_by_source = found


@then("я должен увидеть новый фестиваль в списке")
def step_assert_festival_in_index(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    fest_name = str(getattr(context, "last_festival_name", "") or "").strip()
    if not html:
        raise AssertionError("Не загружена HTML-страница индекса фестивалей")
    if not fest_name:
        raise AssertionError("Не задано имя фестиваля для проверки в индексе")
    if fest_name.lower() not in html.lower():
        raise AssertionError(f"Фестиваль '{fest_name}' не найден в индексе")


@then("ссылка в списке должна вести на страницу фестиваля")
def step_assert_index_links_to_festival(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    fest_url = str(getattr(context, "last_festival_url", "") or "").strip()
    if not fest_url:
        raise AssertionError("Не задана ссылка страницы фестиваля")
    if fest_url in html:
        return
    from urllib.parse import urlparse

    parsed = urlparse(fest_url)
    path = str(parsed.path or "").strip()
    if path:
        if f'href="{path}"' in html:
            return
        if f'href="https://telegra.ph{path}"' in html:
            return
    raise AssertionError("В индексе нет ссылки на страницу фестиваля")


@when('я открываю ссылку "Открыть страницу фестиваля"')
def step_open_festival_page_link(context):
    url = str(getattr(context, "last_festival_url", "") or "").strip()
    if not url:
        url = _extract_recent_named_link(context, "Открыть страницу фестиваля") or ""
    if not url:
        _extract_latest_festival_report(context)
        url = str(getattr(context, "last_festival_url", "") or "").strip()
    if not url:
        raise AssertionError("Не найдена ссылка 'Открыть страницу фестиваля'")
    context.last_festival_url = url
    context.last_opened_html = _fetch_url_html(context, url)


@then("страница должна содержать обложку фестиваля")
def step_assert_festival_cover(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    if "<img" not in html.lower():
        raise AssertionError("На странице фестиваля не найдено ни одного изображения")


@then("страница должна содержать галерею иллюстраций")
def step_assert_festival_gallery(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    img_count = len(re.findall(r"<img\b", html, flags=re.IGNORECASE))
    expected = 2
    fest_name = str(getattr(context, "last_festival_name", "") or "").strip()
    if fest_name:
        conn = sqlite3.connect(_db_path(), timeout=30)
        try:
            cur = conn.cursor()
            cur.execute("SELECT photo_url, photo_urls FROM festival WHERE lower(name)=lower(?) LIMIT 1", (fest_name,))
            row = cur.fetchone()
        finally:
            conn.close()
        if row:
            import json as _json

            photo_url = str(row[0] or "").strip()
            raw = row[1]
            urls = []
            if isinstance(raw, str) and raw.strip():
                try:
                    urls = _json.loads(raw) or []
                except Exception:
                    urls = []
            if len([u for u in urls if str(u or "").strip()]) < 2:
                expected = 1 if photo_url else 1
    if img_count < expected:
        raise AssertionError(f"Ожидали иллюстрации (>= {expected}), найдено: {img_count}")


@when('я открываю страницу фестиваля "{festival_name}" в Telegraph из БД')
def step_open_festival_page_from_db(context, festival_name):
    name = str(festival_name or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT telegraph_url, telegraph_path FROM festival WHERE lower(name)=lower(?) ORDER BY id DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"В БД не найден фестиваль '{name}'")
    url = str(row["telegraph_url"] or "").strip()
    if not url:
        path = str(row["telegraph_path"] or "").strip().lstrip("/")
        if path:
            url = f"https://telegra.ph/{path}"
    if not url:
        raise AssertionError(f"У фестиваля '{name}' нет telegraph_url/telegraph_path")
    context.last_festival_name = name
    context.last_festival_url = url
    context.last_opened_html = _fetch_url_html(context, url)


@then("страница должна содержать короткое описание фестиваля в 1 абзац")
def step_assert_festival_one_paragraph_description(context):
    name = str(getattr(context, "last_festival_name", "") or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля (last_festival_name)")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT description FROM festival WHERE lower(name)=lower(?) ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
    finally:
        conn.close()
    desc = str((row[0] if row else "") or "").strip()
    if not desc:
        raise AssertionError(f"У фестиваля '{name}' отсутствует description в БД")
    if len(desc) < 40:
        raise AssertionError(f"У фестиваля '{name}' слишком короткое описание ({len(desc)} символов)")
    if "\n\n" in desc or desc.count("\n") > 1:
        raise AssertionError(f"У фестиваля '{name}' описание не похоже на 1 абзац (есть переносы строк)")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница фестиваля")
    needle = re.sub(r"\\s+", " ", desc).strip()
    import html as _html

    hay = re.sub(r"\\s+", " ", _html.unescape(html)).strip()
    if needle[:30].lower() not in hay.lower():
        # best-effort check: at least some chunk of the description should be visible
        raise AssertionError("Описание фестиваля из БД не найдено на странице Telegraph (best-effort check)")


@then("страница должна содержать ссылку на источник фестиваля из БД")
def step_assert_festival_source_link_present(context):
    name = str(getattr(context, "last_festival_name", "") or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля (last_festival_name)")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT source_url, source_post_url FROM festival WHERE lower(name)=lower(?) ORDER BY id DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    source_url = str((row[0] if row else "") or "").strip()
    source_post_url = str((row[1] if row else "") or "").strip()
    html = str(getattr(context, "last_opened_html", "") or "")
    if source_url:
        if source_url not in html:
            raise AssertionError(f"На странице фестиваля не найдена ссылка на источник: {source_url}")
        return
    # Source post links from social posts are internal provenance and should not be public.
    if source_post_url and source_post_url in html:
        raise AssertionError(f"На странице фестиваля не должна быть ссылка на пост-источник: {source_post_url}")


@then('страница должна содержать текст "{expected_text}"')
def step_assert_page_contains_text(context, expected_text):
    want = str(expected_text or "").strip()
    if not want:
        raise AssertionError("Пустой expected_text")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница")
    if want.lower() not in html.lower():
        preview = re.sub(r"\\s+", " ", html).strip()[:260]
        raise AssertionError(f"На странице не найден текст: {want}. Превью HTML: {preview}")


@then("страница должна содержать один из текстов:")
def step_assert_page_contains_any_text(context):
    table = getattr(context, "table", None)
    if table is None:
        raise AssertionError("Не передана таблица с text")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница")
    candidates: list[str] = []
    for row in table:
        txt = str(row.get("text") or row.get("phrase") or row.get("value") or "").strip()
        if txt:
            candidates.append(txt)
    if not candidates:
        raise AssertionError("Пустой список text для проверки")
    hay = html.lower()
    for txt in candidates:
        if txt.lower() in hay:
            return
    preview = re.sub(r"\\s+", " ", html).strip()[:260]
    raise AssertionError(
        f"На странице не найден ни один из текстов: {candidates}. Превью HTML: {preview}"
    )


@then('страница должна содержать ссылку "{expected_url}"')
def step_assert_page_contains_link(context, expected_url):
    url = str(expected_url or "").strip()
    if not url:
        raise AssertionError("Пустой expected_url")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница")
    if url not in html:
        raise AssertionError(f"На странице не найдена ссылка: {url}")


@then('страница не должна содержать текст "{forbidden_text}"')
def step_assert_page_not_contains_text(context, forbidden_text):
    forbid = str(forbidden_text or "").strip()
    if not forbid:
        raise AssertionError("Пустой forbidden_text")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница")
    import html as _html

    hay_raw = html.lower()
    hay_unescaped = _html.unescape(html).lower()
    if forbid.lower() in hay_raw or forbid.lower() in hay_unescaped:
        raise AssertionError(f"На странице найден запрещённый текст: {forbid}")


@then("на странице нет видимых служебных маркеров near-festivals")
def step_assert_no_visible_near_festivals_markers(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница")
    forbidden = [
        "&lt;&#33;-- near-festivals:start --&gt;",
        "&lt;&#33;-- near-festivals:end --&gt;",
        "&lt;!-- near-festivals:start --&gt;",
        "&lt;!-- near-festivals:end --&gt;",
        "<!-- near-festivals:start -->",
        "<!-- near-festivals:end -->",
        "<!-- FEST_NAV_START -->",
        "<!-- FEST_NAV_END -->",
        "<!--FEST_NAV_START-->",
        "<!--FEST_NAV_END-->",
    ]
    for f in forbidden:
        if f in html:
            raise AssertionError(f"На странице найден видимый служебный маркер: {f}")


@then("og:image на странице не ссылается на Catbox")
def step_assert_og_image_not_catbox(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница")
    m = re.search(r'property="og:image"\s+content="([^"]+)"', html, flags=re.IGNORECASE)
    if not m:
        raise AssertionError("Не найден meta og:image на странице")
    url = str(m.group(1) or "").strip()
    if "catbox.moe" in url.lower():
        raise AssertionError(f"og:image указывает на Catbox (Telegram кэш может не работать): {url}")


@then("описания событий на странице фестиваля короткие")
def step_assert_festival_event_digests_short(context):
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница фестиваля")
    import html as _html

    # Telegraph renders festival event list as <h4>title</h4><p>digest...</p>...
    pairs = re.findall(r"<h4[^>]*>.*?</h4>\s*<p>(.*?)</p>", html, flags=re.DOTALL | re.IGNORECASE)
    if not pairs:
        raise AssertionError("Не удалось найти блоки описаний событий (<h4>...<p>...) на странице фестиваля")
    max_words = 16
    offenders: list[tuple[int, int, str]] = []
    for p_html in pairs:
        # Convert <br> to newlines so we can isolate the first line (digest).
        p_html2 = re.sub(r"(?i)<br\s*/?>", "\n", p_html)
        txt = _html.unescape(re.sub(r"<[^>]+>", " ", p_html2))
        txt = txt.replace("\u200b", "")
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
        digest = next((ln for ln in lines if ln), "")
        if not digest:
            continue
        if digest.endswith("…") or digest.endswith("..."):
            offenders.append((999, len(digest), digest[:200]))
            continue
        words = [w for w in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", digest) if w.strip()]
        if len(words) > max_words:
            offenders.append((len(words), len(digest), digest[:200]))
    if offenders:
        sample = offenders[:3]
        raise AssertionError(
            "Найдены слишком длинные описания событий на странице фестиваля "
            f"(limit={max_words} слов). Примеры: {sample}"
        )


@then("страница должна содержать события фестиваля:")
def step_assert_festival_page_contains_event_titles(context):
    table = getattr(context, "table", None)
    if table is None:
        raise AssertionError("Не передана таблица с title")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница фестиваля")
    import html as _html

    hay_raw = html.lower()
    hay_unescaped = _html.unescape(html).lower()
    missing: list[str] = []
    for row in table:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        needle = title.lower()
        if needle not in hay_raw and needle not in hay_unescaped:
            missing.append(title)
    if missing:
        raise AssertionError(f"На странице фестиваля не найдены ожидаемые события: {missing}")


@then("страница должна содержать внешние ссылки фестиваля из БД (если есть)")
def step_assert_page_contains_external_links_from_db(context):
    name = str(getattr(context, "last_festival_name", "") or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля (last_festival_name)")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT website_url, vk_url, tg_url FROM festival WHERE lower(name)=lower(?) ORDER BY id DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise AssertionError(f"В БД не найден фестиваль '{name}'")
    website_url = str(row[0] or "").strip()
    vk_url = str(row[1] or "").strip()
    tg_url = str(row[2] or "").strip()
    html = str(getattr(context, "last_opened_html", "") or "")
    for u in (website_url, vk_url, tg_url):
        if u and u not in html:
            raise AssertionError(f"На странице фестиваля отсутствует внешняя ссылка из БД: {u}")


@then('в индексе фестивалей у фестиваля "{festival_name}" есть обложка из БД')
def step_assert_festival_index_has_cover_from_db(context, festival_name):
    name = str(festival_name or "").strip()
    if not name:
        raise AssertionError("Не задано имя фестиваля")
    html = str(getattr(context, "last_opened_html", "") or "")
    if not html:
        raise AssertionError("Не загружена HTML-страница индекса фестивалей")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT photo_url FROM festival WHERE lower(name)=lower(?) ORDER BY id DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    cover = str((row[0] if row else "") or "").strip()
    if not cover:
        raise AssertionError(f"У фестиваля '{name}' нет photo_url в БД")
    if cover not in html:
        raise AssertionError(
            f"В индексе фестивалей не найдена обложка фестиваля '{name}' (photo_url из БД)"
        )


@when("я запускаю фестивальную очередь для Telegram-источников")
def step_run_tg_festival_queue(context):
    step_send_command(context, "/fest_queue --source=tg")


@then("бот сообщает, что Telegram источники обрабатываются через Kaggle")
def step_assert_tg_kaggle_note(context):
    text = _festival_assert_text(context).lower()
    if "telegram источники обрабатываются через kaggle" not in text:
        raise AssertionError("Не найдено сообщение о Kaggle-обработке Telegram источников")


@then('по завершении обработки фестиваль появляется на странице "Фестивали"')
def step_assert_festival_visible_on_index(context):
    step_given_festival_updated_from_queue(context)
    step_open_all_festivals_page(context)
    step_assert_festival_in_index(context)


@given("у серии фестиваля есть несколько выпусков")
def step_given_series_has_multiple_editions(context):
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name, COUNT(*) AS cnt
            FROM festival
            GROUP BY name
            HAVING COUNT(*) >= 2
            ORDER BY cnt DESC, name
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            raise AssertionError("В базе нет серии фестиваля с несколькими выпусками")
        series_name = str(row["name"] or "").strip()
        cur.execute(
            """
            SELECT name, telegraph_url, telegraph_path
            FROM festival
            WHERE name = ?
            ORDER BY start_date DESC, id DESC
            LIMIT 1
            """,
            (series_name,),
        )
        current = cur.fetchone()
    finally:
        conn.close()
    if not current:
        raise AssertionError("Не найден текущий выпуск серии")
    url = str(current["telegraph_url"] or "").strip()
    if not url:
        path = str(current["telegraph_path"] or "").strip().lstrip("/")
        if path:
            url = f"https://telegra.ph/{path}"
    if not url:
        raise AssertionError("У текущего выпуска нет telegraph_url/telegraph_path")
    context.series_name = series_name
    context.last_festival_url = url


@when("я открываю страницу текущего выпуска")
def step_open_current_edition_page(context):
    step_open_festival_page_link(context)


@then("я вижу ссылки на другие выпуски серии")
def step_assert_series_links_present(context):
    series_name = str(getattr(context, "series_name", "") or "").strip()
    html = str(getattr(context, "last_opened_html", "") or "")
    if not series_name:
        raise AssertionError("Не задано имя серии для проверки ссылок")
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM festival WHERE name = ?", (series_name,))
        count = int((cur.fetchone() or [0])[0] or 0)
    finally:
        conn.close()
    if count < 2:
        raise AssertionError("Серия не содержит нескольких выпусков")
    if html.lower().count(series_name.lower()) < 2:
        raise AssertionError("На странице не видно ссылок/блока других выпусков серии")
