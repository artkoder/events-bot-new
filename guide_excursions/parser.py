from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import quote as url_quote


RU_MONTHS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "мая": 5,
    "май": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}

URL_RE = re.compile(r"https?://[^\s<>()]+", re.I)
USERNAME_RE = re.compile(r"(?<!\w)@([a-zA-Z0-9_]{4,64})")
PHONE_RE = re.compile(r"(?:(?:\+7|8)[\s(.-]*)?(?:\d[\s().-]*){10,11}")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?\b")
DATE_WORD_RE = re.compile(
    r"\b(\d{1,2})\s+(январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йяе]|июн[ьяе]?|июл[ьяе]?|август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])(?:\s+(\d{4}))?\b",
    re.I,
)
QUOTE_TITLE_RE = re.compile(r"[«\"]([^\"»\n]{4,160})[»\"]")

NON_SUMMARY_PATTERNS = (
    re.compile(r"\b(?:место встречи|место старта|время|запись|телефон|стоимость|цена|билеты|билет|выезд|сбор)\b", re.I),
    re.compile(r"^\s*(?:☎️|📞|🗓|⏰|📍|💸|🎟|✍️|🚍|🚌)\s*"),
)


@dataclass(slots=True)
class GuideParsedOccurrence:
    block_text: str
    canonical_title: str
    title_normalized: str
    date_iso: str | None
    time_text: str | None
    city: str | None
    meeting_point: str | None
    audience_fit: list[str]
    price_text: str | None
    booking_text: str | None
    booking_url: str | None
    channel_url: str | None
    status: str
    seats_text: str | None
    summary_one_liner: str | None
    digest_blurb: str | None
    digest_eligible: bool
    digest_eligibility_reason: str | None
    is_last_call: bool
    post_kind: str
    availability_mode: str
    guide_names: list[str]
    organizer_names: list[str]
    source_fingerprint: str


def collapse_ws(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_title_key(value: str | None) -> str:
    text = collapse_ws(value).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zа-яё0-9]+", " ", text, flags=re.I)
    text = re.sub(r"\b(?:экскурсия|экскурсии|прогулка|прогулки|тур|маршрут|авторская|пешеходная|поездка|путешествие)\b", " ", text, flags=re.I)
    text = collapse_ws(text)
    return text


def _line_cleanup(line: str) -> str:
    line = str(line or "").replace("\xa0", " ").strip()
    line = re.sub(r"^[•\-\u2022▪▫◾◽]+\s*", "", line)
    return line.strip()


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [collapse_ws(x) for x in raw if collapse_ws(x)]


def _month_from_word(word: str | None) -> int | None:
    raw = (word or "").strip().lower().replace("ё", "е")
    for key, month in RU_MONTHS.items():
        if raw.startswith(key):
            return month
    return None


def _parse_date(day: int, month: int, year: int, *, anchor: date) -> date | None:
    try:
        candidate = date(int(year), int(month), int(day))
    except Exception:
        return None
    if candidate < anchor - timedelta(days=180):
        try:
            candidate = date(candidate.year + 1, candidate.month, candidate.day)
        except Exception:
            pass
    elif candidate > anchor + timedelta(days=300):
        try:
            candidate = date(candidate.year - 1, candidate.month, candidate.day)
        except Exception:
            pass
    return candidate


def extract_date_iso(text: str, *, post_date: datetime) -> str | None:
    anchor = post_date.astimezone(timezone.utc).date()
    low = text.lower()
    if "завтра" in low:
        return (anchor + timedelta(days=1)).isoformat()
    if "сегодня" in low:
        return anchor.isoformat()

    match = DATE_NUMERIC_RE.search(text)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        year_raw = match.group(3)
        year = int(year_raw) if year_raw else anchor.year
        if year < 100:
            year += 2000
        candidate = _parse_date(day, month, year, anchor=anchor)
        if candidate:
            return candidate.isoformat()

    match = DATE_WORD_RE.search(text)
    if match:
        day = int(match.group(1))
        month = _month_from_word(match.group(2))
        year = int(match.group(3)) if match.group(3) else anchor.year
        if month:
            candidate = _parse_date(day, month, year, anchor=anchor)
            if candidate:
                return candidate.isoformat()
    return None


def extract_time_text(text: str) -> str | None:
    match = TIME_RE.search(text)
    if not match:
        return None
    return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"


def extract_meeting_point(text: str) -> str | None:
    patterns = [
        re.compile(r"(?:место встречи|точка встречи|сбор)\s*[:\-]\s*([^\n.]{6,160})", re.I),
        re.compile(r"(?:выезд|отправление)\s+от\s+([^\n.]{4,160})", re.I),
        re.compile(r"встречаемся\s+([^\n.]{6,120})", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return collapse_ws(match.group(1))
    return None


def extract_price_text(text: str) -> str | None:
    patterns = [
        re.compile(r"(?:стоимость|цена|участие|билет(?:ы)?)\s*[:\-]?\s*([^\n.]{2,80})", re.I),
        re.compile(r"(\d[\d\s.,]{0,10}\s*(?:₽|руб(?:\.|ля|лей)?|р\b))", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return collapse_ws(match.group(1))
    return None


def extract_seats_text(text: str) -> str | None:
    patterns = [
        re.compile(r"(осталось\s+\d+\s+[^\n.]{0,30})", re.I),
        re.compile(r"(есть\s+\w+\s+освободив\w*\s+мест\w*)", re.I),
        re.compile(r"(последн\w+\s+\d+\s+мест\w*)", re.I),
        re.compile(r"(мест[ао]?\s+ограничен\w*)", re.I),
        re.compile(r"(места\s+заканчивают\w*)", re.I),
        re.compile(r"(мест\s+нет[^\n.]{0,40})", re.I),
        re.compile(r"(лист\s+ожидания[^\n.]{0,40})", re.I),
        re.compile(r"(уже\s+набран\w*)", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return collapse_ws(match.group(1))
    return None


def extract_booking(text: str) -> tuple[str | None, str | None]:
    urls = URL_RE.findall(text or "")
    if urls:
        booking_text = "Запись по ссылке"
        return booking_text, urls[0]

    usernames = USERNAME_RE.findall(text or "")
    if usernames:
        uname = usernames[0]
        return f"@{uname}", f"https://t.me/{url_quote(uname)}"

    phone = PHONE_RE.search(text or "")
    if phone:
        digits = re.sub(r"[^\d+]", "", phone.group(0))
        return collapse_ws(phone.group(0)), f"tel:{digits}"

    booking_patterns = [
        re.compile(r"(?:запись|записаться|бронирова\w+|бронь)\s*(?:по|в|через)?\s*[:\-]?\s*([^\n.]{4,160})", re.I),
        re.compile(r"(?:в\s+личку|в\s+лс|пишите\s+в\s+личку)", re.I),
    ]
    for pattern in booking_patterns:
        match = pattern.search(text)
        if not match:
            continue
        if match.lastindex:
            candidate = collapse_ws(match.group(1))
            if len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]", candidate)) < 3:
                continue
            return candidate, None
        return collapse_ws(match.group(0)), None
    return None, None


def extract_audience_fit(text: str) -> list[str]:
    low = text.lower()
    out: list[str] = []
    checks = [
        ("семьям", ("семь", "с детьми", "семейн")),
        ("детям", ("дет", "школьник", "квест-экскурсия")),
        ("школьным группам", ("школьн", "класс", "организованные группы школьников")),
        ("взрослым", ("взросл",)),
        ("местным", ("калининградц", "местным", "горожан")),
        ("туристам", ("турист", "гостям города")),
        ("любителям истории", ("истори", "краевед", "району вилл", "кенигсберг")),
        ("любителям природы", ("птиц", "экопрогул", "природ", "залив", "лес")),
        ("для группы", ("организованная группа", "для групп", "группой посетить", "по запросу")),
    ]
    for label, needles in checks:
        if any(needle in low for needle in needles):
            out.append(label)
    return out


def extract_guide_names(text: str, *, fallback_name: str | None = None) -> list[str]:
    out: list[str] = []
    patterns = [
        re.compile(r"авторская\s+экскурсия\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})"),
        re.compile(r"с\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = collapse_ws(match.group(1))
            if len(name) >= 4 and name not in out:
                out.append(name)
    if fallback_name and fallback_name not in out:
        out.insert(0, fallback_name)
    return out[:4]


def extract_organizer_names(text: str, *, source_title: str | None = None) -> list[str]:
    out: list[str] = []
    if source_title:
        out.append(source_title)
    if "профи-тур" in text.lower() and "Профи-тур" not in out:
        out.append("Профи-тур")
    if "хранител" in text.lower() and "Хранители руин" not in out:
        out.append("Хранители руин")
    return out[:4]


def extract_title(text: str, *, source_title: str | None = None) -> str | None:
    quoted = [collapse_ws(m.group(1)) for m in QUOTE_TITLE_RE.finditer(text)]
    quoted = [q for q in quoted if len(q) >= 4]
    if quoted:
        quoted.sort(key=lambda item: (-len(item.split()), len(item)))
        return quoted[0]

    for raw_line in str(text or "").splitlines():
        line = _line_cleanup(raw_line)
        if len(line) < 6:
            continue
        candidate = re.sub(
            r"^[^\wА-Яа-яЁё]*(?:\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?|"
            r"\d{1,2}\s+[А-Яа-яЁё]+\s+в\s+\d{1,2}:\d{2})\s*",
            "",
            line,
            flags=re.I,
        )
        candidate = re.split(r"(?:⏳|💰|📍|✍️|☎️|·\s*\d{1,2}\s+[А-Яа-яЁё]+)", candidate, maxsplit=1)[0].strip()
        if any(pattern.search(candidate) for pattern in NON_SUMMARY_PATTERNS):
            continue
        low = line.lower()
        if "экскурси" in candidate.lower() and len(candidate.split()) <= 14:
            return candidate[:140]
        if "путешествие на " in low:
            start = low.find("путешествие на ")
            value = collapse_ws(line[start:])
            if len(value) >= 10:
                return (value[:1].upper() + value[1:])[:140]

    patterns = [
        re.compile(r"((?:тур|экскурсия|прогулка|путешествие)\s+[A-Za-zА-Яа-яЁё0-9][^\n.!?]{6,140})", re.I),
        re.compile(r"^\s*\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\s+\S+\s+([^\n]{4,140})", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        value = collapse_ws(match.group(1))
        value = re.split(r"(?:⏳|💰|📍|✍️|☎️|·\s*\d{1,2}\s+[А-Яа-яЁё]+)", value, maxsplit=1)[0].strip()
        if re.match(r"^(?:путешествие|экскурсия|прогулка)\s+на\s+", value, flags=re.I):
            return (value[:1].upper() + value[1:])[:140]
        value = re.sub(r"^(?:в|на)\s+", "", value, flags=re.I)
        value = re.sub(r"\s+(?:уже|у\s+вас\s+еще\s+есть|авторская\s+экскурсия).*$", "", value, flags=re.I)
        if len(value) >= 4:
            return value

    for sentence in _sentences(text):
        cleaned = re.sub(r"^[🌿🏰✨🔥📍👣🧭🚌❤️❤♥️\-\s]+", "", sentence).strip()
        cleaned = re.sub(r"^\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\s*", "", cleaned)
        if len(cleaned) < 6:
            continue
        if NON_SUMMARY_PATTERNS[0].search(cleaned):
            continue
        low = cleaned.lower()
        if any(
            token in low
            for token in (
                "сегодня",
                "погоды",
                "больше 10 лет",
                "идея созрела",
                "часть собранных",
                "друзья, приглашаем вас",
                "профи- тур",
                "компании быть",
            )
        ):
            continue
        if cleaned.endswith(":"):
            continue
        if len(cleaned.split()) > 12:
            trimmed = re.split(r"[,.!?]| — | - ", cleaned, maxsplit=1)[0].strip()
            if len(trimmed.split()) > 10:
                trimmed = " ".join(trimmed.split()[:10]).strip()
            cleaned = trimmed
        if len(cleaned.split()) > 10:
            continue
        return cleaned[:120]
    return source_title


def extract_summary_one_liner(text: str, *, title: str | None = None) -> str | None:
    title_norm = normalize_title_key(title)
    for sentence in _sentences(text):
        sentence_norm = normalize_title_key(sentence)
        if title_norm and sentence_norm == title_norm:
            continue
        if any(pattern.search(sentence) for pattern in NON_SUMMARY_PATTERNS):
            continue
        if re.match(r"^\s*[·•-]\s*\d{1,2}\s+[А-Яа-яЁё]+\s*:", sentence):
            continue
        if any(token in sentence.lower() for token in ("мест нет", "лист ожидания", "уже набрана")):
            continue
        if len(sentence.split()) < 6:
            continue
        return sentence[:240]
    return title[:240] if title else None


def detect_on_request(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            "по запросу",
            "организованная группа",
            "организованные группы",
            "для групп",
            "группой посетить",
            "для школьников",
            "для детских групп",
        )
    )


def detect_status(text: str, *, post_dt: datetime) -> tuple[str, bool, str | None]:
    low = text.lower()
    if any(token in low for token in ("не состоится", "отмена", "отменяется")):
        return "cancelled", False, "cancelled"
    if "перенос" in low:
        return "rescheduled", False, "rescheduled"
    seats = extract_seats_text(text)
    if seats:
        return "scheduled", True, seats
    if "завтра" in low and "место встречи" in low:
        return "scheduled", True, extract_seats_text(text)
    if detect_on_request(text) and not extract_date_iso(text, post_date=post_dt):
        return "on_request", False, "on_request"
    return "scheduled", False, None


def classify_post_kind(
    text: str,
    *,
    occurrence_count: int,
    date_iso: str | None,
    status: str,
    is_last_call: bool,
    availability_mode: str,
) -> str:
    low = text.lower()
    if availability_mode == "on_request_private" and not date_iso:
        return "on_request_offer"
    if status in {"cancelled", "rescheduled"} or is_last_call:
        return "status_update"
    if occurrence_count > 1:
        return "announce_multi"
    if date_iso:
        return "announce_single"
    if any(token in low for token in ("сегодня мы", "вчера", "побывала", "были на", "прошла экскурсия")):
        return "reportage"
    return "mixed_or_non_target"


def _is_section_break(line: str) -> bool:
    cleaned = collapse_ws(line)
    if not cleaned:
        return False
    if extract_time_text(cleaned):
        return False
    low = cleaned.lower()
    if extract_meeting_point(cleaned) or extract_price_text(cleaned):
        return False
    if low.startswith(("обзорные экскурсии", "апрельская премьера", "аудиоквест", "бесплатные лекции")):
        return True
    return cleaned.endswith(":") and len(cleaned.split()) <= 6


def _looks_generic_preamble(line: str) -> bool:
    low = collapse_ws(line).lower()
    return any(
        token in low
        for token in (
            "экскурсии и путешествия на",
            "в марте у меня для вас насыщенная программа",
            "весна, идём гулять",
            "весна, идем гулять",
        )
    )


def _has_schedule_anchor(line: str, *, post_date: datetime) -> bool:
    low = line.lower()
    if any(token in low for token in ("мест нет", "лист ожидания", "мест ограничено")):
        return False
    if extract_date_iso(line, post_date=post_date):
        return True
    if "завтра" in low or "сегодня" in low:
        return True
    return False


def split_occurrence_blocks(text: str, *, post_date: datetime) -> list[str]:
    lines = [_line_cleanup(line) for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    blocks: list[list[str]] = []
    current: list[str] = []
    preamble: list[str] = []
    anchors = 0
    section_break_seen = False
    for line in lines:
        if anchors > 0 and current and _is_section_break(line):
            blocks.append(current)
            current = []
            preamble = [line]
            section_break_seen = True
            continue
        if _has_schedule_anchor(line, post_date=post_date) and current:
            blocks.append(current)
            current = [line]
            anchors += 1
            continue
        if _has_schedule_anchor(line, post_date=post_date) and not current:
            anchors += 1
            carry_preamble = [item for item in preamble if not _looks_generic_preamble(item)]
            current = [*carry_preamble, line] if carry_preamble else [line]
            preamble = []
            continue
        if anchors == 0:
            preamble.append(line)
            continue
        current.append(line)
    if current:
        blocks.append(current)

    if anchors <= 1 and not section_break_seen:
        return [collapse_ws(text)]
    return [collapse_ws("\n".join(block)) for block in blocks if collapse_ws("\n".join(block))]


def build_source_fingerprint(
    *,
    title_normalized: str,
    date_iso: str | None,
    time_text: str | None,
) -> str:
    base = "|".join(
        [
            str(title_normalized or ""),
            str(date_iso or ""),
            str(time_text or ""),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def has_public_invite_signal(text: str) -> bool:
    low = collapse_ws(text).lower()
    return any(
        token in low
        for token in (
            "приглашаем",
            "запись",
            "записаться",
            "бронирование",
            "расписание экскурсий",
            "обзорные экскурсии",
            "стоимость",
            "цена",
            "количество мест",
            "мест ограничено",
            "лист ожидания",
            "выезд от",
            "экскурсии будут добавляться",
            "присоединяйтесь",
        )
    )


def looks_operational_only(text: str) -> bool:
    low = collapse_ws(text).lower()
    return any(
        token in low
        for token in (
            "отправление экспресса автобуса",
            "предположительное возвращение",
            "брать с собой хорошее настроение",
            "номер автобуса",
            "автобуса higer",
        )
    )


def looks_context_only(text: str) -> bool:
    low = collapse_ws(text).lower()
    return any(
        token in low
        for token in (
            "стоит единственная экскурсия",
            "на 3-4 вперед закрыта",
            "на 3 4 вперед закрыта",
            "заказчик пропал",
            "читать жеж не будут",
            "не имею прав",
            "отношения не имею",
        )
    )


def looks_closed_without_contact(text: str, *, booking_url: str | None = None) -> bool:
    low = collapse_ws(text).lower()
    has_closed_marker = any(token in low for token in ("мест нет", "лист ожидания", "уже набрана"))
    return has_closed_marker and not collapse_ws(booking_url)


def parse_post_occurrences(
    *,
    text: str,
    post_date: datetime,
    source_kind: str,
    source_title: str | None,
    channel_url: str | None,
    fallback_guide_name: str | None,
) -> list[GuideParsedOccurrence]:
    payload = collapse_ws(text)
    if not payload:
        return []

    blocks = split_occurrence_blocks(text, post_date=post_date)
    post_booking_text, post_booking_url = extract_booking(payload)
    out: list[GuideParsedOccurrence] = []
    for block in blocks:
        date_iso = extract_date_iso(block, post_date=post_date)
        time_text = extract_time_text(block)
        availability_mode = "on_request_private" if detect_on_request(block) and not date_iso else "scheduled_public"
        status, is_last_call, seats_text = detect_status(block, post_dt=post_date)
        title = extract_title(block, source_title=source_title)
        title_norm = normalize_title_key(title)
        if not title or not title_norm or looks_like_noise_title(title):
            continue
        digest_eligible = bool(
            date_iso
            and availability_mode == "scheduled_public"
            and status not in {"cancelled"}
            and has_public_invite_signal(block)
            and not looks_operational_only(block)
            and not looks_context_only(block)
        )
        digest_reason = None
        if not digest_eligible:
            if looks_context_only(block):
                digest_reason = "context_only"
            elif looks_operational_only(block):
                digest_reason = "operational_only"
            elif not has_public_invite_signal(block):
                digest_reason = "weak_public_signal"
            elif not date_iso:
                digest_reason = "missing_date"
            elif availability_mode != "scheduled_public":
                digest_reason = "on_request"
            elif status == "cancelled":
                digest_reason = "cancelled"
        summary = extract_summary_one_liner(block, title=title)
        audience_fit = extract_audience_fit(block)
        booking_text, booking_url = extract_booking(block)
        if not booking_text and post_booking_text:
            booking_text = post_booking_text
        if not booking_url and post_booking_url:
            booking_url = post_booking_url
        if digest_eligible and looks_closed_without_contact(block, booking_url=booking_url):
            digest_eligible = False
            digest_reason = "closed_without_booking"
        occurrence = GuideParsedOccurrence(
            block_text=block,
            canonical_title=title,
            title_normalized=title_norm,
            date_iso=date_iso,
            time_text=time_text,
            city="Калининград" if "зеленоградск" not in block.lower() else "Зеленоградск",
            meeting_point=extract_meeting_point(block),
            audience_fit=audience_fit,
            price_text=extract_price_text(block),
            booking_text=booking_text,
            booking_url=booking_url,
            channel_url=channel_url,
            status=status,
            seats_text=seats_text,
            summary_one_liner=summary,
            digest_blurb=summary,
            digest_eligible=digest_eligible,
            digest_eligibility_reason=digest_reason,
            is_last_call=is_last_call,
            post_kind="announce_single",
            availability_mode=availability_mode,
            guide_names=extract_guide_names(block, fallback_name=fallback_guide_name),
            organizer_names=extract_organizer_names(block, source_title=source_title),
            source_fingerprint=build_source_fingerprint(
                title_normalized=title_norm,
                date_iso=date_iso,
                time_text=time_text,
            ),
        )
        out.append(occurrence)

    if not out:
        return []

    occurrence_count = len(out)
    for idx, item in enumerate(out):
        out[idx].post_kind = classify_post_kind(
            item.block_text,
            occurrence_count=occurrence_count,
            date_iso=item.date_iso,
            status=item.status,
            is_last_call=item.is_last_call,
            availability_mode=item.availability_mode,
        )
    return out


def audience_line(audience_fit: Iterable[str]) -> str | None:
    values = [collapse_ws(x) for x in audience_fit if collapse_ws(x)]
    if not values:
        return None
    return ", ".join(values[:3])


def looks_like_noise_title(value: str | None) -> bool:
    title = collapse_ws(value)
    low = title.lower()
    if not title:
        return True
    if len(re.findall(r"\d", title)) >= 5:
        return True
    if any(
        token in low
        for token in (
            "звоните",
            "ждем вас",
            "ждём вас",
            "по телефонам",
            "место старта",
            "подробности",
            "компании быть",
            "часть собранных",
            "не пропустите",
        )
    ):
        return True
    return False
