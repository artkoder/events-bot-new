    sanitize_telegram_html,
    md_to_html,
    extract_link_from_html,
    extract_links_from_html,
    MONTHS,
    MONTHS_PREP,
    MONTHS_NOM,
    month_name,
    month_name_prepositional,
    month_name_nominative,
    next_month,
    set_month_timezone,

# Re-export imported helpers for backwards compatibility.
_ = (MONTHS_PREP, MONTHS_NOM, extract_link_from_html, extract_links_from_html)
set_month_timezone(LOCAL_TZ)
    set_month_timezone(LOCAL_TZ)
    set_month_timezone(LOCAL_TZ)
        return f"{monday.day}–{sunday.day} {MONTHS[monday.month - 1]}"
        f"{monday.day} {MONTHS[monday.month - 1]} – "
        return f"{saturday.day}–{sunday.day} {MONTHS[saturday.month - 1]}"
        f"{saturday.day} {MONTHS[saturday.month - 1]} – "
def is_valid_url(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.match(r"https?://", text))


