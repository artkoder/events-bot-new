from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from telethon import functions


@dataclass(slots=True)
class GuideScannedPost:
    source_username: str
    source_title: str | None
    message_id: int
    grouped_id: int | None
    post_date: datetime
    source_url: str
    text: str
    views: int | None
    forwards: int | None
    reactions_total: int | None
    reactions_json: dict[str, int] | None
    media_refs: list[dict[str, Any]]


@dataclass(slots=True)
class GuideScannedSourceMeta:
    source_title: str | None
    about_text: str | None
    about_links: list[str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _message_text(message: Any) -> str:
    return str(getattr(message, "message", None) or getattr(message, "text", None) or "").strip()


def _message_has_media(message: Any) -> bool:
    return bool(
        getattr(message, "photo", None)
        or getattr(message, "video", None)
        or getattr(message, "document", None)
        or getattr(message, "media", None)
    )


def _message_media_kind(message: Any) -> str | None:
    if getattr(message, "photo", None):
        return "photo"
    if getattr(message, "video", None):
        return "video"
    doc = getattr(message, "document", None)
    if doc is not None:
        mime = str(getattr(doc, "mime_type", None) or "").lower()
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("image/"):
            return "photo"
    return None


def _reactions_payload(message: Any) -> tuple[int | None, dict[str, int] | None]:
    reactions = getattr(message, "reactions", None)
    if not reactions or not getattr(reactions, "results", None):
        return None, None
    out: dict[str, int] = {}
    total = 0
    for item in getattr(reactions, "results", []) or []:
        count = int(getattr(item, "count", 0) or 0)
        reaction = getattr(item, "reaction", None)
        emoji = str(getattr(reaction, "emoticon", None) or getattr(reaction, "document_id", None) or reaction or "")
        if not emoji:
            emoji = "reaction"
        out[emoji] = count
        total += count
    return total or None, out or None


def _collapse_group(messages: list[Any], *, username: str, source_title: str | None) -> GuideScannedPost | None:
    ordered = sorted(messages, key=lambda item: int(getattr(item, "id", 0) or 0))
    first_with_text = next((msg for msg in ordered if _message_text(msg)), None)
    anchor = first_with_text or ordered[-1]
    anchor_id = int(getattr(anchor, "id", 0) or 0)
    post_date = getattr(anchor, "date", None) or getattr(ordered[-1], "date", None) or _utc_now()
    if post_date.tzinfo is None:
        post_date = post_date.replace(tzinfo=timezone.utc)
    text_parts = [_message_text(msg) for msg in ordered if _message_text(msg)]
    text = "\n".join(text_parts).strip()
    views = max((getattr(msg, "views", None) or 0) for msg in ordered) or None
    forwards = max((getattr(msg, "forwards", None) or 0) for msg in ordered) or None
    reactions_total = None
    reactions_json = None
    for msg in ordered:
        total, payload = _reactions_payload(msg)
        if total is not None and ((reactions_total or -1) < total):
            reactions_total = total
            reactions_json = payload
    media_refs: list[dict[str, Any]] = []
    for msg in ordered:
        kind = _message_media_kind(msg)
        if not kind:
            continue
        media_refs.append(
            {
                "message_id": int(getattr(msg, "id", 0) or 0),
                "kind": kind,
                "grouped_id": int(getattr(msg, "grouped_id", 0) or 0) or None,
            }
        )
    if not text and not media_refs:
        return None
    return GuideScannedPost(
        source_username=username,
        source_title=source_title,
        message_id=anchor_id,
        grouped_id=int(getattr(anchor, "grouped_id", 0) or 0) or None,
        post_date=post_date.astimezone(timezone.utc),
        source_url=f"https://t.me/{username}/{anchor_id}",
        text=text,
        views=views,
        forwards=forwards,
        reactions_total=reactions_total,
        reactions_json=reactions_json,
        media_refs=media_refs,
    )


async def scan_source_posts(
    client: Any,
    *,
    username: str,
    limit: int,
    days_back: int,
) -> tuple[GuideScannedSourceMeta, list[GuideScannedPost]]:
    entity = await client.get_entity(username)
    source_title = str(getattr(entity, "title", None) or getattr(entity, "first_name", None) or "").strip() or None
    about_text = ""
    about_links: list[str] = []
    try:
        full = await client(functions.channels.GetFullChannelRequest(channel=entity))
        about_text = str(getattr(full.full_chat, "about", None) or "").strip()
    except Exception:
        try:
            full = await client(functions.users.GetFullUserRequest(id=entity))
            about_text = str(getattr(full.full_user, "about", None) or "").strip()
        except Exception:
            about_text = ""
    if about_text:
        seen: set[str] = set()
        for token in about_text.replace("\n", " ").split():
            raw = str(token or "").strip("()[]{}<>.,!?:;\"'")
            if raw.startswith("http://") or raw.startswith("https://"):
                if raw not in seen:
                    seen.add(raw)
                    about_links.append(raw)
    messages = await client.get_messages(entity, limit=max(1, int(limit)))
    cutoff = _utc_now() - timedelta(days=max(1, int(days_back)))

    singles: list[Any] = []
    grouped: dict[int, list[Any]] = {}
    for msg in messages:
        msg_date = getattr(msg, "date", None)
        if msg_date is None:
            continue
        if msg_date.tzinfo is None:
            msg_date = msg_date.replace(tzinfo=timezone.utc)
        if msg_date.astimezone(timezone.utc) < cutoff:
            continue
        if getattr(msg, "action", None):
            continue
        gid = int(getattr(msg, "grouped_id", 0) or 0)
        if gid:
            grouped.setdefault(gid, []).append(msg)
        else:
            singles.append(msg)

    out: list[GuideScannedPost] = []
    for msg in singles:
        collapsed = _collapse_group([msg], username=username, source_title=source_title)
        if collapsed:
            out.append(collapsed)
    for _gid, group in grouped.items():
        collapsed = _collapse_group(group, username=username, source_title=source_title)
        if collapsed:
            out.append(collapsed)

    out.sort(key=lambda item: (item.post_date, item.message_id), reverse=True)
    return GuideScannedSourceMeta(source_title=source_title, about_text=about_text or None, about_links=about_links), out
