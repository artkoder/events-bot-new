from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


def test_extract_photo_urls_from_public_tg_html_ignores_avatar_and_neighbor_posts() -> None:
    from source_parsing.telegram.handlers import _extract_photo_urls_from_public_tg_html

    avatar = "https://cdn4.telesco.pe/file/channel_avatar.jpg"
    poster = "https://cdn4.telesco.pe/file/event_poster.jpg"

    html = f"""
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="terkatalk/4487">
    <div class="tgme_widget_message_user">
      <i class="tgme_widget_message_user_photo"><img src="{avatar}"/></i>
    </div>
    <div class="tgme_widget_message_bubble">no media here</div>
  </div>
</div>
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="terkatalk/4488">
    <div class="tgme_widget_message_user">
      <i class="tgme_widget_message_user_photo"><img src="{avatar}"/></i>
    </div>
    <div class="media_supported_cont">
      <a class="tgme_widget_message_photo_wrap blured" href="https://t.me/terkatalk/4488"
         style="width:523px;background-image:url('{poster}')"></a>
    </div>
  </div>
</div>
"""

    assert (
        _extract_photo_urls_from_public_tg_html(
            html,
            username="terkatalk",
            message_id=4487,
            limit=3,
        )
        == []
    )
    assert (
        _extract_photo_urls_from_public_tg_html(
            html,
            username="terkatalk",
            message_id=4488,
            limit=3,
        )
        == [poster]
    )


@pytest.mark.asyncio
async def test_fallback_fetch_posters_uses_direct_cdn_url_when_media_upload_fails(monkeypatch) -> None:
    from source_parsing.telegram.handlers import _fallback_fetch_posters_from_public_tg_page

    poster = "https://cdn4.telesco.pe/file/event_poster.jpg"
    html = f"""
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="meowafisha/6782">
    <div class="media_supported_cont">
      <a class="tgme_widget_message_photo_wrap blured" href="https://t.me/meowafisha/6782"
         style="width:523px;background-image:url('{poster}')"></a>
    </div>
  </div>
</div>
"""
    img_bytes = b"\x89PNG\r\n\x1a\nfake"

    async def fake_http_call(label, method, url, **_kwargs):
        assert method == "GET"
        if label == "tg_public_page":
            assert url.endswith("/meowafisha/6782")
            return SimpleNamespace(status_code=200, content=html.encode("utf-8"))
        if label == "tg_public_img":
            assert url == poster
            return SimpleNamespace(status_code=200, content=img_bytes)
        raise AssertionError(f"unexpected http_call label={label} url={url}")

    async def fake_process_media(_images, *, need_catbox, need_ocr):
        assert need_catbox is True
        assert need_ocr is False
        return [], "upload failed"

    import net
    import poster_media

    monkeypatch.setattr(net, "http_call", fake_http_call)
    monkeypatch.setattr(poster_media, "process_media", fake_process_media)

    posters = await _fallback_fetch_posters_from_public_tg_page(
        username="meowafisha",
        message_id=6782,
        limit=1,
    )

    assert len(posters) == 1
    assert posters[0].catbox_url == poster
    assert posters[0].supabase_url is None
    assert posters[0].sha256 == hashlib.sha256(img_bytes).hexdigest()
