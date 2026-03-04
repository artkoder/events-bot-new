import pytest


@pytest.mark.asyncio
async def test_merge_render_photos_prioritizes_posters_over_site_gallery():
    import main

    poster = "https://files.catbox.moe/poster-test.jpg"
    photos = [f"https://files.catbox.moe/site-{i:02d}.jpg" for i in range(12)]

    ordered = main.merge_render_photos(
        photo_urls=photos,
        poster_urls=[poster],
        cover_url=None,
    )
    assert ordered[0] == poster

    html, _, _ = await main.build_source_page_content(
        "T",
        "Text",
        None,
        None,
        None,
        None,
        None,
        catbox_urls=ordered,
    )
    assert poster in html

    # Control: without prioritization, posters appended after a full 12-image gallery
    # get truncated by build_source_page_content's 12-image limit.
    naive = photos + [poster]
    html2, _, _ = await main.build_source_page_content(
        "T",
        "Text",
        None,
        None,
        None,
        None,
        None,
        catbox_urls=naive,
    )
    assert poster not in html2

