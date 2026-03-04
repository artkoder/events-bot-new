import pytest


@pytest.mark.asyncio
async def test_build_source_page_content_unescapes_backslash_quotes(monkeypatch):
    import main

    # Avoid network uploads.
    async def _fake_upload(images):  # noqa: ANN001 - test helper
        return ([], "")

    monkeypatch.setattr(main, "upload_images", _fake_upload)

    html, _, _ = await main.build_source_page_content(
        "T",
        'В калининградском клубе \\"Сигнал\\" состоится игра \\"Вавилон\\".',
        None,
        None,
        None,
        None,
        None,
    )
    assert "\\\"" not in html
    assert "&quot;Сигнал&quot;" in html
    assert "&quot;Вавилон&quot;" in html

