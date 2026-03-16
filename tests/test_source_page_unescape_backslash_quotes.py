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


@pytest.mark.asyncio
async def test_build_source_page_content_unescapes_backslash_newlines(monkeypatch):
    import main

    async def _fake_upload(images):  # noqa: ANN001 - test helper
        return ([], "")

    monkeypatch.setattr(main, "upload_images", _fake_upload)

    html, _, _ = await main.build_source_page_content(
        "T",
        "> «Мы гордимся своим производством».\\n> — Мария Титова\\n\\n### История\\nТекст раздела.",
        None,
        None,
        None,
        None,
        None,
    )
    assert "\\n" not in html
    assert "<blockquote>" in html
    assert "Мария Титова" in html
    assert "<h3>История</h3>" in html
