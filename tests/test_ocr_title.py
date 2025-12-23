import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from poster_media import PosterMedia, _run_ocr, apply_ocr_results_to_media
from vision_test.ocr import OcrResult, OcrUsage

@pytest.mark.asyncio
async def test_poster_media_ocr_title_extraction():
    # Mock response from run_ocr
    mock_result = OcrResult(
        text="Full OCR Text",
        title="DOMINANT TITLE",
        usage=OcrUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )

    with patch("poster_media.run_ocr", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result

        poster = PosterMedia(data=b"fake_image_data", name="test.jpg")

        await _run_ocr(poster, model="gpt-4o-mini", detail="auto")

        assert poster.ocr_text == "Full OCR Text"
        assert poster.ocr_title == "DOMINANT TITLE"
        assert poster.prompt_tokens == 10

@pytest.mark.asyncio
async def test_apply_ocr_results_to_media_restores_title():
    # Create a mock cache object with a title attribute
    class MockCache:
        text = "Cached Text"
        title = "Cached Title"
        prompt_tokens = 20
        completion_tokens = 10
        total_tokens = 30
        hash = "somehash"

    mock_cache = MockCache()
    poster_media = [PosterMedia(data=b"image_data", name="test.jpg")]

    # Simulate that this poster corresponds to the cached entry
    hash_to_indices = {"somehash": [0]}

    apply_ocr_results_to_media(poster_media, [mock_cache], hash_to_indices=hash_to_indices)

    assert poster_media[0].ocr_text == "Cached Text"
    assert poster_media[0].ocr_title == "Cached Title"
    assert poster_media[0].total_tokens == 30

@pytest.mark.asyncio
async def test_ocr_result_parsing():
    from vision_test.ocr import run_ocr

    # Mock the HTTP session response
    mock_response_data = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "poster_ocr_text": "Parsed Text",
                    "ocr_title": "Parsed Title"
                })
            }
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        },
        "id": "req-123"
    }

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Mock context manager for session.post
    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__.return_value = mock_response
    mock_post_ctx.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_ctx

    # Mock semaphore
    mock_sem = AsyncMock()
    mock_sem.__aenter__.return_value = None
    mock_sem.__aexit__.return_value = None

    import vision_test.ocr
    vision_test.ocr.configure_http(session=mock_session, semaphore=mock_sem)

    # We need to mock os.getenv to return a token
    with patch("os.getenv", return_value="fake_token"):
        # We need valid image bytes for PIL check
        # Minimal 1x1 PNG
        valid_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

        result = await run_ocr(valid_png, model="test-model", detail="auto")

        assert result.text == "Parsed Text"
        assert result.title == "Parsed Title"
        assert result.usage.total_tokens == 150
