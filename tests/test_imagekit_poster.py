import types

from imagekit_poster import (
    PosterGravity,
    PosterProcessingMode,
    _build_transformation,
    process_poster,
)


def test_extend_genfill_prompt_encoding():
    payload = _build_transformation(
        PosterProcessingMode.EXTEND_GENFILL,
        width=320,
        height=240,
        prompt="snow & sun / beach",
        gravity=None,
    )

    assert payload["raw"] == "bg-genfill-prompt-snow%20%26%20sun%20%2F%20beach"


def test_process_poster_accepts_upload_result(monkeypatch):
    class StubUploadResult:
        file_path = "/poster/path.jpg"

        def __getattr__(self, name):  # pragma: no cover - defensive but required by spec
            return None

    class StubResponse:
        content = b"processed"

        def raise_for_status(self):
            return None

    captured = types.SimpleNamespace(url_options=None, requested_url=None)

    class StubImageKit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def upload_file(self, *, file, file_name):
            assert file == b"data"
            assert file_name == "poster.jpg"
            return StubUploadResult()

        def url(self, options):
            captured.url_options = options
            return "https://example.invalid/transformed.jpg"

    def fake_get(url, timeout):
        captured.requested_url = url
        assert timeout == 10.0
        return StubResponse()

    monkeypatch.setenv("IMAGEKIT_PUBLIC_KEY", "public")
    monkeypatch.setenv("IMAGEKIT_PRIVATE_KEY", "private")
    monkeypatch.setenv("IMAGEKIT_URL_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr("imagekit_poster.ImageKit", StubImageKit)
    monkeypatch.setattr("imagekit_poster.requests.get", fake_get)

    result = process_poster(
        b"data",
        mode=PosterProcessingMode.SMART_CROP,
        width=10,
        height=12,
    )

    assert result == b"processed"
    assert captured.url_options == {
        "path": "/poster/path.jpg",
        "transformation_position": "path",
        "transformation": [
            {"w": 10, "h": 12, "fo": PosterGravity.AUTO.value},
        ],
    }
