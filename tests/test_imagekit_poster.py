from imagekit_poster import PosterProcessingMode, _build_transformation


def test_extend_genfill_prompt_encoding():
    payload = _build_transformation(
        PosterProcessingMode.EXTEND_GENFILL,
        width=320,
        height=240,
        prompt="snow & sun / beach",
        gravity=None,
    )

    assert payload["raw"] == "bg-genfill-prompt-snow%20%26%20sun%20%2F%20beach"
