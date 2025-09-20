import json

import main


def _extract_prompt_json(value: str) -> dict:
    json_text = value.rsplit("\n", 1)[-1]
    return json.loads(json_text)


def test_build_prompt_includes_aliases():
    main._prompt_cache.cache_clear()
    prompt = main._build_prompt([
        "Fest B",
        "Fest A",
    ], [
        ("alias", 0),
    ])
    assert "Use the JSON below" in prompt
    data = _extract_prompt_json(prompt)
    assert data == {
        "festival_names": ["Fest A", "Fest B"],
        "festival_alias_pairs": [["alias", 0]],
    }


def test_build_prompt_omits_alias_section_when_empty():
    main._prompt_cache.cache_clear()
    prompt = main._build_prompt(["Fest"], [])
    data = _extract_prompt_json(prompt)
    assert data == {"festival_names": ["Fest"]}
