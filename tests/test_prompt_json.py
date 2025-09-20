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


def test_aliases_bypass_cache_layer():
    main._prompt_cache.cache_clear()
    prompt_initial = main._build_prompt([
        "Fest",
    ], [
        ("old-alias", 0),
    ])
    data_initial = _extract_prompt_json(prompt_initial)
    assert data_initial["festival_alias_pairs"] == [["old-alias", 0]]

    prompt_updated = main._build_prompt([
        "Fest",
    ], [
        ("new-alias", 0),
    ])
    data_updated = _extract_prompt_json(prompt_updated)
    assert data_updated["festival_alias_pairs"] == [["new-alias", 0]]
