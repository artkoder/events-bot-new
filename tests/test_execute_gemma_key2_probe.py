import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "kaggle" / "execute_gemma_key2_probe.py"
SPEC = importlib.util.spec_from_file_location("execute_gemma_key2_probe_local", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_probe_config = MODULE.build_probe_config
build_secret_payload = MODULE.build_secret_payload
candidate_secret_names = MODULE.candidate_secret_names
load_env_file_values = MODULE.load_env_file_values
normalize_model_name = MODULE.normalize_model_name
resolve_secret_value = MODULE.resolve_secret_value


def test_load_env_file_values_parses_basic_assignments(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nGOOGLE_API_KEY2=abc123\nQUOTED='value=with=equals'\nEMPTY=\n",
        encoding="utf-8",
    )

    values = load_env_file_values(env_file)

    assert values == {
        "GOOGLE_API_KEY2": "abc123",
        "QUOTED": "value=with=equals",
        "EMPTY": "",
    }


def test_candidate_secret_names_supports_key2_aliases() -> None:
    assert candidate_secret_names("GOOGLE_API_KEY2") == ["GOOGLE_API_KEY2", "GOOGLE_API_KEY_2"]
    assert candidate_secret_names("GOOGLE_API_KEY_2") == ["GOOGLE_API_KEY_2", "GOOGLE_API_KEY2"]


def test_resolve_secret_value_prefers_environment(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY2=file-value\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_API_KEY2", "env-value")

    value, resolved_name, source = resolve_secret_value("GOOGLE_API_KEY2", [env_file])

    assert value == "env-value"
    assert resolved_name == "GOOGLE_API_KEY2"
    assert source == "environment"


def test_normalize_model_name_adds_models_prefix() -> None:
    assert normalize_model_name("gemma-3-27b-it") == "models/gemma-3-27b-it"
    assert normalize_model_name("models/gemma-3-27b-it") == "models/gemma-3-27b-it"


def test_build_probe_config_and_secret_payload_include_aliases(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_LOCALNAME", "zigomaro")

    config = build_probe_config(
        secret_env_var="GOOGLE_API_KEY2",
        model="gemma-3-27b-it",
        prompt="Reply with exactly: OK",
        max_output_tokens=8,
    )
    payload = build_secret_payload(config, "secret-value")

    assert config["model"] == "models/gemma-3-27b-it"
    assert config["secret_env_aliases"] == ["GOOGLE_API_KEY_2"]
    assert "GOOGLE_API_KEY2" in payload
    assert "GOOGLE_API_KEY_2" in payload
    assert "GOOGLE_API_LOCALNAME" in payload
