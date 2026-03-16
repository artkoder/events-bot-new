from __future__ import annotations

import json
import ssl

import pytest

from guide_excursions import kaggle_service


def _mk_auth_bundle(*, session: str) -> str:
    payload = {
        "session": session,
        "device_model": "Test Device",
        "system_version": "Test OS 1",
        "app_version": "Test App 1",
        "lang_code": "ru",
        "system_lang_code": "ru-RU",
    }
    raw = json.dumps(payload, ensure_ascii=False)
    return kaggle_service.base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def test_build_secrets_payload_includes_guide_llm_gateway_envs(monkeypatch):
    for key in (
        "GUIDE_MONITORING_AUTH_BUNDLE_ENV",
        "TELEGRAM_AUTH_BUNDLE_S22",
        "TELEGRAM_AUTH_BUNDLE_E2E",
        "TELEGRAM_SESSION",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("TG_SESSION", "session")
    monkeypatch.setenv("GOOGLE_API_KEY2", "key2")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("SUPABASE_SCHEMA", "public")
    monkeypatch.setenv("GUIDE_MONITORING_GOOGLE_KEY_ENV", "GOOGLE_API_KEY2")
    monkeypatch.setenv("GUIDE_MONITORING_LLM_TIMEOUT_SEC", "95")
    monkeypatch.setenv("GOOGLE_API_LOCALNAME2", "guide-monitor")

    payload = json.loads(kaggle_service._build_secrets_payload())

    assert payload["GOOGLE_API_KEY2"] == "key2"
    assert payload["GOOGLE_API_LOCALNAME2"] == "guide-monitor"
    assert payload["SUPABASE_URL"] == "https://example.supabase.co"
    assert payload["SUPABASE_SERVICE_KEY"] == "service-key"
    assert payload["SUPABASE_SCHEMA"] == "public"
    assert payload["GUIDE_MONITORING_GOOGLE_KEY_ENV"] == "GOOGLE_API_KEY2"
    assert payload["GUIDE_MONITORING_LLM_TIMEOUT_SEC"] == "95"
    assert payload["TG_SESSION"] == "session"


def test_build_secrets_payload_prefers_s22_bundle(monkeypatch):
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("GOOGLE_API_KEY2", "key2")
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_S22", _mk_auth_bundle(session="session_s22"))
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_E2E", _mk_auth_bundle(session="session_e2e"))
    monkeypatch.delenv("GUIDE_MONITORING_AUTH_BUNDLE_ENV", raising=False)
    monkeypatch.delenv("TG_SESSION", raising=False)
    monkeypatch.delenv("TELEGRAM_SESSION", raising=False)

    payload = json.loads(kaggle_service._build_secrets_payload())

    assert payload["TELEGRAM_AUTH_BUNDLE_S22"]
    assert payload["TG_SESSION"] == "session_s22"


def test_build_secrets_payload_override_maps_selected_bundle_to_s22(monkeypatch):
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("GOOGLE_API_KEY2", "key2")
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_S22", _mk_auth_bundle(session="session_s22"))
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_E2E", _mk_auth_bundle(session="session_e2e"))
    monkeypatch.setenv("GUIDE_MONITORING_AUTH_BUNDLE_ENV", "TELEGRAM_AUTH_BUNDLE_E2E")
    monkeypatch.setenv("GUIDE_MONITORING_ALLOW_NON_S22_AUTH", "1")
    monkeypatch.delenv("TG_SESSION", raising=False)
    monkeypatch.delenv("TELEGRAM_SESSION", raising=False)

    payload = json.loads(kaggle_service._build_secrets_payload())

    assert payload["TELEGRAM_AUTH_BUNDLE_S22"]
    assert payload["TG_SESSION"] == "session_e2e"
    assert payload["GUIDE_MONITORING_AUTH_BUNDLE_ENV"] == "TELEGRAM_AUTH_BUNDLE_E2E"


def test_build_secrets_payload_rejects_e2e_bundle_without_explicit_allow(monkeypatch):
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("GOOGLE_API_KEY2", "key2")
    monkeypatch.delenv("TELEGRAM_AUTH_BUNDLE_S22", raising=False)
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_E2E", _mk_auth_bundle(session="session_e2e"))
    monkeypatch.delenv("GUIDE_MONITORING_ALLOW_NON_S22_AUTH", raising=False)
    monkeypatch.delenv("GUIDE_MONITORING_AUTH_BUNDLE_ENV", raising=False)

    with pytest.raises(RuntimeError, match="requires TELEGRAM_AUTH_BUNDLE_S22"):
        kaggle_service._build_secrets_payload()


def test_stage_repo_bundle_includes_google_ai_package(tmp_path):
    staged_root = kaggle_service.stage_repo_bundle(tmp_path / "bundle")

    assert staged_root.exists()
    assert (staged_root / "google_ai" / "__init__.py").exists()


def test_sync_notebook_entrypoint_embeds_python_runner(tmp_path):
    kernel_path = tmp_path / "GuideExcursionsMonitor"
    kernel_path.mkdir()
    (kernel_path / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "zigomaro/guide-excursions-monitor",
                "code_file": "guide_excursions_monitor.ipynb",
                "kernel_type": "notebook",
            }
        ),
        encoding="utf-8",
    )
    (kernel_path / "guide_excursions_monitor.py").write_text(
        "print('guide notebook smoke')\nVALUE = 42\nasync def main():\n    return 42\n\nif __name__ == \"__main__\":\n    asyncio.run(main())\n",
        encoding="utf-8",
    )

    kaggle_service._sync_notebook_entrypoint(kernel_path)

    notebook = json.loads((kernel_path / "guide_excursions_monitor.ipynb").read_text(encoding="utf-8"))
    assert notebook["cells"][0]["cell_type"] == "markdown"
    source = "".join(notebook["cells"][1]["source"])
    assert "__file__ =" in source
    assert "_GUIDE_EMBEDDED_GOOGLE_AI" in source
    assert "embedded_repo_bundle" in source
    assert "guide notebook smoke" in source
    assert "VALUE = 42" in source
    assert "await main()" not in source
    assert "asyncio.run(main())" not in source
    bootstrap_cell = "".join(notebook["cells"][2]["source"])
    assert "embedded google_ai root" in bootstrap_cell
    run_cell = "".join(notebook["cells"][3]["source"])
    assert "nest_asyncio" in run_cell
    assert "run_until_complete(main())" in run_cell


def test_read_results_run_id(tmp_path):
    path = tmp_path / "guide_excursions_results.json"
    path.write_text(json.dumps({"run_id": "guide-run-42"}), encoding="utf-8")

    assert kaggle_service._read_results_run_id(path) == "guide-run-42"


@pytest.mark.asyncio
async def test_download_results_retries_until_matching_run_id(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def download_kernel_output(self, kernel_ref, path, force=True):
            assert kernel_ref == "owner/kernel"
            self.calls += 1
            out_dir = kaggle_service.Path(path)
            out_dir.mkdir(parents=True, exist_ok=True)
            observed_run_id = "stale-run" if self.calls == 1 else "target-run"
            (out_dir / "guide_excursions_results.json").write_text(
                json.dumps({"run_id": observed_run_id}),
                encoding="utf-8",
            )
            return ["guide_excursions_results.json"]

    async def fake_sleep(seconds):
        assert seconds == 5

    monkeypatch.setattr(kaggle_service.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(kaggle_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(kaggle_service, "RESULTS_MATCH_RETRY_DELAY_SECONDS", 5)

    path = await kaggle_service._download_results(FakeClient(), "owner/kernel", "target-run")

    assert kaggle_service._read_results_run_id(path) == "target-run"


@pytest.mark.asyncio
async def test_poll_kaggle_kernel_retries_transient_ssl_error(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def get_kernel_status(self, kernel_ref):
            assert kernel_ref == "owner/kernel"
            self.calls += 1
            if self.calls == 1:
                raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
            return {"status": "COMPLETE"}

    phases: list[str] = []
    sleeps: list[float] = []

    async def fake_status_callback(phase, _kernel_ref, _status):
        phases.append(str(phase))

    async def fake_sleep(seconds):
        sleeps.append(float(seconds))

    monkeypatch.setattr(kaggle_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(kaggle_service, "STALE_COMPLETE_MIN_SECONDS", 0)

    status, status_data, _duration = await kaggle_service._poll_kaggle_kernel(
        FakeClient(),
        "owner/kernel",
        run_id="guide-run-1",
        timeout_minutes=1,
        status_callback=fake_status_callback,
    )

    assert status == "complete"
    assert status_data == {"status": "COMPLETE"}
    assert "poll_error" in phases
    assert "complete" in phases
    assert sleeps


@pytest.mark.asyncio
async def test_wait_for_remote_kernel_shape_accepts_matching_notebook(monkeypatch):
    calls = {"count": 0}

    async def fake_pull(_client, _kernel_ref):
        calls["count"] += 1
        return {
            "kernel_type": "notebook",
            "code_file": "guide-excursions-monitor.ipynb",
        }

    monkeypatch.setattr(kaggle_service, "_pull_remote_kernel_metadata", fake_pull)

    meta = await kaggle_service._wait_for_remote_kernel_shape(
        client=object(),
        kernel_ref="owner/kernel",
        expected_meta={
            "kernel_type": "notebook",
            "code_file": "guide_excursions_monitor.ipynb",
        },
    )

    assert calls["count"] == 1
    assert meta["kernel_type"] == "notebook"


@pytest.mark.asyncio
async def test_wait_for_remote_kernel_shape_raises_on_script_kernel(monkeypatch):
    phases: list[str] = []
    sleeps: list[float] = []

    async def fake_pull(_client, _kernel_ref):
        return {
            "kernel_type": "script",
            "code_file": "guide-excursions-monitor.py",
        }

    async def fake_status_callback(phase, _kernel_ref, _status):
        phases.append(str(phase))

    async def fake_sleep(seconds):
        sleeps.append(float(seconds))

    monkeypatch.setattr(kaggle_service, "_pull_remote_kernel_metadata", fake_pull)
    monkeypatch.setattr(kaggle_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(kaggle_service, "REMOTE_KERNEL_SHAPE_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(kaggle_service, "REMOTE_KERNEL_SHAPE_RETRY_DELAY_SECONDS", 3.0)

    with pytest.raises(RuntimeError, match="kernel shape mismatch"):
        await kaggle_service._wait_for_remote_kernel_shape(
            client=object(),
            kernel_ref="owner/kernel",
            expected_meta={
                "kernel_type": "notebook",
                "code_file": "guide_excursions_monitor.ipynb",
            },
            status_callback=fake_status_callback,
        )

    assert phases == ["kernel_shape_wait", "kernel_shape_wait"]
    assert sleeps == [3.0]
