import ssl

import pytest

import source_parsing.telegram.service as tg_service


@pytest.mark.asyncio
async def test_poll_kaggle_kernel_retries_on_transient_ssl_errors(monkeypatch):
    monkeypatch.setattr(tg_service, "POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(tg_service, "TIMEOUT_MINUTES", 1)

    class _FakeClient:
        def __init__(self):
            self.calls = 0

        def get_kernel_status(self, kernel_ref: str):  # noqa: ANN001 - test helper
            self.calls += 1
            if self.calls == 1:
                raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
            if self.calls == 2:
                return {"status": "RUNNING"}
            return {"status": "COMPLETE"}

    seen = []

    async def _cb(phase, kernel_ref, status):  # noqa: ANN001 - test helper
        seen.append((phase, (status or {}).get("status")))

    status, status_data, _duration = await tg_service._poll_kaggle_kernel(  # noqa: SLF001
        _FakeClient(),
        "user/kernel",
        run_id="test",
        timeout_minutes=1,
        status_callback=_cb,
    )
    assert status == "complete"
    assert status_data and status_data.get("status") == "COMPLETE"
    assert any(p == "poll_error" for p, _ in seen)
