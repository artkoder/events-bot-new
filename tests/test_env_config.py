import importlib
import main


def test_new_config_defaults(monkeypatch):
    # Clear environment variables to test defaults
    for var in [
        "WEEK_EDIT_MODE",
        "WEEK_EDIT_CRON",
        "CAPTCHA_WAIT_S",
        "CAPTCHA_MAX_ATTEMPTS",
        "CAPTCHA_NIGHT_RANGE",
        "CAPTCHA_RETRY_AT",
    ]:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(main)
    assert main.WEEK_EDIT_MODE == "deferred"
    assert main.WEEK_EDIT_CRON == "02:30"
    assert main.CAPTCHA_WAIT_S == 600
    assert main.CAPTCHA_MAX_ATTEMPTS == 2
    assert main.CAPTCHA_NIGHT_RANGE == "00:00-07:00"
    assert main.CAPTCHA_RETRY_AT == "08:10"
