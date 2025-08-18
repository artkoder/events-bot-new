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
        "VK_WEEK_EDIT_ENABLED",
        "VK_WEEK_EDIT_SCHEDULE",
        "VK_WEEK_EDIT_TZ",
        "VK_CAPTCHA_TTL_MIN",
        "VK_CAPTCHA_QUIET",
    ]:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(main)
    assert main.WEEK_EDIT_MODE == "deferred"
    assert main.WEEK_EDIT_CRON == "02:30"
    assert main.CAPTCHA_WAIT_S == 600
    assert main.CAPTCHA_MAX_ATTEMPTS == 2
    assert main.CAPTCHA_NIGHT_RANGE == "00:00-07:00"
    assert main.CAPTCHA_RETRY_AT == "08:10"
    assert main.VK_WEEK_EDIT_ENABLED is False
    assert main.VK_WEEK_EDIT_SCHEDULE == "02:30"
    assert main.VK_WEEK_EDIT_TZ == "UTC"
    assert main.VK_CAPTCHA_TTL_MIN == 5
    assert main.VK_CAPTCHA_QUIET == "00:00-07:00"
