# Секреты для Kaggle jobs

Цель: безопасно передавать секреты в Kaggle kernels, которые запускаются из бота через Kaggle API.

## Ограничение

Для запусков через API нельзя надёжно рассчитывать на Kaggle Secrets (UI). Поэтому основной механизм доставки секретов — **приватные Kaggle datasets**.

## Базовый паттерн (encrypted split datasets)

1) На сервере:
   - собрать все секреты, нужные ноутбуку (например `TELEGRAM_AUTH_BUNDLE_S22`, `TG_API_ID`, `TG_API_HASH`, `GOOGLE_API_KEY`);
   - зашифровать **весь** секретный payload (например Fernet);
   - обновить/создать два приватных датасета:
     - **cipher dataset**: ciphertext (`secrets.enc`) + (опционально) несекретный `config.json`;
     - **key dataset**: только ключ (`fernet.key`).

2) В `kernel-metadata.json`:
   - `is_private: true`
   - подключить оба dataset в `dataset_sources`.

3) В ноутбуке:
   - прочитать `secrets.enc` и `fernet.key`;
   - расшифровать в памяти;
   - не печатать секреты в stdout.

## Что запрещено

- хранить секреты в plaintext в датасетах (`config.json`, `.env`, ноутбук и т.п.)
- логировать секреты/конфиги в stdout (Kaggle logs)
- делать kernel/datasets публичными

## Практика в репозитории

- Пример “split datasets + Fernet”: `source_parsing/telegram/service.py` + `source_parsing/telegram/split_secrets.py`.
- Ещё один пример “split datasets + Fernet”: `telegraph_cache_sanitizer.py` (Telegraph cache sanitizer / TelegraphCacheProbe).
- `/3di` (`preview_3d/handlers.py` + `kaggle/Preview3D/preview_3d.ipynb`) тоже использует split datasets: `config.json` для non-secret runtime env и `secrets.enc`/`fernet.key` для `SUPABASE_URL` + service key.
- Smoke-probe отдельного Gemma key: `kaggle/GemmaKey2Probe/gemma_key2_probe.ipynb` + `kaggle/execute_gemma_key2_probe.py` (подробный runbook: `docs/features/llm-gateway/README.md`).
- Пример multi-source secrets в Kaggle: `kaggle/UniversalFestivalParser/src/secrets.py` (env → Kaggle Secrets → encrypted datasets).

## Telegram Auth Bundle для Kaggle (ручные запуски)

Если нужно использовать **одну строку Telethon‑сессии** в Kaggle Secrets (например, для ручных запусков ноутбука):

1) Локально скопируйте **всю строку после** `TELEGRAM_AUTH_BUNDLE=` (urlsafe base64).
2) Создайте Kaggle Secret:
   - `TELEGRAM_AUTH_BUNDLE_S22 = <скопированное значение>`

Пример использования в ноутбуке/скрипте:
```python
import base64, json
from kaggle_secrets import UserSecretsClient
from telethon import TelegramClient
from telethon.sessions import StringSession

secrets = UserSecretsClient()
API_ID = int(secrets.get_secret("TELEGRAM_API_ID"))
API_HASH = secrets.get_secret("TELEGRAM_API_HASH")

B64 = secrets.get_secret("TELEGRAM_AUTH_BUNDLE_S22")
bundle = json.loads(base64.urlsafe_b64decode(B64.encode("ascii")).decode("utf-8"))

client = TelegramClient(
    StringSession(bundle["session"]),
    API_ID,
    API_HASH,
    device_model=bundle["device_model"],
    system_version=bundle["system_version"],
    app_version=bundle["app_version"],
    lang_code=bundle["lang_code"],
    system_lang_code=bundle["system_lang_code"],
)
```

Важно: **не запускайте одну и ту же session строку параллельно** в двух процессах (иначе можно словить `AuthKeyDuplicatedError`). Разные session строки для одного аккаунта допустимы.

Рекомендация для разных окружений:

- храните отдельные bundle для prod и dev/e2e;
- для Telegram Monitoring можно явно выбрать, какой env-ключ брать, через
  `TG_MONITORING_AUTH_BUNDLE_ENV` (например `TELEGRAM_AUTH_BUNDLE_E2E`).

Примечание: для **автоматических запусков через Kaggle API** по‑прежнему используйте encrypted datasets (см. базовый паттерн выше), так как Kaggle Secrets не всегда доступны при API‑старте.

## Рекомендации по снижению риска утечки

- держать kernel и datasets строго приватными, без collaborators
- редактировать логи (masking), не печатать JSON-конфиги целиком
- при подозрении на утечку: ротация `TELEGRAM_AUTH_BUNDLE_S22`/API ключей и пересоздание датасетов
