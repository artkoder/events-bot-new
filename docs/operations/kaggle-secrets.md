# Секреты для Kaggle jobs

Цель: безопасно передавать секреты в Kaggle kernels, которые запускаются из бота через Kaggle API.

## Ограничение

Для запусков через API нельзя надёжно рассчитывать на Kaggle Secrets (UI). Поэтому основной механизм доставки секретов — **приватные Kaggle datasets**.

## Базовый паттерн (encrypted split datasets)

1) На сервере:
   - собрать все секреты, нужные ноутбуку (например `TG_SESSION`, `TG_API_ID`, `TG_API_HASH`, `GOOGLE_API_KEY`);
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
- Пример multi-source secrets в Kaggle: `kaggle/UniversalFestivalParser/src/secrets.py` (env → Kaggle Secrets → encrypted datasets).

## Рекомендации по снижению риска утечки

- держать kernel и datasets строго приватными, без collaborators
- редактировать логи (masking), не печатать JSON-конфиги целиком
- при подозрении на утечку: ротация `TG_SESSION`/API ключей и пересоздание датасетов

