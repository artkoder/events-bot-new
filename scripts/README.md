# Scripts

Скрипты для разработки/диагностики/поддержки.

## Правила

- Одноразовые скрипты: `scripts/tmp_*.py` (после использования удалять).
- Переиспользуемые скрипты: читаемое имя без `tmp_` + краткая инструкция в этом файле или в доке фичи/операций.
- Скрипты не должны писать в корень репозитория: результаты складываем в `artifacts/` (см. `artifacts/README.md`).

## Codex

- Wrapper для отчётов (gitignored): `scripts/codex/exec_report.sh`

## Запуск локального бота

- `scripts/run_bot_dev.sh` — запускает бота в polling режиме (`DEV_MODE=1`), подхватывает `.env` (если есть) и использует `DB_PATH` (по умолчанию `db_prod_snapshot.sqlite`).

## Права (DEV/E2E)

- `scripts/seed_dev_superadmin.py` — добавляет права superadmin в sqlite (по `DB_PATH`) для Telethon‑аккаунта из `TELEGRAM_AUTH_BUNDLE_E2E`/`TELEGRAM_SESSION` (нужно, если `/vk` отвечает `Access denied` в локальном тестовом боте).

## Диагностика Supabase / LLM Gateway

- `scripts/inspect/probe_supabase_rpc.py` — короткий probe RPC маршрутов Supabase (`google_ai_reserve`, `google_ai_finalize`, ...).
- `scripts/inspect/sweep_google_ai_stale.py` — запускает `google_ai_sweep_stale(...)` для безопасной очистки зависших `google_ai_requests.status='reserved'` с `sent_at is null`.
