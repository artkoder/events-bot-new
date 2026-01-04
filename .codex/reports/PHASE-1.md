# Фаза 1 — Анализ

## Что нашёл в коде
- `log_token_usage()` в `main.py` создаёт Supabase клиента, формирует строку и запускает внутренний `_log()` через `asyncio.create_task()` без ожидания завершения.
- `_log()` делает `client.table("token_usage").insert(row).execute()` в `asyncio.to_thread()`, затем пишет `log_token_usage success/failed`.
- Все вызовы `log_token_usage()` в проекте уже `await`-ятся, но это ожидание завершается сразу после `create_task()`.
- Основные вызовы из LLM: `parse_event_via_4o()` и `ask_4o()`; также обёртки в `poster_ocr.py` и `vision_test/__init__.py`.

## Проверка гипотезы про create_task
- Да, `create_task()` делает логирование «fire-and-forget». В долгоживущем процессе это обычно ок, но при завершении процесса/цикла (shutdown, перезапуск, однократный job/скрипт) незавершённые задачи могут не выполниться.
- В таком сценарии токены действительно могут не попасть в Supabase.

## Проверка прод-логов
- Команда `flyctl logs -a events-bot-new-wngqia --no-tail` недоступна: `flyctl` не установлен в окружении, сеть ограничена, установить/запустить невозможно в рамках текущих ограничений.
- Поэтому подтвердить наличие/отсутствие `log_token_usage skipped/failed/success` в прод-логах не смог.

## Дополнительные гипотезы
- `SUPABASE_URL`/`SUPABASE_KEY` отсутствуют или `SUPABASE_DISABLED=1` в прод-окружении → `log_token_usage skipped` (логируется на уровне DEBUG, может быть не виден).
- Ошибка вставки в Supabase (RLS/схема/сеть) → `log_token_usage failed`.
- Потенциальная потеря задач на shutdown (особенно если /parse запускается в короткоживущем контексте).
