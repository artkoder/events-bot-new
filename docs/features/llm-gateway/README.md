# LLM Limit Management Framework (LLM Gateway)

> **Linear Task:** [EVE-11](https://linear.app/events-bot-new/issue/EVE-11/llm-rate-limits)
> **Status:** ✅ Implemented
> **Component:** `google_ai.client.GoogleAIClient`

## 1. Цель
Обеспечить надежную работу с LLM (Gemma 2/3, Gemini) в условиях жестких ограничений API (RPM, TPM, Daily Limit), исключая "молчаливые" падения и превышения квот.

## 2. Архитектура
Фреймворк реализован как обертка над `google.generativeai` с централизованным контролем стейта через Supabase.

### 2.1. Ключевые компоненты
*   **GoogleAIClient (`google_ai/client.py`)**: Единая точка входа. Управляет повторными попытками (Retries), логированием и вызовом RPC.
*   **Supabase Database**:
    *   Таблицы `google_ai_*` хранят лимиты/счётчики/аудит. Схема описана в `docs/architecture/eve-arch-phase-1.md`.
    *   *Примечание:* Сами ключи хранятся в ENV, а Supabase возвращает имя переменной окружения для выбранного ключа.
*   **Supabase RPC (`google_ai_reserve`)**: Атомарное резервирование лимитов. Возвращает `env_var_name` (какую переменную среды читать).
*   **Supabase RPC (`google_ai_mark_sent`)**: Помечает, что запрос реально отправлен провайдеру (для диагностики/восстановления).
*   **Supabase RPC (`google_ai_finalize`)**: Фиксирует фактическое потребление токенов и статус провайдера.
*   **Reserve fallback (защита от “вечного fallback в Smart Update”)**:
    * если `google_ai_reserve` временно недоступен из-за миграционного/схемного рассинхрона (`PGRST202`, `Route ... not found`), клиент может перейти в bypass-режим и использовать `GOOGLE_API_KEY` напрямую;
    * управляется ENV `GOOGLE_AI_ALLOW_RESERVE_FALLBACK` (`1` по умолчанию, `0` для strict-режима).
    * fallback больше не “залипает” навсегда в процессе: клиент периодически перепроверяет RPC и автоматически возвращается к Supabase-limiter после восстановления.
    * интервал перепроверки: `GOOGLE_AI_RESERVE_RPC_RECHECK_SECONDS` (по умолчанию `600` сек).
    * при transient сетевых сбоях (`SSL handshake timeout`, `server disconnected`, `EOF`) reserve RPC сначала делает короткие retry и только затем переключается в local fallback:
      - `GOOGLE_AI_RESERVE_RPC_RETRY_ATTEMPTS` (по умолчанию `2`);
      - `GOOGLE_AI_RESERVE_RPC_RETRY_BASE_DELAY_MS` (по умолчанию `350` мс, exponential backoff + jitter).
*   **Совместимость с legacy Supabase-проектами (без миграций):**
    * если отсутствует `google_ai_finalize`, клиент автоматически переключается на `finalize_google_ai_usage`;
    * fallback на legacy finalize применяется не только для первого запроса, а для всех следующих в процессе;
    * это изменение только в коде клиента, без изменения RPC/таблиц в проде.
*   **Stale reservation recovery:**
    * transient RPC errors на `google_ai_mark_sent` / `google_ai_finalize` теперь имеют короткие retry (тот же backoff-профиль, что и reserve RPC);
    * для уже накопившихся записей доступен RPC `google_ai_sweep_stale(p_older_than_minutes, p_limit)`, который компенсирует counters только для безопасного окна `status='reserved' AND sent_at IS NULL`, затем помечает записи как `stale`;
    * ручной запуск из репозитория: `python scripts/inspect/sweep_google_ai_stale.py --use-service --older-than-minutes 30 --limit 500`.

### 2.3. Диагностика PGRST202 (RPC not found / schema cache)

Если локально/в CI вы видите `PGRST202` по `google_ai_reserve`/`google_ai_finalize`, это означает, что PostgREST не видит RPC в текущей схеме или роль не имеет прав на выполнение функции.

Важно: в таком состоянии межсервисный лимитер не работает как единый атомарный контроль (можно превысить общий RPM/TPM/RPD при параллельной нагрузке нескольких сервисов).

Что проверить:

*   Убедитесь, что задан `SUPABASE_SERVICE_KEY` (а не только `SUPABASE_KEY`). Для rate-limit RPC обычно нужен service role.
*   Если RPC лежит не в `public`, выставьте `SUPABASE_SCHEMA` в нужную схему (и проверьте, что PostgREST эту схему экспонирует).
*   Если в проекте есть только `finalize_google_ai_usage`, это допустимо: клиент продолжит работать в режиме legacy finalize без DDL-изменений.

Быстрый probe из репозитория (не печатает секреты, только статус/первые 400 символов ответа):

```bash
python scripts/inspect/probe_supabase_rpc.py google_ai_reserve --schema public
python scripts/inspect/probe_supabase_rpc.py google_ai_reserve --schema public --use-service
```

Если `--use-service` даёт 200/2xx, а без него 404/PGRST202, проблема в правах (нужен service key).

### 2.4. Включение межсервисного лимитера (rollout)

Для проектов, где есть только legacy `finalize_google_ai_usage`, нужно один раз применить SQL-миграцию:

```sql
-- файл из репозитория:
-- migrations/002_google_ai_rpc_rollout.sql
```

Что делает миграция:
*   добавляет недостающие таблицы счётчиков/аудита `google_ai_usage_counters`, `google_ai_requests`, `google_ai_request_attempts`;
*   создаёт RPC `google_ai_reserve`, `google_ai_mark_sent`, `google_ai_finalize`;
*   добавляет только недостающие колонки в уже существующие `google_ai_model_limits/google_ai_api_keys` (без destructive-изменений).

Проверка после применения:

```bash
python scripts/inspect/probe_supabase_rpc.py google_ai_reserve --schema public
python scripts/inspect/probe_supabase_rpc.py google_ai_mark_sent --schema public
python scripts/inspect/probe_supabase_rpc.py google_ai_finalize --schema public
```

Ожидание:
*   больше нет `404/PGRST202` по `google_ai_reserve/google_ai_mark_sent/google_ai_finalize`;
*   `google_ai_reserve` отвечает JSON-объектом (даже если `ok:false` по причине `rpd/rpm/tpm/no_keys`).

### 2.2. Алгоритм работы
1.  **Reserve**: Клиент запрашивает резерв (примерно `max_output_tokens + 1000`).
    *   Для длинных текстовых prompt’ов используется консервативная оценка по байтам **и** символам; это особенно важно для русскоязычных/OCR-heavy запросов, где простой `bytes/4` может занизить реальный input TPM.
    *   *Успех:* Получает `api_key` и разрешение.
    *   *Отказ:* Получает `RateLimitError` (Fail Fast, NO_WAIT).
2.  **Execute**: Вызов API провайдера (Google AI Studio).
    *   *Ошибка:* Если 5xx — ретрай. Если 429 — немедленный проброс ошибки наверх без внутреннего sleep/retry.
    *   *Пустой ответ:* трактуется как `ProviderError(empty_response)` и ретраится.
3.  **Finalize**: Клиент отправляет реальную статистику (`input_tokens`, `output_tokens`) в БД для корректировки квот.

## 3. Возможности
*   **Multi-Account Sharding**: Поддержка ротации ключей/аккаунтов через переменную `GOOGLE_API_LOCALNAME`.
*   **Atomic Counting**: Исключает Race Conditions при параллельных запросах.
*   **Fail Fast**: Не ждет в очереди (чтобы не вешать воркера), а сразу падает, позволяя планировщику (JobOutbox) перезапустить задачу позже.
    * Для provider-side `429` это тоже правило: ждать/деферить должен вызывающий workflow (`event_parse`, `vk_auto_queue`, Smart Update), а не сам `GoogleAIClient`.
*   **Structured Logging**: Все вызовы логируются в формате JSON Lines для анализа.
*   **Operational visibility**: bypass reserve логируется отдельным событием `google_ai.reserve_fallback_no_rpc` — это сигнал, что RPC-схему нужно починить.
*   **Incident alerts**: критические сбои LLM отправляются в админ-чат как инцидент (`notify_llm_incident`).
    * ENV `GOOGLE_AI_INCIDENT_NOTIFICATIONS=0` — выключить инцидент-алерты.
    * ENV `GOOGLE_AI_INCIDENT_COOLDOWN_SECONDS` — антиспам/дедуп уведомлений (по умолчанию 900 сек).
*   **Model fallback chain**: при финальном провале основной модели клиент переключается на запасные модели из `GOOGLE_AI_FALLBACK_MODELS` (через запятую) и логирует `google_ai.model_fallback`.
    * Для текстовых вызовов действует policy качества: `gemma-3-27b` всегда пробуется первой в цепочке.
    * Gemma-модели меньше `12b` (`1b/4b`) автоматически исключаются из цепочки и не используются для текста.

### 3.1. Логирование конкретной модели (обязательно)
В JSON-логах клиента теперь фиксируются **оба** имени модели:

*   `model` — модель для лимитера (rate-limit model, нормализованная для RPC Supabase).
*   `requested_model` — модель, которую запросил вызывающий код.
*   `provider_model` — короткое имя модели у провайдера (например, `gemma-3-27b-it`).
*   `provider_model_name` — фактическое имя, отправленное в API провайдера (например, `models/gemma-3-27b-it`).
*   `invoked_model` — поле для быстрой проверки в логах: фактически вызванная модель.

Это нужно для пост-фактум проверки, что запрос ушёл именно в ожидаемую модель, а не только в её нормализованный alias для квот.

## 4. Использование
```python
from google_ai.client import GoogleAIClient

client = GoogleAIClient(supabase_client=db)
try:
    text, usage = await client.generate_content_async(
        model="gemma-3-27b",
        prompt="Analyze this event..."
    )
except RateLimitError:
    print("Limits exceeded, try again later")
```

## TODO / Risks
- Проверить и зафиксировать статус возможной deprecation `google.generativeai` (источник сигнала: операторский репорт), подготовить план миграции SDK при подтверждении.
