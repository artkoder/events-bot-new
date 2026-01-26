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

### 2.2. Алгоритм работы
1.  **Reserve**: Клиент запрашивает резерв (примерно `max_output_tokens + 1000`).
    *   *Успех:* Получает `api_key` и разрешение.
    *   *Отказ:* Получает `RateLimitError` (Fail Fast, NO_WAIT).
2.  **Execute**: Вызов API провайдера (Google AI Studio).
    *   *Ошибка:* Если 5xx — ретрай. Если 429 — проброс ошибки.
3.  **Finalize**: Клиент отправляет реальную статистику (`input_tokens`, `output_tokens`) в БД для корректировки квот.

## 3. Возможности
*   **Multi-Account Sharding**: Поддержка ротации ключей/аккаунтов через переменную `GOOGLE_API_LOCALNAME`.
*   **Atomic Counting**: Исключает Race Conditions при параллельных запросах.
*   **Fail Fast**: Не ждет в очереди (чтобы не вешать воркера), а сразу падает, позволяя планировщику (JobOutbox) перезапустить задачу позже.
*   **Structured Logging**: Все вызовы логируются в формате JSON Lines для анализа.

## 4. Использование
```python
from google_ai.client import GoogleAIClient

client = GoogleAIClient(supabase_client=db)
try:
    text, usage = await client.generate_content_async(
        model="gemma-2-9b-it",
        prompt="Analyze this event..."
    )
except RateLimitError:
    print("Limits exceeded, try again later")
```
