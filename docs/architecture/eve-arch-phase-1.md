# EVE Frameworks — Фаза 1 (Исследование)

Цель фазы: зафиксировать текущее состояние кода/интеграций и доступную информацию о Supabase-схеме для задач `EVE-11`, `EVE-54`, `EVE-55`.

## 1) Что уже есть в коде про LLM

### 1.1. LLM в основном боте (`main.py`)

- Основной боевой LLM-пайплайн в `main.py` сейчас использует **OpenAI Chat Completions** (модель `"gpt-4o"`) для парсинга событий и вспомогательных задач (например, `parse_event_via_4o`, `ask_4o`).
- Учёт расхода токенов уже реализован: `main.py` содержит `log_token_usage(...)`, который пишет в Supabase таблицу `token_usage`.
  - Вставка делается через `client.table("token_usage").insert(row).execute()` (Supabase python client).
  - Время `at` формируется на стороне сервиса: `datetime.now(timezone.utc).isoformat()` (не “время БД”).

### 1.2. Gemma / Google AI сейчас используется в Kaggle-парсере

В репозитории уже есть отдельный “Kaggle/локальный” пайплайн Universal Festival Parser, который вызывает Google AI (Gemma) напрямую:

- `kaggle/UniversalFestivalParser/src/reason.py`
  - `reason_with_gemma(...)` использует `google.generativeai` и вызывает `GenerativeModel(...).generate_content_async(...)`.
  - `validate_and_enhance(...)` делает второй проход через `generate_content_async(...)`.
- `kaggle/UniversalFestivalParser/src/enrich.py`
  - `extract_ticket_info_with_llm(...)` также использует `google.generativeai` для извлечения цен/статусов билетов с текстов страниц.
- `kaggle/UniversalFestivalParser/src/rate_limit.py`
  - локальный (in-process) лимитер `GemmaRateLimiter` на token bucket с safety-margin; **не глобальный** и не координируется между процессами/сервисами.
- `kaggle/UniversalFestivalParser/src/llm_logger.py`
  - файловый логгер `LLMLogger`, сохраняющий промпты/ответы/оценки токенов в `llm_log.json` рядом с артефактами парсинга.

### 1.3. “Фича-флаг” Festival Parser в `main.py`

В `main.py` есть только проверка наличия `GOOGLE_API_KEY` и лог о том, включён ли Festival Parser; прямых вызовов Google AI API из `main.py` не найдено.

## 2) Где сейчас вызывается Google AI API

Найденные места вызова SDK `google.generativeai`:

- `kaggle/UniversalFestivalParser/src/reason.py`
  - `reason_with_gemma(...)` → `generate_content_async`
  - `validate_and_enhance(...)` → `generate_content_async`
- `kaggle/UniversalFestivalParser/src/enrich.py`
  - `extract_ticket_info_with_llm(...)` → `generate_content_async`

Косвенные entrypoints/обвязки:

- `kaggle/UniversalFestivalParser/universal_festival_parser.py` (основной сценарий пайплайна: Render → Distill → Reason → Validate).
- `scripts/test_festival_parser_local.py` (локальный прогон пайплайна, также вызывает `reason_with_gemma(...)` и опционально `enrich_event_prices(...)`).

## 3) Как сейчас управляются “секреты” для Gemma в Kaggle

`kaggle/UniversalFestivalParser/src/secrets.py` реализует три источника `GOOGLE_API_KEY` (по приоритету):

1. `GOOGLE_API_KEY` в env.
2. Kaggle Secrets (`kaggle_secrets.UserSecretsClient().get_secret("GOOGLE_API_KEY")`).
3. Две приватные Kaggle datasets: одна с `google_api_key.enc`, другая с `fernet.key` (Fernet-расшифровка в памяти).

Это уже “локальный” фреймворк доставки секрета в Kaggle, но он:
- не связан с Supabase пулом ключей (`google_ai_api_keys`);
- не поддерживает балансировку по нескольким ключам;
- не даёт централизованного аудита попыток/блокировок.

## 4) Supabase в проекте (текущее использование)

### 4.1. Клиент Supabase

`main.py` содержит `get_supabase_client()`, который:
- читает `SUPABASE_URL`, `SUPABASE_KEY` (и `SUPABASE_DISABLED`);
- нормализует URL (обрезает `/rest/v1`, `/storage/v1` и т.п.);
- создаёт клиента через `supabase.create_client(...)` и использует `httpx.Client(...)`.

### 4.2. Что уже пишется в Supabase

Из кода активно используется `token_usage` (лог токенов 4o) и Supabase Storage (публикация ICS), но в репозитории **нет** кода, который обращается к таблицам `google_ai_*`.

## 5) Supabase-схема для Google AI (по скриншоту из бэклога)

Схема присутствует в `docs/backlog/linear/assets/eve-11-supabase-schema.png` (в репозитории не найдено SQL-миграций этой схемы).

### 5.1. Таблицы Google AI

**`google_ai_usage_counters`**
- `id` int8
- `api_key_id` uuid
- `model` text
- `minute_bucket` timestamptz
- `day_bucket` date
- `rpm_used` int4
- `tpm_used` int4
- `rpd_used` int4
- `updated_at` timestamptz

**`google_ai_api_keys`**
- `id` uuid
- `key_alias` text
- `provider` text
- `env_var_name` text
- `is_active` bool
- `priority` int4
- `created_at` timestamptz
- `notes` text

**`google_ai_model_limits`**
- `id` int4
- `model` text
- `rpm` int4
- `tpm` int4
- `rpd` int4
- `created_at` timestamptz
- `updated_at` timestamptz

**`google_ai_maintenance_log`**
- `id` int4
- `operation` text
- `tables_affected` _text
- `deleted_count` int4
- `started_at` timestamptz
- `completed_at` timestamptz
- `status` text
- `details` jsonb
- `created_at` timestamptz

### 5.2. Потенциальные неоднозначности схемы (нужно уточнить)

- `google_ai_usage_counters` одновременно содержит `minute_bucket` и `day_bucket`, и поля `rpm_used/tpm_used/rpd_used`.
  - Неочевидно, как предполагается хранить `rpd_used` (суточный счётчик) без отдельной дневной агрегации/таблицы.
  - Неочевидны уникальные ключи/индексы (например, `(api_key_id, model, minute_bucket)` и/или `(api_key_id, model, day_bucket)`), без них атомарный upsert невозможен.

## 6) Переменные окружения и конфиг (что найдено)

- Уже используется:
  - `GOOGLE_API_KEY` (для Kaggle-парсера и “фича-флага” Festival Parser в `main.py`)
  - `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_DISABLED`
- В требованиях (EVE-11) упоминается, но в коде не найдено:
  - `GOOGLE_API_LOCALNAME`

## 7) Промежуточный отчёт (Фаза 1)

Собрано и зафиксировано:
- В основном боте нет прямых вызовов Google AI; есть готовая интеграция с Supabase и таблицей `token_usage`.
- Google AI (Gemma) уже используется в `kaggle/UniversalFestivalParser` через `google.generativeai` (3 места вызовов).
- В Kaggle уже есть локальная схема доставки секрета (`secrets.py`) и локальный лимитер (`rate_limit.py`), но они не решают “глобальные лимиты + пул ключей + атомарность + аудит”.
- Supabase-схема `google_ai_*` доступна только в виде скриншота; SQL/миграции и индексы в репозитории не обнаружены; есть неоднозначности по хранению `rpd_used`.

