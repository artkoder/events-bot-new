# EVE Frameworks — Фаза 4 (Детальный дизайн)

Дата: 2026-01-07

Документ опирается на:
- Фаза 1: `docs/architecture/eve-arch-phase-1.md`
- Фаза 2: `docs/architecture/eve-arch-phase-2.md`
- Фаза 3: `docs/architecture/eve-arch-questions.md`
- Ответы PO: `docs/architecture/eve-arch-answers.md`

Ключевые решения из ответов PO:
- **Нет gateway**: Kaggle делает **прямые** вызовы Google AI.
- Kaggle получает **Supabase credentials через secrets**.
- **Ключи нельзя хранить в Supabase** (только env/secrets).
- **Лимиты считаются “на ключ”** (ключ привязан к аккаунту).
- **RPD в UTC**.
- **WAIT режим не нужен**: при блокировке — **сразу ошибка**.
- **Ретраи только на провайдерные ошибки**, максимум 3.
- Usage токенов берём из ответа Google API (fallback “нет usage” не требуется).
- Каноническое имя модели для БД/логов: `gemma-3-27b`.

---

## 0) Термины и инварианты

**Окна:**
- `minute_bucket`: `date_trunc('minute', now_utc)`.
- `day_bucket`: `date(now_utc)` (UTC).

**Лимиты (per api_key_id + model):**
- `RPM`: requests/minute
- `TPM`: tokens/minute
- `RPD`: requests/day

**Инварианты:**
1. Любая попытка вызова Google AI проходит через атомарный `reserve` в Supabase, который одновременно проверяет и инкрементит RPM/TPM/RPD.
2. При превышении лимитов возвращаем ошибку **сразу** (без ожидания).
3. Ретраи выполняются **только** на провайдерные ошибки и максимум 3; перед каждой попыткой — новый `reserve`.
4. База вычисляет время (`now_utc`) для bucket’ов (защита от clock skew клиентов).
5. Таблица `token_usage` остаётся неизменной; интеграция с ней — через `meta` (обратная совместимость).

---

## 1) Схема данных (Supabase/Postgres)

Ниже — целевая схема с упором на атомарность, идемпотентность и аудит. Она **не требует** изменений в `token_usage` и допускает поэтапное внедрение.

### 1.1. `token_usage` (существующая; не ломаем)

Используется в `main.py:1813` (`log_token_usage(...)`). Для обратной совместимости:
- **не меняем** таблицу и существующие поля;
- для Google AI пишем туда же (опционально) отдельной записью:
  - `bot`: `bot` / `kaggle` / `script`
  - `model`: каноническая строка (`gemma-3-27b`, `gemini-2.5-flash`, …)
  - `endpoint`: `google_ai.generate_content`
  - токены: из usage ответа Google (`input_tokens`, `output_tokens`, `total_tokens`)
  - `meta`: включает `provider=google`, `api_key_id`, `request_uid`, `attempt_no`, `minute_bucket`, `day_bucket`, `account_name`, …

Это обеспечивает единый “сквозной” учёт расхода токенов без миграций `token_usage`.

### 1.2. `google_ai_api_keys` (метаданные ключей; без секретов)

Хранит **только** метаданные и селектор того, где лежит секрет.

Поля:
- `id uuid primary key` — идентификатор ключа (используется для лимитов и аудита).
- `key_alias text not null` — человекочитаемый алиас (например `prod-gemma-1`).
- `provider text not null default 'google'`
- `env_var_name text not null` — имя env var/secrets, где лежит **реальный** API key (например `GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`).
- `account_name text null` — “аккаунт/локальное имя” для меток (`GOOGLE_API_LOCALNAME`), не для расчёта лимитов.
- `is_active bool not null default true`
- `priority int not null default 100` — порядок выбора ключа (меньше = приоритетнее).
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`
- `notes text null`

Индексы/ограничения:
- `unique(provider, key_alias)`
- `index (is_active, priority)`

### 1.3. `google_ai_model_limits` (лимиты моделей; единая точка правды)

Поля:
- `model text primary key` — канонический id модели (например `gemma-3-27b`).
- `rpm int not null`
- `tpm int not null`
- `rpd int not null`
- `tpm_reserve_extra int not null default 0` — безопасный “оверрезерв” (без token counting), чтобы снизить риск превышения TPM при больших промптах.
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Индексы:
- PK по `model` достаточно.

### 1.4. `google_ai_usage_counters` (счётчики окон; атомарные инкременты)

Текущая таблица на скриншоте содержит `minute_bucket` и `day_bucket` одновременно. Чтобы сделать атомарные операции и избежать двусмысленности `rpd_used`, фиксируем семантику как “двухтипные строки” в одной таблице:

- **Минутная строка**: `minute_bucket is not null` + `rpm_used/tpm_used` используются, `rpd_used = 0`.
- **Суточная строка**: `minute_bucket is null` + `rpd_used` используется, `rpm_used=tpm_used=0`.

Поля (минимально совместимо со скриншотной схемой):
- `id bigserial primary key` (или существующий `int8`)
- `api_key_id uuid not null references google_ai_api_keys(id)`
- `model text not null references google_ai_model_limits(model)` (или `text` + FK позже)
- `minute_bucket timestamptz null`
- `day_bucket date not null`
- `rpm_used int not null default 0`
- `tpm_used int not null default 0`
- `rpd_used int not null default 0`
- `updated_at timestamptz not null default now()`

Ключевые индексы (для атомарных upsert’ов):
- Частичный unique индекс минутных строк:
  - `unique(api_key_id, model, minute_bucket) where minute_bucket is not null`
- Частичный unique индекс дневных строк:
  - `unique(api_key_id, model, day_bucket) where minute_bucket is null`
- Индексы для обслуживания/ретеншена:
  - `index (minute_bucket) where minute_bucket is not null`
  - `index (day_bucket) where minute_bucket is null`

Почему это важно: без уникального ключа по окну невозможен корректный `INSERT ... ON CONFLICT DO UPDATE` в `reserve`.

### 1.5. `google_ai_requests` (идемпотентность + аудит уровня “запрос”)

Назначение:
- обеспечить `request_uid` (идемпотентность reserve/finalize);
- хранить статусы “зарезервировали”, “отправили”, “финализировали”;
- дать основу для обработки падений “между reserve и finalize”.

Поля:
- `request_uid uuid primary key` — генерируется клиентом (SDK) один раз на logical-request.
- `consumer text not null` — `bot` / `kaggle` / `script` / …
- `account_name text null` — метка (из secrets/env).
- `provider text not null default 'google'`
- `model text not null`
- `api_key_id uuid not null`
- `minute_bucket timestamptz not null`
- `day_bucket date not null`
- `reserved_rpm int not null default 1`
- `reserved_tpm int not null` — сколько токенов зарезервировали в минутном окне (см. 2.1).
- `reserved_rpd int not null default 1`
- `status text not null` — `reserved` | `sent` | `succeeded` | `failed_provider` | `failed_limit` | `finalized` | `stale`
- `attempts int not null default 0`
- `last_error_kind text null` — `provider` / `limit` / `internal`
- `last_error_code text null`
- `last_error_message text null`
- `sent_at timestamptz null`
- `finalized_at timestamptz null`
- `usage_input_tokens int null`
- `usage_output_tokens int null`
- `usage_total_tokens int null`
- `meta jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Индексы:
- `index (created_at)`
- `index (consumer, created_at)`
- `index (api_key_id, minute_bucket)`
- `index (status, updated_at)` (для “свипера” stale)

### 1.6. `google_ai_request_attempts` (аудит уровня “попытка”)

Назначение:
- запись каждой попытки (включая blocked по лимитам);
- хранение latencies, retryable, ошибок провайдера.

Поля:
- `id bigserial primary key`
- `request_uid uuid not null references google_ai_requests(request_uid)`
- `attempt_no int not null` (начиная с 1)
- `status text not null` — `reserved` | `blocked` | `sent` | `succeeded` | `failed_provider` | `finalized`
- `blocked_reason text null` — `rpm` | `tpm` | `rpd`
- `retry_after_ms int null` — подсказка для клиента (даже если WAIT не реализуем).
- `provider_status int null`
- `provider_error_type text null`
- `provider_error_code text null`
- `provider_error_message text null`
- `reserved_tpm int not null`
- `usage_total_tokens int null`
- `started_at timestamptz not null default now()`
- `completed_at timestamptz null`
- `duration_ms int null`
- `meta jsonb not null default '{}'::jsonb`

Индексы:
- `unique(request_uid, attempt_no)`
- `index (started_at)`
- `index (status, started_at)`

---

## 2) Алгоритм контроля лимитов (NO_WAIT, атомарно)

### 2.1. Входные данные для `reserve`

SDK перед вызовом провайдера формирует:
- `request_uid` (UUID v4) — общий на logical-request.
- `attempt_no` — 1..3.
- `consumer` (`bot` / `kaggle` / …).
- `account_name` (из secrets/env, только для меток).
- `model` (каноническая строка, например `gemma-3-27b`).
- `candidate_key_ids` (опционально):
  - если consumer имеет **один** ключ → передаёт ровно его `api_key_id`;
  - если consumer имеет пул ключей → передаёт список ключей, которые реально доступны в env/secrets данного runtime.
- `reserved_tpm`:
  - базово: `max_output_tokens` (из generation config) + `google_ai_model_limits.tpm_reserve_extra`;
  - если `max_output_tokens` не задан — используем консервативное значение из конфигурации SDK (например `DEFAULT_MAX_OUTPUT_TOKENS`), либо запрещаем вызов без него.

Примечание по ответам PO (“planned_tokens не нужно”):
- мы **не считаем** точные токены промпта;
- но для TPM всё равно нужен **консервативный резерв**, иначе пред-проверка TPM невозможна.

### 2.2. RPC `google_ai_reserve(...)` (основной атомарный шаг)

Семантика:
- вычислить `minute_bucket/day_bucket` по **времени БД** (UTC);
- выбрать `api_key_id` (см. 2.3);
- атомарно:
  - инкрементнуть минутные `rpm_used += 1`, `tpm_used += reserved_tpm`;
  - инкрементнуть дневные `rpd_used += 1`;
  - создать/обновить `google_ai_requests` и добавить строку `google_ai_request_attempts`.
- если лимит превышен на всех кандидатах — вернуть `blocked` (без ожидания).

Псевдо-логика (внутри функции, в одной транзакции):
1. `now_utc := timezone('utc', now())`
2. `minute_bucket := date_trunc('minute', now_utc)`
3. `day_bucket := (now_utc)::date`
4. `limits := select rpm,tpm,rpd,tpm_reserve_extra from google_ai_model_limits where model=$model`
5. `reserved_tpm_effective := $reserved_tpm` (уже включает extra)
6. `candidates := $candidate_key_ids` или `select id from google_ai_api_keys where is_active order by priority, id`
7. Для каждого `api_key_id`:
   - в под-транзакции (savepoint):
     - upsert минутную строку и условно инкрементнуть:
       - `... WHERE rpm_used + 1 <= rpm_limit AND tpm_used + reserved_tpm_effective <= tpm_limit`
     - upsert дневную строку и условно инкрементнуть:
       - `... WHERE rpd_used + 1 <= rpd_limit`
     - upsert `google_ai_requests` (идемпотентно по `request_uid`):
       - если `request_uid` уже существует и совпадает `(model, consumer)` → вернуть существующую резервацию (защита от повторов клиента);
       - иначе — ошибка конфликтов (request_uid reuse).
     - insert `google_ai_request_attempts` для `(request_uid, attempt_no)` (идемпотентно).
     - return success с `api_key_id`, bucket’ами, лимитами и текущими used.
8. Если не удалось ни на одном ключе:
   - вернуть `blocked_reason`:
     - если упёрлись в `rpd` → `blocked_rpd`
     - иначе `blocked_rpm_or_tpm`
   - `retry_after_ms` вычислить только для минутного окна: `ms до начала следующей минуты` (подсказка, хотя WAIT не делаем).

### 2.3. Выбор ключа

Так как секреты ключей не в БД, возможны 2 режима:

**Режим 1 (рекомендуется для Kaggle): fixed key**
- Kaggle runtime знает, какой конкретно ключ доступен (один секрет) → передаёт `candidate_key_ids = [<that id>]`.
- RPC либо резервирует на этом ключе, либо сразу блокирует.

**Режим 2 (для бота/внутренних задач): pool**
- runtime хранит несколько ключей в env/secrets → передаёт список `candidate_key_ids`.
- RPC выбирает первый подходящий по `priority` (и/или round-robin на уровне БД, если понадобится позже).

Отдельно: `account_name` не участвует в выборе ключа и не влияет на лимиты; это метка для логов/аналитики.

### 2.4. Вызов провайдера и `finalize`

Шаги SDK на попытке:
1. `reserve` через Supabase RPC → получить `api_key_id` и `env_var_name` (через join или вторым запросом).
2. `mark_sent` (обычный update `google_ai_requests.sent_at` + попытка `status=sent`) **до** реального вызова провайдера.
3. Вызвать `google.generativeai` с фактическим ключом из env/secrets.
4. Извлечь usage из ответа Google (`input_tokens`, `output_tokens`, `total_tokens`).
5. `finalize` через RPC:
   - сохранить usage в `google_ai_requests`/`google_ai_request_attempts`;
   - выполнить reconcile TPM: `tpm_used += (actual_total_tokens - reserved_tpm)` для `minute_bucket` и `api_key_id`.
6. Записать `token_usage` (существующий механизм) с `meta`, чтобы сохранить обратную совместимость.

RPC `google_ai_finalize(...)` должен быть **идемпотентным**:
- повторный вызов с тем же `request_uid` не должен “два раза” менять counters;
- если `finalized_at` уже заполнен — вернуть текущую запись без изменений.

### 2.5. Ретраи (только provider errors, max 3)

SDK retry loop:
- `attempt_no` от 1 до 3
- для каждой попытки:
  1) `reserve` (с новым attempt_no)
  2) провайдерный вызов
  3) `finalize`
- при `RateLimitExceeded` (blocked_reason rpm/tpm/rpd) → **сразу возвращаем ошибку**, не ретраим.
- при `ProviderRetryableError` → ретраим до 3 попыток (с backoff, например 250/500/1000ms + jitter).
- при `ProviderNonRetryableError` → сразу ошибка.

---

## 3) Компоненты системы

### 3.1. SDK библиотека (общая для бота и Kaggle)

Суть: python-библиотека-обёртка над `google.generativeai`, которая:
- выбирает ключ (через Supabase + `env_var_name`);
- делает `reserve → call → finalize`;
- пишет аудит (таблицы `google_ai_*`) и совместимый лог в `token_usage`;
- реализует retry policy (только provider errors, max 3).

Публичный интерфейс (эскиз):
- `GoogleAIClient.generate_content(...)`
- `GoogleAIClient.generate_content_async(...)`
- `RateLimitError` (rpm/tpm/rpd, includes `retry_after_ms`)
- `ProviderError` (retryable flag)

### 3.2. Потребители

**Бот**
- использует SDK как библиотеку;
- получает `SUPABASE_*` и все нужные `GOOGLE_API_KEY*` через env/secrets деплоя.

**Kaggle notebook**
- использует SDK (как vendor-код или pip-install из репозитория/артефакта);
- получает через Kaggle Secrets:
  - `SUPABASE_URL`
  - `SUPABASE_KEY` (см. 3.3)
  - `GOOGLE_API_KEY` (или набор ключей)
  - `GOOGLE_API_LOCALNAME` (для меток, если нужно)

### 3.3. Supabase credentials для Kaggle (минимизация прав)

PO подтвердил, что credentials можно пробрасывать в Kaggle. Рекомендуется:
- не давать `service_role` в Kaggle, если можно;
- создать отдельный ключ/роль с правами:
  - вызывать RPC `google_ai_reserve`, `google_ai_finalize`;
  - читать `google_ai_model_limits`, `google_ai_api_keys` (только метаданные);
  - вставлять в `token_usage` (или выполнять это через RPC тоже).

Если “анон ключ + RLS” невозможен для RPC-логики, использовать выделенный “backend key” с ограниченными grants.

---

## 4) Формат логов и аудит

### 4.1. Runtime-логи (stdout, JSON lines)

Единый формат событий (пример полей):
- `ts` (ISO8601 UTC)
- `event`:
  - `google_ai.reserve_ok`
  - `google_ai.reserve_blocked`
  - `google_ai.call_start`
  - `google_ai.call_ok`
  - `google_ai.call_error`
  - `google_ai.finalize_ok`
- `request_uid`, `attempt_no`
- `consumer`, `account_name`
- `model`, `provider`
- `api_key_id`, `key_alias`
- `minute_bucket`, `day_bucket`
- `limits`: `{rpm,tpm,rpd}`
- `reserved`: `{rpm:1,tpm:<n>,rpd:1}`
- `usage`: `{input,output,total}` (только после ответа)
- `duration_ms`
- `blocked_reason`, `retry_after_ms`
- `error`: `{type,code,message,retryable}`

Важно:
- не логировать полный prompt/response; максимум — длины и хэши (например `prompt_sha256`), чтобы не утекали данные.

### 4.2. Аудит в БД

Запись в БД даёт две “гранулярности”:
- `google_ai_requests`: итог по logical-request (status, ключ, usage).
- `google_ai_request_attempts`: каждая попытка (blocked/success/provider errors) с latency и причиной.

Плюс — `token_usage` для совместимости/аналитики токенов в одном месте.

---

## 5) Edge cases и обработка

### 5.1. Падение между `reserve` и `finalize`

Разделяем два случая:

1) **Успели `reserve`, но не успели `sent`** (падение до реального вызова провайдера)
- В `google_ai_requests.status = reserved`, `sent_at is null`.
- Политика: через “свипер” пометить как `stale` и **компенсировать** counters:
  - `rpm_used -= 1`, `tpm_used -= reserved_tpm`, `rpd_used -= 1` для соответствующих bucket’ов.
- Требование безопасности: компенсируем только если точно знаем, что `sent_at is null`.

2) **Успели `sent`, но не успели `finalize`**
- Риск: провайдер мог обработать запрос и потратить токены, но мы не записали usage.
- Политика: **не компенсировать** counters автоматически (консервативно), пометить `status=sent`/`stale` и оставить “ручной/фоновой” reconcile (если usage можно восстановить из артефактов/логов).

### 5.2. Clock skew

Все bucket’ы считаются на стороне БД (`now()`), поэтому клиентские часы не влияют.

### 5.3. Конкурентность / гонки

Защита:
- частичные unique индексы по `(api_key_id, model, minute_bucket)` и `(api_key_id, model, day_bucket)` для корректного upsert;
- условный `DO UPDATE ... WHERE used + delta <= limit` гарантирует, что при параллельных запросах только часть транзакций сможет обновить строку.

### 5.4. Двойные списания (повторы клиента)

Защита:
- `google_ai_requests.request_uid` — PK.
- `google_ai_request_attempts` — `unique(request_uid, attempt_no)`.
- RPC должны быть идемпотентными: повторный `reserve`/`finalize` не должен повторно менять counters.

### 5.5. Провайдер вернул ошибку после `reserve`

Политика:
- RPM/RPD считаем потраченными (попытка была).
- TPM:
  - если usage не получен (ошибка) — считаем потраченным **reserved_tpm** (консервативно);
  - если провайдер возвращает usage даже на ошибке (маловероятно) — reconcile по факту.

---

## 6) План тестирования

### 6.1. Unit tests (SDK)

- `reserved_tpm` вычисление (учёт `max_output_tokens` + `tpm_reserve_extra`).
- Retry policy:
  - лимитные ошибки → без ретраев;
  - provider retryable → до 3 попыток, корректные backoff/jitter;
  - provider non-retryable → без ретраев.
- Идемпотентность на уровне клиента:
  - повторный `finalize` не меняет локально состояние и корректно обрабатывает “already finalized”.
- Формат лог-событий: обязательные поля присутствуют, нет утечек prompt/response.

### 6.2. Integration tests (Postgres/Supabase)

- `google_ai_reserve`:
  - конкурентные параллельные резервации на один ключ → не превышаем rpm/tpm/rpd;
  - корректный выбор ключа из `candidate_key_ids`;
  - `blocked_reason` и `retry_after_ms` для минутного окна.
- `google_ai_finalize`:
  - reconcile TPM (delta положительная/отрицательная);
  - идемпотентность finalize.
- “Падение” сценарии:
  - `reserved` без `sent` → компенсация “свипером” (если реализован);
  - `sent` без finalize → не компенсируем.

### 6.3. E2E (бот + Kaggle)

- Серия запросов до 80% лимита → все success, корректные counters.
- Упёрлись в минутный RPM/TPM → сразу ошибка `RateLimitError` (без WAIT).
- Упёрлись в дневной RPD → сразу ошибка.
- Параллельная нагрузка (например 50 одновременных вызовов SDK) → нет превышений, корректная блокировка.

---

## 7) Фреймворк секретов (EVE-54)

### 7.1. Проблема и цель

Kaggle Secrets (UI) удобны для ручной настройки, но **Kaggle Secrets API недоступен** для программного обновления из бота. Поэтому нужен механизм, который позволяет:
- безопасно **передавать/обновлять** секреты из бота в Kaggle;
- поддерживать **несколько типов секретов** (не только `GOOGLE_API_KEY`);
- поддерживать **пулы** (несколько `GOOGLE_API_KEY*`);
- работать в Kaggle runtime без записи секретов на диск и без утечек в логи.

Исходная MVP-реализация уже существует: `kaggle/UniversalFestivalParser/src/secrets.py` (Fernet + 2 приватных датасета).

### 7.2. Набор секретов (минимальный)

Фреймворк должен поддерживать как минимум:
- `SUPABASE_URL`
- `SUPABASE_KEY` (ограниченный по правам ключ/роль для вызова RPC `google_ai_*`)
- `GOOGLE_API_KEY*` (пул ключей; реестр метаданных в `google_ai_api_keys.env_var_name`)
- любые будущие секреты (строковые значения), без изменения механизма доставки

Рекомендуется хранить **только значения секретов** (без метаданных) и поддерживать версионирование пакета на уровне транспорта.

### 7.3. Единый API и порядок fallback’ов

На стороне runtime (бот и Kaggle) секреты должны читаться через единый интерфейс, например:
- `get_secret(name: str) -> str | None`
- `get_secret_pool(prefix: str) -> list[str]` (например `GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`, …)

Порядок источников (fallback chain), единый для всех секретов:
1) **env** (`os.getenv`)
2) **Kaggle Secrets** (`kaggle_secrets.UserSecretsClient`) — может быть недоступен/пуст в некоторых контекстах
3) **Kaggle Datasets (EVE-54)**: приватные датасеты + шифрование (см. 7.4)

Важно: SDK лимитов должен получать секреты через этот фреймворк, чтобы Kaggle мог работать даже при отсутствии/необновляемости Kaggle Secrets.

### 7.4. Transport: “датасеты + шифрование” (2 датасета, Fernet)

#### Инварианты безопасности
- **Cipher** и **ключ(и)** лежат в **разных приватных** датасетах.
- Расшифровка происходит **только в памяти**, секреты не пишутся на диск.
- Владельцем датасетов является **доверенный Kaggle аккаунт**; доступ — минимум необходимый.
- Логи/аудит не содержат значений секретов.

#### Структура датасетов (рекомендуемая, расширяемая)

Датасет A (cipher), например `eve-secrets-cipher`:
- `secrets.enc` — один бинарный blob, шифрованный Fernet, внутри — JSON с секретами:
  - `{"SUPABASE_URL":"...","SUPABASE_KEY":"...","GOOGLE_API_KEY":"...","GOOGLE_API_KEY_2":"...", ...}`
  - можно добавить `schema_version`, `created_at`, `active_key_id` (внутри зашифрованного payload)

Датасет B (key ring), например `eve-secrets-key`:
- `fernet.keys` — key ring (по одной base64-строке на ключ), **минимум 1 ключ**

Почему key ring: позволяет **ротацию Fernet ключа без даунтайма** (см. 7.6).

Примечание про текущий MVP:
- Сейчас `kaggle/UniversalFestivalParser/src/secrets.py` ожидает `google_api_key.enc` и `fernet.key` в датасетах `gemma-cipher` / `gemma-key`.
- Фреймворк EVE-54 расширяет это до “bundle + key ring”; миграция возможна без изменения принципов безопасности.

### 7.5. Обновление датасетов из бота (Kaggle API/CLI)

Бот обновляет датасеты через Kaggle API (или Kaggle CLI как thin-wrapper над API):
- аутентификация: `KAGGLE_USERNAME` + `KAGGLE_KEY` (доверенный аккаунт-владелец датасетов)
- операция: **dataset create version** (публикация новой версии датасета)

Практический паттерн (совместим с текущими хелперами в репозитории, например `video_announce/kaggle_client.py`):
1) Сформировать во временной директории:
   - `dataset-metadata.json` (id, title, licenses)
   - файлы датасета (`secrets.enc` или `fernet.keys`)
2) Вызвать Kaggle API `dataset_create_version(...)` (или `kaggle datasets version -p <dir> -m "<msg>"`)
3) Очистить временную директорию (в ней нет plaintext секретов; есть только ciphertext или key ring)

### 7.6. Ротация: значения секретов, пул ключей и Fernet ключ

**Ротация значений (без смены Fernet ключа)**:
- перешифровать bundle тем же активным Fernet ключом;
- опубликовать новую версию cipher-датасета.

**Ротация Google API keys как “пул”**:
- секреты `GOOGLE_API_KEY*` обновляются в bundle;
- метаданные/порядок использования управляются в Supabase через `google_ai_api_keys`:
  - включение/выключение ключа (`is_active`)
  - приоритет (`priority`)
- runtime (бот/Kaggle) выбирает доступные ключи по `env_var_name` через secrets framework и передаёт их как `candidate_key_ids` в `google_ai_reserve`.

**Ротация Fernet ключа (без простоя)** — через key ring:
1) Сгенерировать новый Fernet ключ и **добавить его** в `fernet.keys`, не удаляя старые.
2) Опубликовать новую версию key-датасета (key ring расширился).
3) Перешифровать `secrets.enc` новым ключом и опубликовать новую версию cipher-датасета.
4) После окна миграции удалить старые ключи из `fernet.keys` (опционально).

Свойство: в любой момент времени Kaggle runtime сможет расшифровать bundle, потому что key ring содержит **и старый, и новый** ключи в переходный период.

### 7.7. Runbook (операционные шаги)

**Добавить новый секрет**:
1) Добавить имя секрета в “реестр” (док/код фреймворка), определить где он используется.
2) Обновить bundle (`secrets.enc`) и опубликовать новую версию cipher-датасета.
3) В Kaggle runtime убедиться, что датасеты подключены как inputs, и секрет читается через fallback chain.

**Добавить новый Google API key в пул**:
1) Добавить метаданные ключа в `google_ai_api_keys` (`env_var_name=GOOGLE_API_KEY_N`, `priority`, `is_active=true`).
2) Добавить значение `GOOGLE_API_KEY_N` в bundle и обновить cipher-датасет.
3) Верифицировать, что Kaggle/bot видят ключ (secrets provider) и что reserve выбирает ключ (аудит в `google_ai_requests`).

**Ротировать (заменить) значение секрета**:
1) Обновить bundle и опубликовать новую версию cipher-датасета.
2) Проверить, что Kaggle kernel подтянул новую версию input dataset (новый run).

**Ротировать Fernet ключ**: см. 7.6.

**Компрометация**:
1) Отключить ключ в `google_ai_api_keys.is_active=false` (немедленная остановка использования).
2) Выпустить новый ключ, обновить bundle, затем удалить/отозвать старый у провайдера.
3) При необходимости — выполнить ротацию Fernet ключа (key ring) и перевыпустить оба датасета.
