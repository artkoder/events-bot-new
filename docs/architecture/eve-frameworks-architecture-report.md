# EVE Frameworks — Итоговый сводный архитектурный отчёт (handoff на реализацию)

Дата: 2026-01-07

Связанные задачи: `EVE-11` (лимиты), `EVE-54` (секреты Kaggle), `EVE-55` (Gemma).

Источник решений: `docs/architecture/eve-arch-answers.md`. Детальный дизайн: `docs/architecture/eve-arch-design.md`.

---

## 1) Что строим (одной фразой)

**Единый Python SDK** (для бота и Kaggle), который оборачивает `google.generativeai` и перед каждым вызовом делает **атомарный reserve лимитов** (RPM/TPM/RPD) в Supabase, а после ответа — **finalize + аудит**, без gateway и без хранения ключей в БД.

---

## 2) Зафиксированные решения и ограничения (PO)

- Kaggle делает **прямые** вызовы Google AI API (нет gateway).
- Kaggle получает **Supabase credentials через secrets** (env/Kaggle Secrets/датасеты — см. EVE-54).
- Google API keys **нельзя хранить в Supabase** (только env/secrets).
- Лимиты считаются **на ключ**.
- RPD считается в **UTC**.
- При превышении лимита — **сразу ошибка** (NO_WAIT).
- Ретраи **только** на провайдерные ошибки, максимум 3.
- Google API возвращает usage токенов в ответе.
- Каноническое имя модели: `gemma-3-27b` (так хранить в БД/логах).

---

## 3) Компоненты и взаимодействие

### 3.1. Компоненты

1) **`eve_google_ai` SDK (Python)**
- Обёртка над `google.generativeai`.
- Делает `reserve → call → finalize`.
- Реализует retry policy (provider errors only, max 3).
- Пишет аудит в `google_ai_*` и (опционально) usage в `token_usage` для обратной совместимости.

2) **Supabase Postgres**
- Хранит лимиты моделей, метаданные ключей (без секретов), счётчики окон и аудит запросов.
- Даёт атомарность через RPC (Postgres functions).

3) **Потребители**
- **Бот** (prod/runtime env): имеет Supabase credentials и нужные Google API keys в env/secrets.
- **Kaggle notebook**: получает секреты через общий фреймворк (env → Kaggle Secrets → encrypted datasets, EVE-54).

### 3.2. Runtime flow (на попытку)

1. SDK → Supabase RPC `google_ai_reserve(...)` (атомарно check+increment RPM/TPM/RPD).
2. SDK → update `google_ai_requests.sent_at` (или RPC `google_ai_mark_sent`) **до** вызова провайдера.
3. SDK → `google.generativeai` (с ключом из env/secrets).
4. SDK → Supabase RPC `google_ai_finalize(...)` (сохраняем usage, reconcile TPM, закрываем аудит).
5. (опционально) SDK → вставка строки в `token_usage` (совместимость).

При превышении лимитов на шаге 1 — **немедленный `RateLimitError`**, без ожидания.

---

## 4) Контракты и политика ошибок

### 4.1. Ошибки лимитов (без ретраев)

`RateLimitError`:
- `blocked_reason`: `rpm | tpm | rpd`
- `retry_after_ms`: вычисляется для минутного окна как подсказка (но SDK не ждёт автоматически)
- `api_key_id` (если известен), `model`, `minute_bucket`, `day_bucket`

Политика: **не ретраить**.

### 4.2. Провайдерные ошибки (с ретраями до 3)

`ProviderError`:
- `retryable=true|false` (классификация SDK)
- retries: максимум 3, с backoff + jitter

Политика: перед каждой попыткой делаем **новый reserve** (новый `attempt_no`).

---

## 5) Схема данных (что должно быть в Supabase)

### 5.1. Таблицы (обязательные)

1) `google_ai_model_limits`
- PK `model`
- `rpm`, `tpm`, `rpd`
- `tpm_reserve_extra` (опционально, но рекомендовано)

2) `google_ai_api_keys`
- метаданные ключей, **без секретов**
- `env_var_name` указывает, где лежит реальный ключ в env/secrets
- `priority`, `is_active`

3) `google_ai_usage_counters`
- одна таблица для минутных и дневных счётчиков (через `minute_bucket NULL/non-NULL`)
- **частичные unique индексы**:
  - `unique(api_key_id, model, minute_bucket) where minute_bucket is not null`
  - `unique(api_key_id, model, day_bucket) where minute_bucket is null`

4) `google_ai_requests`
- PK `request_uid` (идемпотентность)
- хранит статус `reserved/sent/succeeded/...`, ключ, bucket’ы и итоговый usage

5) `google_ai_request_attempts`
- `unique(request_uid, attempt_no)`
- хранит каждую попытку, включая blocked по лимитам

### 5.2. `token_usage` (обратная совместимость)

`token_usage` не меняется. Для Google AI записи идут туда же (по необходимости) с деталями в `meta`.

---

## 6) RPC (Supabase) — минимально необходимый набор

### 6.1. `google_ai_reserve`

Вход:
- `request_uid`, `attempt_no`, `consumer`, `account_name`
- `model`
- `reserved_tpm`
- `candidate_key_ids uuid[]` (опционально; для Kaggle обычно 1 элемент)

Выход (успех):
- `ok=true`
- `api_key_id`, `env_var_name`, `key_alias` (по join)
- `minute_bucket`, `day_bucket`
- `limits` и `used_after` (для логов/аудита)

Выход (блокировка):
- `ok=false`, `blocked_reason`, `retry_after_ms`, `minute_bucket/day_bucket`

Свойства:
- атомарно инкрементит minute+day счётчики
- идемпотентен по `request_uid` (повторный вызов возвращает уже созданную резервацию, не списывает повторно)

### 6.2. `google_ai_finalize`

Вход:
- `request_uid`, `attempt_no`
- `usage_input_tokens`, `usage_output_tokens`, `usage_total_tokens`
- `provider_status`, `provider_request_id` (если доступно), `error_*` (если ошибка)

Поведение:
- сохраняет usage в аудит
- reconcile TPM: `delta = usage_total_tokens - reserved_tpm`, применяет к minute counter
- идемпотентен (повторный finalize не меняет counters)

### 6.3. (Опционально, но рекомендовано) `google_ai_mark_sent`

Зачем: разделить случаи “reserve, но не отправили” vs “отправили, но не финализировали”.

### 6.4. (Опционально) `google_ai_sweep_stale`

Зачем: компенсировать “reserve, но не sent” по TTL и помечать “sent, но не finalize” как `stale` без компенсации.

---

## 7) Edge cases (принятые политики)

- Падение **между reserve и sent** → можно компенсировать counters (если `sent_at is null`).
- Падение **между sent и finalize** → **не компенсируем** автоматически (консервативно).
- Clock skew → bucket’ы считаются по времени БД.
- Конкурентность → решается условными upsert’ами + unique индексами.
- Дубли/повторы клиента → `request_uid` + идемпотентные RPC.

---

## 8) Секреты и конфигурация (handoff)

### 8.1. Bot runtime (env/secrets)

- `SUPABASE_URL`
- `SUPABASE_KEY` (желательно ограниченный по правам)
- `GOOGLE_API_LOCALNAME` (метка)
- один или несколько `GOOGLE_API_KEY*` (в соответствии с `google_ai_api_keys.env_var_name`)

### 8.2. Kaggle runtime (secrets resolution)

Порядок fallback’ов (единый для `SUPABASE_*`, `GOOGLE_API_KEY*` и будущих секретов):
1) env
2) Kaggle Secrets (UI)
3) encrypted datasets (EVE-54; Fernet + 2 приватных датасета)

Детали транспорта/безопасности/ротации и runbook: `docs/architecture/eve-arch-design.md` (раздел 7) + текущая MVP-реализация `kaggle/UniversalFestivalParser/src/secrets.py`.

---

## 9) План внедрения (по шагам)

1) Миграции Supabase:
- добавить/проверить таблицы `google_ai_model_limits`, `google_ai_api_keys`, `google_ai_usage_counters`, `google_ai_requests`, `google_ai_request_attempts`
- добавить частичные unique индексы для атомарных upsert’ов
- добавить RPC `google_ai_reserve`, `google_ai_finalize` (+ опционально `mark_sent`, `sweep_stale`)

2) Реализация SDK:
- client Supabase (с таймаутами/ретраями на сеть)
- wrapper над `google.generativeai`
- retry policy (provider errors only)
- structured logging
- запись в `token_usage` (через существующий формат + `meta`)

3) Интеграция в Kaggle:
- подключить SDK в `kaggle/UniversalFestivalParser` вместо локального `GemmaRateLimiter`
- провайдер секретов: env → Kaggle Secrets → encrypted datasets (EVE-54)

4) Интеграция в боте:
- начать с одного use-case (например festival parser), затем расширять

---

## 10) Тестирование (минимальный чеклист для “готово к запуску”)

- Unit: retry policy, классификация ошибок, расчёт `reserved_tpm`, формат логов.
- Integration (DB): конкурентные reserve, блокировки RPM/TPM/RPD, идемпотентность reserve/finalize, reconcile TPM.
- E2E: серия запросов + параллельность, “упёрлись в лимит” (сразу ошибка), ретраи на provider errors.
