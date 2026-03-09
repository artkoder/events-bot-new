# Smart Update Opus Consultation Review

Источник внешней консультации:
- `/home/vscode/.gemini/antigravity/brain/7484d4ae-fe08-4290-8d62-3d8498a1525c/smart-update-consultation.md.resolved`
- копия в репозитории: `artifacts/codex/smart-update-consultation-opus-20260306.md`

Контекст:
- базовый кейсбук: `docs/reports/smart-update-duplicate-casebook.md`
- long-run quality benchmark: `docs/reports/smart-update-identity-longrun.md`
- brief для внешних моделей: `docs/reports/smart-update-cross-llm-brief.md`

## 1. Короткий итог

Внешняя консультация Opus в целом подтверждает выбранное направление:
- quality-first identity resolution внутри Smart Update;
- переход от бинарного `merge|create` к `merge|gray|create|skip`;
- pairwise LLM triage на structured payload вместо fat-shortlist финального арбитража;
- deterministic scoring и hard guards до LLM.

При этом один совет из консультации нужно скорректировать под наши реальные инциденты:
- нельзя автоматически подменять извлечённую venue на `default_location` канала при конфликте;
- именно такой override уже породил критичный дубль `Собакусъел`.

## 2. Что принимаем без изменений

1. Добавить 4-state decision model:
- `merge`
- `gray_create_softlink`
- `create`
- `skip_non_event`

2. Убрать merge-bias из prompt-инструкций.
- Текущая формулировка в `smart_event_update.py` действительно смещает LLM к forced match.

3. Разделить ответственность:
- deterministic evidence scoring и hard guards до LLM;
- LLM как judge в серой зоне, не как единственный decider.

4. Явно учитывать multi-event контекст.
- Для schedule/multi-event детей требуется более строгая политика, чтобы не склеивать umbrella и child events.

5. Вести мониторинг по quality-метрикам.
- отдельно по false merge;
- отдельно по duplicate rate;
- отдельно по доле `gray` и её итоговой ручной развязке.

## 3. Что принимаем с корректировкой

### 3.1. Soft-link для `gray`

Суть рекомендации верная: `gray` должен быть first-class состоянием.
Корректировка для текущего этапа:
- сначала внедрить `gray` в runtime-решении и логах;
- таблицу/интерфейс soft-link вводить второй фазой, после стабилизации правил.

### 3.2. Deterministic score-пороги

Рекомендация про `HIGH/LOW` пороги верная, но веса и пороги из консультации нельзя переносить 1:1.
Нужно калибровать на нашем benchmark и кейсбуке:
- `ticket_link` у нас часто generic и не всегда strong identity;
- `poster_hash` не всегда заполнен;
- time/venue конфликты требуют исключений (`doors` vs `start`, venue aliases, extraction noise).

### 3.3. Strict policy для multi-event

Верно, что multi-event требует отдельной политики.
Но правило «для multi-event запретить LLM match полностью» слишком жёсткое для наших кейсов.
Безопаснее:
- не запрещать LLM полностью;
- но запускать его с отдельным policy-профилем для `multi_event`, где default — `gray/create`.

## 4. Что не принимаем в текущем виде

### 4.1. Авто-фолбэк к `default_location` при конфликте extraction

Это противоречит нашим подтверждённым инцидентам.

Кейс `Собакусъел` показал:
- source text явно указывал `ТёркаситиХолл`;
- channel-level `default_location` (`Сигнал`) перетёр извлечённую venue;
- shortlist потерял правильный existing event;
- получился критичный дубль с неверной локацией.

Поэтому корректная политика:
- `default_location` может быть только weak hint;
- при явной venue в source/OCR нельзя перетирать её каналным default;
- при конфликте venue нужен `gray`/human-safe path, а не silent override.

## 5. Прямой impact на план внедрения

На основе Opus review и наших данных финальный план остаётся:

1. Hard guards + normalization:
- venue alias normalization;
- city alias normalization (`Гурьевский городской округ` -> `Гурьевск`);
- linked-source `expected_event_id` invariant;
- single-event source ownership guard;
- non-event skip filters (включая giveaway-only промо).

2. Identity resolver внутри Smart Update:
- deterministic scoring;
- routing в `merge|gray|create`;
- отдельный риск-флаг для same-source multi-child schedule/holiday.

3. LLM layer:
- pairwise triage;
- structured evidence JSON;
- prompt без forced-match bias;
- явный `uncertain -> gray`.

4. Rollout:
- shadow mode;
- сравнение current vs proposed на реальном потоке;
- постепенное включение auto-merge по классам кейсов.

## 6. Открытые вопросы после консультации

1. Где держать `gray` связь в первой итерации:
- отдельная таблица `event_soft_link` сразу;
- или сначала lightweight JSON/audit в `event_source_fact`/logs.

2. Нужен ли отдельный multi-event triage prompt:
- возможно да, чтобы не мешать политику single-event и multi-event.

3. Как фиксировать canonical time correction:
- для кейсов типа `Гараж` нужен отдельный merge-mode (`time_correction`) и audit trail.
