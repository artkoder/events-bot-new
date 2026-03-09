# Smart Update Opus Gemma Event Copy Pattern Redesign Brief

Дата: 2026-03-07

## 1. Зачем нужен этот раунд

Предыдущий `prompt audit` полезен, но сам по себе не закрывает задачу.

Наша цель не в том, чтобы аккуратно отполировать старые prompts.
Наша цель:

- поднять качество итогового текста;
- сделать формирование описаний менее шаблонным;
- перейти от набора разрозненных prompt surfaces к более осмысленной pattern-driven системе письма;
- сохранить fact-first discipline, groundedness и разумный call-budget.

Иными словами: `audit` здесь только подзадача.
Основная задача: **переход к более сильному формированию текста через паттерны, routing и better extraction for copy**.

## 2. Что считаем целевым качеством текста

Нужен текст, который:

- звучит по-человечески, а не как LLM-template;
- естественно объясняет, о чём событие;
- при достаточной фактуре умеет показать, почему событие может быть интересно;
- не скатывается в рекламный нажим;
- не выдумывает деталей;
- остаётся профессиональным, понятным и умеренно лаконичным;
- допускает вариативность композиции:
  - scene-led;
  - topic-led;
  - program-led;
  - rare quote-led;
  - value-led / `why go`, но только когда это оправдано данными.

## 3. Что именно нужно от Opus

Нужен **комплексный redesign всего text flow**, а не только optimization existing prompts.

Opus должен сделать сразу четыре связанных вещи.

### A. Prompt audit

Да, нужен prompt-by-prompt review:

- что оставить;
- что переписать;
- что объединить;
- что сделать transitional;
- что потом удалить.

Но это только первый слой.

### B. Pattern system design

Нужно предложить библиотеку паттернов формирования текста.

Минимально ожидаем:

- список основных narrative patterns;
- короткое объяснение, когда применять каждый;
- когда pattern противопоказан;
- какие признаки во входных данных должны запускать pattern;
- как избежать шаблонности внутри одного и того же pattern.

Примеры классов паттернов, которые точно стоит рассмотреть:

- `topic_led`
- `scene_led`
- `program_led`
- `value_led`
- `quote_led` как rare/high-threshold mode
- `person_led` / `subject_led`, если есть сильная фигура или герой
- `compact_fact_led` для бедных источников

### C. Extraction redesign for better writing

Очень вероятно, что текущего extraction недостаточно для сильного текста.

Нужно оценить и предложить, какие дополнительные поля или blocks стоит извлекать помимо atomic facts.

Например:

- `dominant_angle`
- `theme_frame`
- `program_highlights`
- `experience_signals`
- `why_go_evidence`
- `quote_candidates`
- `scene_cues`
- `contrast_or_tension`
- `named_entities_for_lead`
- `safe_descriptors`
- `format_signal`
- `audience_value_signal`

Нас интересует не “красивое обогащение ради красоты”, а такие extraction outputs, которые реально помогают писать лучше и при этом остаются grounded in source.

### D. Prompt redesign aligned to the pattern system

После pattern design хотим, чтобы Opus предложил:

- как переписать extraction prompts;
- как переписать main generation prompts;
- как переписать coverage / revise;
- как переписать short description / search digest;
- какие shared rule blocks вынести;
- какие support prompts оставить только как repair layer.

То есть prompts должны redesign’иться **под систему паттернов**, а не независимо друг от друга.

## 4. Что входит в scope

Нас интересуют все prompts, связанные с обработкой и формированием текста в `smart_event_update.py`:

- extraction;
- fact-first description generation;
- coverage / revise;
- support cleanup prompts;
- bundled prompts;
- merge prompts;
- short description;
- search digest.

Inventory уже собран в:

- `artifacts/codex/opus_gemma_event_copy_prompt_inventory_latest.md`

## 5. Какие материалы нужно использовать

### Внутренние и внешние паттерны

- `artifacts/codex/opus_gemma_event_copy_casebook_latest.md`
- `artifacts/codex/opus_gemma_event_copy_lexicon_latest.md`
- `artifacts/codex/tg_announcement_patterns_2026-03-01.md`

### Предыдущий цикл с Opus

- `docs/reports/smart-update-opus-gemma-event-copy-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-followup-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-followup-response-review.md`

### Runtime surface

- `smart_event_update.py`

## 6. Ключевые требования и ограничения

### 6.1. Fact-first и anti-hallucination нельзя терять

Нельзя жертвовать:

- groundedness;
- source-backed phrasing;
- coverage discipline;
- запретом на домысливание;
- запретом на логистический мусор внутри narrative.

### 6.2. Variability нужна через композицию, а не через декоративность

Нас интересует прежде всего:

- разный порядок подачи;
- разный тип лида;
- разный тип смыслового центра;
- разная степень объяснения `зачем идти`.

Нас не интересует простая подмена одних клише другими.

### 6.3. `Why go` нужен, но не всегда

Отдельно нужен разбор:

- когда стоит явно формулировать `зачем идти`;
- когда лучше встроить ценность внутрь абзаца без отдельного блока;
- когда этого не нужно делать вовсе;
- какие evidence thresholds должны это разрешать.

### 6.4. Нужно уважать TPM / latency / call-budget

Нельзя строить решение, которое:

- слишком раздувает prompt count;
- тащит лишние repair-calls;
- становится тяжёлым для Gemma;
- даёт красивую идею, но operationally слишком дорого.

## 7. Что хотим получить от Opus

Ожидаемый ответ должен содержать не только audit, но и redesign.

### 1. `Pattern Library`

- 6-10 основных narrative patterns;
- для каждого:
  - когда использовать;
  - когда не использовать;
  - какие evidence triggers нужны;
  - какой тип структуры он даёт.

### 2. `Extraction Redesign`

- какие новые fields / blocks нужно извлекать;
- что можно объединить с текущим facts extraction;
- что должно оставаться optional;
- где schema лучше prose.

### 3. `Prompt Surface Matrix`

- по каждому prompt:
  - keep / tune / rewrite / merge / remove-later;
  - short reason;
  - priority.

### 4. `Rewritten Prompt Family`

- конкретные revised prompts или replacement fragments;
- уже выстроенные вокруг pattern-driven logic.

### 5. `Routing Logic`

- как выбирать pattern;
- что делать при бедных источниках;
- что делать при конфликтующих сигналах;
- как не перепутать `program_led`, `topic_led`, `value_led`, `quote_led`.

### 6. `Quality Controls`

- какие coverage / revise / deterministic checks нужны;
- какие support prompts стоит оставить;
- какие support prompts становятся redundant после redesign.

### 7. `Implementation Order`

- low-risk now;
- medium-risk branch;
- later / research-only.

## 8. Bottom line

Если коротко:

- нам недостаточно просто оптимизировать старые prompts;
- мы хотим от Opus redesign всей текстовой подсистемы в сторону **pattern-driven, less-template, more-human generation**;
- prompt audit нужен как часть этой работы, а не как отдельная самоцель.
