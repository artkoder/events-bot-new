# Smart Update Opus Gemma Event Copy Prompt Audit Brief

Дата: 2026-03-07

## 1. Зачем нужен ещё один раунд

После двух раундов с Opus у нас уже есть приемлемый architectural direction:

- low-risk prompt/coverage improvements;
- medium-risk ветка с `facts + copy_assets`;
- `quote_led` и tone-adaptive generation отложены в research-only.

Но сейчас нужен **не очередной архитектурный ответ**, а полный **prompt audit** всего текстового флоу.

Цель нового раунда:

- пройтись по всем prompt surfaces, которые участвуют в обработке текста;
- сократить лишний шум и дубли правил;
- предложить более чистые, Gemma-friendly и экономные формулировки;
- явно сказать, какие prompts нужно:
  - оставить;
  - слегка подправить;
  - серьёзно переписать;
  - слить друг с другом;
  - в будущем убрать.

## 2. Что именно считаем scope этого аудита

Нас интересуют **все промпты, связанные с обработкой текста и генерацией текста** внутри Smart Update.

Это включает:

### A. Extraction / source-backed text processing

- atomic facts extraction;
- create-bundle prompts;
- merge prompts;
- match-or-create prompts в той части, где формируются `bundle.title/description/facts/search_digest/short_description`.

### B. Fact-first formation

- main description prompt;
- coverage prompt;
- revise prompt;
- targeted cleanup prompts;
- missing-facts integration;
- paragraph reflow.

### C. Support / fallback text prompts

- journalistic rewrite;
- logistics removal;
- quote/block-quote enforcement;
- shrink-to-budget.

### D. Derived text fields

- `short_description`
- `search_digest`

## 3. Чего мы хотим от Opus

Нужен не “совет в общих словах”, а prompt-by-prompt audit.

По каждому prompt surface хотим:

1. `verdict`
- keep / tune / rewrite / merge / remove-later

2. `reason`
- что в текущем prompt хорошо;
- что мешает Gemma;
- где prompt раздут, повторяется или тянет модель не туда.

3. `optimized prompt text`
- конкретный новый текст prompt'а или prompt-fragment'а.

4. `shared fragments`
- какие rule blocks надо унифицировать между несколькими prompt'ами;
- что лучше держать как общий reusable block, а не копировать 5 раз.

5. `call-budget / complexity posture`
- стоит ли этот prompt оставлять как отдельный call;
- можно ли что-то слить;
- где уменьшение prompt noise даст выигрыш без потери качества.

## 4. Что особенно важно

### 4.1. Можно спорить с текущими prompt'ами

Мы не просим “вежливо полировать существующее”.

Если prompt:

- перегружен;
- повторяет одни и те же запреты в трёх местах;
- смешивает слишком много задач;
- мешает Gemma;
- делает output более шаблонным;

то хотим, чтобы Opus сказал это прямо.

### 4.2. Но нельзя потерять уже наработанные guardrails

Нельзя предлагать “красивое упрощение”, если оно ломает:

- fact-first discipline;
- anti-hallucination;
- запрет логистики в narrative;
- coverage discipline;
- сохранение списков и цитат;
- bounded latency/call count.

### 4.3. Нужна Gemma-specific оптимизация

Хотим от Opus взгляд именно под Gemma runtime:

- где prompt слишком длинный;
- где абстрактные инструкции можно заменить на более конкретные;
- где enum / boolean / compact JSON лучше длинного prose;
- где негативные инструкции избыточны;
- где текущие prompts конфликтуют друг с другом stylistically.

## 5. Приоритеты внутри аудита

### P0: active current fact-first path

Это highest priority:

- facts extraction
- fact-first description generation
- coverage
- revise
- short_description
- search_digest

### P1: support prompts, которые реально влияют на качество

- integrate missing facts
- reflow
- remove infoblock logistics
- shrink to budget
- remove_posv

### P2: fallback / bundled / legacy-adjacent text prompts

- create_bundle
- match_create_bundle
- merge
- rewrite
- blockquote enforcement

Их тоже надо покрыть, но можно с меньшей глубиной, если Opus считает часть из них transitional.

## 6. Готовые материалы для Opus

- Inventory prompt surfaces:
  - `artifacts/codex/opus_gemma_event_copy_prompt_inventory_latest.md`
- Current accepted direction:
  - `docs/reports/smart-update-opus-gemma-event-copy-followup-response.md`
  - `docs/reports/smart-update-opus-gemma-event-copy-followup-response-review.md`
- Previous response chain:
  - `docs/reports/smart-update-opus-gemma-event-copy-response.md`
  - `docs/reports/smart-update-opus-gemma-event-copy-response-review.md`
- Ready-to-send prompt:
  - `artifacts/codex/opus_gemma_event_copy_prompt_audit_prompt_latest.md`
- Send manifest:
  - `artifacts/codex/opus_gemma_event_copy_prompt_audit_manifest_latest.md`

## 7. Ожидаемый формат ответа Opus

Предпочтительно так:

1. `Prompt Surface Matrix`
- по каждому prompt: keep / tune / rewrite / merge / remove-later

2. `Shared Rule Blocks`
- что вынести в общие fragments

3. `Rewritten Prompts`
- конкретные тексты или replacement fragments

4. `Simplification / Merge Opportunities`
- какие prompts можно слить или упростить

5. `Gemma-specific Risks`
- где новые тексты могут навредить

6. `Implementation Order`
- что внедрять first / second / later

## 8. Bottom line

Если коротко:

- архитектурный спор уже largely закончен;
- теперь нужен surgical prompt audit;
- хотим от Opus не новую “концепцию”, а полный разбор и оптимизацию **всех** text-processing и text-generation prompts этого флоу.
