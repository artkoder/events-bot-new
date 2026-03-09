# Smart Update Opus Gemma Event Copy Preservation Matrix Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-preservation-matrix-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-final-impl-calibration-response-review.md`
- `artifacts/codex/opus_gemma_event_copy_preservation_matrix_prompt_latest.md`
- `smart_event_update.py`

## 1. Краткий verdict

Этот ответ Opus **закрывает последний pre-code consultation gap**.

Главное:

- он не пытается снова изобретать redesign;
- он отдельно и системно разбирает, что именно нельзя потерять из текущего runtime;
- он даёт внятную `keep / adapt / replace / remove-later` рамку;
- он не ломает quality-first direction, а связывает его с уже работающими защитными слоями.

Мой текущий вывод:

- **ещё один консультационный этап до dry-run не нужен**;
- рациональный следующий шаг: **реальный Gemma dry-run на новых событиях**;
- следующий Opus-раунд имеет смысл уже **по живым примерам**, а не по теории.

## 2. Что в ответе Opus особенно сильное

### 2.1. Он действительно сделал migration audit, а не очередной redesign pitch

Это именно то, чего не хватало после прошлого раунда.

Opus отдельно прошёлся по:

- upstream fact pipeline;
- epigraph / opening logic;
- cleanup и repair layers;
- policy checks;
- shared rules;
- prompt/runtime boundary.

То есть теперь у нас есть не просто набор новых идей, а карта перехода, где видно, что сохраняется и где это живёт в новой архитектуре.

### 2.2. Полезные механизмы текущего кода не потеряны

Это главный practical plus ответа.

Opus явно сохраняет или адаптирует именно те вещи, которые реально помогают текущему quality:

- `_facts_text_clean_from_facts`
- `_sanitize_fact_text_clean_for_prompt`
- `_pick_epigraph_fact`
- `_find_missing_facts_in_description`
- `_llm_integrate_missing_facts_into_description`
- `_fact_first_remove_posv_prompt`
- `_cleanup_description`
- `_collect_policy_issues`
- compact program list rule
- visitor conditions rule

Это сильнее и полезнее, чем предыдущие ответы, где часть redesign-логики ещё слишком легко могла “перекрыть” уже работающие эвристики.

### 2.3. Runtime / prompt boundary стал заметно чище

Ответ хорошо разводит:

- что должно оставаться в prompt contract;
- что лучше держать в runtime;
- что должно проверяться coverage/policy layer;
- что должно жить в cleanup/repair.

Это особенно важно для Gemma, потому что здесь нельзя рассчитывать, что один сильный prompt заменит все защитные слои.

### 2.4. Конфликт `epigraph vs pattern lead` решён правильно

Это одна из самых ценных конкретизаций.

Формула:

- blockquote epigraph при наличии сильного факта остаётся выше pattern lead;
- pattern определяет lead после epigraph, а не вместо него.

Это позволяет не потерять одну из самых живых сильных сторон текущего fact-first flow.

## 3. Где ответ всё ещё не идеален

### 3.1. `Style C -> replace` правильно по архитектуре, но practically это надо доказывать только dry-run'ом

На бумаге замена `Style C` на pattern family выглядит логично.
Но текущий `Style C` уже даёт определённый compositional lift.

Значит в implementation нельзя трактовать `replace` как право безоговорочно выкинуть всё существующее prose behavior.
Это надо валидировать живыми примерами.

### 3.2. `program_highlights` и перенос program-list logic в extraction всё ещё надо делать аккуратно

Идея здравая.
Но тут всё ещё есть риск:

- раздуть extraction schema;
- получить почти-дубликаты между facts и copy assets;
- ухудшить prompt discipline ради “полезного богатства”.

То есть не возражение по сути, а предупреждение к implementation.

### 3.3. `quality_flags` как расширение `_collect_policy_issues` надо делать без расползания субъективности

Текущее преимущество `_collect_policy_issues` в том, что он mostly structural.
Если туда добавить слишком много fuzzy quality checks, можно получить шум и лишние revise-calls.

Поэтому это скорее осторожное `adapt`, а не carte blanche на большое число мягких флагов.

## 4. Нужен ли ещё один этап консультаций сейчас

**Нет.**

На текущем этапе новый консультационный раунд до dry-run уже даёт мало пользы.

Почему:

- conceptual architecture уже достаточно прояснена;
- migration/preservation вопрос отдельно закрыт;
- дальнейшие споры без живых кейсов будут снова уходить в предпочтения и гипотезы;
- у нас уже есть явное намерение после кода ещё раз отправить промпты в Opus на fine-tuning.

Это означает, что лучший следующий источник информации теперь не ещё один theoretical round, а **реальные outputs текущего и нового flow**.

## 5. Что делать дальше

Следующий шаг я считаю таким:

1. провести dry-run на 5 новых реальных событиях через текущий Gemma fact-first flow;
2. по каждому кейсу показать:
   - source mix;
   - `facts_text_clean`;
   - итоговый description;
   - заметные пропуски / forbidden leakage / structural issues;
3. использовать этот пакет как grounded basis:
   - для локальной implementation работы;
   - и для следующей критической консультации с Opus уже по реальным результатам.

## 6. Bottom line

Preservation-matrix ответ Opus принят.

Он не делает систему автоматически готовой к продакшен-коду, но **снимает необходимость в ещё одном pre-code brainstorming round**.

Теперь правильнее идти в реальный dry-run и переносить дискуссию с уровня “теории промптов” на уровень “вот факты, вот текст, вот где качество действительно проседает”.
