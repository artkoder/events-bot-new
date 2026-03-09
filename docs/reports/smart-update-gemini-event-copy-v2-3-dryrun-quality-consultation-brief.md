# Smart Update Gemini Event Copy V2.3 Dry-Run Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-review-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_3_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_3_2026_03_07.py`
- `docs/reports/smart-update-gemini-event-copy-first-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-first-consultation-response-review.md`

## 1. Цель консультации

Нам нужна quality-first консультация по реальному `v2.3` dry-run для Gemma event-copy.

Главная цель остаётся прежней:

- естественный;
- профессиональный;
- связный;
- точный;
- grounded;
- невыдуманный текст описания события для Telegraph-страницы.

Нам нужен текст уровня хорошего культурного редактора:

- ясный;
- не рекламный;
- не шаблонный;
- без synthetic prose;
- в меру образный;
- в меру лаконичный;
- но при этом не теряющий важные факты.

## 2. Что важно в этой задаче

Приоритеты в таком порядке:

1. качество итогового текста;
2. естественность и профессиональность;
3. полнота и точность фактов;
4. логичность и связанность повествования;
5. практическая реализуемость для LLM Gemma в рамках runtime / TPM.

Важно:

- мы не хотим выдумывания;
- не хотим unsupported embellishment;
- не хотим рекламного tone drift;
- не считаем минимальность prompt/pipeline самоцелью;
- готовы принять умеренный рост latency/token usage, если качество текста действительно растёт.

## 3. Что уже было сделано до этого

До `v2.3` мы прошли несколько раундов:

- baseline current flow;
- `v1`;
- `v2`;
- `v2.1`;
- `v2.2`;
- первый раунд консультации с Gemini по `v2.2`.

После первой Gemini-консультации локально были проверены несколько узких изменений.

## 4. Что именно менялось в `v2.3`

`v2.3` — это не новый большой redesign, а узкий prompt+facts patch pack поверх предыдущей линии.

Главные изменения:

1. Ослаблен brittle lexical ban на `посвящ*`:
   - вместо жёсткого запрета добавлены positive examples естественных starts.

2. Для sparse cases убраны обязательные headings:
   - compact branch теперь предпочитает `1-2` коротких связных абзаца.

3. Добавлено правило body self-sufficiency:
   - ключевой субъект не должен жить только в эпиграфе.

4. Перед generation добавлена очень осторожная fact pre-consolidation:
   - чтобы уменьшить fragmentation и почти-дубли.

5. При этом не трогались базовые grounded guardrails:
   - fact-first core;
   - cleanup pipeline;
   - service/CTA/metatext checks;
   - preservation floor.

## 5. Что показал `v2.3`

Результат mixed, но уже не нулевой.

Что выглядит promising:

- `2660`: вернулся к baseline-level coverage и стал чище по prose;
- `2745`: sparse case стал редакторски здоровее;
- `2687`: один из лучших кейсов ветки;
- `2673`: явный выигрыш от pre-consolidation, хотя всё ещё не final-quality.

Что остаётся блокером:

- `2734`: coverage regression и возврат `посвящ*`.

То есть у нас уже есть не только regressions, но и реальные признаки того, что часть направления работает.

## 6. Что Gemini получает на вход

Материалы уже содержат полный процесс, а не только итоговые descriptions.

### 6.1. В markdown-report есть

Для каждого из 5 событий:

- исходные тексты постов;
- raw facts;
- `extracted_facts_initial`;
- `facts_text_clean`;
- `copy_assets`;
- side-by-side descriptions:
  - baseline;
  - `v1`;
  - `v2`;
  - `v2.1`;
  - `v2.2`;
  - `v2.3`;
- deterministic diagnostics;
- branch routing;
- runtime.

### 6.2. В raw JSON есть

Machine-readable слой с:

- `source_payload`;
- `raw_facts`;
- `extracted_facts_initial`;
- `facts_text_clean`;
- `copy_assets`;
- `branch_name`;
- `epigraph_fact`;
- `description`;
- `missing_deterministic`;
- `forbidden_reasons`;
- previous version outputs.

### 6.3. В harness-коде есть

Полная логика `v2.3` prototype:

- extraction hints;
- cleanup / floor;
- pre-consolidation;
- routing;
- generation;
- revise;
- reporting.

## 7. Что именно нам нужно от Gemini

Нужен не общий opinion, а максимально практический quality-first разбор.

Просим Gemini:

1. Критически оценить реальное качество текстов `v2.3` по сравнению с baseline / `v1` / `v2` / `v2.1` / `v2.2`.
2. Проверить:
   - не потеряны ли важные факты;
   - не появилось ли лишнее editorial broadening;
   - насколько логично и связно устроено повествование;
   - насколько тексты профессиональны и естественны;
   - насколько headings, lists, epigraph и pattern moves уместны.
3. Разложить quality problems по стадиям:
   - extraction / fact representation;
   - pre-consolidation;
   - routing;
   - generation;
   - revise / cleanup;
   - evaluation mismatch.
4. Свободно предложить любые изменения, которые объективно повысят качество.

## 8. Особенно важный запрос: конкретные правки промптов для Gemma

Это критический пункт.

Мы просим Gemini не ограничиваться общими советами вроде:

- “сделайте generation точнее”;
- “смягчите headings”;
- “усильте coverage”.

Нужны **конкретные Gemma-friendly prompt edits**:

1. Какие формулировки менять в extraction prompt.
2. Какие формулировки менять в standard generation prompt.
3. Какие формулировки менять в compact generation prompt.
4. Какие формулировки менять в revise prompt.
5. Что стоит удалить из prompt contract.
6. Что стоит добавить.
7. Что должно быть expressed as positive example, а что как explicit rule.

Если возможно, просим давать:

- готовые rewritten prompt blocks;
- короткие patch-style replacements;
- или чёткие before/after варианты.

Нам особенно важны такие вещи:

- как формулировать тему/смысл события без synthetic prose;
- как не терять факты при richer cases;
- как не скатываться в generic headings;
- как удерживать тело текста самодостаточным;
- как писать так, чтобы это реально работало именно на Gemma, а не только на более сильной модели.

## 9. Ожидаемый тип ответа

Нам нужен ответ, который можно почти напрямую превратить в следующий experimental patch pack.

То есть в нём должны быть:

- quality verdict;
- event-by-event critique;
- pipeline diagnosis;
- concrete prompt changes for Gemma;
- next-step dry-run plan.

Главная ценность сейчас:

- не защитить уже сделанную линию;
- а честно понять, что из `v2.3` работает, что нет, и какие точные prompt changes дадут следующий реальный прирост качества.
