# Smart Update Gemini Event Copy V2.4 Regression Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-review-2026-03-07.md`
- `docs/reports/smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_4_2026_03_07.py`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_4_5events_2026-03-07.json`

## 1. Зачем нужна эта консультация

Нам нужна новая Gemini-консультация после реального regression round.

Главная цель не изменилась:

- естественный;
- профессиональный;
- связный;
- grounded;
- невыдуманный;
- редакторски сильный текст описания события для Telegraph.

Но теперь вопрос стоит уже уже не как “что улучшить вообще”, а как:

- почему `v2.4` по факту стал хуже `v2.3`;
- какие именно изменения стоит сохранить;
- какие изменения надо откатить;
- какой следующий локальный шаг реально может поднять качество итогового текста.

## 2. Что произошло до этого

У нас был сильный `v2.3`, который впервые дал mixed-positive signal:

- sparse cases стали живее;
- body self-sufficiency помогла;
- prose в ряде кейсов стала заметно человеческой.

Потом был consultation round с Gemini по `v2.3`.
Мы не приняли его ответ на веру, а сделали наш критический review и собрали узкий `v2.4` patch pack.

В `v2.4` проверялись только 3 идеи:

1. anti-conversational headings;
2. stronger program preservation;
3. stronger anti-`посвящ*` contract.

## 3. Что показал `v2.4`

Итог неприятный:

- `v2.4` по сумме хуже `v2.3`;
- `2660` и `2745` потеряли `compact_fact_led`;
- `2687` регресснул;
- `2673` избавился от question-headings, но стал тяжелее и слабее;
- `2734` только частично восстановился и всё ещё содержит `посвящ*`.

То есть:

- часть идей Gemini оказалась полезной;
- но их текущая реализация ухудшила общий quality balance.

## 4. Что именно Gemini должен увидеть

В этот раз мы специально даём полный operational context:

### 4.1. Реальные тексты и факты

В отчётах уже есть:

- source texts;
- raw facts;
- `extracted_facts_initial`;
- `facts_text_clean`;
- `copy_assets`;
- side-by-side descriptions по версиям;
- deterministic diagnostics.

### 4.2. Промпты и алгоритм

В `experimental_pattern_dryrun_v2_4_2026_03_07.py` есть:

- extraction prompt;
- generation prompts;
- revise prompts;
- routing;
- pre-consolidation;
- policy checks;
- repair path;
- reporting logic.

То есть Gemini видит не только outputs, но и actual prompt/algorithm contract.

## 5. Что мы уже считаем вероятной проблемой

Это важно: мы не просим Gemini просто повторить прошлые мысли.

Наш текущий рабочий диагноз такой:

1. `v2.4` сломал sparse/rich balance, потому что changed fact preservation altered routing.
2. Anti-conversational headings были верной идеей, но patch pack оказался слишком широким.
3. Anti-`посвящ*` в форме `forbidden_marker(посвящ*)` слишком машинный и плохо работает как revise signal.
4. Program preservation полезен, но restored items не гарантированно проходят downstream intact.
5. Проблема уже не только в generation prompt, а в связке:
   - extraction / facts layer;
   - routing;
   - revise issue wording.

Но это всё ещё гипотезы, а не догма.

## 6. Что мы хотим от Gemini сейчас

Нужен не просто review, а grounded failure diagnosis.

Просим Gemini:

1. Критически сравнить `v2.3` и `v2.4`.
2. Сказать, где именно `v2.4` ухудшил качество текста и почему.
3. Разделить:
   - верную идею;
   - неудачную реализацию идеи;
   - ложную идею.
4. Предложить только те next-step changes, которые реально worth testing.

## 7. Самое важное требование

Нужны **конкретные правки для Gemma prompts и локального algorithm contract**.

Нас интересуют:

- extraction prompt changes;
- compact generation changes;
- standard generation changes;
- revise prompt changes;
- routing changes;
- policy issue wording changes.

И очень важно:

- не просто “сделайте лучше”;
- а именно patchable changes;
- желательно в формате rewritten blocks или before/after.

## 8. Чего мы НЕ хотим

- Не хотим абстрактной теории.
- Не хотим защищать предыдущие рекомендации только потому, что они уже были даны.
- Не хотим снова раздувать architecture.
- Не хотим, чтобы внешняя модель принимала metrics за абсолютную истину.

Главная ценность сейчас:

- трезво понять, что именно стоит сохранить из `v2.3`;
- как исправить локальные провалы `v2.4`;
- и как выйти на следующий quality-first dry-run без очередного общего regressions burst.
