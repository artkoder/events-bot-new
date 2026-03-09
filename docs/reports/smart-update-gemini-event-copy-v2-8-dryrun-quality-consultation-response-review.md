# Smart Update Gemini Event Copy V2.8 Dry-Run Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-brief.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-review-2026-03-07.md`

## 1. Краткий verdict

Это сильный post-run response, но пока не окончательный.

Gemini правильно прочитал main shape `v2.8`:

- rollback от `v2.7` был концептуально верным;
- `2734` действительно подтверждает ценность data-point extraction;
- `2660 / 2745 / 2687 / 2673` не стали quality win;
- main bottleneck остаётся в extraction/fact shaping.

Но response всё ещё неполный по одному важному practical reason:

- он переоценивает роль extraction prompt alone;
- и недоучитывает уже существующий prompt-facing sanitizer из текущего runtime, который сам может превращать `... посвящена ...` в `Тема: ...`.

Из-за этого один узкий follow-up round всё же нужен.

## 2. Что в ответе Gemini принимаю

### 2.1. `v2.8` — mixed round, не pure failure

Это принимаю.

`2734` действительно нельзя игнорировать.

### 2.2. Branch inflation и fact inflation — главный structural failure

Это тоже принимаю полностью.

Практический сигнал очень сильный:

- `2660` и `2745` потеряли compact path;
- `2687` и `2673` раздулись ещё сильнее;
- generation уже не может cleanly переварить такой facts layer.

### 2.3. Template-overuse control worth keeping

Согласен.

Этот слой не решает корневую проблему, но как hygiene rule он полезен и cheap.

### 2.4. `anti-splitting rule` как next hypothesis

Идея сильная.

Нам действительно не хватает прямого требования:

- не дробить один смысл на 3-4 facts;
- держать data points плотными.

## 3. Что беру только с поправками

### 3.1. `Aggressive hints` нельзя принимать на веру

Gemini предлагает перейти к тону:

- `КРИТИЧЕСКАЯ ОШИБКА`
- `СТРОГО ЗАПРЕЩЕНО`

Это worth testing, но только как experimental change.

Риск:

- Gemma может лучше реагировать на error markers;
- но может и начать literal-copy / panic-style obedience without nuanced shaping.

То есть это `accept with modification`, а не blind accept.

### 3.2. `Anti-splitting` должен быть content-aware

Если формулировать его грубо как:

- `одна мысль = один факт`

мы рискуем снова схлопнуть:

- program items;
- named people clusters;
- multi-part lecture/program content.

Правильнее:

- merge paraphrastic or overlapping facts;
- never collapse distinct named items or separate program details.

## 4. Что не принимаю буквально

### 4.1. `Survival of labels` объясняется не только extraction prompt

Это сейчас самый важный practical correction.

После dry-run я отдельно проверил текущий support layer и нашёл следующее:

- `smart_event_update._sanitize_fact_text_clean_for_prompt`
- в common case переписывает `<event> посвящена/посвящён ...` в `Тема: <topic>.`

Именно этот слой затем вызывается в harness через `_safe_prompt_fact_sanitize`.

То есть часть label-style artifacts:

- может рождаться не в extraction;
- а уже в prompt-facing sanitize layer current runtime.

Следствие:

- Gemini response useful;
- но next-step patch pack без калибровки этого sanitizer будет неполным.

### 4.2. `Revise safety net` не должен маскировать root cause

Да, короткий scaffold `Лекция о ...` полезен.
Но нельзя снова уехать в pure late-fix logic.

Если fact layer polluted, revise это не вылечит надёжно.

## 5. Что реально пойдёт дальше

Из этого ответа беру:

1. anti-splitting hypothesis для extraction;
2. более жёсткий ban на label/intent-style data shapes;
3. сохранить `template-overuse control`;
4. сохранить отказ от `v2.7` safe-positive wrappers.

Но перед `v2.9` нужен ещё один узкий Gemini follow-up по конкретному вопросу:

- как менять patch pack с учётом существующего `_sanitize_fact_text_clean_for_prompt`, который сам генерирует `Тема: ...`.

## 6. Bottom line

Этот response не закрывает цикл окончательно, но существенно помогает.

Практический вывод:

- **response quality**: strong;
- **blind implementation**: нет;
- **нужен узкий follow-up**: да, только по sanitizer interaction и updated `v2.9` patch pack.
