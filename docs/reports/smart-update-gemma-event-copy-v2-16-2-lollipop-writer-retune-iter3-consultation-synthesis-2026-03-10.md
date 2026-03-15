# Smart Update Gemma Event Copy V2.16.2 Lollipop Writer Retune Iter3 Consultation Synthesis

Дата: 2026-03-10

## 1. Inputs

- author brief: `artifacts/codex/tasks/smart-update-lollipop-v2-16-2-writer-retune-iter3-consultation-brief-2026-03-10.md`
- `Opus` consultation: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-retune-iter3-consultation-opus-2026-03-10.raw.json`
- `Gemini 3.1 Pro Preview` critique: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-retune-iter3-consultation-gemini-3.1-pro-preview-2026-03-10.raw.json`
- comparison corpus: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-final-iter2-vs-baseline-2026-03-10.md`
- timing context: [smart-update-gemma-event-copy-v2-16-2-lollipop-pipeline-profiling-2026-03-10.md](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-pipeline-profiling-2026-03-10.md)

## 2. Consensus

- Главный новый regression class живёт не в `writer.final_4o`, а в `facts.prioritize.lead` плюс слишком плоском `editorial.layout`.
- Оба консультанта сошлись, что `2673`, `2659` и `2747` объединяет один паттерн:
  - title недостаточно явно сообщает формат события;
  - lead открывается content/cast fact’ом вместо event-action anchor;
  - prose начинает читаться как справка о предмете, а не как анонс события.
- Оба консультанта признают, что потеря heading structure стала реальным reader-facing минусом, а не просто harmless cleanup.
- Оба консультанта считают `writer_pack.compose` не главным местом фикса.
- `writer.final_4o` допустим как узкий safety net, но не как owner semantic fix.

## 3. Main Disagreement

`Opus` прав в диагностике, но часть его remedy слишком механическая:

- он предлагает numeric heading floors;
- он предлагает soft length target `70-85%` of baseline;
- он допускает новый `event_format_anchor` field в `writer_pack.compose`.

`Gemini` подтверждает root cause, но отвергает именно эту механику:

- length target посчитан хрупким и провоцирующим padding;
- heading floor посчитан риском для awkward micro-sections;
- `event_format_anchor` признан лишним contract bloat на этом шаге.

Каноническое решение по спору:

- root-cause diagnosis `Opus` принимается;
- numeric percentage targets не принимаются;
- schema expansion в `writer_pack.compose` пока не принимается;
- structural recovery делается через semantic layout rules, а не через fixed quotas.

## 4. Canonical Change Set

### `facts.prioritize`

Внедрить title-opacity-aware lead selection:

- если title bare или не даёт явного сигнала event format, lead обязан начинаться с event-action / event-format anchor;
- для `presentation` нельзя открывать текст только фактом о том, что проект собой представляет;
- для `кинопоказ` нельзя открывать bare-title case через `people_and_roles`, если нет явного screening anchor.

Практический смысл по кейсам:

- `2673`: lead должен сначала объявить презентацию проекта, а уже потом объяснять, что такое `Собакусъел`;
- `2659`: lead должен объявить показ фильма, а не начинаться с Франсуа Озона;
- `2747`: `Киноклуб` prefix помогает, но cast-first lead всё равно нужно отодвинуть вниз.

### `editorial.layout`

Ослабить текущую compression bias и вернуть structure по semantic shift, а не по формуле:

- headings нужны там, где narrative реально переключается между разными смысловыми блоками;
- для `кинопоказ` разумный split: screening/event framing vs film/cast/synopsis;
- для `presentation` разумный split: что за событие / что будет / для кого / программа;
- не вводить fixed rule вида `>=3 facts => 1 heading`, но вернуть layout обязанность не схлопывать multi-theme events в один paragraph blob.

### `writer.final_4o`

Добавить только узкий prompt carry:

- если title stylized, bare или opaque, первое предложение должно снять неоднозначность event format;
- без reopen writer-stage semantics и без новых variant mechanics.

### `writer_pack.compose`

На этом раунде не менять канонический schema contract.

Причина:

- `writer_pack.compose` почти бесплатен по latency, но оба консультанта не видят в нём root cause;
- extra `event_format_anchor` можно вернуть только если `facts.prioritize + editorial.layout + writer.final_4o` safety line не закроют ambiguity.

## 5. What This Means For The Metrics

Текущие корпусные числа:

- average baseline chars: `735.7`
- average iter2 chars: `445.9`
- `11/12` events shorter than baseline
- `11/12` events with heading loss vs baseline

Эти числа использовать как smoke signals, а не как explicit target:

- нужно восстановить clarity и scan-ability;
- не нужно учить pipeline “добивать до 70-85% baseline length”.

## 6. Recommended Next Step

Следующий practical round:

1. retune `facts.prioritize.lead` для opaque-title `presentation` / `кинопоказ`;
2. retune `editorial.layout` на semantic block restoration без numeric floors;
3. добавить узкий `writer.final_4o` first-sentence format rule;
4. rerun full `12`-event corpus как новый downstream iteration;
5. сравнить не только `2673/2659/2747`, но и corpus-level:
   - heading recovery,
   - average length drift,
   - отсутствие regressions на уже починенных `2498/2657/2734`.

## 7. Verdict

Итоговый verdict этого consultation round:

- `facts.prioritize`: `required`
- `editorial.layout`: `required`
- `writer.final_4o`: `optional narrow carry`
- `writer_pack.compose`: `not first-line change`

То есть следующий retune должен идти не в сторону “сильнее переписать prose”, а в сторону “лучше выбрать первый anchor и вернуть минимально нужную structure”.
