# Smart Update Opus Gemini Event Copy V2.11 Complex Consultation Review

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/tasks/event-copy-v2-11-complex-consultation-brief.md`
- `artifacts/codex/reports/event-copy-v2-11-complex-consultation-claude-opus.json`
- `artifacts/codex/reports/event-copy-v2-11-complex-consultation-claude-opus.md`
- `artifacts/codex/reports/event-copy-v2-11-complex-consultation-gemini-3.1-pro.md`
- `docs/reports/smart-update-gemini-event-copy-v2-10-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-review-2026-03-08.md`
- `/home/vscode/.gemini/tmp/events-bot-new/chats/session-2026-03-08T09-35-33a9fda4.json`

## 1. Проверка моделей

`Opus` был вызван в strict one-shot режиме через `claude -p --model claude-opus-4-6 --tools "" --output-format json`.

Проверка `modelUsage` в raw JSON вернула ровно один ключ:

- `claude-opus-4-6`

`Gemini` была вызвана через `gemini -m gemini-3.1-pro-preview`. Session log подтверждает модель:

- `gemini-3.1-pro-preview`

## 2. Где Opus и Gemini сошлись

Обе внешние модели независимо подтвердили четыре тезиса:

1. `v2.10` ломался не только в extraction prompt, а и на post-merge contamination:
   в `facts_text_clean` возвращались более сырые baseline / merge facts с `посвящ*` и intent/metatext формулами.
2. Нужен generation-side `anti-quote` control:
   кейс `2660` частично проваливался не из-за фактов как таковых, а из-за цитатной упаковки фактов в prose.
3. Нужны clause-style nominalization examples:
   текущие `[ПЛОХО] -> [ХОРОШО]` примеры были слишком простыми для кейсов типа `зачем появился / как устроена / какую проблему решает`.
4. `list consolidation` нельзя оставлять универсальным:
   оно полезно для перечислений имён / треков / пунктов программы, но не для смысловых agenda blocks.

## 3. Где Opus и Gemini разошлись

Главное расхождение:

- `Opus` предлагал отдельный deterministic filter именно под `посвящ*`.
- `Gemini` возразила, что отдельный regex-патч под одно слово слишком хрупок для русского синтаксиса и технически хуже, чем общий semantic/token-overlap dedup.

Второе расхождение:

- `Opus` предлагал жёстче вырезать soft/marketing facts.
- `Gemini` справедливо указала, что для художественных событий это может вырезать смысловую ткань события, а не мусор.

## 4. Что я принял

В `v2.11` я взял только пересечение рекомендаций и safe subset:

- post-merge semantic cleanup с приоритетом cleaner facts над dirty metatext/intent facts;
- generation/revise `anti-quote` rule;
- clause-style nominalization examples для extraction;
- scoped `list consolidation` только для enumerations, не для `зачем/как/почему`.

## 5. Что я сознательно не взял

- отдельный regex rewrite вроде `посвящ* -> ...` как primary solution;
- жёсткий soft/marketing fact filter для художественных кейсов;
- новый intermediate LLM stage;
- новый router или новый repair pass.

## 6. Итоговый pre-run verdict

Комплексная консультация была полезной и дала сильный corrective signal:

- проблема действительно уже не сводится к “нужно ещё лучше переписать extraction prompt”;
- главный practical bottleneck оказался в связке `extraction -> preserve floor / merge-back -> generation`;
- `v2.11` нужно было делать как узкий corrective round, а не как ещё один широкий redesign.

Именно этот verdict и лёг в основу experimental harness `v2.11`.
