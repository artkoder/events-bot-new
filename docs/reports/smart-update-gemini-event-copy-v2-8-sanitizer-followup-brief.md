# Smart Update Gemini Event Copy V2.8 Sanitizer Follow-Up Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-review-2026-03-07.md`

## 1. Зачем нужен follow-up

После первого post-run Gemini round выявился один важный runtime interaction, который не был достаточно явно виден в предыдущем docs-only context.

В текущем `smart_event_update.py` есть prompt-facing sanitizer:

- `_sanitize_fact_text_clean_for_prompt`

Он умеет делать такое преобразование:

- `<event> посвящена/посвящён ...` -> `Тема: <topic>.`

И этот sanitizer реально применяется в experimental harness перед generation prompt.

## 2. Почему это важно

Это означает, что часть artifacts в `v2.8`:

- label-style facts;
- странные `Тема: ...`;

может рождаться не только в extraction prompt, но и уже в support layer текущего runtime.

То есть если мы будем собирать `v2.9` только как prompt-only patch pack без этой калибровки, диагноз останется неполным.

## 3. Что нужно понять

Нужно переоценить:

- насколько первый Gemini response меняется с учётом этого sanitizer;
- надо ли для `v2.9` менять сам sanitizer;
- надо ли его отключать / ужесточать / делать conditionally safer;
- как теперь должен выглядеть настоящий minimal `v2.9 patch pack`.

## 4. Самое важное требование

Нужен очень узкий ответ:

- без нового redesign;
- без general brainstorming;
- только re-calibration `v2.9` с учётом sanitizer interaction.
