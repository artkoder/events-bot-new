# Smart Update Gemini Event Copy V2.9 Hypotheses Consultation Brief

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-review-2026-03-07.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-sanitizer-followup-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-sanitizer-followup-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-prompt-context.md`

## 1. Зачем нужна эта консультация

Мы начинаем `v2.9`.

`v2.8` закрыл диагностический этап:

- стало ясно, что `v2.7` был placement error;
- стало ясно, что чистый rollback к `v2.6` недостаточен;
- отдельно выяснилось, что часть мусора рождается в prompt-facing sanitizer текущего runtime.

Теперь нужен новый узкий pre-run round перед локальной сборкой `v2.9`.

## 2. Главная цель

Нужно собрать такой `v2.9`, который:

- не вернёт `v2.7`-style narrative shaping в Extraction;
- не будет снова плодить `Тема: ...` и `расскажут о ...`;
- сохранит gain `2734`;
- повысит шансы `2660` и `2745` снова попасть в более удачную compact-like dynamics;
- наконец начнёт лечить `2687` и `2673`.

## 3. Текущий practical bottleneck

После `v2.8` root cause map выглядит так:

1. Extraction всё ещё недостаточно контролирует:
   - splitting одной мысли на 2-4 факта;
   - label-style data shapes;
   - intent-style data shapes.
2. Prompt-facing sanitizer current runtime сам генерирует `Тема: ...`.
3. Generation и revise не успевают reliably исправить fact-layer pollution downstream.

## 4. Наш `v2.9 hypothesis pack`

Ниже гипотезы, которые нужно критически проверить.

### H1. `v2.9` надо строить от `v2.8`, а не заново от `v2.6`

Причина:

- `v2.8` уже содержит полезные вещи, которые не хочется откатывать:
  - anti-bureaucracy / natural framing balance;
  - template-overuse control;
  - clean rollback away from `v2.7`.

### H2. Надо убрать только branch `<event> посвящена ... -> Тема: ...` из prompt-facing sanitizer

То есть:

- не вырубать весь sanitize layer;
- убрать именно rewrite, который рождает label-style facts.

### H3. Extraction нужен explicit `content-aware anti-splitting`

Не в форме vague `один факт = одна мысль`, а в более точном виде:

- не дроби один и тот же смысл на несколько близких facts;
- но не схлопывай distinct names, program items и отдельные meaningful details.

### H4. Extraction должен получить более жёсткие data-shape constraints

Именно:

- жёсткий запрет на label-style facts;
- жёсткий запрет на intent-style facts;
- но примеры должны быть data-shaped, а не sentence-shaped, чтобы не вернуть `v2.7`-style template copying.

### H5. `_pre_extract_issue_hints` и post-extract hints стоит усилить до error-style, но без shouting

То есть:

- `ОШИБКА: ...`
- а не мягкое `критическое требование`, которое Gemma игнорирует;
- но и не `АЛЯРМ / СБОЙ / КАПС`, чтобы не провоцировать literal-copy.

### H6. Стоит сохранить generation-side improvements `v2.8`

Пока hypothesis такая:

- `template-overuse control` оставить;
- anti-bureaucracy + natural framing оставить;
- generation не трогать радикально.

### H7. Возможен узкий LLM-first pre-generation fact quality gate

Это новый пункт.

Если после sanitize слоя в prompt-facing facts всё ещё остались:

- `посвящ*`,
- label-style facts,
- intent-style facts,

то вместо того чтобы кормить этим Generation, можно сделать один targeted LLM repair call именно по facts layer.

Важно:

- это не deterministic rewrite;
- это не новый большой pipeline;
- это один узкий quality gate только when needed.

## 5. Что особенно важно для `v2.9`

Нас интересует не только general quality, но и конкретные operational goals:

- `2660`: убрать regressions и вернуть более compact, less repetitive prose;
- `2745`: не раздувать sparse case и не терять density;
- `2734`: не потерять gain;
- `2687`: убрать `посвящ*` и канцелярит;
- `2673`: убрать service tone и лишнее дробление.

## 6. Самое важное требование к ответу

Нужны **конкретные Gemma-friendly prompt changes**, а также verdict по H1-H7:

- `accept`
- `accept with modification`
- `defer`
- `reject`

Без нового большого redesign и без возврата уже опровергнутых идей.
