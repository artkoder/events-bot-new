# Smart Update Event Copy V2.16.1 Ice-Cream Duel Iteration 3 Consultation Synthesis

Дата: 2026-03-09

Основание:

- [duel iter3 report](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-ice-cream-duel-iter3-2026-03-09.md)
- [Opus review](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-16-1-ice-cream-duel-iter3-opus-2026-03-09.md)
- [Gemini single-launch failure note](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-16-1-ice-cream-duel-iter3-gemini-3.1-pro-preview-2026-03-09.md)

## 1. Что показал сам duel

`iter3` улучшил aggregate coverage относительно baseline:

- total missing: `27 -> 21`
- total enhanced missing: `32 -> 20`

Но этот выигрыш не равен runtime-ready улучшению:

- `2731` дал `unsupported=2`
- `2498` дал `unsupported=2`
- `2673` регрессировал по raw missing: `6 -> 7`
- `2659` не улучшился вообще
- `2687` остался по сути flat
- из безопасных win остался только умеренный локальный плюс на `2734`

То есть aggregate-цифры снова вводят в заблуждение: главные coverage-win пришли из blocking-failed кейсов.

## 2. Что подтвердил локальный stage-profile

Самый ценный новый вывод этого раунда:

- `audit_narrative_core` на `2731` и `2498` сказал `unsupported=0`
- `repair_narrative_core` не был вызван
- `final_audit` на тех же текстах потом сказал `unsupported=2`

Это не абстрактная “LLM ошиблась”, а конкретный контрактный разрыв внутри pipeline:

- generation производит modality/scope drift;
- промежуточный audit этого не видит;
- final audit видит;
- следовательно pipeline не умеет сам себя чинить в главной зоне риска.

Локальная проверка JSON это подтверждает буквально:

- `2731` final audit зарубил:
  - `На встрече не нужно учить слова заранее`
  - `На вечеринке откроется полный список песен, которые будут исполняться`
- `2498` final audit зарубил:
  - `Спектакль погружает в мир закулисных отношений и трагической любви, разворачивающихся на фоне Нюрнбергского процесса.`
  - `В центре внимания — любовь, разделенная идеологией.`

## 3. Что добавил Opus

`Opus` дал самый полезный сдвиг в формулировке проблемы:

- проблема не только в “weak audit prompt”;
- основная ошибка generation для двух чувствительных shape не просто `fact expansion`, а:
  - `modality shift`
  - `scope drift`
  - `interpretive reframing`
- текущие anti-expansion examples бьют в слишком узкий класс ошибок и не покрывают реальные fail-паттерны;
- для `party_theme_program` и `theater_history` generation не должен пытаться писать “живой narrative prose” в текущем виде;
- для этих shape правильнее перейти к более сухому `structured fact arrangement`:
  - одно предложение = один факт
  - минимум связок
  - без лид-магнетизма и immersive phrasing

Самые сильные рекомендации `Opus`:

1. Добавить deterministic post-generation screen для high-risk phrases/patterns.
2. Для `party_theme_program` и `theater_history` переключить generation из narrative-режима в structural arrangement mode.
3. Дать intermediate audit больше контекста (`title/date/venue`), чтобы сократить разрыв с final audit.
4. Не продолжать бесконечно наращивать похожие anti-expansion examples: Gemma может читать их как исчерпывающий список.
5. Отдельно stress-test repair path, а не считать его автоматически рабочим.

## 4. Что произошло с Gemini

`Gemini` review этим раундом не получен.

Зафиксировано:

- был выполнен один canonical launch `gemini-3.1-pro-preview`;
- CLI не вернул usable report;
- stderr показал repeated provider `429 RESOURCE_EXHAUSTED`;
- explicit reason: `MODEL_CAPACITY_EXHAUSTED`;
- внутри этого же single launch Gemini CLI ещё логировал `read_file` error на локальный harness path из-за ignore patterns;
- повторный user-level запуск не делался.

## 5. Итоговое решение по iter3

`ice-cream iter3` остаётся только `dry-run/duel` веткой.

Baseline `Smart Update` по-прежнему остаётся production эталоном.

Следующий раунд не должен быть ещё одним общим tightening-промптов.
Нужен более узкий `iter4` пакет:

1. `party_theme_program` и `theater_history` перевести в structural generation mode вместо narrative mode.
2. Между `audit_narrative_core` и `repair_narrative_core` добавить deterministic screen на modality / interpretation / scope-drift паттерны.
3. В intermediate audit добавить contextual frame (`title`, возможно площадка/формат), чтобы сократить разницу с final audit.
4. Отдельно прогнать repair stress-test на уже известных unsupported claims.
5. Только после этого снова гнать duel на том же 6-event наборе.

## 6. Мой собственный вывод

С `Opus` я здесь в основном согласен.

Самый важный новый сдвиг по сравнению с `iter2`:

- раньше основной разговор был про anti-hallucination tightening;
- теперь видно, что на двух опасных shape сама цель “написать narrative core” уже задаёт Gemma неправильный режим.

То есть следующий шаг — не просто “ещё жёстче запретить выдумки”, а для части shape сменить сам тип генеративного контракта.
