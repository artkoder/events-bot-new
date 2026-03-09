# Smart Update Gemini Event Copy V2.8 Hypotheses Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-8-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-hypotheses-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_7_2026_03_07.py`

## 1. Краткий verdict

Это сильный pre-run response.

Главное:

- Gemini правильно не тянет нас в новый redesign;
- хорошо подтверждает rollback-to-`v2.6` direction;
- согласен не добавлять deterministic support на этом этапе;
- даёт достаточно узкий `v2.8` patch pack, чтобы идти в локальный dry-run.

Мой итог:

- **ещё один pre-run consultation round не нужен**;
- response достаточно strong, чтобы собирать `v2.8`;
- но несколько формулировок нужно брать с умеренной правкой.

## 2. Что в ответе Gemini принимаю

### 2.1. `v2.8` должен строиться от `v2.6`

Это принимаю полностью.

`v2.7` был слишком contaminated, чтобы его патчить дальше.

### 2.2. Extraction = плотные data points

Это тоже принимаю полностью.

Ключевой принцип:

- не single-word fragments;
- не narrative-ready wrappers;
- а компактные self-contained factual propositions.

### 2.3. Deterministic support пока не нужен

С этим согласен.

На этом раунде любой смысловой sanitizer скорее создаст silent risk, чем поможет.

## 3. Что беру только с поправками

### 3.1. `В центре внимания — ...` как allowed framing всё ещё нужно дозировать

Gemini прав про limit, но это не единственный risky template.

Я бы трактовал rule шире:

- не злоупотреблять любой одной вводной конструкцией,
- а не только `В центре ...`.

### 3.2. `Искусство` как пример direct topic extraction слишком коротко

С этим уже не согласен буквально.

Нам нужны:

- плотные;
- но self-contained facts.

То есть лучше:

- `Жизнь и творчество художниц`

чем:

- `Искусство`.

## 4. Что реально пойдёт в `v2.8`

Из ответа Gemini беру:

1. rollback extraction к data-point mode;
2. simplified `_pre_extract_issue_hints` без sentence templates;
3. сохранить anti-bureaucracy и natural framing в generation;
4. добавить template-overuse guardrail в generation wording;
5. сделать `посвящ*` revise issue короче и жёстче.

## 5. Bottom line

Этот Gemini response закрывает pre-run consultation stage для `v2.8`.

Практический вывод:

- **идти в локальный `v2.8` dry-run**: да;
- **дополнительная pre-run консультация**: нет.
