# Smart Update Gemini Event Copy V2.7 Dry-Run Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-7-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-7-dryrun-quality-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_7_2026_03_07.py`

## 1. Краткий verdict

Это сильный и достаточно жёсткий Gemini response.

Главное:

- он не пытался оправдать `v2.7`;
- правильно признал раунд regression;
- очень точно локализовал root cause как `placement error`: мы перенесли syntactic/narrative shaping в Extraction;
- дал clear rollback direction без нового needless redesign.

Мой итог:

- **ещё один консультационный раунд сейчас не нужен**;
- response достаточно strong, чтобы переходить к локальному `v2.8`;
- часть формулировок нужно брать с санитарной правкой, но direction в целом правильный.

## 2. Что в ответе Gemini принимаю

### 2.1. Главная ошибка `v2.7` — narrative-ready extraction

Это принимаю полностью.

Именно это лучше всего объясняет:

- `Выставка носит название ...`
- `В центре встречи ...`
- `facts 11 -> 14`
- collapse of compact density.

То есть ключевой вывод:

- extraction должен возвращать плотные data points;
- generation должен отвечать за human-sounding phrasing.

### 2.2. Rollback agenda-safe framing out of Extraction

Это тоже принимаю.

Наша попытка сделать extraction “человечнее” на самом деле:

- размывала representation;
- плодила boilerplate;
- ухудшала downstream quality.

### 2.3. Оставить anti-bureaucracy и natural framing в Generation

С этим согласен.

То есть полезно сохранить:

- запрет на weak bureaucratic metatext;
- allow-list для естественного event framing.

Это правильное место для таких правил.

## 3. Что беру только с поправками

### 3.1. `вместо "Тема: искусство" -> "Искусство"` слишком агрессивно

Здесь нужна осторожность.

Полностью голый noun типа:

- `Искусство`

может быть слишком атомарным и плохо работать как fact.

Лучше:

- `Жизнь и творчество художниц`
- `История любви Магомаева и Синявской`
- `Устройство платформы и её возможности`

То есть rollback narrative wrapping не должен превращать facts в single-word fragments.

### 3.2. `На встрече разберут ...` как generation allow-list стоит использовать умеренно

Это нормальная natural formula, но если Gemma будет злоупотреблять ей, мы просто поменяем один шаблон на другой.

Поэтому идея Gemini про limit на `В центре ...` правильная, но я бы шире трактовал её как:

- следить за overuse любых one-template openings.

## 4. Что не беру буквально

- Не беру ultra-atomic replacements вроде одного существительного без контекста.
- Не беру допущение, что проблема полностью закрыта только extraction rollback; revise для `посвящ*` всё ещё важен.

## 5. Что реально пойдёт в `v2.8`

Из ответа Gemini я беру следующее:

1. откатить `narrative-ready` и `agenda-safe sentence templates` из Extraction;
2. упростить `_pre_extract_issue_hints` до извлечения чистой сути без шаблонов;
3. оставить anti-bureaucracy в Generation;
4. ограничить template overuse;
5. сделать `посвящ*` revise issue короче и жёстче.

## 6. Bottom line

Этот Gemini response закрывает `v2.7` consultation cycle.

Если коротко:

- **accept**: placement error diagnosis, rollback of narrative extraction, keep anti-bureaucracy in generation;
- **accept with modification**: examples for denser fact forms, template-overuse controls;
- **no more consultation before v2.8**: согласен.

Следующий шаг уже локальный: собирать `v2.8` и снова проверять его на тех же 5 событиях.
