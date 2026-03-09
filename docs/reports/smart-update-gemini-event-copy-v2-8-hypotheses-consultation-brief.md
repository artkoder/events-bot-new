# Smart Update Gemini Event Copy V2.8 Hypotheses Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-7-review-2026-03-07.md`
- `docs/reports/smart-update-gemini-event-copy-v2-7-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-7-dryrun-quality-consultation-response-review.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_7_2026_03_07.py`

## 1. Зачем нужна эта консультация

Нам нужен узкий pre-run round перед `v2.8`.

После `v2.7` картина уже не про поиск “магического паттерна”, а про корректный rollback одной конкретной ошибки:

- мы случайно перенесли narrative shaping в Extraction;
- получили fact inflation;
- сломали рабочие кейсы;
- и всё равно не вылечили `2687` / `2673`.

## 2. Что уже подтверждено

Сейчас довольно надёжно подтверждено следующее:

- `v2.6` был net-positive, но не production-ready;
- `v2.7` стал regression round;
- safe-positive examples сами по себе не решение, если они стоят в Extraction;
- anti-bureaucracy и natural event framing useful, но это layer Generation / Revise;
- `посвящ*` всё ещё остаётся stubborn failure marker.

## 3. Наш текущий hypothesis pack для `v2.8`

### H1. `v2.8` должен строиться от `v2.6`, а не от `v2.7`

То есть:

- брать `v2.6` как рабочую base;
- переносить только те идеи из `v2.7`, которые действительно survived evidence.

### H2. Extraction надо вернуть к плотным factual propositions

Наша гипотеза:

- extraction не должен писать `narrative-ready` или `agenda-safe` предложения;
- он должен возвращать плотные publishable data points;
- нельзя плодить packaging facts вроде:
  - `Выставка носит название ...`
  - `Автор выставки — ...`
  - `В центре встречи — ...`

### H3. `_pre_extract_issue_hints` надо упростить и очистить от sentence templates

Наша гипотеза:

- hints должны требовать извлечения самой сути;
- но не должны навязывать готовые sentence-level patterns;
- особенно нельзя снова тащить в hints:
  - `в центре ...`
  - `на встрече разберут ...`

### H4. Anti-bureaucracy надо оставить в Generation, а не в Extraction

Наша гипотеза:

- generation может и должен банить:
  - `мероприятие анонсирует...`
  - `проект призван стать...`
  - `будет представлен обзор...`
- generation может и должен разрешать natural event framing:
  - `лекция о ...`
  - `спектакль исследует ...`
  - `на встрече разберут ...`

### H5. Нужен мягкий control on template overuse

Гипотеза:

- проблема не только в самом шаблоне `В центре ...`, а в его repeated use;
- значит, возможно, generation prompt надо учить не столько одному allowed phrase, сколько rule:
  - не повторять один opening/frame more than once.

### H6. Revise для `посвящ*` должен стать короче и механичнее

Гипотеза:

- Gemini прав, что previous wording было слишком сложным;
- нужен короткий blocking instruction:
  - удалить предложение целиком;
  - переписать суть иначе;
  - без косметических замен.

### H7. Возможно, нужен очень узкий deterministic support layer, но только если он не меняет смысл

Это open question, а не принятая истина.

Мы рассматриваем только low-risk support, например:

- детектировать empty packaging wrappers;
- не позволять фактам становиться single-word fragments;
- не менять смысл вручную.

Если Gemini сочтёт, что даже такой слой пока вреден, мы его не берём.

### H8. После этой консультации надо сразу идти в локальный `v2.8` dry-run

То есть новый theory loop не нужен, если hypothesis pack уже достаточно сильный.

## 4. Что мы хотим проверить у Gemini

Нас интересует:

1. Достаточно ли strong этот rollback-oriented pack.
2. Где мы всё ещё рискуем повторить `v2.7` under другой формулировкой.
3. Нужен ли вообще хоть какой-то deterministic support.
4. Какие конкретные Gemma-friendly prompt edits реально worth testing next.

## 5. Самое важное требование

Нужен не общий opinion, а hypothesis-by-hypothesis critique с конкретными prompts/block edits.

Особенно важно:

- исключать всё, что уже было проверено и сломалось;
- не предлагать unsafe rewrites;
- не предлагать needless complexity.
