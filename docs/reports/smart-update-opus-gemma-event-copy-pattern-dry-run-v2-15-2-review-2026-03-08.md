# Smart Update Gemma Event Copy V2.15.2 Review

Дата: 2026-03-08

Основание:

- [v2.15.2 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-5-events-2026-03-08.md)
- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)
- [v2.13 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-review-2026-03-08.md)
- [v2.14 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-14-review-2026-03-08.md)

## 1. Короткий verdict

`v2.15.2` — это самый сильный `2.15.x` round на данный момент по сумме `coverage + hygiene`.

По сухим цифрам:

- baseline total missing = `22`
- `v2.13` total missing = `14`
- `v2.14` total missing = `14`
- `v2.15.2` total missing = `11`
- `v2.15.2` total forbidden = `0`

Итоговая оценка:

- против `v2.14`: `v2.15.2` лучше overall;
- против `v2.13`: `v2.15.2` лучше как system candidate, но не по всем текстам лучше редакторски;
- против baseline: `v2.15.2` лучше как candidate architecture и coverage-safe branch, но это ещё не clean editorial win против baseline на всех 5 кейсах.

Причина последней оговорки: `2687` остаётся явным regress-case по prose quality из-за ложного эпиграфа, а `2660` показывает leak формата вопреки собственному plan.

## 2. Что реально стало лучше

- `full-floor normalization + tiny planner + dynamic generation` дали первый `2.15.x` round, где одновременно:
  - missing стал лучше baseline, `v2.13` и `v2.14`;
  - forbidden-маркеры ушли в `0`;
  - `2673` наконец лучше называет событие как презентацию проекта, а не уходит в абстрактное explanation prose.
- Sparse cases снова выглядят компактно и без явного бюрократического расползания:
  - `2660`
  - `2745`
- `2734` держит хороший coverage без старого service leakage.
- `2673` заметно лучше по factual framing, чем в `v2.13` и `v2.14`: теперь текст прямо говорит, что это презентация социальной сети для профессионалов креативных индустрий.

## 3. Главные проблемы

### 3.1. `2687` сломан ложным quote extraction

`2687` — главный blocker этого раунда.

Что произошло:

- quote extractor вытащил `План Ле Корбюзье: Буэнос-Айрес` из соседнего элемента музейного дайджеста;
- tiny planner из-за этого выбрал `quote_led`;
- в финальный текст попал нерелевантный эпиграф, который не относится к лекции `Художницы`.

Это не мелкая stylistic проблема, а architectural failure:

- quote gate сейчас недостаточно привязан к фактическому предмету события;
- false quote может отравить и pattern selection, и lead.

### 3.2. `2660` показывает format-plan leak

Для `2660` plan был:

- `use_headings=false`
- branch=`compact_fact_led`

Но финальный текст всё равно сгенерировал `###`-секции.

Следствие:

- generation не до конца слушается plan;
- validation не ловит сам факт запретного heading leakage;
- compact branch пока не полностью safe.

### 3.3. `2734` всё ещё переигрывает с цитатой/эпиграфом

В `2734` текст уже лучше по coverage и overall hygiene, но остаётся structural artifact:

- отдельный blockquote `«Янтарный соловей»` выглядит не как сильный эпиграф, а как случайный обрывок;
- финальный абзац частично повторяет уже сказанное выше.

То есть program-rich case в целом оздоровился, но `quote/block` policy всё ещё слишком loose.

### 3.4. `2673` улучшился, но ещё не стал по-настоящему естественным

Самое важное улучшение:

- текст наконец ясно говорит, что это презентация проекта / социальной сети;
- user-facing проблема предыдущих раундов здесь частично решена.

Что ещё не дотянуто:

- headings всё ещё слишком agenda-like;
- prose местами остаётся explanation-heavy;
- `quote_led` тут уже работает лучше, чем раньше, но всё ещё не выглядит best-fit pattern.

## 4. По кейсам

### `2660`

- coverage: хорошо
- prose: mixed
- verdict: лучше `v2.14`, примерно на уровне `v2.13`, но не clean win из-за heading leakage и повторов

### `2745`

- coverage: хорошо
- prose: лучший кейс этого раунда
- verdict: лучше baseline, `v2.13` и `v2.14`

### `2734`

- coverage: хорошо
- prose: mixed-positive
- verdict: лучше `v2.14`, сопоставимо или немного лучше baseline по общему впечатлению, но ещё не clean editorial finish

### `2687`

- coverage: лучше baseline
- prose: хуже baseline и хуже `v2.13`
- verdict: главный regression-case из-за ложного эпиграфа

### `2673`

- coverage: лучше `v2.13`, на уровне лучшей недавней линии
- prose: лучше `v2.13` и `v2.14`, но ещё не достаточно естественно
- verdict: meaningful improvement, но не закрытый case

## 5. Архитектурный вывод

`v2.15.2` подтверждает правильность нескольких решений:

- semantic core должен оставаться `LLM-first`;
- `full-floor normalization` полезнее старого `subset extraction`;
- dynamic prompt assembly работает лучше giant prompt wall;
- tiny planner допустим, если остаётся structural-only.

Но текущий round также показал, что следующие проблемы нельзя решать наращиванием regex-semantics:

- quote grounding;
- strict format-plan enforcement;
- heading / epigraph validation;
- pattern overreach для lecture/presentation cases.

Нужный следующий тип улучшений:

- более узкий and safer quote gate;
- deterministic validation именно на mismatch `plan -> output`;
- более короткая и жёсткая execution contract для `quote_led`;
- сдвиг `2673` от agenda-like sectioning к более естественной project presentation prose.

## 6. Production verdict

В production переносить `v2.15.2` пока рано.

Почему:

- `2687` даёт слишком заметную editorial failure;
- `2660` доказывает, что plan enforcement пока не надёжен;
- `2673` ещё не достиг нужного качества финального текста.

Но как исследовательский результат это сильный раунд:

- лучший `coverage + forbidden` профиль за весь цикл;
- архитектурно полезный candidate;
- реальная база для post-run Gemini consultation уже есть.
