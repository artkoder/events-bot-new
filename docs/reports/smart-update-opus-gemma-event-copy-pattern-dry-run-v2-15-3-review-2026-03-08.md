# Smart Update Gemma Event Copy V2.15.3 Review

Дата: 2026-03-08

Основание:

- [v2.15.3 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-3-5-events-2026-03-08.md)
- [v2.15.3 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-design-brief-2026-03-08.md)
- [v2.15.3 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-prompt-context-2026-03-08.md)
- [Opus prompt pack review](/workspaces/events-bot-new/docs/reports/smart-update-opus-v2-15-3-prompt-pack-review-2026-03-08.md)
- [v2.15.2 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-review-2026-03-08.md)

## 1. Короткий verdict

`2.15.3` не стал новым best round.

По сумме результата:

- против baseline: лучше по aggregate coverage и в целом современнее по prompt discipline;
- против `v2.14`: лучше overall;
- против `v2.15.2`: хуже overall.

Чётко по цифрам:

- baseline total missing = `22`
- `v2.13` total missing = `14`
- `v2.14` total missing = `14`
- `v2.15.2` total missing = `11`
- `v2.15.3` total missing = `13`

По forbidden:

- `v2.15.2`: `0`
- `v2.15.3`: не ноль, потому что `2687` снова пропустил `посвящ*`

Итог:

- `2.15.3` полезен как prompt-pack learning round;
- production-ready: нет;
- strongest current base всё ещё остаётся `2.15.2`, а не `2.15.3`.

## 2. Что реально подтвердилось

### 2.1. Gemma-specific prompt repack действительно влияет

Это не был cosmetic round.

Что реально улучшилось:

- `2745` стал лучшим кейсом этой итерации;
- `2687` больше не разваливается на ложном эпиграфе из соседнего digest item;
- `2.15.3` держит более дисциплинированный self-contained prompt pack;
- по части format/quote hygiene round стал осмысленнее, чем старые `2.15.x` попытки.

### 2.2. Event-local quote discipline стала лучше

Главный провал `2.15.2`:

- ложный эпиграф на `2687`.

В `2.15.3` этот specific failure mode реально подрезан.

Это значит:

- tighter quote gate был не зря;
- prompt-pack redesign под Gemma сработал хотя бы в этой зоне.

### 2.3. Dynamic prompt assembly полезен

Generation в `2.15.3` уже меньше похож на старую giant prompt wall.

Это видно по тому, что:

- sparse cases не всегда тянут ненужную структуру;
- execution hints более узкие;
- main prose prompt стал operationally clearer.

## 3. Что стало хуже

### 3.1. `2660` регресснул против `2.15.2`

Это заметный минус.

Вместо достаточно цельного короткого текста `2.15.3` ушёл в более общий и объясняющий ход:

- "исследуют тему двойственности"
- "сложные философские вопросы"
- "визуальный язык формы и фактуры"

Это звучит более abstract-AI и хуже держит groundedness sparse-case.

И по метрике тоже регресс:

- `v2.15.2 missing=1`
- `v2.15.3 missing=3`

### 3.2. `2734` потерял sharpness

`2734` в `2.15.2` был одним из лучших candidate-cases.

В `2.15.3` текст чище структурно, но слабее по содержательной точности:

- program-led angle стал более общим;
- пропала часть safe structural силы прошлого round;
- coverage регресснул:
  - `v2.15.2 missing=1`
  - `v2.15.3 missing=3`

### 3.3. `2673` всё ещё не решён

Это по-прежнему главный editorial blocker.

Да, в `2.15.3` текст лучше обозначает, что речь о проекте/платформе.
Но prose всё ещё слабый:

- начинается со сцены программы, а не с сути события;
- фраза `платформа предоставляет задачи и возможности` звучит неестественно и фактически плохо собрана;
- сохраняется explanation-heavy тон;
- последний блок почти agenda recap, а не нормальная project-presentation copy.

`2673` по-прежнему хуже baseline по naturalness и всё ещё слабее, чем должен быть лучший `2.15.x` candidate.

### 3.4. `2687` не добил anti-`посвящ*`

Хотя false epigraph устранён, `2687` всё равно пропускает `посвящ*`.

Это значит:

- prompt repack сам по себе не решил stubborn lexical issue;
- repair/validation contract всё ещё недостаточно жёсток для этой проблемы.

## 4. По кейсам

### `2660`

- против baseline: лучше по общей чистоте текста, но хуже держит groundedness sparse-case
- против `v2.15.2`: хуже
- verdict: regression

### `2745`

- против baseline: лучше
- против `v2.15.2`: лучше
- verdict: strongest local win round

### `2734`

- против baseline: лучше overall
- против `v2.15.2`: хуже
- verdict: mixed regression

### `2687`

- против baseline: лучше overall, потому что исчез false quote catastrophe
- против `v2.15.2`: mixed
- verdict: hygiene gain, but not fully fixed

### `2673`

- против baseline: всё ещё не clean editorial win
- против `v2.15.2`: не лучше
- verdict: unresolved project-case failure

## 5. Архитектурный вывод

`2.15.3` важен не как новый best version, а как доказательство следующего:

- Gemma-specific repack по всем main prompts действительно нужен;
- research-driven sectioned prompts полезны;
- но одного repack недостаточно, если:
  - sparse-case grounding ослабевает;
  - program/project cases получают слишком общий prose;
  - validation/repair не закрывают stubborn lexical problems.

То есть следующий шаг нельзя сводить к:

- ещё одному ban-list;
- ещё одному regex-layer;
- ещё одному одному generation prompt tweak.

Нужно улучшать:

- planner semantics;
- project / program execution cards;
- quote/epigraph gating;
- issue-specific repair enforcement.

## 6. Что это значит для следующего шага

Если брать `2.15.3` как learning round, то фактический carry-forward такой:

Сохранить:

- self-contained sectioned prompts;
- dynamic prompt assembly;
- tighter quote provenance;
- LLM-first normalization core.

Доработать:

- sparse-case grounding, чтобы не уходить в abstract explanation prose;
- `program_led` sharpness;
- project presentation prose для `2673`;
- hard anti-`посвящ*` enforcement уже после generation/repair.

## 7. Production verdict

`2.15.3` в production переносить нельзя.

Причины:

- aggregate weaker than `2.15.2`;
- `2673` остаётся слабым;
- `2660` и `2734` показали regressions;
- `2687` всё ещё не полностью clean.

Но сам round полезный:

- он подтвердил, что Gemma-specific prompt-pack redesign был правильным ходом;
- он отделил полезные repack gains от remaining text-quality blockers;
- он даёт хороший материал для следующей консультации с Gemini уже по реальным текстам, а не по теории.
