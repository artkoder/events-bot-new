# Smart Update Opus Gemma Event Copy Text Quality Improvements Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-text-quality-improvements.md`
- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-followup-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `smart_event_update.py`

## 1. Краткий verdict

Это хороший каталог улучшений.

Но как **непосредственный input в production prompts** он пока слишком широкий.

Главная проблема не в качестве идей, а в упаковке:

- сильных предложений действительно много;
- но если тащить их в prompts почти wholesale, для Gemma это почти наверняка обернётся instruction noise.

Поэтому правильный verdict такой:

- документ полезен как `idea bank`;
- но перед локальной реализацией его нужно **жёстко ужать и приоритизировать**.

## 2. Самые сильные предложения

Ниже — идеи, которые выглядят high-signal и реально могут поднять качество.

### 2.1. Anti-embellishment block

Очень сильная идея.

Особенно полезно:

- явный ban list на unsupported prose;
- принцип `можно перестроить фразу, нельзя достроить смысл`.

Это прямой удар по одному из самых частых defects.

### 2.2. Anti-metatext rule

Тоже один из strongest items.

Запрет на:

- `лекция расскажет о...`
- `спектакль рассказывает...`
- `концерт представляет собой...`

имеет очень высокий quality impact.

Это действительно один из самых сильных способов убрать “учебниковый” и шаблонный тон.

### 2.3. Anti-duplication prompt/runtime pair

Это уже совпадает с follow-up response и выглядит must-have.

### 2.4. Heading palette / generic heading ban

Очень полезно.

Причём не только как запрет `О событии`, но и как positive signal:

- конкретные heading names;
- содержательные секции вместо meta headings.

### 2.5. Filler phrase ban

Тоже high value.

Gemma действительно хорошо реагирует на explicit denylist формулировок.

## 3. Что полезно, но требует сужения

### 3.1. Lead engineering

Сильная идея, но в текущем виде слишком широкая.

Что я бы оставил:

- concrete-detail lead
- quote lead
- contrast lead

Что я бы не принимал в v2 как prompt mandate:

- question-led openings

Они слишком легко скатываются в gimmick и synthetic engagement.

### 3.2. Sentence variety rule

Как общая идея правильно.
Но в текущей формулировке слишком много micro-style instructions:

- чередуй длинные и короткие;
- простое → сложное → назывное → вопросительное;
- не начинай 2 предложения подряд...

Для Gemma это многовато.

Сильное ядро здесь только одно:

- не начинать подряд несколько предложений с одного и того же слова / конструкции.

Остальное лучше не тащить в первую v2 волну.

### 3.3. Density-aware prompts

Идея здравая.
Но как first v2 step я бы не делал 3 fully separate prompt variants.

Практичнее:

- sparse branch
- non-sparse branch

То есть simplified two-tier version.

### 3.4. Few-shot examples

Очень high-leverage идея, но и high-risk.

Риски:

- token overhead;
- structural copying;
- style freezing;
- prompt overweight for Gemma.

Поэтому:

- не reject;
- но скорее `defer or micro-shot only`.

Если и брать, то:

- один очень короткий micro-example;
- и только после core blockers fixed.

## 4. Что выглядит рискованно или premature

### 4.1. Question-led openings

Я бы не вёл это в v2.

Слишком высокий риск:

- artificial engagement;
- rhetorical gimmick;
- неестественный тон для event copy.

### 4.2. Heavy paragraph quality gates

Параграфные и sentence-level quality gates из разделов C.3 / C.5 выглядят умно,
но risk/reward пока сомнительный.

Проблемы:

- высокая fuzziness;
- false positives;
- усложнение runtime;
- сложнее отлаживать, чем простые structural checks.

Это скорее later refinement, не v2 core subset.

### 4.3. Слишком большой composite style block

В самом конце документа есть сильный composite quality block.
Но как единая вставка он уже довольно тяжёлый.

Если поверх этого добавить:

- pattern rules
- completeness rules
- logistics bans
- list rules
- epigraph rules

можно легко перегрузить prompt.

То есть idea хорошая, но нужна **компрессия**.

## 5. Практический приоритет

### 5.1. Accept now

1. anti-embellishment rule
2. anti-metatext rule
3. anti-duplication rule
4. filler phrase ban
5. generic heading ban / heading palette

### 5.2. Accept with modification

1. lead engineering
   - keep only: concrete-detail / quote / contrast
   - drop question-led

2. sentence variety
   - keep only anti-repeated-start rule
   - drop broader rhetoric choreography

3. density-aware prompting
   - reduce to sparse vs non-sparse

4. few-shot examples
   - maybe one tiny example later
   - not a must-have for first v2 patch

### 5.3. Defer

1. paragraph quality gate
2. sentence-level prose-quality gate
3. richer stylistic runtime diagnostics beyond core blockers

## 6. Что это значит для следующего этапа

Следующий consultation round должен быть не про поиск новых идей,
а про **compression and prioritization**:

- какие 5-7 text-quality improvements действительно входят в `v2 patch pack`;
- как они формулируются компактно;
- что остаётся за бортом v2, чтобы не утонуть в prompt noise.

## 7. Bottom line

`text-quality-improvements.md` — полезный документ.

Но как есть это скорее:

- широкий каталог сильных идей;
- чем готовый к внедрению patch set.

Для Gemma нам нужен не richest possible prompt,
а **shortlist of the highest-signal improvements**.
