# Smart Update Gemma Event Copy V2.16.2 Lollipop Seed-Bank Consultation Synthesis

Дата: 2026-03-09

Основание:

- [lollipop design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-funnel-design-brief-2026-03-09.md)
- [lollipop salvage matrix](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-salvage-matrix-2026-03-09.md)
- [Opus review](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-16-2-lollipop-seed-bank-opus-2026-03-09.md)
- [Gemini review](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-16-2-lollipop-seed-bank-gemini-3.1-pro-preview-2026-03-09.md)

## 1. Что показала сама salvage matrix

Матрица подтвердила уже на внутреннем уровне:

- extraction-side carry из `2.15.5+` у нас сильный и конкретный;
- baseline infoblock/list reuse уже можно считать устойчивым инвариантом;
- главная слабая зона будущего `lollipop v1` не в отсутствии extractors, а в том, что composition-side (`hook / pattern / layout / pack / final writer`) пока почти не имеет такого же проверенного банка stage-версий.

Именно это и нужно было проверить через внешнюю консультацию.

## 2. Где Opus и Gemini реально сошлись

Ниже — strongest shared signals.

### 2.1. Extraction-bank у нас уже сильнее, чем composition-bank

Обе модели по сути подтвердили одну и ту же мысль:

- extraction/history carry из поздних Gemma rounds накоплен хорошо;
- downstream composition layer пока недоопределён;
- если не усилить `hook / layout / pack / writer contract`, хороший upstream pack не превратится автоматически в хороший текст.

### 2.2. `hook` нельзя делать как prose-drafting на Gemma

Это главное совпадение консультантов.

Обе модели согласны:

- Gemma не должна писать 2-3 “красивых” hook-кандидата;
- это снова затолкает нас в проблему `ice-cream`: mediocre prose upstream;
- вместо этого `hook` должен стать structured stage family:
  - `hook seed`
  - `angle discovery`
  - `anchor fact`
  - `why this fact matters`

И уже из этого `4o` пишет реальный лид.

### 2.3. `merge_select` нельзя оставлять монолитным

Обе модели считают это одним из самых рискованных мест.

Общий вывод:

- один большой `facts.merge_select` будет слишком тяжёлым и хрупким;
- нужен более дробный merge-подход;
- как минимум нужно разделить:
  - простой deterministic/cheap dedup
  - LLM-conflict-resolution только для реально конфликтных или ambiguous случаев

### 2.4. `writer.final_4o` нельзя оставлять “без спецификации”

Это ещё один сильный консенсус.

Пока `writer.final_4o` — это только идея, а не stage-spec.

Обе модели по сути говорят:

- без явного payload contract для `4o` весь upstream funnel будет впадать в “чёрный ящик”;
- значит `writer.final_4o` надо проектировать рано, а не “добавить потом”.

### 2.5. `extract_plot` нельзя считать безопасным carry для `v1`

Обе модели трактуют его как слишком рискованный для первой рабочей итерации.

Практический вывод:

- `extract_plot` из `v2.15.8` не должен входить в `seed-bank v1` как normal path;
- максимум он может жить отдельно как later experiment.

## 3. Где они разошлись

### 3.1. `pattern` в `v1`

`Opus`:

- считает, что `pattern.route / pattern.select` не нужно тащить в `Gemma v1`;
- предлагает передавать `shape + pattern_hint` финальному `4o`.

`Gemini`:

- не хочет убирать pattern совсем;
- но предлагает radically сузить его роль:
  - не эстетический выбор паттерна,
  - а логическое `fact-density / slot-fit` routing.

Мой вывод:

- полноценный creative pattern choice действительно не стоит делать Gemma-stage в `v1`;
- но совсем выбрасывать pattern signal тоже рано;
- компромиссный carry:
  - в `v1` держать не `pattern.select prose style`,
  - а `pattern.signal / pattern_hint` как structural hint family.

То есть в `v1` pattern остаётся, но в сильно суженной и более логической форме.

### 3.2. Насколько сильно надо консолидировать extractors

`Opus` тянет к сокращению количества extractors.

`Gemini` на уровне общего диагноза сначала спорит с консолидацией как с риском loss of atomicity, но дальше сама же предлагает укрупнить overlapping families до более компактного набора.

Этот ответ внутренне неоднороден, поэтому я считаю его здесь weak signal.

Мой вывод:

- не надо резко схлопывать весь extraction-bank;
- но overlapping families действительно нужно пересобрать в registry аккуратно, чтобы не плодить лишний merge burden;
- значит нужен не “mass consolidation now”, а controlled registry review:
  - что реально остаётся отдельной стадией,
  - что становится alias / sibling version / subfamily.

## 4. Что я считаю итоговым составом `seed-bank v1`

После матрицы и консультаций я бы зафиксировал такой состав.

### 4.1. Обязательные stage families

- `scope`
- `facts.extract`
- `facts.merge.tier1`
- `facts.merge.tier2`
- `facts.type`
- `facts.priority`
- `hook.seed`
- `hook.select`
- `pattern.signal`
- `layout.plan`
- `pack.compose`
- `pack.select`
- `writer.final_4o.spec`
- `writer.final_4o`
- `trace.stage_profile`

### 4.2. Что это значит относительно старой схемы

Это не тот же `seed-bank`, который был в первом draft.

Изменения такие:

- `facts.merge_select` разбит на `tier1 + tier2`;
- добавлен `facts.priority`, потому что внутри bucket тоже нужен выбор того, что становится prose-core;
- `hook.candidates` заменён на `hook.seed`;
- `pattern.route/select` заменён на более узкий `pattern.signal`;
- `pack.compose` больше не одинокий stage, а получает `pack.select`;
- `writer.final_4o.spec` становится отдельным обязательным design artifact/stage family.

## 5. Как это соотносится с quality bar

### 5.1. Что нужно для factual side

Чтобы текст был:

- grounded
- полный по фактам
- без сервисного мусора
- со списками там, где они нужны

нужны:

- `scope`
- `facts.extract`
- `facts.merge.tier1`
- `facts.merge.tier2`
- `facts.type`
- `facts.priority`
- baseline `infoblock`
- baseline `lists`

### 5.2. Что нужно для editorial side

Чтобы текст был:

- естественным
- профессиональным
- нешаблонным
- вариативным
- с сильным лидом
- с осмысленными подзаголовками
- без AI clichés
- без бюрократического drift

нужны:

- `hook.seed`
- `hook.select`
- `pattern.signal`
- `layout.plan`
- `pack.compose`
- `pack.select`
- `writer.final_4o.spec`
- `writer.final_4o`

Вывод:

- `quality bar` действительно недостижим только extraction-side банком;
- downstream presentation-bank должен существовать уже в `v1`, но не как Gemma prose-writing, а как structured presentation planning.

## 6. Что я бы изменил прямо сейчас в канонике

### 6.1. Переименовать hook-family

Было:

- `hook.candidates`

Должно стать:

- `hook.seed`

### 6.2. Разбить merge-family

Было:

- `facts.merge_select`

Должно стать:

- `facts.merge.tier1`
- `facts.merge.tier2`

### 6.3. Добавить приоритизацию внутри buckets

Нужен новый stage family:

- `facts.priority`

Он решает:

- какие facts обязаны попасть в prose;
- какие остаются only in list / infoblock / support.

### 6.4. Ослабить pattern-family

Было:

- `pattern.route`
- `pattern.select`

Для `v1` лучше:

- `pattern.signal`

То есть:

- не полноценный stylistic chooser;
- а structural hint для final writer.

### 6.5. Сделать `writer.final_4o.spec` обязательным артефактом

Нельзя считать `writer.final_4o` stage-family готовым, пока не описано:

- exact input payload
- fields
- anti-pattern payload
- layout contract
- grounding rules

## 7. Что делаю дальше на этой основе

Следующий инженерный порядок я считаю таким:

1. Обновить `lollipop brief` под новые family names.
2. Построить `stage registry` из salvage matrix.
3. Выкинуть `extract_plot` из `v1`.
4. Подготовить первый `seed-bank registry`:
   - с exact version ids
   - provenance
   - known strengths
   - known risks
5. После этого уже запускать первый narrow tuning round только для:
   - `scope`
   - `facts.extract`
   - `facts.type`
   - `facts.merge.tier1/tier2`

И отдельно:

6. Спроектировать `writer.final_4o.spec`, не дожидаясь полного stage-bank.

## 8. Короткий вердикт

`salvage matrix` в целом правильная и полезная.

Но после консультации `seed-bank v1` должен быть уточнён так:

- меньше prose-like Gemma stages;
- больше structured selection stages;
- явный two-tier merge;
- явный `facts.priority`;
- `hook seed`, а не hook prose;
- `pattern signal`, а не full pattern prose choice;
- ранняя спецификация `writer.final_4o`.
