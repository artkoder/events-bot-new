# Smart Update Gemini Event Copy First Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-first-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-first-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-2-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-2-review-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_2_5events_2026-03-07.json`

## 1. Краткий verdict

Ответ Gemini сильный и практически полезный.

Главное:

- он хорошо прочитал реальный `v2.2` dry-run;
- правильно сместил фокус с “ещё одного redesign” на `fact representation + prompt contract`;
- дал достаточно узкий набор изменений, который реально можно проверить локально.

Мой итог:

- **ещё один этап дискуссии с Gemini сейчас не нужен**;
- следующий шаг рационально делать уже локально: узкий `v2.3` dry-run;
- к Gemini есть несколько поправок, но они не блокируют implementation round.

## 2. Что в ответе Gemini особенно сильное

### 2.1. Правильная локализация bottleneck

Это самая ценная часть ответа.

Gemini правильно увидел, что текущий тупик уже не про “ещё один паттерн”, а про то, что:

- `facts_text_clean` часто слишком атомарный;
- generation вынуждена склеивать дробные факты в prose;
- из-за этого появляются репетиции, сухость или синтаксически неловкие ходы.

Это особенно хорошо совпадает с кейсом `2673`, а частично и с `2734`.

### 2.2. Сильный тезис про dynamic formatting

Это очень полезная калибровка.

Gemini убедительно сформулировал, что:

- sparse cases не должны автоматически получать подзаголовки;
- короткий текст часто лучше работает как 2-3 связных абзаца;
- headings в коротких кейсах часто делают prose более бюрократичным, а не более профессиональным.

Это хорошо совпадает с нашими наблюдениями по `2745` и частично `2660`.

### 2.3. Правильная критика hard negative constraints

Это ещё один сильный пункт.

Запреты вида:

- “не используй `посвящ*`”

локально действительно ломали русский синтаксис и вели к уродливым заменам вроде:

- `в центре внимания великой любви`.

Gemini прав, что для Gemma positive examples почти наверняка полезнее, чем грубые lexical bans.

### 2.4. Самодостаточность основного текста

Замечание про эпиграф точное и практичное.

Эпиграф полезен, но он не должен становиться костылём, который держит missing subject/context.
Кейс `2660` Gemini прочитал правильно.

## 3. Что я принимаю в работу

### 3.1. Accept now

1. Делать следующий шаг как узкий `v2.3` dry-run, а не как новую дискуссионную фазу.
2. Убрать hard lexical anti-`посвящ*` из generation contract.
3. Заменить его на positive start-pattern examples.
4. Добавить правило самодостаточности основного текста:
   - субъект / автор / коллектив должны звучать в body, даже если есть epigraph.
5. Сделать dynamic formatting:
   - sparse cases без обязательных Markdown headings;
   - списки только когда реально есть program/list payload.
6. Оставаться в линии “не возвращать сложный repair pipeline”.

### 3.2. Accept with modification

1. `Fact pre-consolidation`

Direction сильный, но его нельзя брать грубо.

Нужно:

- только в experimental ветке;
- с явным preservation floor;
- без схлопывания program items;
- без смешивания разных смысловых ролей в один “красивый” мегa-факт.

То есть да, но не “сжать всё до 5 фактов”, а нормализовать repetition / fragmentation.

2. `2745` as success case

Gemini прав, что `v2.2` тут чище prose-wise.
Но это не значит, что baseline здесь точно хуже по сумме качества.
Я бы трактовал этот кейс так:

- `v2.2` выигрывает по cleanliness;
- baseline всё ещё держит неплохую informational density;
- значит это case for sparse-format tuning, а не просто “`v2.2` уже победил”.

3. `No LLM revisor`

Тут я согласен по сути, но не буквально.

Правильный вывод:

- не надо возвращать отдельный сложный repair-pass;
- не надо наращивать новый многошаговый revise pipeline.

Но это не означает, что нужно автоматически выкинуть уже существующий лёгкий revise step, если он полезен.

## 4. Где Gemini местами переупростил картину

### 4.1. `Pre-consolidation -> brilliant prose` слишком оптимистично

Это хорошая гипотеза, но не гарантия.

Даже после нормализации фактов остаются:

- routing problems;
- overly broad `copy_assets`;
- weak lead shaping;
- tendency к generic editorial fillers.

То есть fact consolidation — likely high-impact, но не серебряная пуля.

### 4.2. Не все negative constraints плохи

Gemini прав против brittle lexical bans.
Но из этого не следует, что вообще любые negative guardrails вредны.

Сохранять стоит:

- anti-CTA;
- anti-metatext;
- anti-service contamination;
- anti-unsupported embellishment.

Откатывать нужно не весь класс запретов, а именно те, что заставляют Gemma ломать грамматику.

### 4.3. Часть rankings остаётся редакторски субъективной

Например:

- `2660`: baseline действительно сильный, но назвать `v2.2` однозначно худшим вариантом — уже оценочное суждение;
- `2745`: `v2.1 / v2.2` как joint-best — спорно, потому что `v2.1` всё же тек через service contamination.

Это не дискредитирует ответ, но значит, что rankings Gemini надо использовать как editorial signal, а не как hard oracle.

## 5. Нужен ли ещё один раунд с Gemini сейчас

**Нет.**

Причина простая:

- ответ уже достаточно grounded;
- он не уходит в теоретизирование;
- он даёт понятный и ограниченный next step;
- дополнительный раунд сейчас даст меньше пользы, чем реальный новый dry-run.

Рациональная последовательность такая:

1. Локально собрать `v2.3`.
2. Повторить dry-run на тех же 5 событиях.
3. Уже потом, если будет нужно, вернуться к Gemini или к другому внешнему консультанту с новым evidence set.

## 6. Что я предлагаю как следующий шаг

Следующий шаг: **сразу делать правки для `v2.3` dry-run**, без дополнительной дискуссии с Gemini на этом этапе.

Ядро `v2.3` должно быть узким:

1. Ослабить brittle lexical ban на `посвящ*` и заменить его positive examples.
2. Ввести sparse formatting contract:
   - короткие кейсы без обязательных headings.
3. Добавить body self-sufficiency rule:
   - имя/субъект не должен уезжать только в epigraph.
4. Добавить очень аккуратную pre-consolidation / dedup нормализацию facts перед generation.
5. Не трогать:
   - current grounded fact-first core;
   - cleanup pipeline;
   - content-preservation floor;
   - service/CTA/metatext guardrails.

## 7. Bottom line

Gemini дал не просто opinion, а usable correction vector.

Самое ценное в его ответе:

- он не пытается снова “архитектурно спасать” всё сразу;
- он правильно указывает на `fact fragmentation + rigid prompting` как на главную проблему prose quality.

Поэтому мой ответ на твой вопрос такой:

- review принят;
- **ещё один раунд дискуссии с Gemini сейчас не нужен**;
- дальше лучше **сразу идти в локальные правки и новый `v2.3` dry-run**.
