# Smart Update Opus V2.15.3 Prompt Pack Review

Дата: 2026-03-08

Основание:

- Raw strict Opus JSON: [event-copy-v2-15-3-opus-gemma-prompt-pack-2026-03-08.json](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-15-3-opus-gemma-prompt-pack-2026-03-08.json)
- Extracted Opus prompt pack: [smart-update-opus-v2-15-3-prompt-pack-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-opus-v2-15-3-prompt-pack-2026-03-08.md)
- Brief for the round: [event-copy-v2-15-3-opus-gemma-prompt-pack-brief-2026-03-08.md](/workspaces/events-bot-new/artifacts/codex/tasks/event-copy-v2-15-3-opus-gemma-prompt-pack-brief-2026-03-08.md)

## 1. Проверка режима

Strict `Opus` launch корректный:

- raw JSON сохранён;
- `modelUsage` содержит только `claude-opus-4-6`;
- fallback или второй `Opus` запуск не делался.

## 2. Короткий verdict

Это полезный и practically strong deliverable.

Самое ценное:

- `Opus` реально написал полный prompt pack по всем Gemma steps;
- он правильно удержал `LLM-first` architecture и не увёл решение в regex-first;
- он хорошо попал в нужный уровень конкретики: короткие секционированные prompts, pattern cards, dynamic blocks, per-step decoding.

Но переносить ответ буквально нельзя.

Главные причины:

- в deliverable есть внутренние несостыковки;
- часть советов слишком буквальна и может породить новый template drift;
- некоторые поля/контракты задекларированы в диагнозе и carry-forward списке, но не доведены до фактического JSON schema prompt text.

Итог:

- **worth taking**
- **accept with modification**
- использовать как базу для реализации `2.15.3`, но не как literal source of truth

## 2.1. Насколько `Opus` соблюл Gemma research carry

В этом раунде `Opus` был проинструктирован не в пустоте: brief уже включал carry из Gemma research и из наших ретроспектив.

Что он действительно соблюл:

- self-contained sectioned prompts;
- short-step architecture вместо giant prompt;
- per-step decoding discipline;
- dynamic prompt assembly;
- facts-backed repair;
- более позитивную и operational form правил вместо одной стены запретов.

Что он соблюл только частично:

- stronger quote provenance он описал концептуально, но не довёл до полностью согласованного schema contract;
- typed metadata предложены полезно, но местами слишком жёстко и могут вернуть label-style drift;
- anti-cliche governance он сделал компактнее, но всё ещё местами перегнул в сторону rule density.

Что пришлось доделывать локально уже в реализации:

- выровнять `has_verified_quote` / `quote_source` contract;
- ужать style-rule density;
- ослабить риск label-style grouped facts;
- сильнее подстроить planner и generation под реальные `2.15.x` failure cases.

## 3. Что принимаю

### 3.1. Full prompt repack across all Gemma steps

Это главный полезный результат.

`Opus` не вернулся к "подправим один generation prompt", а действительно разложил prompt pack по шагам:

- `normalize_floor`
- `shape_and_format_plan`
- `generate_description`
- `targeted_repair`

Это совпадает с актуальной логикой `2.15.3`.

### 3.2. Секционированные self-contained prompts

Это сильный carry из Gemma research, и `Opus` его хорошо реализовал:

- `РОЛЬ`
- `ПРАВИЛА`
- `ВЫХОД`
- `ФАКТЫ / ПЛАН`

Для Gemma это выглядит правильным operational shape.

### 3.3. Pattern cards + optional blocks

Эта часть особенно полезна для `generate_description`.

Хорошо, что `Opus` развёл:

- core prompt;
- pattern card;
- optional blocks:
  - epigraph
  - headings
  - list
  - `why_it_matters`

Это лучше старой монолитной prompt wall.

### 3.4. Жёстче defined epigraph contract

Правильный direction:

- `use_epigraph` не должен жить как loose stylistic whim;
- генератор должен использовать только конкретный `epigraph_fact_id`;
- repair должен скорее удалять плохой epigraph, чем "чинить" его творчески.

### 3.5. Facts-backed repair

Это тоже правильный carry:

- `targeted_repair` получает issues + facts + current description;
- repair больше не должен быть blind rewrite.

### 3.6. Per-step decoding discipline

В целом принимаю:

- low temperature для `normalize_floor` / planner / repair;
- более свободный режим только для main prose generation.

Это хорошо совпадает и с research, и с прошлой практикой.

## 4. Что беру только с поправками

### 4.1. `quote_source` / `has_verified_quote`

В диагнозе и top-10 `Opus` говорит про:

- `quote_source`
- `has_verified_quote`

Но в самом prompt text для `normalize_floor` этих полей нет.

Есть только:

- `type=quote`
- `quote_author`
- `confidence`

Это внутренняя несостыковка deliverable.

Правильный вывод:

- идею stronger quote provenance принимаю;
- literal current prompt text надо доработать;
- либо добавить отдельное поле вроде `quote_origin="event_local | unclear"`,
- либо явно описать, как `confidence` и local relevance должны кодировать quote safety.

### 4.2. Typed fact schema itself

`Opus` предлагает richer typed facts:

- `what`
- `who`
- `when`
- `where`
- `program`
- `program_block`
- `detail`
- `quote`
- etc.

Это потенциально полезно для planner, но есть риск:

- typed facts легко превращаются в новый rigid ontology layer;
- если generation начнёт слишком буквально смотреть на эти labels, prose снова станет mechanical.

Принимаю только частично:

- types можно использовать как support metadata;
- `text` должен оставаться главным semantic carrier;
- planner может выиграть от type hints;
- generation не должна писать "по label-ам".

### 4.3. Stop-phrase block in generation

`Opus` сам пишет, что основную enforcement лучше переложить на deterministic validation.

Но в финальном generation prompt он всё равно оставляет заметный in-prompt stop-phrase block.

Это не катастрофа, но и не идеальная consistency.

Принимаю с поправкой:

- короткий in-prompt anti-cliche hint можно оставить;
- wall-of-bans снова раздувать нельзя;
- основной enforcement должен остаться downstream.

### 4.4. Universal Telegram bold rule

`Opus` предлагает:

- `**жирный**` для ключевых имён/названий при первом упоминании

Это потенциально полезно, но не должно стать обязательным universal behavior.

Причина:

- на части sparse / delicate cases жирное форматирование может выглядеть слишком mechanical;
- baseline structured discipline полезна, но не любой bold automatically good.

Это стоит брать как optional style rule, не как unconditional invariant.

### 4.5. Example-heavy pattern cards

В целом pattern cards сильные, но примеры надо брать осторожно.

Причина:

- Gemma любит прилипать к ритму примера;
- особенно это опасно на лидах.

Поэтому:

- structural hint принимаю;
- отдельные example sentences лучше сильно укоротить или сделать optional.

## 5. Что не принимаю буквально

### 5.1. Слишком жёсткая default-логика по headings

`Opus` местами слишком резко толкает:

- богатая структура -> headings true
- лекции / концерты / выставки -> headings false

Это useful bias, но не должно становиться догмой.

Нам нужен:

- stronger heading gate;
- но не blanket rule по event type alone.

### 5.2. `program_block` as text with prefixes

Формулировки типа:

- `В программе: ...`
- `Темы: ...`

могут быть полезны внутри normalized floor как support structure,
но если их без контроля перенести в generation, это легко вернёт label-style prose.

Поэтому:

- grouped block idea принимаю;
- literal text prefixes надо фильтровать аккуратно.

### 5.3. `80-200` words as universal target

Это слишком broad and slightly templatic.

Практически у нас уже есть понимание:

- sparse case может быть короче;
- rich case может быть длиннее;
- word target лучше оставлять как soft guide, не как tight universal contract.

## 6. Practical synthesis for implementation

Если превращать ответ `Opus` в реальный `2.15.3`, то правильно брать из него следующее:

1. Repack all main prompts in sectioned self-contained form.
2. Keep dynamic prompt assembly with:
   - one core generate prompt
   - one pattern card
   - only active optional blocks
3. Strengthen quote provenance in `normalize_floor`.
4. Make planner more semantically explicit, but still structural-only.
5. Keep repair issue-scoped and facts-backed.
6. Trim the in-prompt anti-cliche block so it stays compact.
7. Keep deterministic enforcement for:
   - headings mismatch
   - list mismatch
   - blockquote mismatch
   - forbidden phrases

## 7. Final conclusion

`Opus` delivered what was needed:

- not theory;
- not only critique;
- a real prompt pack.

This is enough to move into `2.15.3` implementation work.

But the response still needs local engineering judgment before use.

Best framing:

- `Opus` prompt pack = **primary implementation input**
- local brief + research + our retrospective = **guardrails against literal over-acceptance**
