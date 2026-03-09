# Smart Update Opus Gemma Event Copy Pattern Redesign Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-followup-response-review.md`
- `artifacts/codex/opus_gemma_event_copy_prompt_inventory_latest.md`

## 1. Краткий verdict

Это на данный момент **самый сильный ответ Opus в этом цикле**.

Главное:

- pattern-driven direction принимается;
- response уже похож не на brainstorming, а на implementation blueprint;
- переход к локальному design/prototype после этого ответа выглядит оправданным;
- новый большой Opus-раунд не обязателен.

Но есть важная оговорка:

- **medium-risk contract в текущем виде ещё слишком широкий**;
- перед runtime implementation его лучше немного сузить и сделать жёстче по scope.

Иными словами:

- **направление принимается**;
- **план фаз принимается**;
- **часть schema/routing деталей стоит подрезать или уточнить**.

## 2. Что в ответе Opus действительно сильное

### 2.1. Вариативность наконец описана через композицию, а не через “красивые слова”

Это главное улучшение относительно ранних раундов.

Opus правильно сместил фокус:

- с cosmetic prompt cleanup;
- на narrative patterns;
- routing;
- signal-based choice of structure.

Это именно тот тип вариативности, который реально может уменьшить ощущение шаблонности.

### 2.2. Extraction redesign привязан к задачам письма, а не к декоративности

Хороший сигнал в ответе:

- он не предлагает засорять extraction “прилагательными”;
- он отбрасывает `safe_descriptors` как вторичное поле;
- он привязывает enrichment к composition logic:
  - `core_angle`
  - `program_highlights`
  - `experience_signals`
  - `why_go_candidates`
  - `voice_fragments`
  - `scene_cues`

Это редакторски более зрелый ход, чем попытка просто расширить style prompt.

### 2.3. Low-risk phase сформулирована практично

Phase 1 выглядит как реально исполнимый пакет:

1. shared rule blocks;
2. heading stop-list;
3. anti-redundancy;
4. lead variety rules;
5. compact sizing;
6. extended coverage flags;
7. anti-template self-check.

Это бьёт в самые видимые дефекты current output и не требует schema migration.

### 2.4. Quality controls выглядят взрослее, чем в прошлых раундах

Особенно полезны:

- `template_feel`
- `weak_lead`
- `weak_heading`
- `redundancy`

Это уже не “ещё чуть-чуть стилистики”, а нормальные quality gates для fact-first текста.

### 2.5. Lifecycle support prompts описан честно

Opus не делает вид, что все repair layers можно удалить сразу.

Это правильный engineering posture:

- часть repair prompts остаётся;
- часть становится transitional;
- удаление привязано к residual rate, а не к желанию упростить систему на бумаге.

## 3. Что всё ещё требует коррекции или осторожности

### 3.1. Medium-risk schema в v1 перегружена

Это самый важный remaining risk.

В текущем ответе слишком многое тащится в первый enriched contract:

- `tone_hint`
- `routing_features`
- `contrast_or_tension`
- богатый набор supporting fields

Проблема не в том, что поля плохие сами по себе.
Проблема в том, что для первой рабочей ветки это может:

- увеличить шум extraction;
- ухудшить стабильность базовых `facts`;
- усложнить parsing и runtime debugging;
- затянуть первый usable A/B.

Практический вывод:

- v1 schema лучше делать уже, чем у Opus в текущем варианте;
- особенно это касается `tone_hint` и части routing-oriented helpers.

### 3.2. Есть внутренняя непоследовательность вокруг `scene_cues`

В основном design ответе:

- `scene_led` описан как один из основных patterns;
- `scene_cues` включены в enrichment schema;
- decision tree использует `scene_cues`.

Но в implementation order:

- `scene_cues` и `scene_led` фактически отодвинуты на later refinement.

Это нужно развести.

Нельзя одновременно:

- делать `scene_led` частью основного routing design;
- и откладывать его ключевой signal на позднюю фазу.

Нужен один из двух честных вариантов:

1. либо `scene_led` входит в medium-risk v1;
2. либо он целиком откладывается и не участвует в initial routing.

### 3.3. Часть anti-template heuristics слишком жёсткая

Особенно спорны такие правила:

- “lead не начинается со слова из title”;
- “weak_lead лечится стартом с core_angle, не с title”.

Проблема:

- для `person_led` старт с имени может быть как раз правильным;
- для artist-led / speaker-led событий proper noun из title часто и есть лучший lead anchor;
- слишком жёсткий запрет даст ложные срабатывания.

То есть это должно быть не hard ban, а soft heuristic:

- избегать механического повторения title;
- но не запрещать естественный proper-noun lead там, где он действительно уместен.

### 3.4. `value_led` сейчас потенциально переоценён относительно `scene_led`

В decision tree / conflict rules `value_led` получает приоритет над `scene_led`.

Это логично с точки зрения utilitarian reasoning, но есть риск:

- текст станет чуть более “объясняющим, почему это важно”;
- и чуть менее живым и естественным;
- особенно там, где хороший grounded scene cue уже есть.

Для задачи “сделать текст человечнее” это неочевидный tradeoff.

Практически я бы не принимал этот приоритет как истину без A/B:

- в части событий `scene_led` может давать более natural opening;
- даже если `why_go` formally available.

### 3.5. `routing_features` как LLM-output могут дублировать runtime logic

Это второй важный contract-risk.

Если LLM одновременно возвращает:

- primitive signals;
- и готовые routing booleans;

то появляется риск drift:

- signals говорят одно;
- booleans говорят другое;
- debug становится сложнее.

Рациональнее:

- максимум routing выводить runtime-логикой из primitive fields;
- LLM-booleans оставлять только там, где они действительно дают выигрыш и проходят validation.

### 3.6. `tone_hint` пока выглядит premature

Сам Opus оставляет tone-adaptive generation в research-only.

Поэтому `tone_hint` в первой enriched schema сейчас выглядит как поле без достаточно ясного payoff.

Для v1 practical posture лучше:

- либо убрать его;
- либо оставить как nullable optional without routing dependency.

## 4. Практическая позиция после этого ответа

### 4.1. Что можно принять уже сейчас

Без большого спора можно принять:

1. pattern-driven framing как основной direction;
2. Phase 1 low-risk package почти целиком;
3. расширенные coverage flags;
4. heading stop-list;
5. shared rule blocks;
6. переход к enriched extraction как medium-risk branch;
7. `quote_led` и tone-adaptive generation как research-only.

### 4.2. Что я бы изменил перед medium-risk implementation

Для первой рабочей ветки я бы сузил enriched contract до более компактного `copy_assets v1`.

Более безопасный v1:

- `core_angle`
- `format_signal`
- `program_highlights`
- `experience_signals`
- `why_go_candidates`
- `credibility_signals`
- `voice_fragments` как optional
- `scene_cues` как optional

С осторожностью или позже:

- `routing_features`
- `tone_hint`
- `contrast_or_tension`

### 4.3. Как я бы поставил routing для первой ветки

Для first medium-risk iteration стоит держать routing проще, чем у Opus:

1. `quote_led` не активировать;
2. `program_led` только по сильному participatory gate;
3. `compact_fact_led` оставить очень консервативным;
4. `person_led`, `topic_led`, `value_led` оставить как основные;
5. `scene_led` либо включать сразу осознанно, либо честно отложить.

То есть первую рабочую ветку лучше делать не “полной красивой картой”, а controllable subset.

## 5. Нужен ли ещё один Opus-раунд

Большой раунд — скорее нет.

Ответ уже достаточно сильный, чтобы:

- перейти к локальному implementation design;
- собрать narrowed medium-risk contract;
- и работать дальше без ещё одного broad brainstorming цикла.

Но возможен **один узкий clarifying round**, если хотим, чтобы Opus сам разрешил 4 оставшихся вопроса:

1. минимальный `copy_assets v1` против слишком широкой schema;
2. судьба `scene_led` в первой medium-risk ветке;
3. смягчение overly rigid anti-template rules;
4. нужна ли `routing_features` / `tone_hint` в initial contract.

## 6. Bottom line

Если коротко:

- это сильный и полезный ответ;
- направление pattern-driven redesign принимается;
- low-risk пакет можно считать практически готовым к локальной реализации;
- medium-risk слой стоит брать в работу в более узком и жёстком варианте, чем предлагает Opus;
- следующая главная работа уже не в очередном широком запросе к Opus, а в аккуратной локальной сборке implementation subset.
