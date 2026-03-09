# Smart Update Event Copy External Consultation Retrospective

Дата: 2026-03-08

Цель документа:

- собрать в одном месте самые полезные предложения `Opus` и `Gemini` по линии `baseline -> v2.15`;
- отделить recurring strong signals от разовых или спорных идей;
- дать готовый source of truth для `2.15.2`, чтобы не разыскивать мысли внешних моделей по десяткам документов.

Важно:

- это не пересказ всех консультаций подряд;
- это synthesis only;
- здесь зафиксировано, что из внешних советов реально worth carrying forward.

## 1. Самые устойчивые точки согласия `Opus` и `Gemini`

### 1.1. Semantic core должен оставаться LLM-first

Обе модели по разным путям пришли к одному practical limit:

- deterministic support layer полезен;
- но regex/deterministic смысловой core для event-copy не масштабируется.

Recurring accepted take:

- LLM должен делать primary смысловую трансформацию;
- deterministic слой должен заниматься hygiene, filtering, validation, caps, dedup.

### 1.2. `full-floor normalization` сильнее старого `subset extraction -> dirty merge`

Это, пожалуй, самый важный stable external signal.

Recurring take:

- нельзя выбирать "лучшее подмножество" facts и потом смешивать его с dirty floor;
- лучше нормализовать всю relevant fact base в clean publishable floor.

### 1.3. Большие negative ban-lists плохо работают на Gemma

Обе модели многократно возвращались к одному и тому же:

- когда промпт перегружен запретами, Gemma начинает исполнять их нестабильно;
- особенно это заметно на:
  - `посвящ*`
  - label-style facts
  - anti-metatext requirements.

Recurring take:

- меньше wall-of-bans;
- больше конкретных positive transformations и concrete style rules.

### 1.4. Same-model full editorial rewrite pass — хрупкий и малополезный

Recurring take:

- same-model editorial pass повторяет blind spots generation;
- он увеличивает latency;
- но не даёт proportional quality gain.

### 1.5. Patterning само по себе полезно, но не должно становиться source of semantic drift

Обе модели, пусть и по-разному, указывали:

- plain one-style generation даёт шаблонность;
- но pattern layer опасен, если он:
  - пишет prose слишком рано;
  - начинает сам выбирать смысл;
  - создаёт новые generic forms.

Recurring take:

- patterns нужны, но они должны жить в generation-layer;
- patterns должны управлять формой, а не смыслом.

## 2. Самые полезные предложения `Gemini`

### 2.1. Sparse formatting без forced headings

Recurring Gemini signal:

- sparse cases не надо автоматически дробить на секции;
- 1-2 плотных абзаца часто естественнее и профессиональнее.

Accepted:

- использовать compact branch без обязательных headings.

### 2.2. Positive transformation patterns вместо brittle lexical bans

Recurring Gemini signal:

- Gemma лучше следует примеру переписывания, чем абстрактному "не используй слово X".

Accepted with modification:

- не убирать все negative constraints;
- но brittle lexical bans заменять clearer alternatives.

### 2.3. Anti-bureaucracy и anti-metatext как отдельный слой

Recurring Gemini signal:

- `Лекция расскажет...`
- `Презентация посвящена...`
- `Мероприятие анонсирует...`

это не мелкие stylistic issues, а отдельный family of failures.

Accepted:

- это must-have prompt and validation layer.

### 2.4. Fact fragmentation как реальный bottleneck prose quality

Accepted:

- overly atomic fact layers делают generation сухой и repetitive;
- значит нужно чище собирать floor до generation.

### 2.5. Syntax-level prompt rules полезнее абстрактной persona

Сильный поздний Gemini signal:

- "культурный журналист" слишком абстрактно;
- Gemma лучше реагирует на concrete writing rules:
  - active voice
  - direct entry
  - strong verbs
  - no philosophical opener

Accepted:

- это один из strongest carries into `2.15.2`.

### 2.6. Stop-phrase / anti-filler layer

Gemini дал самый practically useful list для русского AI-prose:

- `мероприятие`
- `данное событие`
- `погрузиться в атмосферу`
- `уникальная возможность`
- `никого не оставит равнодушным`
- `будет интересно как..., так и ...`
- `не просто X, а настоящее Y`

Accepted with modification:

- как validation/prompt blacklist;
- не как deterministic rewriting engine.

## 3. Самые полезные предложения `Opus`

### 3.1. Anti-duplication как P0 text-quality issue

Recurring Opus signal:

- revise-loop и rich generation часто лечат coverage ценой повторов;
- duplication — не вторичный defect, а central blocker quality.

Accepted:

- anti-duplication нужен и в prompt, и в validation.

### 3.2. Preserve baseline strengths, not just baseline architecture

Recurring Opus signal:

- нельзя увлечься redesign и потерять:
  - эпиграф;
  - списки;
  - Telegraph readability;
  - фактическую собранность baseline.

Accepted:

- baseline strengths должны жить в новой architecture как modules or references.

### 3.3. Coverage floor and content-preservation discipline

Recurring Opus signal:

- нельзя улучшать prose ценой незаметной потери content;
- нужен хотя бы content-preservation floor.

Accepted with modification:

- не raw fact count floor;
- а preservation of publishable content.

### 3.4. Outline должен быть structural-only

Сильный поздний Opus signal:

- prose inside outline creates bureaucracy;
- outline должен работать через structure / fact ids, not mini-prose.

Accepted:

- this directly shaped the `2.15` direction.

### 3.5. Prompt simplification over prompt accretion

Recurring Opus signal:

- новые раунды не должны превращаться в accumulation of bans and rules;
- prompts должны становиться короче и чище, если хотим стабильного поведения Gemma.

Accepted:

- one of the key rules for `2.15.2`.

## 4. Предложения, которые repeated often and should become `2.15.2` defaults

### 4.1. Text-quality defaults

- no generic headings;
- no question-headings;
- no `посвящ*`;
- no `лекция расскажет / спектакль рассказывает / мероприятие будет интересно`;
- no promo filler;
- no decorative list use;
- no decorative epigraph use;
- direct subject entry in lead;
- self-contained body.

### 4.2. Prompt-design defaults

- prompts short and role-separated;
- each prompt has one main responsibility;
- positive examples > long negative enumerations;
- syntax-level rules > abstract style labels;
- optional modules must be gated by evidence.

### 4.3. Architecture defaults

- LLM-first semantic transformation;
- deterministic validation/support only;
- pattern-layer in generation, not in extraction;
- no prose-outline;
- no full editorial pass by default.

## 5. Что repeatedly оказывалось слабой идеей

### 5.1. Heavy negative prompting

Recurring failure:

- Gemma начинает игнорировать часть запретов или выполняет их уродливо.

### 5.2. Semantic regex fixes

Recurring failure:

- они быстро становятся non-scalable;
- ломают русский;
- создают новые артефакты.

## 6. Late-stage prompt-profiling carries for `2.15.x`

Этот блок добавлен уже после отдельного prompt-profiling раунда `Opus` по `2.15.2`.

Здесь не новые "большие идеи", а late-stage corrections, которые стоит считать особенно полезными для Gemma.

### 6.1. Dynamic prompt assembly > one always-fat generation prompt

Сильный новый сигнал:

- optional heading / epigraph / list rules не должны всегда сидеть в одном generation prompt;
- prompt должен собираться из core block + active blocks.

Почему это worth carrying forward:

- это снижает overload без semantic drift;
- это не делает architecture rule-heavy в плохом смысле;
- это хорошо масштабируется.

### 6.2. Stop-phrase duplication was real waste

Новый сильный carry:

- не надо одновременно держать длинную stop-phrase wall inside generation prompt и ещё раз в deterministic validation;
- lexical enforcement должен жить в validator / repair layer, а не отъедать tokens у main prose prompt.

### 6.3. Register works better than persona for Gemma

Late stable carry:

- короткий register anchor типа `раздел культуры городской газеты` полезнее абстрактной role-play persona;
- это помогает prose, но не раздувает prompt.

### 6.4. Repair without facts is too weak

Новый carry:

- narrow repair prompt должен получать исходные facts;
- иначе замены banned phrases часто уходят в другой канцелярит.

### 6.5. Planning must stay structural-only, but can be hybrid

Синтез `Opus` + our own review:

- planning должен быть tiny and structural-only;
- но полностью убирать его рано;
- лучшая форма для `2.15.x` сейчас — deterministic / tiny-hybrid planner, not full prose planner and not giant routing table.

### 6.6. Execution patterns can be fewer than editorial pattern ideas

Полезное позднее уточнение:

- для Gemma execution-layer лучше иметь меньше, но sharper structural patterns;
- при этом editorial library может быть богаче на уровне aliases / cards / narrative intent.

Это важное различие:

- не надо заставлять Gemma выбирать между десятком тонко различимых abstract patterns;
- но и не надо терять саму идею pattern-driven variability.

### 5.3. Over-abstract pattern naming

Recurring failure:

- модель превращает abstract pattern into generic formula.

### 5.4. Full rewrite repair

Recurring failure:

- вместо локальной правки происходит новый unstable draft.

### 5.5. Free-text intermediate planning

Recurring failure:

- bureaucracy enters before generation even starts.

## 6. Что именно стоит вынести в `2.15.2`

### Must carry

- `full-floor normalization`
- syntax-level generation rules
- anti-bureaucracy / anti-metatext layer
- anti-duplication layer
- semantic heading rules
- epigraph/list evidence gates
- pattern library as structural generation cards
- stop-phrase bank for Russian AI-prose

### Carry only with modification

- `why it matters`
- deterministic pattern selection
- stopword DB
- tiny optional extra calls for complex cases

### Do not carry verbatim

- regex-first semantics
- universal ban-only prompt style
- shape-heavy branching tree
- full same-model editorial pass
- prose-like outline

## 7. Bottom line

Если сжать все лучшие советы внешних моделей в один practical source of truth, то `2.15.2` должен строиться так:

- small atomic Gemma prompts;
- short contracts;
- clear division of responsibility;
- patterning only at generation;
- deterministic support only for hygiene/validation;
- text-quality rules concrete and lexical, not abstract and poetic.

Именно это — самый полезный accumulated external signal за весь цикл.
