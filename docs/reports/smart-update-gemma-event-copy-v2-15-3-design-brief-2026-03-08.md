# Smart Update Gemma Event Copy V2.15.3 Design Brief

Дата: 2026-03-08

Основание:

- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)
- [v2.15.2 dry-run review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-review-2026-03-08.md)
- [Gemini review on v2.15.2 texts](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-v2-15-2-dryrun-text-consultation-review-2026-03-08.md)
- [Gemma deep research impact on v2.15.2](/workspaces/events-bot-new/docs/reports/smart-update-gemma-deep-research-impact-on-v2-15-2-2026-03-08.md)
- [master retrospective](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-opus-gemini-recommendations-master-retrospective-2026-03-08.md)

## 0. Как читать этот brief

Этот документ не заменяет базовые правила `v2.15` и `2.15.2`, а сужает следующий implementation round.

Важно:

- `2.15.3` — это additive brief;
- все базовые non-negotiables по качеству текста и архитектуре из `v2.15` и `2.15.2` остаются в силе;
- узкие patch-goals ниже не имеют права подменять главную цель: качественный, естественный и профессиональный итоговый текст на масштабируемой архитектуре.

## 1. Непотеряемые цели, которые переходят в `2.15.3`

Ниже — не новый wishlist, а carry-forward базовых требований, которые нельзя забывать в следующих итерациях.

### 1.1. Цель по итоговому тексту

Итоговый текст должен быть одновременно:

- естественным;
- профессиональным;
- нешаблонным;
- grounded;
- полным по смыслу;
- ясным;
- уместно мотивирующим, но не рекламным;
- вариативным по подаче;
- с сильным лидом;
- с осмысленными заголовками;
- с функциональным форматированием;
- чистым от AI clichés;
- чистым от сервисного мусора;
- лёгким по ритму;
- масштабируемым на тысячи разных исходников.

### 1.2. Editorial quality важнее локальной оптимизации метрик

`missing`, `forbidden`, `plan compliance` и другие метрики нужны, но они вторичны по отношению к quality bar текста.

Это значит:

- нельзя улучшать только `missing`, если prose становится хуже;
- нельзя выигрывать только на hygiene, если текст становится мёртвым или бюрократичным;
- нельзя двигаться в узкую песочницу 2-3 показателей и терять главный user-facing результат.

### 1.3. Coverage остаётся обязательным, но не единственным критерием

Нельзя терять:

- subject / author / коллектив;
- program items;
- grouped names;
- agenda blocks;
- project specifics;
- факты, реально объясняющие, что это за событие.

Но и простое перечисление фактов не считается хорошим результатом, если prose выглядит роботизированным.

### 1.4. Pattern idea остаётся core частью решения

Мы не отказываемся от мысли, что живой editorial output должен использовать несколько крупных паттернов подачи.

Значит:

- pattern library остаётся;
- execution patterns для Gemma могут быть компактнее;
- но сама идея variативной human-like подачи не теряется.

### 1.5. Baseline wins и прошлые удачи тоже сохраняются

Нельзя потерять сильные находки прошлых раундов:

- baseline structured discipline;
- safe use of list blocks;
- safe use of epigraphs;
- сильный prose ambition из `v2.6`;
- hygiene-safe branch из `v2.13`;
- лучший coverage/hygiene профиль из `v2.15.2`.

## 2. Архитектурные инварианты, которые тоже нельзя терять

### 2.1. `LLM-first` остаётся базовой политикой

Semantic core не должен уходить в regex-first.

Детерминированный слой допустим только как:

- validation;
- sanitization;
- format enforcement;
- safe support logic.

### 2.2. Short-step architecture остаётся правильной

Сохраняется линия:

- `normalize -> tiny plan -> generate -> validate -> narrow repair`

Мы не откатываемся:

- к giant prompt;
- к prose-outline stage;
- к full editorial rewrite pass.

### 2.3. Prompts должны быть self-contained и role-separated

Для Gemma это не опционально.

Каждый шаг:

- self-contained;
- короткий;
- секционированный;
- с одной главной задачей.

### 2.4. Prompt discipline важнее усложнения логики

Если что-то не работает, сначала усиливаем:

- quote gate;
- schema semantics;
- prompt packaging;
- per-step decoding;
- validation.

А не добавляем новый heavy stage по умолчанию.

## 3. Что такое `2.15.3`

`2.15.3` — это не новый architectural reset, но и не просто узкий patch на 4-5 багов.

После новых Gemma research документов стало ясно:

- проблема не только в нескольких локальных failure cases;
- проблема ещё и в том, как упакован весь prompt stack для Gemma;
- если оставить старую prompt packaging логику почти без изменений, мы снова будем лечить симптомы поверх того же instruction-overload.

Поэтому `2.15.3` = **prompt-pack repack round внутри существующей `LLM-first` architecture**.

Что это значит practically:

- архитектурные шаги `normalize -> tiny plan -> generate -> validate -> narrow repair` сохраняются;
- но prompts всех основных шагов подлежат переупаковке под Gemma-specific discipline;
- локальные fixes вроде false quote / format leakage / project prose остаются, но уже как часть более широкого prompt repack.

Цель `2.15.3`:

- сохранить лучший за цикл `coverage + forbidden` профиль;
- убрать главные text-quality blockers;
- переупаковать весь prompt stack так, чтобы Gemma исполняла инструкции стабильнее;
- не сломать `LLM-first` architecture;
- не скатиться обратно в regex-heavy semantic core.

## 4. Текущий baseline для сравнения

По dry-run:

- baseline total missing = `22`
- `v2.13` total missing = `14`
- `v2.14` total missing = `14`
- `v2.15.2` total missing = `11`
- `v2.15.2` forbidden = `0`

Практический статус:

- `v2.15.2` лучше baseline overall как system candidate;
- `v2.15.2` лучше `v2.14` overall;
- `v2.15.2` лучше `v2.13` по hygiene/coverage, но местами слабее по prose quality;
- `v2.15.2` не production-ready.

## 5. Главные blockers, которые должен чинить `2.15.3`

### 3.1. False quote / false epigraph

Главный blocker:

- `2687` вытянул quote из соседнего digest item;
- `2673` выдал hallucinated philosophical quote;
- `2734` показал, что даже source-backed fragment ещё не гарантирует хороший epigraph.

Вывод:

- quote gating и epigraph discipline сейчас важнее, чем новые stylistic experiments.

### 3.2. Plan-to-output leakage

`2660` показал:

- plan запрещал headings;
- final output всё равно их дал.

Вывод:

- generation prompt сам по себе недостаточен;
- нужна output-side deterministic enforcement для format contract.

### 3.3. Project prose still sounds agenda-like

`2673` уже лучше factual framing-wise, но текст всё ещё местами выглядит как unpacked agenda:

- `устройство и цели платформы`
- `формат встречи`

Вывод:

- project/presentation branch всё ещё требует более human editorial execution.

## 6. Что сохраняем из `2.15.2`

- `full-floor normalization`
- `LLM-first` semantic core
- compact execution pattern set
- tiny/hybrid planner instead of prose-outline
- dynamic prompt assembly
- self-contained prompts
- per-step decoding discipline
- deterministic validation as support layer
- targeted repair as narrow optional step

## 6.1. Что именно должны поменять Gemma research документы

Gemma research не должен остаться "фоном" или парой мелких carry.

Он должен materially повлиять на весь prompt stack:

- `normalize_floor`
- `tiny planner`
- `generate_description`
- `targeted_repair`

Влияние должно быть таким:

- все prompts self-contained;
- все prompts sectioned (`ROLE / RULES / OUTPUT / FACTS`);
- у каждого prompt одна главная обязанность;
- negative rules не должны доминировать над positive transformations;
- JSON/planner prompts должны получить stronger semantic field guidance;
- generation prompt должен стать короче, sharper и более pattern-executable;
- repair prompt должен быть facts-backed и issue-specific, а не blind rewrite.

## 7. Что меняем в `2.15.3`

### 7.0. Full prompt repack for Gemma

Это главный structural change итерации.

Меняем не только local rules, но и prompt packaging всех рабочих шагов.

#### `normalize_floor`

Нужно переупаковать prompt так, чтобы он:

- был полностью self-contained;
- был sectioned;
- опирался на короткие positive transformations;
- меньше полагался на длинные negative walls;
- сильнее различал:
  - subject facts
  - program facts
  - project/presentation facts
  - quote candidates

#### `tiny planner`

Planner prompt должен быть перепакован так, чтобы:

- choices в schema были семантически объяснены;
- `pattern`, `use_epigraph`, `use_headings`, `use_list_block` получали short purpose hints;
- planner не производил prose-like decisions;
- output оставался compact structural JSON.

#### `generate_description`

Generation prompt должен быть перепакован наиболее заметно.

Он не должен быть просто "старый prompt + ещё 5 запретов".

Нужна новая упаковка:

- self-contained;
- sectioned;
- с короткими pattern execution cards;
- с короткими positive `плохо -> хорошо` примерами;
- с коротким stop-phrase bank;
- с более явным syntax-level guidance по leads, headings, epigraphs и list blocks.

#### `targeted_repair`

Repair prompt тоже должен быть перепакован:

- issue-specific;
- facts-backed;
- без blind rewriting;
- с явным приоритетом удаления dubious quote/epigraph над творческой починкой.

### 7.1. Quote gate becomes event-local and stricter

Новые принципы:

- quote candidate должен быть привязан к самому event fragment, а не к соседнему пункту дайджеста;
- titles, fragments и short labels не должны включать `quote_led`;
- для `presentation/project` cases epigraph по умолчанию запрещён, если нет очень сильной source-backed прямой цитаты;
- `quote_led` не должен активироваться только потому, что в источнике есть кавычки.

### 7.2. Anti-hallucination epigraph contract

Main generation и repair должны жёстко держать:

- эпиграф можно использовать только дословно из `quote_text`;
- нельзя расширять, дописывать или придумывать цитату;
- если `quote_text` слабый или сомнительный, лучше не делать epigraph вообще.

### 7.3. Deterministic format enforcement

После generation:

- если `use_headings=false`, строки `### ...` не проходят;
- если blockquote не стоит первым блоком, он удаляется;
- если epigraph совпадает с weak fragment / title-like artifact, он удаляется.

Это safe deterministic layer:

- он не меняет смысл;
- он не придумывает текст;
- он только защищает format contract.

### 7.4. Project prose cleanup

Для `presentation/project` cases:

- generation prompt должен толкать к нормальному presentation lead;
- agenda-like headings и bureaucratic section names надо ослабить;
- `why_it_matters` и project framing должны встраиваться естественно, а не через checklist tone.

### 7.5. Stronger semantic hints in planner schema

Для tricky planner fields:

- `pattern`
- `use_epigraph`
- `use_headings`

нужны не только type constraints, но и короткие semantic hints/examples.

## 8. Что не делать в `2.15.3`

- не делать новый redesign
- не тащить новый giant prompt
- не усиливать semantic regex rewrites
- не добавлять новый heavy editorial pass
- не превращать generation в visible step-by-step reasoning
- не откатывать pattern idea
- не сводить Gemma research к "одной-двум локальным правкам"

## 9. Prompt Repack And Patch Pack For Implementation

`2.15.3` не должен быть большим architectural rebuild, но он уже не может оставаться только маленьким patch pack.

Правильная формулировка:

- architecture stays;
- full prompt repack happens;
- deterministic support changes stay narrow.

### Prompt Repack A. `normalize_floor`

- self-contained packaging;
- section labels;
- short positive transformations;
- quote candidate discipline;
- weaker dependence on broad ban-lists.

### Prompt Repack B. `tiny planner`

- stronger semantic field hints;
- clearer pattern/format contracts;
- no prose leakage.

### Prompt Repack C. `generate_description`

- shorter prompt body;
- execution cards instead of abstract editorial philosophy;
- syntax-level lead/heading/list/epigraph rules;
- positive examples and stop-phrase bank.

### Prompt Repack D. `targeted_repair`

- issue-scoped;
- facts-backed;
- conservative on quotes and headings.

### Patch 1. Event-local quote filtering

- сузить quote extraction;
- отключать digest-leak quotes;
- titles/fragments не считать epigraph candidates.

### Patch 2. Hard epigraph contract

- generation prompt: использовать quote только дословно;
- repair prompt: если quote dubious, удалять epigraph, а не чинить его творчески.

### Patch 3. Hard format compliance

- post-generation strip for forbidden headings when `use_headings=false`;
- blockquote position lock.

### Patch 4. Project presentation execution card

- улучшить именно `project/presentation` prose guidance;
- меньше agenda sectioning;
- сильнее акцент на natural project lead.

### Patch 5. Planner schema semantics

- короткие descriptions/examples для pattern and formatting decisions.

## 10. Success bar for `2.15.3`

Минимум:

- не хуже `v2.15.2` по total missing
- forbidden = `0`
- false epigraph regressions = `0`

Quality bar:

- `2687` должен перестать быть broken quote case
- `2660` должен реально obey `use_headings=false`
- `2673` должен звучать естественнее, чем в `v2.15.2`
- итоговые тексты не должны становиться более шаблонными, чем `v2.15.2`
- quality bar оценивается не только по цифрам, но и по human reading

Editorial bar:

- не ухудшить лучший sparse-case quality;
- не вернуть явные AI clichés;
- не обменять естественность на checklist-like prose;
- не потерять variативность pattern-driven подачи.

System bar:

- сохранить `LLM-first`
- не увеличить happy-path call count
- не ввести новый brittle semantic regex core

## 11. Следующий практический шаг

После этого brief логика такая:

1. локально собрать `2.15.3` в experimental harness, включая repack всех основных Gemma prompts
2. прогнать те же 5 событий
3. сравнить с baseline / `v2.13` / `v2.14` / `v2.15.2`
4. отдельно проверить, что проблема не "сдвинулась" из prose в planner/repair
5. только после этого идти в следующий внешний consultation round

То есть сейчас правильный следующий шаг — implementation + dry-run именно с full prompt repack, а не с narrow bugfix-only round.
