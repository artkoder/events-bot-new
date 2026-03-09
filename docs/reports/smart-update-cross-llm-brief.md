# Smart Update Cross-LLM Brief

Цель этого документа: дать другой нейросети полный, но компактный контекст для внешней консультации по качеству Smart Update, дедупликации событий, Gemma prompt design и устройству LLM-слоя внутри Smart Update.

Статус контекста:
- база расследований: свежий prod-snapshot от `2026-03-06`;
- основной кейсбук: `docs/reports/smart-update-duplicate-casebook.md`;
- расширенная выборка Session 2: раздел `7` кейсбука + `artifacts/codex/opus_session2_casepack_latest.json`;
- итоговый long-run benchmark: `docs/reports/smart-update-identity-longrun.md`.
- сводная карта для Opus Session 2: `docs/reports/smart-update-opus-session2-material-map.md`.
- воспроизводимый dry-run runbook: `docs/operations/smart-update-opus-dryrun.md`.
- ready-to-send handoff для Opus Session 2: `docs/reports/smart-update-opus-session2-handoff.md`.

## 1. Что такое Smart Update

Smart Update читает новый источник события и принимает решение:
- слить его с уже существующим `event`;
- создать новый `event`;
- обогатить существующий `event` как linked/additional source;
- не создавать `event`, если это non-event контент.

Каноническая документация:
- `docs/features/smart-event-update/README.md`
- `docs/features/smart-event-update/fact-first.md`
- `docs/features/llm-gateway/README.md`
- `docs/llm/prompts.md`

Runtime entrypoints:
- `smart_event_update.py`
- `source_parsing/telegram/handlers.py`

## 2. Главная проблема

Система создаёт слишком много ошибочных дублей, но при этом риск ошибочной склейки ещё опаснее, чем дубли.

Приоритеты задачи:
- приоритет номер один: не делать скрытых ложных merge;
- дубли желательно уменьшить, но не ценой более агрессивной и опасной автосклейки;
- LLM должен оставаться обязательной частью решения;
- важнее качество и управляемый TPM, чем минимальная latency;
- можно сделать обработку одного спорного события медленнее, если итог лучше и аккуратнее.

## 3. Текущее прод-поведение

Упрощённо текущий прод-пайплайн выглядит так:

```text
Extractor / importer
  -> Smart Update candidate
  -> source_url / anchor checks
  -> shortlist by date(+city)
  -> early location/time filtering
  -> deterministic exact/related-title checks
  -> fat-shortlist LLM match_or_create
  -> binary decision:
       merge/update
       or create
  -> facts merge / rewrite / Telegraph / queues
```

Слабые места текущего подхода:
- бинарная модель `merge | create`, без нормального промежуточного состояния;
- shortlist может потерять правильный existing event ещё до LLM;
- часть ошибок вызывается слишком ранними blocker’ами:
  - location mismatch;
  - time mismatch;
  - broken title/location extraction;
  - channel default location override;
- current prompt в спорных кейсах слишком охотно выбирает `merge`;
- `multi_event` и `single_event` контекст downstream не всегда различим;
- linked-source enrichment иногда создаёт новый `event`, хотя не должен.

## 4. Целевое поведение

Цель не в том, чтобы “агрессивнее мерджить всё похожее”.

Цель: внутри Smart Update ввести quality-first identity-resolution слой, который умеет различать:
- `merge`
- `gray_create_softlink`
- `create`
- `skip_non_event`

Упрощённо целевой пайплайн:

```text
Extractor / importer
  -> Smart Update candidate + source metadata
  -> hard guards
       expected_event_id
       single-event source ownership
       source_kind propagation
       safe venue/city normalization
  -> shortlist by broad anchors
  -> identity resolver inside Smart Update
       deterministic evidence scoring
       pairwise LLM triage
       merge / gray / different / skip routing
  -> final decision:
       merge
       gray_create_softlink
       create
       skip_non_event
  -> facts merge / rewrite / Telegraph / queues
```

Принцип:
- ошибочная склейка хуже дубля;
- поэтому merge должен требовать сильного identity-proof;
- `gray` должен быть нормальным исходом, а не скрытым `create`;
- LLM должен быть не fat-shortlist decider, а pairwise quality judge на compact structured payload.

## 5. Что уже показал benchmark

См.:
- `docs/reports/smart-update-identity-longrun.md`
- `artifacts/codex/smart_update_identity_longrun_20260306_v7.json`
- `artifacts/codex/smart_update_identity_targeted_checks_20260306.json`

Итоги:
- current baseline acceptable только `20/32`;
- current prompt по сути тянет в `merge` все `32/32` кейса;
- quality-first v7 дал `32/32` acceptable на расширенном gold-наборе:
  - `20 merge`
  - `11 gray`
  - `1 different`

Ключевой residual risk после tuning:
- `2743 / 2744 / 2745` — umbrella holiday program и её child events из одного музейного поста;
- same-source multi-child schedule / holiday cases всё ещё требуют отдельного guardrail.

## 6. Ключевые кейсы, которые обязательно надо учитывать

Полный набор: `docs/reports/smart-update-duplicate-casebook.md`

Ниже condensed selection для внешнего анализа.

### 6.1. Подтверждённые дубли, которые должны стать одним event

1. Форты Кёнигсберга
- `2729 / 2732`
- `2730 / 2733`
- причина: `Форт №11` vs `Форт № 11`
- класс: venue normalization failure

2. Шамбала
- `2799 / 2843 / 2844`
- один и тот же event, но разные title framing:
  - бренд события;
  - lineup;
  - linked-source повторно создал event
- класс: title alias + linked enrichment bug

3. Собакусъел
- `2793 / 2810`
- правильная локация: `ТёркаситиХолл`
- неправильная локация `Сигнал` появилась из channel `default_location`
- класс: channel default override broke shortlist

4. Громкая связь
- `2667 / 2792`
- одно событие, но `19:30` это `doors`, а `20:00` это `start`
- плюс `Bar Sovetov` vs `Бар Sovetov`
- класс: `door_time` vs `start_time` + venue alias

5. Художницы
- `2541 / 2675 / 2779 / 2801 / 2838`
- один официальный слот Третьяковки, импортированный из ticket page, teaser posts, digest, Telegram
- класс: canonical ticket event + title aliases + schedule-derived child

6. Праздник у девчат
- `2789 / 2802 / 2803`
- broken extraction:
  - address попал в title;
  - часть description попала в location;
  - один single-event source URL породил два active event
- класс: extraction corruption + single-event owner guard failure

7. Маленькие женщины
- `2761 / 2815 / 2816 / 2817`
- movie title vs club-show title;
- один Telegram source URL создал сразу два active event
- класс: brand vs item title + source owner guard failure

8. Гараж
- `2546 / 2554`
- один canonical repertory source и одна дата, но две карточки с разным временем;
- правильное время выглядит как `18:00`
- класс: occurrence time correction

9. Сергей Маковецкий / Чехов
- `2758 / 2759`
- одинаковый source narrative, одинаковая OCR-афиша, разные title framing;
- в БД время осталось `00:00`, хотя OCR говорит `19:00`
- класс: title framing + incomplete time extraction

10. Детские онкологи, Светлогорск
- `2710 / 2721`
- это один и тот же city occurrence;
- `2712` Зеленоградск — отдельный event и merge делать нельзя
- класс: regional campaign -> city occurrence

### 6.2. Похожие кейсы, которые опасно автосклеивать

См. раздел `2` кейсбука.

Особенно важны:
- `2714 / 2835` — лекционный цикл, но разные лекции;
- `2741 / 2742` — два child event из одного repertoire post;
- `1390 / 1414` — похожий стендап-бренд, но merge опасен;
- `758 / 759` — один оркестр/серия, но разные концертные программы.

### 6.3. Multi-event / schedule посты

См. раздел `3` кейсбука.

Ключевая проблема:
- downstream child-event нередко уже не выглядит schedule-like;
- значит `source_kind` и `was_split_from_parent_post` нужно протаскивать из extractor в Smart Update явно.

### 6.4. False-positive non-event

`event 2701`
- Telegraph: `https://telegra.ph/Rozygrysh-biletov-na-match-Baltika--CSKA-03-05`
- source: `https://vk.com/wall-86702629_7354`
- это giveaway promo, а не attendable event source;
- правильный исход: `skip_non_event`, а не `create`.

### 6.5. Канонизация города

`Гурьевский городской округ` должен нормализоваться в `Гурьевск`.

Это нужно:
- для группировки month page;
- для matching;
- для единообразия canonical city в `event.city`.

## 7. Ограничения и инварианты

Это обязательные условия для предлагаемого решения.

1. LLM обязателен.
- Решение не должно превращаться в purely deterministic dedup.
- Но LLM не должен быть единственным и последним арбитром без guardrail’ов.

2. False merge хуже duplicates.
- Если неуверенность остаётся, предпочтение у `gray`, а не у `merge`.

3. TPM важнее суммарной стоимости.
- Можно увеличить latency для спорных кейсов.
- Нельзя бездумно раздувать fat-shortlist prompt и выбивать лимиты.

4. `single_event` и `multi_event` должны жить по разным policy.
- Для `single_event` можно вводить owner guard и более жёсткую идемпотентность.
- Для `multi_event` это сломает легитимные child-event.

5. Linked enrichment не должен создавать новый event.
- Если есть `expected_event_id`, linked-source обязан вернуться в него или упасть в safe-fail, но не создать новый active event.

6. Broken extraction нельзя использовать как жёсткую истину.
- corrupted `title/location` должны понижаться в доверии;
- channel default location не должна слепо перетирать явно извлечённую venue.

7. Gemma живёт в жёстком TPM-режиме.
- primary runtime в Smart Update принудительно Gemma;
- prompts должны оставаться компактными и массово применимыми;
- fat-shortlist и длинные raw payload нельзя бездумно раздувать;
- проекту важнее стабильный потоковый throughput, чем “идеальный” разовый prompt на одном кейсе.

## 8. Что изменилось к текущему раунду Opus

По сравнению с первым handoff теперь добавлен не только кейсбук проблемных мартовских дублей, но и отдельная refresh-выборка из базы на recurring/schedule/false-merge controls:
- `artifacts/codex/opus_session2_casepack_latest.json`
- `artifacts/codex/opus_session2_casepack_latest.md`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.json`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.md`

Эта refresh-выборка нужна, чтобы Opus проектировал policy не “под Форт/Шамбалу/Собакусъел”, а под массовый поток:
- same source URL -> много legal events;
- same poster/hash -> разные даты или разные showtimes;
- same ticket link -> разные события;
- same slot + same venue -> всё ещё не identity-proof.

## 8. Что именно нужно проанализировать внешней модели

Нужен не общий обзор, а конкретная инженерная рекомендация.

Просьба проанализировать:

1. Правильную архитектуру LLM внутри Smart Update.
- Должен ли это быть pairwise triage?
- Какие deterministic шаги должны идти до LLM?
- Где должно быть состояние `gray`?

2. Prompt design для Gemma.
- Почему current prompt так склонен к merge?
- Как лучше разделить:
  - shortlist matching;
  - pairwise identity triage;
  - follow-up / correction / same-source alias / brand-item bridge.

3. Structured evidence payload.
- Какие поля обязательно давать LLM?
- Какие сигналы слишком шумные и лучше оставлять deterministic?
- Как лучше кодировать schedule risk / same-source multi-child risk / follow-up / time correction.

4. Работа с ограничениями.
- Как удержать TPM ниже текущих пиков?
- Какой prompt budget нужен для pairwise triage?
- Стоит ли делать 1-stage или 2-stage LLM policy?

5. Rollout strategy.
- shadow mode;
- logging;
- какие метрики и алерты собирать;
- как безопасно включать auto-merge по классам кейсов.

6. Residual risk classes.
- same-source multi-child holiday program;
- venue alias vs hall-level child;
- title brand vs конкретный item;
- campaign-wide post vs city-specific child;
- non-event promo with event-like date mention.

## 9. Ожидаемый формат ответа внешней модели

Желательно получить:
- чёткое сравнение `current` vs `recommended`;
- архитектурную схему;
- рекомендации по Gemma prompt structure;
- список hard guards;
- список признаков для structured payload;
- policy по `merge / gray / create / skip_non_event`;
- прогноз по TPM / latency / quality;
- rollout plan по этапам.

## 10. Copy-Paste Prompt Для Другой Нейросети

Ниже готовый prompt-бриф, который можно отдать другой модели.

```text
Ты выступаешь как внешний reviewer архитектуры Smart Update в системе событийного импорта.

Контекст:
- Smart Update решает: merge existing event / create new event / linked enrichment / skip non-event.
- В системе много ошибочных duplicates, но ложный merge считается более тяжёлой ошибкой.
- LLM обязателен, но должен работать quality-first.
- Важнее контролировать TPM, чем суммарный расход токенов.
- Можно пожертвовать latency спорных кейсов ради более высокого качества.

Канонические документы проекта:
- docs/features/smart-event-update/README.md
- docs/features/smart-event-update/fact-first.md
- docs/features/llm-gateway/README.md
- docs/llm/prompts.md
- docs/reports/smart-update-duplicate-casebook.md
- docs/reports/smart-update-identity-longrun.md

Текущее прод-поведение:
- candidate -> shortlist -> early location/time filters -> LLM match_or_create on fat shortlist -> merge or create
- состояния gray нет
- часть правильных existing events вылетает из shortlist ещё до LLM
- current prompt слишком склонен к merge

Тестируемая target idea:
- Smart Update остаётся центром пайплайна
- внутри него появляется quality-first identity resolver
- deterministic evidence scoring + pairwise LLM triage
- итоговые состояния: merge / gray_create_softlink / create / skip_non_event

Benchmark:
- current baseline acceptable only 20/32
- current prompt effectively wants to merge all 32/32
- tuned quality-first longrun on gold set: 32/32 acceptable
- residual risk remains for same-source multi-child holiday/schedule posts

Ключевые real-world кейсы:

1. Must merge:
- Форты Кёнигсберга: venue noise `№11` vs `№ 11`
- Шамбала: title alias `brand` vs `lineup`, linked-source duplicate
- Собакусъел: wrong location injected by channel default venue
- Громкая связь: `doors 19:30` vs `start 20:00`, `Bar` vs `Бар`
- Художницы: ticket page + teasers + digest + Telegram all for one official slot
- Праздник у девчат: broken extraction, same single-event source created 2 active events
- Маленькие женщины: club brand title vs film title, same TG source created 2 active events
- Гараж: canonical occurrence time correction, one source/date/title/venue but two times
- Сергей Маковецкий / Чехов: same poster and narrative, different title framing, DB time stayed 00:00
- Детские онкологи Светлогорск: general campaign post + city-specific post describe same city occurrence

2. Must NOT auto-merge:
- different lectures from one lecture series
- different theatre child events extracted from one repertoire post
- similar standup family / orchestra family / schedule child events
- same-source multi-child holiday programs

3. Must skip:
- giveaway promo accidentally imported as event (`Розыгрыш билетов на матч Балтика—ЦСКА`)

4. Additional normalization:
- `Гурьевский городской округ` should canonicalize to `Гурьевск`

Сделай глубокий engineering review и ответь:
1. Как должна быть устроена LLM-подсистема внутри Smart Update?
2. Как бы ты переработал Gemma prompts и структуру LLM-вызовов?
3. Какие hard guards должны быть deterministic и идти до LLM?
4. Какие evidence fields нужно передавать в pairwise triage payload?
5. Как безопасно обрабатывать single_event vs multi_event vs schedule vs linked_enrichment?
6. Как удержать TPM ниже текущих пиков?
7. Какой rollout plan и какие метрики нужны?
8. Какие кейсы всё ещё опасны даже после proposed quality-first design?

Формат ответа:
- Current vs Recommended
- Proposed pipeline diagram
- Hard guards
- LLM prompt architecture
- Structured payload schema
- Decision policy for merge / gray / create / skip_non_event
- TPM and latency expectations
- Rollout plan
- Residual risks
```
