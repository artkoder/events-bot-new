# Smart Update Opus Stage 03 Production Layer

Дата: 2026-03-06

Этот документ фиксирует результат локального Stage 03 прогона:
- без нового внешнего ответа;
- без LLM в decision loop;
- только на безопасном deterministic слое, который мы считаем кандидатом на `production-ready baseline`.

Связанные материалы:
- stage index: `docs/reports/smart-update-opus-stage-index.md`
- расширенный кейсбук: `docs/reports/smart-update-duplicate-casebook.md`
- расширенный casepack: `artifacts/codex/opus_session2_casepack_latest.json`
- dry-run артефакт: `artifacts/codex/smart_update_stage_03_production_ready_dryrun_latest.json`

## 1. Что такое Stage 03 layer

Это не “финальная модель merge”.

Это специально консервативный слой, который:
- умеет делать только самые безопасные `merge`;
- жёстко отсекает самые безопасные `different`;
- всё рискованное оставляет в `gray`.

Слой intentionally не пытается решить всё.
Его задача:
- доказать, что можно локально сдвинуть качество без роста hidden false merge;
- показать, какие классы кейсов действительно надо отдавать в следующий слой или в LLM.

## 2. Что вошло в safe deterministic baseline

В текущем Stage 03 dry-run мы разрешили только такие действия:

1. `date_blocker`
2. `city_blocker`
3. gated `time_blocker`
4. safe `time_correction` merge
5. safe `follow_up` merge
6. exact-title safe merge через:
- same source;
- specific ticket;
- same text
7. `venue_noise -> gray`, а не auto-merge
8. `title mismatch same slot -> gray/different` в зависимости от дополнительных сигналов

Что намеренно НЕ включалось:
- агрессивный source rescue;
- широкие score thresholds;
- новые LLM prompt changes;
- risky exceptions для long-running / multi-event beyond `gray`.

## 3. Результат dry-run

Артефакт:
- `artifacts/codex/smart_update_stage_03_production_ready_dryrun_latest.md`

Итог на текущем расширенном casepack:
- `35` кейсов;
- `72` pairwise-проверки;
- `9` must-merge pairs resolved сразу;
- `30` must-not-merge pairs resolved сразу;
- `29` must-merge pairs intentionally оставлены в `gray`;
- `4` must-not-merge pairs intentionally оставлены в `gray`;
- `0` false merges;
- `0` false differents.

Главный вывод:
- safe-layer уже полезен;
- он реально двигает систему в сторону качества;
- но он слишком консервативен, чтобы быть конечным решением сам по себе.

## 4. Что safe-layer уже уверенно закрывает

1. Same-day double shows и recurring-day multi-slot controls.
2. Exact duplicate с безопасным same-source или specific-ticket proof.
3. Date-sensitive schedule repeats.
4. Базовые city/date blockers.

Особенно показательные Stage 03 controls:
- `buratino_double_show_same_source`
- `severnoe_siyanie_same_day_triple_show`
- `zoo_free_range_schedule_repeat`
- `organ_booklet_exact_ticket_duplicate`

## 5. Что safe-layer специально оставляет на следующий слой

Сейчас в `gray` уходят именно те классы, где агрессивный deterministic merge был бы опасен:

1. `venue_noise`
- `sobakusel`
- `gromkaya`
- `prazdnik_u_devchat`
- `little_women`
- `oncologists_svetlogorsk`

2. `same-source anomaly`
- `led_hearts`
- `womanhood`

3. `semantic duplicate without strict source proof`
- `matryoshka_exhibition_duplicate`
- `plastic_nutcracker_cross_source_duplicate`
- `makovetsky_chekhov_duplicate`
- часть `hudozhnitsy`

4. `safe-not-merge but still too similar for deterministic-only`
- `museum_holiday_program_multi_child`
- `cathedral_shared_ticket_false_friend`

Это и есть основной материал для следующего Opus-раунда.

## 6. Что Stage 03 меняет в понимании задачи

До этого мы спорили в основном про архитектуру.

После локального Stage 03 dry-run видно более конкретно:
- жёсткий deterministic baseline можно сделать уже сейчас;
- его реально можно держать на нулевом false-merge и нулевом false-different уровне на текущем casepack;
- но цена за это — большой `gray` хвост.

Значит следующий рациональный вопрос уже не:
- “нужен ли pairwise LLM?”

А такой:
- “какие именно gray-классы надо поднимать следующими правилами, а какие навсегда оставить за LLM/manual review?”

## 7. Что нужно от следующего ответа Opus

Нужен уже не новый общий redesign, а более узкий и практичный ответ:

1. Какие `gray`-классы стоит пытаться понижать deterministic rules.
2. Какие `gray`-классы лучше принципиально не трогать без LLM.
3. Какой следующий минимальный слой правил можно добавить поверх Stage 03, чтобы:
- не открыть false merges;
- но уменьшить `gray` хвост.
4. Какие prompt-hints и pairwise evidence нужны именно для оставшихся `gray`-кластеров.
