# Smart Update Session 2 Deep Consultation Review

Дата: 2026-03-06

Цель документа:
- зафиксировать, что из последнего deep-consultation от Opus уже выглядит надёжным;
- отдельно вынести спорные места и фактические неточности;
- привязать возражения к реальным данным snapshot, а не к абстрактным рассуждениям;
- подготовить follow-up пакет для ещё одного раунда консультации.

Связанные материалы:
- исходный отчёт Opus: `docs/reports/smart-update-session2-deep-consultation.md`
- полный кейсбук: `docs/reports/smart-update-duplicate-casebook.md`
- casepack: `artifacts/codex/opus_session2_casepack_latest.json`
- unified bundle: `artifacts/codex/opus_consultation_bundle_latest.json`

## 1. Что уже можно считать сильной частью отчёта

Без серьёзного спора принимаются такие направления:

1. Убрать merge-bias из `_llm_match_event`.
2. Убрать давление на `action=match` из `_llm_match_or_create_bundle`.
3. Уйти от fat-shortlist prompt в pairwise identity triage.
4. Использовать `4-state verdict`, а не бинарный `match/create`.
5. Разделить `judge` и `decider`.
6. Убрать `default_location` как venue-override и оставить его только слабым hint.
7. Явно защищать recurring pages, same-day double shows, generic-ticket false friends и multi-event sources.

Это направление согласуется и с кейсбуком, и с уже имеющимися dry-run / longrun артефактами.

## 2. Что в отчёте Opus требует коррекции или дополнительного ответа

### 2.1. Фактическая ошибка в кейсе `Собакусъел`

В отчёте перепутан источник `default_location`.

Что реально видно в snapshot:
- у `telegram_source.username='meowafisha'` `default_location = NULL`;
- у `telegram_source.username='signalkld'` `default_location = Сигнал, Леонова 22, Калининград`.

Это важно, потому что сам вывод Opus правильный:
- venue override по `default_location` опасен;
- но фактическое объяснение в отчёте сейчас опирается на неверный канал.

Что нужно от Opus в follow-up:
- переписать этот кусок уже без фактической ошибки;
- отдельно подтвердить, что policy должна запрещать override явной venue через channel default.

### 2.2. В отчёте названы кейсы, которых нет в текущем casepack

В тексте фигурируют:
- `must_not_nutcracker_two_shows`;
- `must_not_lecture_cycle`.

В актуальном `artifacts/codex/opus_session2_casepack_latest.json` таких ключей нет.

Проблема не в названии как таковом, а в том, что follow-up решение должно ссылаться на реальный пакет, который у нас потом станет regression baseline.

Что нужно от Opus:
- дать исправленный список case keys строго по актуальному casepack;
- не опираться на придуманные названия, если они не совпадают с пакетом.

### 2.3. `date_mismatch -> different` требует явного порядка применения

Само правило в целом здравое, но его нельзя формулировать без оговорок.

Контрпример:
- `led_hearts_same_post_triple_duplicate` из кейсбука;
- один Telegram post `https://t.me/signalkld/9906` породил три active event;
- у одного из них дата ошибочно расползлась на `2026-03-08`.

Если правило `date_mismatch -> different` поставить раньше:
- same-source extraction rescue;
- `expected_event_id`;
- single-event source-owner guard,

то система закрепит ошибочно созданный дубль вместо исправления bad extraction.

Что нужно от Opus:
- чётко расписать ordering rules;
- отдельно сказать, когда hard blocker работает безусловно, а когда после single-event rescue.

### 2.4. `same_source_url + same_date` нельзя расширять за пределы `single_event`

В отчёте этот сигнал местами звучит слишком сильно.

Проблема:
- same-source guard безопасен только в узком контуре `single_event`;
- как только его вынести шире, он ломает:
  - legal same-day double shows;
  - recurring repertory pages;
  - часть multi-event schedule posts.

Контрольные кейсы:
- `treasure_island_double_show`
- `frog_princess_double_show`
- `nutcracker_two_shows_same_post`
- `backstage_tour_weekly_run`
- `dramteatr_number13_recurring`

Что нужно от Opus:
- явно ограничить область действия source-owner guard;
- отдельно сформулировать, как runtime должен отличать `single_event` от `multi_event/recurring`.

### 2.5. `generic ticket` нельзя сводить к `owner_count > 5`

В отчёте heuristic для generic ticket пока недостаточно сильный.

Почему:
- кейс `cathedral_shared_ticket_false_friend` уже показывает проблему;
- у него owner count мал, но ссылка всё равно generic, потому что это scheme-level routing страницы площадки, а не event-specific identity.

Значит для `ticket_is_generic` нужны не только counts, но и:
- доменные паттерны площадки;
- path-level heuristics;
- признак отсутствия event-specific slug / stable item id;
- возможно allowlist/denylist по известным билетным маршрутам.

Что нужно от Opus:
- дать более точную, прикладную спецификацию `ticket_is_generic`;
- не ограничиваться только owner count.

### 2.6. `multi_event uncertain -> create` пока недостаточно убедительно

Для quality-first системы это спорное место.

Риск:
- голый `create` безопасен против false merge;
- но он снова увеличивает вероятность лишних дублей на digest/update-потоке.

В ряде кейсов нам ближе промежуточный маршрут:
- `gray_create_softlink`;
- либо `gray_no_automerge_but_keep_linkage`.

Что нужно от Opus:
- объяснить, почему именно `create`, а не более мягкий `gray` маршрут;
- отдельно разделить `multi_event parent`, `multi_event child`, `recurring_page`, `schedule aggregate`.

### 2.7. Числовые thresholds пока выглядят как гипотеза, а не калибровка

Пороговые значения из отчёта полезны как рабочий draft, но не как готовая policy.

Проблема:
- часть самых рискованных кейсов ломается не из-за “маленькой ошибки в балле”, а из-за неверной семантики признаков;
- сначала нужно стабилизировать blockers и scope правил;
- только потом закреплять `score >= ...` как runtime threshold.

Что нужно от Opus:
- явно отделить provisional thresholds от production-ready thresholds;
- сказать, какой минимум cross-check нужен перед включением auto-merge по score.

## 3. Небольшое расширение окна кейсов

После deep-consultation в пакет добавлены ещё 4 реальных кейса из snapshot.

| Case key | Ожидание | Текущий deterministic preview | Зачем нужен |
| --- | --- | --- | --- |
| `matryoshka_exhibition_duplicate` | `must_merge` | `gray` | merge без `same_source_url`, чтобы система умела склеивать одну и ту же выставку по смыслу и периоду |
| `museum_overlap_exhibitions_same_period` | `must_not_merge` | `gray` | контроль против склейки разных выставок в одном музее при одинаковом периоде |
| `womanhood_exhibition_time_noise_duplicate` | `must_merge` | `different` | проверка, что time blocker нельзя бездумно применять к long-running exhibition с перечислением экскурсионных слотов |
| `nutcracker_two_shows_same_post` | `must_not_merge` | `different` | same-source control: одинаковый post и title не должны склеивать легальные `14:00` и `19:00` |

Это расширение сделано специально не ради “ещё больше кейсов”, а ради проверки трёх спорных тем:
- scope same-source guard;
- качество time blockers;
- работа policy на long-running exhibition vs double-show.

## 4. Что уже ясно после этого этапа

1. Архитектурный вектор Opus в целом верный.
2. Но policy ещё нельзя считать финальной спецификацией к внедрению.
3. Самые опасные места сейчас:
- ordering blockers vs rescue;
- область действия same-source правил;
- generic ticket heuristics;
- трактовка `multi_event uncertain`;
- premature score thresholds.

## 5. Что нужно спросить у Opus в follow-up

1. В каком точном порядке должны применяться:
- `expected_event_id`;
- same-source single-event rescue;
- hard blockers `date/time/venue`;
- LLM verdict.

2. Как именно ограничить `same_source_url` / source-owner guard, чтобы он:
- помогал в `led_hearts`;
- но не ломал `nutcracker_two_shows_same_post`, `treasure_island_double_show`, `frog_princess_double_show`.

3. Как должна выглядеть production-ready спецификация `ticket_is_generic` для:
- Cathedral-style scheme URLs;
- recurring repertory pages;
- venue-root links.

4. Нужно ли `multi_event` делить минимум на:
- `parent/digest`;
- `child from digest`;
- `recurring page`;
- `schedule aggregate`.

5. Какие из новых 4 кейсов Opus считает:
- true merge;
- true different;
- legitimate gray;
- и почему.

6. Какие thresholds он считает provisional, а какие реально готов предлагать в rollout.

## 6. Ожидаемый результат follow-up раунда

От следующего ответа Opus нужен не новый общий обзор, а конкретные исправления к его же отчёту:

1. Исправленные фактические места.
2. Явная ordering policy.
3. Более узко определённый source-owner guard.
4. Улучшенная спецификация `ticket_is_generic`.
5. Позиция по новым 4 кейсам.
6. Чёткое разделение:
- что уже production-ready;
- что ещё остаётся гипотезой и требует дополнительной калибровки.
