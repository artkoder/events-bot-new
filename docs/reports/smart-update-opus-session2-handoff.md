# Smart Update Opus Session 2 Handoff

Цель: подготовить для Opus второй раунд консультации на реальных данных так, чтобы он мог:
- увидеть свои рекомендации из первого ответа без потерь;
- проверить, какие из них были учтены, а какие нет;
- поднять первичные тексты постов из подготовленного пакета;
- сам прогнать dry-run и предложить доработанный дизайн prompt/policy.

## 1. Важный reset перед Session 2

Нужно опираться не на промежуточные пересказы, а снова на исходный ответ Opus v1:
- исходная консультация: `artifacts/codex/smart-update-consultation-opus-20260306.md`
- разбор с нашей интерпретацией: `docs/reports/smart-update-opus-consultation-review.md`

Ключевой риск прошлого раунда:
- часть рекомендаций Opus v1 по prompt-дизайну была зафиксирована, но не была вынесена в достаточно жёсткий handoff для повторной консультации;
- поэтому во втором раунде нужно явно показать Opus, какие именно prompt-идеи из v1 мы хотим перепроверить на реальном контенте.

## 2. Что Opus v1 сказал именно про prompt и это нельзя потерять

Это главные рекомендации, к которым нужно вернуться.

1. Убрать forced-match bias из `_llm_match_event`.
- Проблемная строка: `Не возвращай null, если есть правдоподобный матч...`
- Opus v1 прямо указал, что это главный источник merge-bias.

2. Убрать агрессивное `action=match` давление из `_llm_match_or_create_bundle`.
- Проблемная идея: если есть похожий якорь, толкать модель к `match`.
- Opus v1 прямо отметил, что это опасно для parallel events и schedule posts.

3. Перестроить LLM в pairwise evidence judge.
- Не “найди лучший match”, а “оцени пару candidate vs existing”.
- Вернуть structured evidence по сигналам, а не один общий `confidence`.

4. Увести verdict в 4-состояния.
- `same_event`
- `likely_same`
- `different`
- `uncertain`

5. Разделить judge и decider.
- LLM оценивает evidence.
- Runtime сам маппит verdict в `merge|gray|create|skip`.

6. Жёстче вести себя на `multi_event`.
- Это рекомендация Opus v1, но у нас она partially disputed:
  - строгий full-disable LLM на `multi_event` мы не принимаем как готовое решение;
  - но Opus должен ещё раз проверить, не нужна ли более жёсткая политика именно на части multi-event кейсов.

## 3. Что из Opus v1 у нас остаётся спорным

Это нужно показать Opus явно, а не “между строк”.

1. `default_location` fallback.
- Opus v1 предложил использовать `default_location` при конфликте extraction.
- Мы это не принимаем из-за кейса `Собакусъел`.
- Во втором раунде Opus должен анализировать этот кейс уже на реальном raw text из bundle.

2. Полный запрет LLM на `multi_event`.
- Возможно слишком жёстко.
- Мы хотим, чтобы Opus посмотрел реальные кейсы `must_not_merge` и сказал:
  - где нужен полный deterministic-only режим;
  - где допустим gray через LLM;
  - где всё же возможен safe merge.

3. Веса scoring “как есть”.
- Нельзя переносить без калибровки.
- Нужен quality-first совет именно на наших данных.

## 4. Ready-to-send data package

Основные документы:
- `docs/reports/smart-update-opus-session2-brief.md`
- `docs/reports/smart-update-opus-session2-material-map.md`
- `docs/reports/smart-update-opus-consultation-review.md`
- `docs/operations/smart-update-opus-dryrun.md`

Кейсбук и первичный контекст:
- `docs/reports/smart-update-duplicate-casebook.md`
- `artifacts/codex/opus_session2_casepack_latest.json`
- `artifacts/codex/opus_session2_casepack_latest.md`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.json`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.md`

Упаковка реальных данных для Session 2:
- `artifacts/codex/opus_consultation_bundle_latest.json`
- `artifacts/codex/opus_consultation_bundle_latest.md`
- `artifacts/codex/opus_session2_prompt_ready.md`

Dry-run / benchmark артефакты:
- `artifacts/codex/quality_first_dedup_dryrun_latest.json`
- `artifacts/codex/quality_first_dedup_dryrun_latest.md`
- `artifacts/codex/gemma_match_prompt_eval_latest.json`
- `artifacts/codex/smart_update_identity_longrun_latest.json`
- `artifacts/codex/smart_update_identity_longrun_latest.md`

Исходный ответ Opus v1:
- `artifacts/codex/smart-update-consultation-opus-20260306.md`

## 5. Где в пакете лежат реальные тексты постов

Это важно проговорить Opus явно.

1. Telegram/VK тексты из snapshot БД:
- `artifacts/codex/opus_consultation_bundle_latest.json`
- секция: `db_extract.event_sources[*].source_text`

2. Текст и event payload из `telegram_results.json`:
- `artifacts/codex/opus_consultation_bundle_latest.json`
- секция: `tg_extract.matched_messages[*]`

3. Связь кейсов с event/source URL:
- `artifacts/codex/opus_session2_casepack_latest.json`
- `artifacts/codex/opus_consultation_bundle_latest.json`

## 6. Как Opus может сам перепроверить dry-run

Инструкция уже подготовлена:
- `docs/operations/smart-update-opus-dryrun.md`

Что там есть:
- SQL-команды для вытаскивания source text из БД;
- пример извлечения текста конкретного Telegram post из `telegram_results.json`;
- команда сборки unified consultation bundle;
- команда deterministic dry-run;
- команда Gemma prompt eval;
- команда longrun quality-first с LLM.

## 7. Что именно нужно получить от Opus во втором раунде

Нужен не общий обзор, а повторная глубокая консультация на реальных данных.

Ожидаемые deliverables:
- новый prompt redesign для `_llm_match_event`;
- новый prompt redesign для `_llm_match_or_create_bundle`;
- decision-policy для `merge|gray|create|skip`;
- policy для `single_event` vs `multi_event`;
- позиция по `default_location` конфликтам на основании реальных кейсов;
- позиция по `doors/start`, `time_correction`, `follow-up`, `brand vs item`, `same-source multi-child`;
- план безопасного rollout после второго раунда.

## 8. Что нужно подчеркнуть в сообщении Opus

1. Предыдущий ответ был полезен, но его prompt-рекомендации нужно теперь перепроверить на реальном контенте.
2. Сейчас у него есть полный пакет:
- raw source texts;
- casebook;
- benchmark;
- expanded sample refresh;
- runbook;
- bundle для ручного анализа.
3. Нужно, чтобы он не только предложил архитектуру, но и сам верифицировал её на приложенных кейсах и дал более жёсткие, конкретные рекомендации.

## 9. Что нового обязательно проверить во втором раунде

По итогам расширения выборки появились кейсы, которые нельзя игнорировать при проектировании policy.

1. Same-source duplicate с bad extraction.
- `led_hearts_same_post_triple_duplicate`
- Один Telegram post породил сразу три active event.
- Здесь нужен ответ Opus, как сочетать:
  - same-source owner guard;
  - rescue для bad date/time extraction;
  - и защиту от ошибочной склейки в массовом потоке.

2. Recurring / repertory controls.
- `backstage_tour_weekly_run`
- `dramteatr_number13_recurring`
- `actopus_three_day_run`
- Тут один и тот же `source_url`, title и poster могут быть нормой для разных дат.

3. Same-day double-show controls.
- `treasure_island_double_show`
- `frog_princess_double_show`
- Opus должен явно сформулировать, какие сигналы защищают от merge при legal `11:00 + 14:00` в один день.

4. Generic ticket / schedule controls.
- `oceania_march_lecture_series`
- `cathedral_shared_ticket_false_friend`
- Здесь один ticket URL и даже один source URL не должны тащить модель в merge.

5. Same-slot false-merge controls.
- `dramteatr_same_slot_cross_title`
- плюс уже существующие `1390/1414`, `758/759`, `2714/2835`
- Здесь нужен консервативный ответ: какие сигналы вообще могут разрешить merge при одном и том же слоте, а какие должны отправлять в `different` или `gray`.

## 10. Конкретные вопросы к Opus Session 2

1. Какие именно prompt-фразы в Gemma нужно добавить, чтобы она в recurring/schedule кейсах не переоценивала:
- `same poster`
- `same source_url`
- `same ticket_link`
- `same venue/date/time`

2. Как должна выглядеть compact `pairwise payload` для Gemma при жёстком TPM-лимите:
- какие поля обязательны;
- какие поля нужно clip/drop;
- какие derived hints нужно передавать заранее.

3. Какие deterministic rules можно сделать hard-safe для `single_event`, не ломая recurring pages и multi-event schedule posts?

4. Как Gemma должна объяснять verdict в терминах evidence:
- чтобы downstream runtime мог безопасно отличать `merge` от `gray_create_softlink`.

5. Какие кейсы из expanded sample он считает:
- true duplicates;
- true different;
- legitimate gray;
- и где наш текущий gold set стоит пересмотреть.
