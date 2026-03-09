# Smart Update Opus Session 2 Brief

Цель: провести вторую консультационную сессию с Opus по пограничным merge-кейсам на реальном контенте (source posts + OCR + факты), с фокусом на quality-first и снижением false merge.

Контекст:
- кейсбук: `docs/reports/smart-update-duplicate-casebook.md`
- long-run benchmark: `docs/reports/smart-update-identity-longrun.md`
- разбор первой консультации: `docs/reports/smart-update-opus-consultation-review.md`
- casepack с исходным контентом: `artifacts/codex/opus_session2_casepack_latest.json` и `artifacts/codex/opus_session2_casepack_latest.md`
- expanded sample refresh: `artifacts/codex/opus_session2_sample_refresh_results_latest.json` и `artifacts/codex/opus_session2_sample_refresh_results_latest.md`

## 1. Что уже подтверждено и принимается

Из первой консультации принимаем как baseline:
- переход к 4-state модели `merge | gray_create_softlink | create | skip_non_event`;
- deterministic guards + scoring до LLM;
- pairwise LLM triage на structured payload вместо fat-shortlist final decider;
- отдельная политика для multi-event/schedule контекста;
- мониторинг качества отдельно по false merge и duplicate rate.

## 2. Что осталось спорным и требует детальной донастройки

1. `default_location` override:
- auto-подмена извлечённой venue на channel default ломает матчи и порождает критичные дубли;
- кейс: `sobakusel_default_location_conflict`.

2. Граница между `must_merge` и `must_not_merge` в серийных/программных кейсах:
- один источник, общая программа/серия, но разные дочерние события;
- кейсы: `museum_holiday_program_multi_child`, `oncologists_zelenogradsk_separate`.

3. Time semantics:
- `doors` vs `start`;
- canonical `time_correction` для того же occurrence;
- кейсы: `gromkaya_doors_vs_start`, `garage_time_correction`.

4. Broken extraction fields:
- corrupted `title/location` не должны блокировать правильный merge;
- кейс: `prazdnik_u_devchat_broken_extraction`.

5. Title framing ambiguity:
- бренд/формат vs конкретный item;
- кейсы: `shambala_cluster`, `little_women_cluster`, `makovetsky_chekhov_duplicate`.

6. Массовые recurring/schedule false-merge controls:
- один `source_url` -> много легальных событий;
- один poster hash -> много разных дат или showtimes;
- same-day double show;
- кейсы: `backstage_tour_weekly_run`, `actopus_three_day_run`, `treasure_island_double_show`, `frog_princess_double_show`, `oceania_march_lecture_series`.

## 3. Конкретные возражения к рекомендациям из Opus v1

### 3.1. Не принимаем fallback `extracted venue -> default_location`

Причина: противоречит реальному инциденту и ухудшает качество.

Фактура:
- `source_parsing/telegram/handlers.py` сейчас перетирает extracted venue на default location при mismatch (`handlers.py` около `2878..2894`);
- на кейсе `Собакусъел` это породило неверную локацию и дубль.

Рамка для Opus v2:
- `default_location` должен быть weak hint, а не overwrite policy;
- при конфликте extracted venue vs default location решение должно идти в risk-aware path (`gray`), не в silent override.

### 3.2. Не принимаем жесткий запрет LLM для multi-event

Причина: слишком жёстко, теряется полезный triage в сложных кейсах.

Рамка для Opus v2:
- нужен отдельный policy-профиль для `multi_event`, где default outcome — `gray/create`;
- но полный запрет LLM может ухудшить recall на безопасных merge-кейсах.

### 3.3. Не принимаем перенос scoring-весов “как есть”

Причина: наши данные отличаются:
- `ticket_link` часто generic;
- `poster_hash` неполный;
- много alias/noise-кейсов по venue/time/title.

Рамка для Opus v2:
- предложить не фиксированные веса “из коробки”, а калибруемую схему с guardrail-признаками и separate thresholds для different policies.

## 4. Prompt-узкие места в текущем коде (для анализа Opus)

### 4.1. `_llm_match_event` merge bias

Файл: `smart_event_update.py` (`6428..6442`).

Критичная строка:
- `Не возвращай null, если есть правдоподобный матч: лучше выбрать наиболее вероятное и снизить confidence.`

Проблема:
- толкает LLM к forced match;
- увеличивает риск скрытых ложных merge.

### 4.2. `_llm_match_or_create_bundle` binary pressure

Файл: `smart_event_update.py` (`6541..6558`).

Проблемные места:
- жёсткая бинарная маршрутизация через threshold;
- фраза “если хотя бы одно событие совпадает по якорям ... выбирай `action=match`” слишком смело для schedule/series/umbrella-кейсов.

## 5. Что нужно получить от Opus во второй сессии

1. Конкретный redesign промптов:
- как переписать `_llm_match_event` и `_llm_match_or_create_bundle` без forced-match bias;
- как ввести `uncertain`/`gray` semantics в LLM output.

2. Политику решений для пограничных кейсов:
- `must_merge` vs `must_not_merge` на однотипных источниках;
- `single_event` vs `multi_event` policy;
- `time_correction` vs separate occurrence.

3. Предложение по structured evidence payload:
- какие признаки обязательны;
- какие признаки должны быть weak/strong;
- как кодировать extraction confidence и source kind.

4. Чёткий mapping:
- LLM verdict -> runtime action (`merge|gray|create|skip`) с правилами при конфликте с deterministic blockers.

5. Мини-план внедрения prompt-изменений:
- какая последовательность безопаснее;
- какие метрики смотреть в shadow режиме;
- какие regression-tests добавить по casepack.

## 6. Пакет материалов для Opus Session 2

Обязательные материалы:
- `artifacts/codex/opus_session2_casepack_latest.json`
- `artifacts/codex/opus_session2_casepack_latest.md`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.json`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.md`
- `docs/operations/smart-update-opus-dryrun.md`
- `docs/reports/smart-update-duplicate-casebook.md`
- `docs/reports/smart-update-identity-longrun.md`
- `docs/reports/smart-update-opus-consultation-review.md`
- `docs/reports/smart-update-opus-session2-material-map.md`
- `docs/reports/smart-update-opus-session2-handoff.md`
- `smart_event_update.py` (секции `_llm_match_event`, `_llm_match_or_create_bundle`)
- `source_parsing/telegram/handlers.py` (логика `default_location` override)

## 7. Наша позиция после первой консультации

1. Мы не спорим с quality-first направлением, а уточняем его по реальным поломкам.
2. Главный принцип остаётся: false merge хуже duplicate.
3. Но для нас критично не потерять способность закрывать дубли в “грязных” данных:
- venue aliases;
- doors/start split;
- broken extraction fields;
- branding vs item title framing.
4. Поэтому ожидаем от Opus не абстрактную архитектуру, а конкретные prompt-переписывания и decision-policy для описанных casepack-кейсов.

## 8. Candidate Prompt Draft (что хотим отревьюить у Opus)

Ниже не финальная реализация, а draft-кандидаты для второй консультации.

### 8.1. `_llm_match_event` (`smart_event_update.py`, блок `6428..6443`)

Текущее проблемное место:
- forced-match инструкция `Не возвращай null, если есть правдоподобный матч...`.

Draft-намерение замены:
- убрать forced-match;
- явно разрешить `null` в неуверенных кейсах;
- зафиксировать принцип `false merge worse than duplicate`.

Черновая формулировка:
- `Если нет сильного identity-proof, возвращай match_event_id=null и confidence<=0.55.`
- `Если кейс похож на schedule/series/umbrella-vs-child, не форсируй match без явного item-level совпадения.`
- `Ошибочная склейка хуже дубля.`

### 8.2. `_llm_match_or_create_bundle` (`smart_event_update.py`, блок `6541..6558`)

Текущая проблема:
- бинарное давление `confidence >= threshold -> match`, `иначе create`;
- якорная фраза про “если хотя бы одно событие совпадает... выбирай match” слишком агрессивна в пограничных кейсах.

Draft-намерение замены:
- смягчить бинарный порог для неоднозначных матчей;
- в неуверенности в этой версии лучше предпочесть `create` (до появления runtime `gray` в этом шаге).

Черновая формулировка:
- `Если признаки противоречивы или неполные, выбирай action=create (safe fallback), а reason_short отмечай как uncertain_duplicate.`
- `Выбирай action=match только при сильном identity-proof (не менее двух независимых сильных сигналов: specific ticket / shared poster hash / exact or alias title + same occurrence anchors).`
- `Для multi-event/schedule контекста общий источник и похожий текст сами по себе не являются доказательством дубля.`

### 8.3. `default_location` disambiguation (`source_parsing/telegram/handlers.py`, `2878..2894`)

Текущее поведение:
- при mismatch extracted venue vs default location происходит silent override в default.

Draft-намерение замены:
- `default_location` использовать как weak hint;
- при явной venue в source/OCR не перетирать extracted venue;
- mismatch отправлять в risk-aware path (лог + metadata), а не в silent override.

Черновая формулировка:
- `Если извлечённая локация явно присутствует в source_text/OCR и конфликтует с default_location, сохраняй extracted как primary, default как hint.`
- `При низкой уверенности extraction помечай confidence/flag, но не делай принудительный overwrite.`
