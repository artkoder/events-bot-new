# Smart Update Opus Session 2 Material Map

Цель: дать для второй консультации с Opus один компактный документ со всем спорным контекстом:
- что именно оспариваем/принимаем из Opus v1;
- какие ограничения есть у текущего Gemma-runtime;
- на какие реальные кейсы и исходные посты нужно опираться;
- какой итоговый результат мы ожидаем от консультации.

Базовые документы:
- `docs/reports/smart-update-opus-session2-brief.md`
- `docs/reports/smart-update-opus-consultation-review.md`
- `docs/reports/smart-update-opus-session2-handoff.md`
- `docs/operations/smart-update-opus-dryrun.md`
- `artifacts/codex/opus_session2_casepack_latest.md`
- `artifacts/codex/opus_session2_casepack_latest.json`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.md`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.json`

Техническая упаковка:
- `scripts/inspect/prepare_opus_consultation_bundle.py` (bundle из casepack + snapshot + telegram_results)

## 1. Матрица рекомендаций Opus v1

| Рекомендация из Opus v1 | Статус | Почему | Доказательства на кейсах |
|---|---|---|---|
| 4-state модель `merge/gray/create/skip` | Принимаем | Убирает бинарную перегрузку `match/create`, снижает и false-merge, и шумные дубли | `shambala_cluster`, `museum_holiday_program_multi_child`, `giveaway_non_event` |
| Убрать merge-bias из prompt | Принимаем | Текущая формулировка форсит match даже при низкой уверенности | `smart_event_update.py` `_llm_match_event` |
| Pairwise evidence judge вместо “fat shortlist decider” | Принимаем | Повышает управляемость в пограничных кейсах | `gromkaya_doors_vs_start`, `garage_time_correction`, `makovetsky_chekhov_duplicate` |
| При конфликте venue брать `default_location` канала | Отклоняем | Уже приводило к критичной ошибке локации и дублю | `sobakusel_default_location_conflict` |
| Полный запрет LLM-match для `multi_event` | Частично принимаем | Нужен отдельный policy-профиль, но полный disable убирает полезный triage | `museum_holiday_program_multi_child`, `oncologists_zelenogradsk_separate` |
| Перенос фиксированных весов scoring “как есть” | Частично принимаем | Весовые коэффициенты нужно калибровать под наши данные и шум | `fort_*`, `prazdnik_u_devchat_broken_extraction`, `little_women_cluster` |

## 2. Gemma Runtime Constraints (важно для prompt-профилирования)

Ниже факты текущего runtime, которые Opus должен учитывать в рекомендациях.

1. Smart Update принудительно использует Gemma как primary.
- `SMART_UPDATE_LLM` форсится в `gemma`.
- `SMART_UPDATE_MODEL` форсится в Gemma-семейство (по умолчанию `gemma-3-27b-it`).
- 4o используется только fallback при ошибках Gemma.

2. JSON-output обязателен по schema.
- Для `_llm_match_event`: `match_event_id`, `confidence`, `reason_short`.
- Для `_llm_match_or_create_bundle`: `action`, `match_event_id`, `confidence`, `reason_short`, `bundle`.
- Нестабильный JSON у Gemma компенсируется внутренним “fix JSON” проходом.

3. Ретраи и rate-limit поведение уже есть в runtime.
- `SMART_UPDATE_GEMMA_RETRIES` (default 3).
- Экспоненциальная пауза.
- Отдельный wait-budget при 429, чтобы не сжигать fallback без необходимости.

4. Текущий routing в match/create зависит от threshold.
- Обычно `threshold=0.6`.
- Для `allow_parallel && shortlist>1` используется `0.85`.
- Это усиливает значимость prompt-инструкций вокруг `confidence`.

5. Текущее узкое место промптов.
- `_llm_match_event`: есть forced-match инструкция “не возвращай null…”.
- `_llm_match_or_create_bundle`: есть бинарное давление “confidence>=threshold -> match”.

6. У Gemma здесь реально жёсткий throughput-budget.
- локальный limiter по умолчанию держит `tpm=12000`, `rpm=20`, `rpd=5000`;
- reserve делается заранее по estimate `input + output + reserve_extra`;
- значит даже “просто чуть-чуть увеличить prompt” на массовых постах быстро упирается в TPM раньше, чем в суммарную стоимость.

7. Из этого следует жёсткое требование к дизайну решения.
- нельзя делать большой single-call shortlist на много кандидатов;
- нельзя подсовывать LLM весь raw description/ocr без budget policy;
- нужно проектировать компактный pairwise payload и вызывать его только на действительно спорных случаях.

## 3. Реальные кейсы и указатели на исходные посты

Полный материал: `artifacts/codex/opus_session2_casepack_latest.md`.
Ниже минимальный набор, который обязательно включать в консультацию.

| Case key | Ожидаемый исход | Telegraph | Источники постов |
|---|---|---|---|
| `fort_excursion_duplicate` | must_merge | `https://telegra.ph/EHkskursiya-v-forty-Kyonigsberga-03-05` + `...-03-05-2` | `https://vk.com/wall-78248807_5900`, `https://vk.com/wall-220261025_393` |
| `fort_night_duplicate` | must_merge | `https://telegra.ph/Nochnaya-ehkskursiya-v-Fort-11-Dyonhoff-03-05` + `...-03-05-2` | `https://vk.com/wall-78248807_5900`, `https://vk.com/wall-220261025_393` |
| `shambala_cluster` | must_merge | `https://telegra.ph/Vlada-Klepcova-Vika-Kozlova-Sasha-Dosaeva-Ulyana-Bartkus-i-Oleg-Tarasov-03-06`, `https://telegra.ph/SHambala-03-06`, `https://telegra.ph/SHambala-03-06-2` | `https://t.me/mesto_sily_bar/1678`, `https://t.me/meowafisha/6841` |
| `sobakusel_default_location_conflict` | must_merge | `https://telegra.ph/Sobakusel-03-06`, `https://telegra.ph/Sobakusel-prezentaciya-novogo-proekta-03-06` | `https://t.me/meowafisha/6817`, `https://t.me/terkatalk/4513`, `https://t.me/terkatalk/4527`, `https://t.me/signalkld/9891` |
| `gromkaya_doors_vs_start` | must_merge | `https://telegra.ph/Gromkaya-svyaz-komedijnoe-shou-03-04`, `https://telegra.ph/Gromkaya-svyaz-tehnicheskaya-vecherinka-ot-LOCO-Stand-Up-Club-03-06` | `https://vk.com/wall-214027639_10783`, `https://t.me/locostandup/3171` |
| `prazdnik_u_devchat_broken_extraction` | must_merge | см. casepack | см. casepack (Telegram/VK источники по cluster) |
| `little_women_cluster` | must_merge | см. casepack | см. casepack |
| `garage_time_correction` | must_merge (time correction) | см. casepack | см. casepack |
| `makovetsky_chekhov_duplicate` | must_merge | `https://telegra.ph/Sergej-Makoveckij-Skripka-Rotshilda-i-rasskazy-CHehova-03-05`, `https://telegra.ph/Literaturno-muzykalnyj-vecher-Sergeya-Makoveckogo-Skripka-Rotshilda-03-05` | см. casepack |
| `oncologists_svetlogorsk_duplicate` | must_merge | `https://telegra.ph/Besplatnye-konsultacii-detskih-onkologov-03-05`, `https://telegra.ph/Besplatnyj-priyom-detskogo-onkologa-03-05` | `https://vk.com/wall-30777579_14694`, `https://vk.com/wall-211997788_2805`, `https://vk.com/wall-151577515_24685` |
| `oncologists_zelenogradsk_separate` | must_not_merge | `https://telegra.ph/Besplatnye-konsultacii-detskih-onkologov-03-05`, `https://telegra.ph/Besplatnye-konsultacii-detskih-onkologov-v-Zelenogradske-03-05` | `https://vk.com/wall-30777579_14694`, `https://vk.com/wall-211997788_2805` |
| `museum_holiday_program_multi_child` | must_not_merge | `https://telegra.ph/8-Marta-v-Muzee-izobrazitelnyh-iskusstv-03-05`, `https://telegra.ph/Akciya-Vam-lyubimye-v-Muzee-izobrazitelnyh-iskusstv-03-05`, `https://telegra.ph/Besplatnaya-ehkskursiya-v-Muzej-izobrazitelnyh-iskusstv-03-05` | `https://vk.com/wall-9118984_23596` |
| `sisters_followup_post` | must_merge | `https://telegra.ph/Syostry-03-04-2`, `https://telegra.ph/Sestry-03-04-3` | `https://vk.com/wall-194968_17306`, `https://vk.com/wall-194968_17338` |
| `giveaway_non_event` | must_skip | `https://telegra.ph/Rozygrysh-biletov-na-match-Baltika--CSKA-03-05` | `https://vk.com/wall-86702629_7354` |
| `city_alias_guryevsk` | normalize_city_only | `https://telegra.ph/EHkskursiya-po-fortam-Kyonigsberga-SHtajn-i-Dyonhoff-03-05`, `https://telegra.ph/Lekciya-po-istorii-kraya-v-zamke-Nojhauzen-03-05` | `https://vk.com/wall-78248807_5900`, `https://vk.com/wall-220261025_393`, `https://vk.com/wall-222073295_7494`, `https://t.me/castleneuhausen/3113` |

Дополнительно для Session 2 sample refresh обязательно учитывать ещё `10` кейсов из БД:
- `led_hearts_same_post_triple_duplicate`
- `zoikina_missing_time_duplicate`
- `oceania_march_lecture_series`
- `backstage_tour_weekly_run`
- `actopus_three_day_run`
- `treasure_island_double_show`
- `frog_princess_double_show`
- `dramteatr_number13_recurring`
- `cathedral_shared_ticket_false_friend`
- `dramteatr_same_slot_cross_title`

Сводка текущего preview по ним лежит в:
- `artifacts/codex/opus_session2_sample_refresh_results_latest.md`
- `artifacts/codex/opus_session2_sample_refresh_results_latest.json`

## 4. Что ожидаем от Opus Session 2 (конкретный deliverable)

1. Точный rewrite текстов для:
- `_llm_match_event`;
- `_llm_match_or_create_bundle`.

2. Decision policy:
- mapping `verdict -> merge|gray|create|skip`;
- где deterministic блокеры сильнее LLM;
- отдельные правила `single_event` и `multi_event`.

3. Structured schema evidence:
- признаки strong/weak;
- как кодировать `doors vs start`, `time_correction`, venue alias, extraction confidence.

4. Разбор по кейсам:
- почему конкретно `must_merge/must_not_merge/must_skip`.

5. План rollout:
- shadow-метрики;
- пороги;
- регрессионные тесты по casepack.

## 5. Наша позиция после Opus v1

1. Направление quality-first подтверждено и остаётся основой.
2. Главный инвариант сохраняется: false merge хуже дубля.
3. Но часть рекомендаций Opus v1 надо адаптировать к реальным поломкам:
- `default_location` только weak hint;
- multi-event не через тотальный запрет LLM, а через отдельный осторожный policy;
- scoring-пороги и веса только после калибровки на нашем casepack и long-run benchmark.
