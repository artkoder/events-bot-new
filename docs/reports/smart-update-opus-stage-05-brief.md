# Smart Update Opus Stage 05 Brief

Дата: 2026-03-06

Stage 05 = финальный внешний раунд перед внедрением narrowed deterministic subset и проектированием residual gray LLM layer.

Это уже не новый redesign.
Это попытка закрыть последние содержательные вопросы в доказательном формате.

Связанные материалы:
- `docs/reports/smart-update-opus-stage-index.md`
- `docs/reports/smart-update-stage-04-competitive-response.md`
- `docs/reports/smart-update-stage-04-followup-response.md`
- `docs/reports/smart-update-stage-04-competitive-response-review.md`
- `docs/reports/smart-update-duplicate-casebook.md`
- `artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.md`
- `artifacts/codex/smart_update_stage_04_competitive_validation_latest.md`

## 1. Что к этому моменту уже почти consensus

Локально подтверждён widened deterministic candidate:

| Candidate | must_merge_resolved | must_merge_gray | must_not_merge_resolved | must_not_merge_gray | false_merges | false_differents |
| --- | --- | --- | --- | --- | --- | --- |
| `run_04b_plus_both` | `16` | `22` | `34` | `0` | `0` | `0` |

Его состав:

1. `generic_ticket_false_friend`
- уже с cleanup `GENERIC_TITLE_TOKENS`

2. `same_post_exact_title`

3. `same_post_longrun_exact_title`

4. `broken_extraction_address_title`

5. `specific_ticket_same_slot`

6. `doors_start_ticket_bridge`
- уже в tightened виде:
- `ticket_same`
- и `(both tickets non-empty OR same_source_url)`

7. `multi_event_source_blocker`
- `source_url_owner_pair_max >= 4 and not title_exact -> different`

8. `cross_source_exact_match`
- exact title/date/time/venue match across different sources

По состоянию на конец Stage 04 follow-up:
- `multi_event_source_blocker` практически agreed как production-safe;
- `cross_source_exact_match` тоже почти agreed, но operational posture у нас немного осторожнее, чем у Opus;
- broad venue alias, broad title mismatch, cluster merge и relaxed same-source rescue остаются rejected.

## 2. Что ещё остаётся открытым

### 2.1. Operational posture для `cross_source_exact_match`

Содержательный спор почти исчез.
Остался вопрос rollout posture:
- сразу runtime с `alert-on-fire`;
- или сначала fresh-snapshot / short shadow pass.

### 2.2. Как именно проектировать LLM слой для residual gray

После `run_04b_plus_both` остаются `22` must-merge gray pairs:

- `shambala_cluster` × `2`
- `sobakusel_default_location_conflict` × `1`
- `hudozhnitsy_5way_cluster` × `10`
- `prazdnik_u_devchat_broken_extraction` × `2`
- `little_women_cluster` × `1`
- `makovetsky_chekhov_duplicate` × `1`
- `oncologists_svetlogorsk_duplicate` × `1`
- `led_hearts_same_post_triple_duplicate` × `3`
- `matryoshka_exhibition_duplicate` × `1`

Главный принцип не меняется:
- false merge хуже дубля;
- deterministic rules не должны лезть сюда broad эвристиками;
- LLM нужен именно для residual semantic/extraction/noise territory.

Но есть два still-open design questions:

1. baseline должен оставаться `compact pairwise`, как и раньше;
2. Opus предложил cluster-call для кластеров типа `hudozhnitsy`, но это пока не согласовано как runtime-база.

## 3. Что мы отдельно добавили в casebook перед финальным раундом

### 3.1. `museum_holiday_program_multi_child`

Добавлен как явный structural control для `multi_event_source_blocker`.

Смысл:
- один `source_post_url`;
- один payload;
- один generic ticket;
- но это три разных child event;
- здесь `same source` не доказательство merge, а ловушка.

### 3.2. `led_hearts_same_post_triple_duplicate`

Добавлен как контрпример против слишком широкой интерпретации `source_url_owner_pair_max`.

Смысл:
- `source_url_owner_pair_max = 3` здесь не multi-event page;
- это bad duplicate extraction из одного Telegram post;
- значит порог `4` сейчас выглядит безопаснее, чем `3`;
- а сам кейс лучше уводить в LLM/extraction-fix, а не лечить broad deterministic rule.

## 4. Чего мы хотим от финального Opus-раунда

Не нужен ещё один общий brainstorm.

Нужен финальный ответ по трём вещам:

1. Final sign-off по narrowed deterministic subset.
2. Final recommendation по residual gray LLM layer в рамках compact pairwise baseline.
3. Только последние реальные blockers, если они ещё остались.

Если Opus предлагает что-то сверх pairwise baseline:
- это должно идти как optional appendix;
- а не как замена базового решения под текущие runtime-ограничения.
