# Smart Update Opus Stage 04 Consensus Prep

Дата: 2026-03-06

Stage 04 = не новый большой redesign, а узкий локальный этап после Stage 03:
- мы прогнали три локальных consensus dry-run;
- проверили несколько deterministic гипотез поверх Stage 03 baseline;
- оставили только те формулировки, которые не открывают `false merge` и `false different` на текущем casepack;
- подготовили narrowed dispute set для следующего раунда с Opus.

Связанные артефакты:
- `artifacts/codex/smart_update_stage_03_production_ready_dryrun_latest.json`
- `artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.json`
- `artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.md`
- `artifacts/codex/opus_stage_04_prompt_latest.md`

## 1. Что именно было проверено

Мы сознательно не ставили цель убрать `gray` полностью.

Цель Stage 04:
- найти узкий deterministic пакет, который и локально выглядит safe, и который можно защищать перед Opus уже не словами, а dry-run фактами;
- убрать самые жёсткие разногласия по broad rules;
- сократить residual gray только там, где доказательства уже достаточно сильны.

Проверялись три последовательных bundles:

| Bundle | Идея |
| --- | --- |
| `run_01_minimal_consensus` | generic-ticket false-friend blocker + exact same-post exact-title rescue + long-running same-post time-noise rescue |
| `run_02_consensus_plus_extraction` | bundle 01 + broken extraction rescue + specific-ticket same-slot merge при shared payload |
| `run_03_preprod_candidate` | bundle 02 + узкий doors-vs-start bridge |

## 2. Итог трёх прогонов

Базовый Stage 03 baseline:
- `must_merge_resolved = 9`
- `must_merge_gray = 29`
- `must_not_merge_resolved = 30`
- `must_not_merge_gray = 4`
- `false_merges = 0`
- `false_differents = 0`

Результаты Stage 04 consensus dry-run:

| Run | must_merge_resolved | must_merge_gray | must_not_merge_resolved | must_not_merge_gray | false_merges | false_differents |
| --- | --- | --- | --- | --- | --- | --- |
| `run_01_minimal_consensus` | `11` | `27` | `31` | `3` | `0` | `0` |
| `run_02_consensus_plus_extraction` | `14` | `24` | `31` | `3` | `0` | `0` |
| `run_03_preprod_candidate` | `15` | `23` | `31` | `3` | `0` | `0` |

Главный вывод:
- мы нашли узкий local preprod candidate;
- он даёт `+6` must-merge resolution и `+1` must-not-merge resolution против Stage 03 baseline;
- при этом на текущем casepack всё ещё держит `0 false merges` и `0 false differents`.

## 3. Что уже выглядит как consensus-safe

Ниже правила, где disagreement surface заметно сузился.

### 3.1. Strong-safe уже сейчас

1. `same_post_exact_title`
- Разрешает exact same-post duplicate без time conflict.
- Локально закрыл `shambala_cluster [2843, 2844]`.

2. `same_post_longrun_exact_title`
- Разрешает exact-title duplicate из одного long-running source при time-noise.
- Локально закрыл `womanhood_exhibition_time_noise_duplicate [2755, 2756]`.

3. `generic_ticket_false_friend`
- Но только в narrowed виде:
- same slot;
- `ticket_same`;
- обе ссылки generic;
- `title_related = false`;
- `0` overlap по значимым title tokens.
- В таком виде rule больше не ломает `hudozhnitsy`, но закрывает `cathedral_shared_ticket_false_friend [1979, 2278]`.

### 3.2. Preprod-candidate после ещё одного runtime dry-run

1. `broken_extraction_address_title`
- same source;
- same date;
- same payload;
- один title выглядит как address-like corruption.
- Локально закрыл `prazdnik_u_devchat_broken_extraction [2802, 2803]`.

2. `specific_ticket_same_slot`
- same slot;
- same non-generic specific ticket;
- pair ещё и делит payload (`same_source_url` или `text_same/text_containment`).
- Локально закрыл:
- `little_women_cluster [2815, 2816]`
- `little_women_cluster [2815, 2817]`

3. `doors_start_ticket_bridge`
- very narrow bridge для pairs вида doors/start:
- same date;
- `door_vs_start_pair = true`;
- `ticket_same = true`;
- `title_related = true`;
- `venue_noise_rescuable = true`.
- Локально закрыл `gromkaya_doors_vs_start [2667, 2792]`.

## 4. Какие broad идеи Stage 04 НЕ подтвердили

Это важнее, чем список удачных rules.

1. Broad `title_mismatch -> different` не подтвердился.
- В первой формулировке он немедленно дал ложный `different` на `hudozhnitsy [2779, 2801]`.
- Значит rule допустим только в узком generic-ticket false-friend контуре.

2. Broad venue-alias deterministic merge не подтвердился как отдельная safe корзина.
- `sobakusel`, `prazdnik`, `oncologists`, часть `shambala` и `little_women` внешне похожи на venue-noise, но фактически это смесь:
- default-location conflict;
- broken extraction;
- brand-vs-item framing;
- city/venue granularity.
- Значит “venue_alias_table solves 9 pairs” — это слишком оптимистично.

3. Relaxed same-source rescue по Opus-логике пока не подтверждён.
- `led_hearts` всё ещё не имеет доказательно safe deterministic rescue без extraction-bug signature.
- broad rule тут слишком близко подходит к same-post double-show risk.

4. Cluster-aware merge shortcut сейчас не проходит как preprod-safe.
- `hudozhnitsy` по-прежнему остаётся LLM/dispute territory;
- часть пар внутри кластера даже сейчас не держит `title_related=true`.

## 5. Что именно удалось убрать из критических разногласий

По состоянию на конец Stage 04 уже можно предметно договориться с Opus о следующем:

1. Не нужен общий спор “надо ли aggressively снимать gray”.
- Нет, не надо.
- Нужен узкий deterministic пакет + LLM на residual gray.

2. Generic ticket false-friend можно формализовать уже не абстрактно, а в safe narrowed rule.

3. Same-post rescue нужно обсуждать только в very narrow subclasses:
- exact-title exact-post duplicate;
- long-running same-post time-noise duplicate;
- explicit broken extraction corruption.

4. Broad venue alias / broad source rescue / cluster merge пока не проходят как доказательно safe.

Это и есть главное сужение disagreement surface перед новым Opus-раундом.

## 6. Residual gray после `run_03_preprod_candidate`

После strongest local candidate всё ещё остаются:

### 6.1. Must-merge gray

- `shambala_cluster [2799, 2843]`
- `shambala_cluster [2799, 2844]`
- `sobakusel_default_location_conflict [2793, 2810]`
- весь residual `hudozhnitsy_5way_cluster` кроме уже не тронутых pairwise-safe wins
- `prazdnik_u_devchat_broken_extraction [2789, 2802]`
- `prazdnik_u_devchat_broken_extraction [2789, 2803]`
- `little_women_cluster [2761, 2815]`
- `makovetsky_chekhov_duplicate [2758, 2759]`
- `oncologists_svetlogorsk_duplicate [2710, 2721]`
- весь `led_hearts_same_post_triple_duplicate`
- `matryoshka_exhibition_duplicate [2725, 2726]`
- `plastic_nutcracker_cross_source_duplicate [1603, 1622]`

### 6.2. Must-not-merge gray

- `museum_holiday_program_multi_child [2743, 2744]`
- `museum_holiday_program_multi_child [2743, 2745]`
- `museum_holiday_program_multi_child [2744, 2745]`

## 7. Что теперь рационально спросить у Opus

Новый Opus-раунд должен идти уже не по старой логике “предложи Stage 04 вообще”, а по более узкому контракту:

1. Считает ли Opus acceptable `run_03_preprod_candidate` как Stage 04A preprod subset?
2. Если нет, то какое именно правило из этого пакета он считает рискованным и на каком конкретном case class?
3. Какие residual gray пары он считает:
- `LLM-only`;
- `keep-gray`;
- `needs more local deterministic evidence`.
4. Какие из оставшихся споров он готов защищать именно против наших control cases:
- `museum_holiday_program_multi_child`
- `led_hearts_same_post_triple_duplicate`
- `hudozhnitsy_5way_cluster`
- `sobakusel_default_location_conflict`

## 8. Практический вывод

Stage 04 уже дал meaningful preprod result:
- есть reproducible script;
- есть три проверенных прогона;
- есть narrowed rule pack;
- есть список broad идей, которые не выдержали local cross-check.

Следующий Opus-раунд теперь должен обсуждать не “архитектуру вообще”, а:
- endorse / reject `run_03_preprod_candidate`;
- residual gray strategy для оставшихся 26 pairwise случаев.
