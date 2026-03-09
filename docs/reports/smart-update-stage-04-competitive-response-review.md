# Smart Update Stage 04 Competitive Response Review

Дата: 2026-03-06

Контекст:
- Opus в `smart-update-stage-04-competitive-response.md` принял `run_03_preprod_candidate` с двумя оговорками и предложил ещё два narrow deterministic rules.
- Мы перепроверили эти оговорки и оба новых rules на том же casepack `35 cases / 72 pairs` и том же DB snapshot, что использовались в Stage 04.

Связанные материалы:
- `docs/reports/smart-update-stage-04-competitive-response.md`
- `docs/reports/smart-update-opus-stage-04-consensus-prep.md`
- `artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.md`
- `artifacts/codex/smart_update_stage_04_competitive_validation_latest.md`

## 1. Краткий verdict

- `run_03_preprod_candidate` остаётся accepted.
- Обе оговорки Opus выглядят корректно и не ухудшают покрытие:
  - убрать case-specific слова из `GENERIC_TITLE_TOKENS`;
  - сделать `doors_start_ticket_bridge` явным: `both tickets non-empty OR same_source_url`.
- Оба новых rules Opus проходят текущий casepack без новых ошибок:
  - `multi_event_source_blocker`
  - `cross_source_exact_match`

Комбинированный локальный кандидат после этих уточнений:

| Candidate | must_merge_resolved | must_merge_gray | must_not_merge_resolved | must_not_merge_gray | false_merges | false_differents |
| --- | --- | --- | --- | --- | --- | --- |
| `run_04b_plus_both` | `16` | `22` | `34` | `0` | `0` | `0` |

Главный вывод:
- disagreement surface с Opus сильно сузился;
- по текущему casepack у нас уже есть рабочий `Stage 04A` candidate и сильный `Stage 04B` candidate;
- residual gray больше не нужно aggressively выбивать deterministic'ом любой ценой.

## 2. Что локально подтвердилось

### 2.1. `GENERIC_TITLE_TOKENS` cleanup

Из denylist безопасно убрать:
- `английская`
- `придворная`
- `века`
- `королева`
- `фей`

Проверка на `cathedral_shared_ticket_false_friend [1979, 2278]`:
- до cleanup meaningful token overlap = `0`;
- после cleanup meaningful token overlap всё равно = `0`;
- rule `generic_ticket_false_friend` продолжает корректно давать `different`.

Значит замечание Opus про overfit здесь принято.

### 2.2. `doors_start_ticket_bridge` guard

Текущее narrow rule срабатывает только на одном pair:
- `gromkaya_doors_vs_start [2667, 2792]`

Для него:
- у обеих сторон ticket non-empty;
- обе ссылки совпадают на одном и том же `t.me/stolik_na_standup_bot`.

Поэтому добавление явного guard'а:
- `pair["ticket_same"]`
- и `(left.ticket_link and right.ticket_link) or pair["same_source_url"]`

не уменьшает текущий coverage и делает контракт правила чище.

### 2.3. `multi_event_source_blocker`

Предлагаемая логика:

```python
if source_url_owner_pair_max >= 4 and not title_exact:
    return "different"
```

Фактические срабатывания на текущем casepack:
- `museum_holiday_program_multi_child` × `3` pairwise gray cases
- `oceania_march_lecture_series` × `6` pairs, но они и так уже baseline `different`

Что это даёт:
- `must_not_merge_resolved: 31 -> 34`
- `must_not_merge_gray: 3 -> 0`
- `false_merges: 0`
- `false_differents: 0`

На текущем наборе это выглядит как hard-safe blocker для obvious multi-event sources, а не как новый merge heuristic.

### 2.4. `cross_source_exact_match`

Предлагаемая логика:

```python
if (
    title_exact
    and same_date
    and venue_match
    and not time_conflict
    and left.time
    and right.time
    and left.time == right.time
    and not same_source_url
):
    return "merge"
```

Фактические срабатывания на текущем casepack:
- `plastic_nutcracker_cross_source_duplicate [1603, 1622]` as new gray -> merge win
- ещё `7` already-correct must-merge pairs, которые и так были resolved

Что это даёт:
- `must_merge_resolved: 15 -> 16`
- `must_merge_gray: 23 -> 22`
- `false_merges: 0`
- `false_differents: 0`

Дополнительный control-check:
- все must-not-merge pairs с `title_exact=true` на текущем casepack всё равно не проходят это rule, потому что там либо `same_date=false`, либо `time_conflict=true`, либо `same_source_url=true`.

Вывод:
- на текущем casepack rule выглядит clean;
- но по risk profile оно всё же слабее `multi_event_source_blocker`, потому что опирается не на structural blocker, а на "coincidental exact match is unlikely".

## 3. Где мы теперь реально согласны с Opus

1. `run_03_preprod_candidate` valid и не требует отката назад.
2. Broad venue alias, broad title mismatch, cluster merge и relaxed same-source rescue действительно не проходят quality-first cross-check.
3. `generic_ticket_false_friend`, `same_post_exact_title`, `same_post_longrun_exact_title`, `broken_extraction_address_title`, `specific_ticket_same_slot` остаются sound.
4. `museum_holiday_program_multi_child` действительно лучше решать narrow deterministic blocker'ом, а не LLM.
5. `led_hearts`, `hudozhnitsy`, `shambala [2799,*]`, `makovetsky`, `matryoshka`, `oncologists` по-прежнему остаются LLM-first territory.

## 4. Где остаётся осторожность

1. `multi_event_source_blocker` выглядит на текущем casepack более production-safe, чем `cross_source_exact_match`.
2. `cross_source_exact_match` уже выглядит good `Stage 04B` candidate, но перед runtime auto-merge лучше ещё раз перепроверить его на fresh snapshot / shadow log.
3. Даже после `run_04b_plus_both` остаются `22` must-merge gray pairs. И это нормально: текущая цель не "убить gray", а не открывать false merge class.

## 5. Практический следующий шаг

Рациональная последовательность после нового ответа Opus:

1. `Stage 04A runtime candidate`
- `run_03_preprod_candidate`
- плюс cleanup `GENERIC_TITLE_TOKENS`
- плюс explicit guard в `doors_start_ticket_bridge`

2. `Stage 04B deterministic expansion`
- почти без спора можно нести `multi_event_source_blocker`
- `cross_source_exact_match` разумно держать как preprod/shadow candidate до свежего snapshot dry-run

3. `Residual gray strategy`
- remaining `22` must-merge gray pairs не добивать broad deterministic эвристиками
- переводить их в compact pairwise LLM triage

Практически это уже выглядит не как "спор с Opus", а как согласованный rollout ladder:
- `run_03_tightened`
- затем `+ multi_event_source_blocker`
- затем, если fresh snapshot не откроет новый риск, `+ cross_source_exact_match`
