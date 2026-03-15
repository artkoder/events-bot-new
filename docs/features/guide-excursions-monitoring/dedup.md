# Guide Excursions Dedup

Каноническая схема дедупликации для guide excursions track.

Статус: partially implemented in live digest path.

## Что сломано сейчас

Текущий merge occurrence работает только по `source_fingerprint = sha256(title_normalized | date | time)`.

Это хорошо ловит:

- повторный импорт того же поста;
- одинаковый title в одном и том же выходе;
- агрегаторный mirror, если у него совпал title.

Но схема пропускает важный класс дублей:

- один и тот же конкретный выход экскурсии оформлен разными постами внутри одного канала;
- один пост даёт master-анонс, другой даёт teaser/focus на часть маршрута;
- один пост даёт полный анонс, другой — last-call/status update;
- агрегатор публикует тот же выход под другой редактурой.

## Подтверждённые live findings

Аудит по `artifacts/test-results/guide_excursions_live_20260314.sqlite` показывает минимум один сильный false split:

- `ruin_keepers/5054`
  - title: `юго-запад области в Мамоново и к легендарному замку Бальга`
- `ruin_keepers/5055`
  - title: `Два легендарных музея в одном путешествии!`

Почему это почти наверняка один и тот же выход:

- один и тот же источник;
- одна и та же дата: `2026-03-14`;
- второй пост прямо ссылается на поездку `в рамках нашего путешествия на юго-запад области`;
- общие route anchors: `Мамоново`, `юго-запад области`, музеи в Мамоново;
- первый пост содержит master logistics, второй — focus teaser на музейную часть этой же поездки.

Есть и важный negative case:

- `vkaliningrade/4563` vs `vkaliningrade/4565`

У них совпадают дата/время и телефон записи, но это не делает их дублем автоматически. Один пост про `Хаусмарки, барельефы, арки`, другой про стрит-арт/Анастасию Туз. Значит `same booking + same time` недостаточно для merge.

## Целевой компонент

Новый dedup stage для guide track: `Route Matchmaker v1`.

Его задача:

- находить кандидаты на semantic duplicate после обычного occurrence extract;
- отличать `same concrete occurrence` от `same template but different outing` и от `distinct excursions`;
- выбирать canonical occurrence;
- переводить остальные occurrence в supporting members, чтобы они не дублировались в digest.

## Что уже реализовано

В текущем runtime `Route Matchmaker v1` уже стоит между shortlist и render digest:

- строит deterministic candidate pairs по same-date buckets;
- сначала применяет heuristic rules;
- потом best-effort вызывает Gemma pair judge;
- если LLM недоступен, остаётся на conservative no-merge fallback;
- в digest идут только canonical rows, а suppressed member ids помечаются опубликованными вместе с canonical cluster.

Уже закрытые live duplicate patterns:

- `ruin_keepers/5054` vs `5055`
- `ruin_keepers/5065` + same-day teaser/update posts
- `alev701/631` vs `636` (`schedule rollup` vs `departure update`)

## Правильная стадийность

1. `Trail Scout`
   - scan source posts, OCR, raw extraction.
2. `Route Weaver`
   - создаёт/обновляет occurrence по hard fingerprint.
3. `Route Matchmaker`
   - делает soft semantic dedup между occurrence-кандидатами.
4. `Trail Digest`
   - берёт только canonical occurrence.

Dedup не должен жить внутри `extract_title()` или одного deterministic fingerprint. Это отдельная identity stage.

## Candidate generation

LLM нельзя вызывать на все пары. Нужен deterministic prefilter.

### Bucket rules

- same source + same date;
- same source + adjacent date only if one side looks like reschedule/update;
- aggregator involved + same date;
- same booking contact + same date;
- same route anchors + same date.

### Soft signals

- token overlap по `title + summary_one_liner`;
- overlap по route anchors: районы, города, кирхи, замки, музеи, природные точки;
- same guide / organizer;
- same meeting point;
- same booking contact;
- same price/duration;
- same transport shape (`автобус`, `пешеходная`, `экопрогулка`);
- same audience fit.

### Hard anti-signals

- разные даты без явного reschedule;
- разное время при одинаковом канале и разных заголовках, если оба выглядят как самостоятельные выходы;
- разные route anchors без общей master-route структуры;
- один и тот же booking phone у целой сетки разных экскурсий.

## LLM decision taxonomy

`Route Matchmaker v1` должен возвращать только одно из решений:

- `same_occurrence`
- `same_occurrence_update`
- `same_template_other_occurrence`
- `distinct`
- `uncertain`

Дополнительно relation type:

- `master_announce_vs_focus_teaser`
- `master_announce_vs_last_call`
- `master_announce_vs_aggregator_mirror`
- `master_announce_vs_status_update`
- `shared_template_but_different_date`
- `same_contact_but_different_product`

## Merge policy

Если решение `same_occurrence` или `same_occurrence_update`:

- выбирается canonical occurrence;
- secondary occurrence не идёт в digest как отдельная карточка;
- secondary post остаётся в `guide_occurrence_source` как supporting source;
- факты из secondary source можно дозаливать в canonical occurrence;
- digest link по умолчанию ведёт на canonical source post.

### Canonical selection priority

1. non-aggregator source above aggregator;
2. post с более полными logistics above teaser;
3. post с booking link above post без booking link;
4. post с ценой/meeting point above пост без них;
5. более ранний master-анонс above поздний reminder, если facts не хуже.

Для кейса `ruin_keepers/5054 + 5055` canonical должен быть `5054`, а `5055` — supporting teaser.

## Data model recommendation

Не стоит просто переписывать `source_fingerprint`.

Нужны отдельные dedup-артефакты:

- `guide_occurrence.semantic_cluster_id`
- `guide_occurrence.canonical_occurrence_id`
- `guide_occurrence.dedup_state`
  - `canonical | merged | distinct | uncertain`
- `guide_occurrence.dedup_relation`
- `guide_occurrence.dedup_confidence`
- `guide_occurrence.dedup_decided_by`
  - `heuristic | llm | manual`

Если нужен более чистый вариант для cluster history:

- `guide_occurrence_cluster`
- `guide_occurrence_cluster_member`

## Digest rules

- digest family `new_occurrences` и `last_call` берут только `dedup_state='canonical'`;
- secondary members не показываются как отдельные карточки;
- optional future enhancement: в карточке можно дать строку `ещё 1 анонс/подробности`.

## Frozen eval cases

Минимальный набор для разработки и калибровки:

### Positive

- `ruin_keepers/5054` + `ruin_keepers/5055`
  - expected: `same_occurrence`
  - relation: `master_announce_vs_focus_teaser`

### Negative

- `vkaliningrade/4563` + `vkaliningrade/4565`
  - expected: `distinct`
  - relation: `same_contact_but_different_product`

### Future mandatory cases

- master announce + last call внутри одного канала;
- source post + aggregator mirror;
- два выхода одной типовой экскурсии в разные даты;
- reschedule pair.

## Metrics

Для dedup нельзя смотреть только на raw pair accuracy.

Нужны:

- `pair_precision` на frozen set;
- `pair_recall` на frozen positives;
- `overmerge_rate`;
- `digest_duplicate_rate`;
- `aggregator_duplicate_escape_rate`;
- `canonical_selection_accuracy`.

## Tools

Локальный аудит кандидатов:

- `python scripts/inspect/audit_guide_excursion_duplicates.py --db <path>`

Prompt contract:

- [Guide Excursions Dedup Prompts](/workspaces/events-bot-new/docs/llm/guide-excursions-dedup.md)
