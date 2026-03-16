# Guide Excursions Eval Pack

> **Status:** Frozen pre-implementation reference set  
> **Goal:** иметь измеримый эталон до начала разработки и до tuning prompt-ов, чтобы потом сравнивать `expected vs actual` на реальных постах кейсбука.

Связанные документы:

- casebook: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- LLM-first pack: `docs/backlog/features/guide-excursions-monitoring/llm-first.md`
- MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- digest spec: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- E2E: `docs/backlog/features/guide-excursions-monitoring/e2e.md`
- source artifacts:
  - `artifacts/codex/excursion_posts_2026-03-14.json`
  - `artifacts/codex/guide_channel_excursions_profitour_2026-03-14.json`

## 1. Как использовать этот pack

Eval pack нужен на трёх стадиях:

1. до реализации:
   - как frozen contract для notebook output и merge behavior;
2. во время реализации:
   - как ручной acceptance checklist;
3. после реализации:
   - как baseline для prompt tuning и regression checks.

Важно: здесь фиксируются не “идеальные тексты”, а ожидаемые domain decisions и required facts.

## 2. Evaluation Layers

### Layer A. Candidate screening

Нужно проверить:

- post вообще распознан как relevant или irrelevant;
- правильно выбран `extract_mode`;
- `on_request` не утёк в public announce;
- `status_update` не создал новый occurrence.

### Layer B. Kaggle `Tier 1` extraction

Нужно проверить:

- сколько occurrences извлечено;
- какие fields обязательны;
- какие fields допускаются как unknown;
- какие status claims выходят без server-state;
- какие fields намеренно ещё не появляются до enrichment.

### Layer C. Server bind / enrich / materialization

Нужно проверить:

- new occurrence vs status patch vs template-only;
- status claim корректно привязался к active occurrence;
- enrichment выставил `audience_fit`, `availability_mode`, `digest_eligible`;
- aggregator fallback не становится source of truth без причины;
- past occurrence rows не создаются и не висят в БД.

### Layer D. Digest rendering

Нужно проверить:

- в digest попадают только `digest_eligible=yes`;
- `summary_one_liner` остаётся grounded;
- `audience_line` появляется только когда действительно есть сигнал;
- `last_call` строится из status deltas, а не из новых announces.

## 3. Acceptance Metrics Before Coding

### 3.1. Screening targets

- `prefilter_recall >= 0.95` on frozen relevant set
- `screen_exact_class_match >= 0.85`
- `status_update_no_create_accuracy = 1.00`
- `on_request_false_public_rate = 0.00`

### 3.2. Extraction targets

- `tier1_occurrence_count_exact_match >= 0.80`
- `tier1_critical_field_recall >= 0.90` for:
  - `title`
  - `date_local`
  - `time_local`
  - `status`
- `status_claim_field_recall >= 0.85` for:
  - `title_fragment`
  - `date_hint`
  - `update_type`
  - `meeting_point`
  - `seats_text`
- `operational_field_recall >= 0.80` for:
  - `meeting_point`
  - `price_text`
  - `booking_target`
- `server_enrichment_recall >= 0.80` for:
  - `digest_eligible`
  - `availability_mode`
  - `audience_fit_tags`

### 3.3. Merge/materialization targets

- `aggregator_wrong_primary_rate = 0.00`
- `past_occurrence_persist_rate = 0.00`
- `template_only_public_digest_leak_rate = 0.00`
- `status_bind_exact_match >= 0.85`
- `reschedule_link_integrity = 1.00` for frozen reschedule set

### 3.4. Digest targets

- `digest_card_factuality = 1.00` on manual review of frozen digest set
- `digest_omission_error <= 1` critical field per 10 cards
- `media_bridge_success_rate >= 0.80` on test publishing path
- `split_render_integrity = 1.00` on over-4096 / continuation cases

## 4. Frozen Reference Cases

### 4.1. Screening and extraction set

### `GE-S01` Multi-announce schedule from original guide

- Source: `https://t.me/tanja_from_koenigsberg/3895`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
  - `extract_mode=announce`
- Expected extraction:
  - `occurrence_count=4`
  - titles:
    - `Город К. Женщины, которые вдохновляют`
    - `У Тани на районе: Закхайм и окрестности`
    - `Анатомия калининградской архитектуры: как это построено`
    - `Полесский край. Версия 3:0`
  - first three are `digest_eligible=yes`
  - `Полесский край. Версия 3:0` may be `digest_eligible=yes`, but `price_text=уточняется`
- Required facts:
  - guides extracted where explicit
  - `quantity places limited` stored as scarcity signal
  - booking contacts attached per route where explicit
- Must not:
  - collapse four routes into one occurrence

### `GE-S02` Multi-announce with meeting point and reserve-only nuance

- Source: `https://t.me/tanja_from_koenigsberg/3873`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - `occurrence_count=2`
  - `Innenstadt: жизнь в кольце` with:
    - `date_local=2026-02-14`
    - `time_local=11:00`
    - `meeting_point=площадь Победы`
    - `duration_text=около 3 часов`
    - `price_text=2000 руб`
  - `Моя королевская улица` with:
    - `date_local=2026-02-15`
    - `time_local=11:00`
    - `meeting_point=у Мировых часов на Нижнем пруду`
    - `price_text=2200 руб`
    - note `только в резерв`
- Must not:
  - drop `meeting_point`
  - rewrite `только в резерв` into fake sold-out status

### `GE-S03` Schedule plus explicit reschedule reference

- Source: `https://t.me/tanja_from_koenigsberg/3830`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - one new/public occurrence:
    - юбилейная экскурсия по улице Комсомольской/Луизеналлее
  - one status claim / template signal:
    - `Innenstadt: жизнь в кольце` moved from `2026-01-31` to `2026-02-14`
- Expected server behavior:
  - new юбилейный маршрут materialized as occurrence;
  - reschedule signal binds to existing `Innenstadt` occurrence and creates forward-linked rescheduled row if needed.
- Must not:
  - create two future occurrences for `Innenstadt`
  - lose multi-guide composition of the юбилейный маршрут

### `GE-S04` Pure status update with last-call signal

- Source: `https://t.me/gid_zelenogradsk/2705`
- Expected screen:
  - `decision=status_update`
  - `post_kind=status_update`
  - `extract_mode=status`
- Expected extraction:
  - `occurrence_count=0`
  - `status_claim_count=1`
  - claim kind includes:
    - `few_seats`
    - `meeting_point_update`
  - fields:
    - `seats_text=Есть одно освободившееся место`
    - `meeting_point=у супермаркета Спар, ул. Тургенева, 1Б`
    - `time_local=12:00`
    - `duration_text=4+ часа`
- Expected server behavior:
  - one active occurrence receives the bound delta;
  - no new occurrence is created.
- Must not:
  - create new occurrence

### `GE-S05` Mixed post: lectures + excursions + waitlist + premiere + audioquest

- Source: `https://t.me/gid_zelenogradsk/2684`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - public occurrences:
    - `Зеленоградск: прогулка с русским путешественником в 19 веке` on `2026-03-07 12:00`
    - `Расширенная экскурсия по Зеленоградску` on `2026-03-12 12:00`
    - `Расширенная экскурсия по Зеленоградску` on `2026-03-18 12:00`
  - template-only signal:
    - `Гранц - Нахимовск - Зеленоградск: как всё начиналось`
  - ignore:
    - lecture blocks
    - `КОТастрофа` audioquest as non-target
- Required facts:
  - `12 марта` and `18 марта` should carry sold-out/waitlist signal
- Must not:
  - create occurrences for lectures
  - treat audioquest as excursion occurrence

### `GE-S06` Mixed gastro + excursions + sold-out + repeat

- Source: `https://t.me/amber_fringilla/5739`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - ignore `История в Тарелке` as non-excursion block
  - public excursion occurrences:
    - `Город К. Женщины, которые вдохновляют`
    - `У Тани на районе: Закхайм и окрестности`
    - `Анатомия калининградской архитектуры: как это построено`
    - `Орнитологическая коса`
- `Орнитологическая коса`:
  - `status=sold_out` or equivalent
  - repeat signal `повтор будет 4 апреля`
- Expected merge behavior:
  - overlapping collaborative routes attach as duplicates/supporting sources when matching original guide rows;
  - `Орнитологическая коса` stays primary in `amber_fringilla`, not in collaborator channels.
- Must not:
  - create gastro-set occurrence

### `GE-S07` Pure reschedule signal from collaborator source

- Source: `https://t.me/amber_fringilla/5676`
- Expected screen:
  - `decision=status_update`
  - `post_kind=status_update`
- Expected extraction:
  - no new occurrence
  - one raw `rescheduled` claim for `Innenstadt: жизнь внутри кольца`
  - from `2026-01-31` to `2026-02-14`
- Expected server behavior:
  - claim binds to existing active/historical chain;
  - no duplicate standalone occurrence is created.
- Must not:
  - duplicate already known `Innenstadt` occurrence

### `GE-S08` Sparse schedule post with one full and two future items

- Source: `https://t.me/alev701/631`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - `occurrence_count=3`
  - `На Восток-2` with status `sold_out|full`
  - `МК по Ратсхофу`
  - `МК Марауненхоф, ч.I`
- Must not:
  - ignore the post only because it is very sparse
  - fabricate missing price or meeting point

### `GE-S09` Group-only school program without date

- Source: `excursions_profitour` post `845`
- Expected screen:
  - `decision=template_only`
  - `post_kind=on_demand_offer`
  - `extract_mode=template`
- Expected extraction:
  - `occurrence_count=0`
  - one template candidate
  - `availability_mode=on_request_private`
  - `digest_eligible_default=no`
  - `audience_fit_tags` include `school_groups` and `children`
- Must not:
  - create public digest item

### `GE-S10` Organized group fixed slot but still non-public offer

- Source: `excursions_profitour` post `847`
- Expected screen:
  - `decision=template_only` or `announce` with `digest_eligible_default=no`
- Expected extraction:
  - if occurrence is extracted, it must be:
    - `digest_eligible=no`
    - clearly marked as `group-only`
  - `audience_fit_tags` include `school_groups`
  - `availability_mode=mixed` or `on_request_private`
- Must not:
  - appear in public `new_occurrences`

### `GE-S11` Educational/proforientation operator program

- Source: `excursions_profitour` post `851`
- Expected screen:
  - `decision=template_only`
  - `post_kind=on_demand_offer`
- Expected extraction:
  - template candidate for educational/proforientation route
  - `audience_fit_tags` include `school_groups`
  - `availability_mode=on_request_private`
- Must not:
  - create public occurrence

### `GE-S12` Scheduled and digest-eligible operator excursion

- Source: `excursions_profitour` post `857`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_single`
- Expected extraction:
  - one occurrence
  - `date_local=2026-03-15`
  - `time_local=11:00`
  - `title` around ферма осетра и улиток / гастрономическое путешествие
  - `price_text=2500руб`
  - `status=planned`
  - `digest_eligible=yes`
- Required facts:
  - `meeting_point=Дом Советов`
  - price and included bundle
- Must not:
  - collapse the route into template-only just because operator source publishes many group offers

### `GE-S13` Grouped album without caption on linked message

- Source: `https://t.me/katimartihobby/1842`
- Expected screen:
  - blank message alone must not be treated as non-target if grouped album contains caption in sibling message
- Expected behavior:
  - grouped album collapse restores caption from album peer
  - resulting candidate proceeds through normal screening
- Must not:
  - mark the album as empty or ignore solely because linked `message_id` has no text

### `GE-S14` Aggregator meeting-point reminder

- Source: `vkaliningrade` post `4585`
- Expected screen:
  - `decision=status_update`
  - `post_kind=status_update`
- Expected extraction:
  - one raw `meeting_point_update` claim
  - no new occurrence
  - source quality marked as aggregator/fallback
- Expected server behavior:
  - claim can patch meeting point only if original guide row matches;
  - original guide ownership stays primary.
- Must not:
  - overwrite original guide ownership

### `GE-S15` Emoji-heavy multi-announce with mixed non-target block

- Source: `https://t.me/tanja_from_koenigsberg/3754`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - public occurrences include:
    - `Архитектурная прогулка «Всё дело в Хоппе»` on `2025-11-16 11:00`
    - экскурсия по улице `9 Апреля` on `2025-11-30 11:00`
  - keep booking contacts from text where explicit;
  - ignore `История в тарелке` dinner block as non-target for public excursion digest.
- Must not:
  - fail screening because of decorative hearts/emoji;
  - promote гастросет to excursion occurrence by default.

### `GE-S16` Fixed-date operator post with group-size pricing and private-group constraint

- Source: `excursions_profitour` post `803`
- Expected screen:
  - `decision=template_only` or `announce` with non-public outcome
  - `post_kind=on_demand_offer` or group-only `announce_single`
- Expected extraction:
  - captures:
    - `date_local=2025-11-21`
    - `time_local=13:00`
    - `price_text=1250₽/чел. при группе от 25 до 30 человек`
    - `group_only` signal from `ТОЛЬКО ДЛЯ ОРГАНИЗОВАННЫХ ГРУПП`
  - `digest_eligible=no`
- Must not:
  - leak into public `new_occurrences`;
  - lose the group-size pricing nuance.

### `GE-S17` Mixed nature schedule with partial details and later-detail placeholders

- Source: `https://t.me/amber_fringilla/5806`
- Expected screen:
  - `decision=announce`
  - `post_kind=announce_multi`
- Expected extraction:
  - concrete occurrences:
    - `Экопрогулка в Южный парк` on `2026-03-22 09:00`
    - `знакомство с историей растительного мира на острове Канта` on `2026-04-05 10:00`
    - `весенняя буковая роща` on `2026-04-16` with weekday note `(четверг)` and no invented time
  - template/detail-pending signals:
    - `Самбия: от 0 до 60 м`
    - `Весенняя Роминта`
  - price nuances preserved:
    - `500/300 руб взрослые/дети, пенсионеры`
    - `1000 руб + билеты`
- Must not:
  - invent exact time for routes where it is not given;
  - collapse detail-pending routes into fully specified occurrences.
  - lose later schedule blocks (`16 апреля`, `26 апреля`) just because the first blocks already produced valid occurrences.

### `GE-S18` Mixed-region generic travel calendar must not become excursion occurrence

- Source: `https://t.me/twometerguide/2761`
- Expected screen:
  - `decision=ignore` or `decision=template_only`
  - `post_kind=mixed_or_non_target`
- Expected extraction:
  - `occurrence_count=0`
- Rationale:
  - post is a bloom/travel calendar across `Калмыкия / Поволжье / Крым`, not a concrete regional excursion announcement;
  - high-engagement guide-project source does not override region fit or public-announcement requirements.
- Must not:
  - publish this post into guide digest;
  - materialize out-of-region generic travel content as `GuideOccurrence`.

### 4.2. Digest rendering reference set

### `GE-D01` New digest card from rich public occurrence

- Base source: `GE-S01` first occurrence (`Город К. Женщины, которые вдохновляют`)
- Expected card must include:
  - title
  - guides
  - date/time
  - one-sentence summary
  - meeting point
  - price
  - booking contact
- Summary must not repeat all logistics inside prose.

### `GE-D02` New digest card from sparse but still valid occurrence

- Base source: `GE-S08` second or third occurrence
- Expected card behavior:
  - keep title/date
  - omit unknown fields instead of hallucinating them
- Must not:
  - invent meeting point or price

### `GE-D03` Last-call digest card

- Base source: `GE-S04`
- Expected card must prioritize:
  - status delta
  - time
  - meeting point
  - booking contact
- Must not:
  - render as a brand-new excursion announcement

### `GE-D04` On-demand offer exclusion

- Base source: `GE-S09` or `GE-S10`
- Expected behavior:
  - no appearance in public digest
  - may appear only in admin review surfaces

### `GE-D05` Continuation split for long digest

- Base set: aggregated publish set from `GE-S01 + GE-S05 + GE-S17`
- Expected behavior:
  - first text message contains at most `8` cards;
  - continuation message has header `Продолжение ... (2/2)`;
  - media companion bundle is sent only once for the first part;
  - card numbering remains global and does not restart.

## 5. Expected Database Behavior

### Active/future occurrence policy

- already-past occurrences must not be materialized;
- if a post is already historical, it may still enrich template/profile facts, but not `guide_occurrence`;
- cleanup must remove occurrences after they move into the past.

### Reschedule integrity

For cases `GE-S03` and `GE-S07` expect:

- old occurrence and new occurrence remain linked through `rescheduled_from_id`;
- operator can reconstruct the chain without losing the original announcement context.

### Booking link future-proofing

For cases `GE-S01`, `GE-S02`, `GE-S04`, `GE-S12` expect:

- raw `booking_target_url` or contact preserved;
- no click rewriting yet;
- future redirect/click tracking remains a deferred extension, not an active MVP column set.

## 6. How To Compare Expected vs Actual

После появления рабочего notebook/import path у каждого frozen case сравниваем:

1. screening decision
2. occurrence count
3. critical fields
4. digest eligibility
5. new occurrence vs status patch vs template-only
6. digest output presence/absence

Результат удобно вести как таблицу:

| Case | Expected | Actual | Verdict | Notes |
|---|---|---|---|---|
| `GE-S01` | `4 occurrences` |  |  |  |
| `GE-S04` | `status only` |  |  |  |
| `GE-S09` | `template-only` |  |  |  |

Это и есть стартовый tuning ledger для prompt-ов и эвристик.
