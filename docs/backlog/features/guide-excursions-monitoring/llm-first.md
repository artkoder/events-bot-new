# Guide Excursions LLM-First Pack

> **Status:** Canonical implementation handoff  
> **Goal:** закрыть white spots в LLM-first слое для `guide excursions`, чтобы после этого можно было переходить к разработке notebook/import/digest без повторного проектирования prompt architecture.

Связанные документы:

- feature overview: `docs/backlog/features/guide-excursions-monitoring/README.md`
- architecture: `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- digest spec: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- eval pack: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- Smart Update canon: `docs/features/smart-event-update/README.md`
- LLM-first policy: `docs/llm/request-guide.md`
- lollipop funnel canon: `docs/llm/smart-update-lollipop-funnel.md`
- live `Opus` audit artifact: `artifacts/codex/reports/guide-excursions-monitoring-opus-audit-2026-03-14.md`
- `Opus`/`Gemini` syntheses:
  - `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-editorial-layout-plan-consultation-synthesis-2026-03-10.md`
  - `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-writer-pack-compose-consultation-synthesis-2026-03-10.md`
  - `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-writer-final-4o-consultation-synthesis-2026-03-10.md`

## 1. Audit Verdict After Live `Opus` Review

Общая схема фичи была выбрана верно:

- raw Telegram scan в Kaggle;
- facts-first merge на сервере;
- digest как downstream surface;
- reuse текущего `Telegram monitoring -> Smart Update -> /digest`.

Live `Opus` audit подтвердил базовую схему и поднял те правки, которые нужно внести до начала разработки.

До этой ревизии оставались четыре белых пятна:

1. не был зафиксирован **конкретный prompt family**, а значит реализация могла скатиться в один giant extraction prompt;
2. не была зафиксирована **граница между notebook-LLM и server-LLM**;
3. не было frozen **eval-набора на реальных posts**, а значит не было чем мерить качество до начала разработки;
4. не были дожаты два guardrail-вопроса:
   - уже прошедшие occurrences пока не храним;
   - booking links нужно закладывать так, чтобы позже можно было включить click tracking без перепридумывания домена.

После live-аудита дополнительно фиксируем:

- Kaggle не матчится к server-state: `status` из notebook выходит только как raw claim;
- extraction делится на `Tier 1` в Kaggle и server-side enrichment;
- digest shell не переводится в fully deterministic режим, но per-card LLM call тоже не допускается;
- `rescheduled_from_id`, occurrence-level `digest_eligible`, title normalization и split rules становятся обязательной частью каноники.

## 2. Каноническая LLM Chain

Для MVP нельзя делать один LLM call “распознай всё, смёржь всё и напиши текст”.

Каноническая цепочка такая:

```text
deterministic prefilter
-> trail_scout.screen.v1
-> trail_scout.announce_extract_tier1.v1 | trail_scout.status_claim_extract.v1 | trail_scout.template_extract.v1
-> deterministic shortlist
-> route_weaver.status_bind.v1 (only if status claim is ambiguous)
-> route_weaver.occurrence_match.v1 (only if ambiguous)
-> route_weaver.enrich.v1
-> deterministic materialize
-> lollipop_trails.digest_batch.v1
```

Что важно:

- notebook отвечает за **candidate understanding + concrete Tier 1 extraction**;
- сервер отвечает за **entity resolution, status binding, semantic enrichment and merge**;
- digest writer видит **fact pack**, а не raw source text;
- deterministic слой форматирует shell и guardrails, но не заменяет semantic writing полностью.

## 3. Opus-Derived Design Principles

Ниже зафиксированы не только локальные переносы из старых syntheses, но и прямые выводы live `Opus` audit из `2026-03-14`.

### 3.1. Что переносим без изменений

- prompts для `Gemma` должны быть короткими, self-contained и stage-specific;
- один prompt = одна задача;
- schema должна быть маленькой и жёсткой;
- то, что можно безопасно precompute в Python, нужно precompute в Python, а не просить `Gemma` считать это внутри prompt;
- после каждого LLM stage нужен deterministic validator;
- downstream writing должен получать structure/fact pack, а не raw source noise;
- не нужно делать giant ban-list wall prompts;
- не нужно перекладывать semantic merge в regex-first слой.

### 3.2. Что особенно важно именно для экскурсионного домена

- не просить модель “самой догадаться”, public ли это excursion или group-only offer: это отдельный explicit output field;
- не смешивать в одном prompt:
  - screening,
  - occurrence extraction,
  - template/profile accumulation,
  - digest copy;
- для `status_update` нужен отдельный prompt family, иначе модель будет постоянно создавать ложные новые occurrences;
- `audience_fit` должен быть first-class output, а не free-text хвостом;
- `on_request` и `scheduled_public` должны быть explicit fields, а не выводом из текстового summary.

### 3.3. Что принимаем из live `Opus`, а что нет

Принимаем полностью:

- `status` binding переносим на сервер;
- Kaggle extraction режем до concrete Tier 1 facts;
- для переносов добавляем явную связь между occurrences;
- правила split-message, transport contract и title normalization фиксируем заранее.

Принимаем частично:

- `Opus` прав, что per-card LLM write для MVP избыточен;
- но в fully deterministic digest не уходим, потому что guide-domain слишком неровный по формулировкам, `audience fit`, нюансам маршрута и rescued titles;
- итоговая каноника: deterministic shell + batched LLM copy from fact pack.

## 4. Где работает LLM

### 4.1. Kaggle / `Trail Scout v1`

В Kaggle живут только такие LLM-задачи:

- candidate screening;
- concrete Tier 1 extraction;
- template/profile hints;
- raw status-claim extraction.

### 4.2. Server / `Route Weaver v1` + `Lollipop Trails v1`

На сервере живут только такие LLM-задачи:

- status-claim binding fallback;
- ambiguous occurrence match;
- semantic enrichment from Tier 1 snippets;
- batched digest short copy from fact pack.

### 4.3. Чего в MVP не делаем

- guide-page synthesis;
- template-page synthesis;
- freeform narrative rewrite from raw posts;
- LLM-based “smart ranking”.

Ranking в MVP остаётся deterministic + metrics-driven.

## 5. LLM Gateway And Rate-Limit Policy

Эта фича не должна обходить existing limit-control framework.

Каноника:

- и в Kaggle, и на сервере использовать existing LLM Gateway;
- Kaggle side для guide-monitoring использует `GoogleAIClient`, `GOOGLE_API_KEY2` и отдельный guide account label `GOOGLE_API_LOCALNAME2`;
- server-side `Route Weaver` / `Lollipop Trails` используют более сильную модель, уже разрешённую политикой текущего Smart Update стека;
- limiter остаётся Supabase-backed, как в `docs/features/llm-gateway/README.md`;
- для guide-monitoring notebook primary secret = `GOOGLE_API_KEY2`;
- для guide-monitoring notebook account/audit label = `GOOGLE_API_LOCALNAME2`;
- labels запросов должны быть отдельными:
  - `guide_scout_screen`
  - `guide_scout_tier1_extract`
  - `guide_status_bind`
  - `guide_occurrence_match`
  - `guide_occurrence_enrich`
  - `guide_digest_batch`

Fail policy:

- provider-side `429` не лечить длинным sleep внутри prompt stage;
- notebook должен помечать candidate как `llm_deferred_rate_limit` и завершать run частично, а не висеть;
- partial run допустим, если это видно в admin report и в `/general_stats`.

## 6. Evidence Model For Prompts

Чтобы extraction не расплывался, notebook до LLM должен готовить evidence pack.

### 6.1. Evidence chunk ids

- `T1..Tn` — текстовые chunks из post text;
- `O1..On` — OCR chunks;
- `M1..Mn` — media/meta hints;
- `P1..Pn` — precomputed flags / source metadata.

LLM не должен копировать длинный source text назад.  
Вместо этого он должен возвращать `fact_refs`.

### 6.2. What to precompute deterministically

До prompt нужно посчитать:

- `source_kind`
- `post_date_local`
- `has_date_signal`
- `has_time_signal`
- `has_price_signal`
- `has_booking_signal`
- `has_status_signal`
- `has_group_signal`
- `has_children_school_signal`
- `has_relative_date_words`
- `has_excursion_keywords`
- `grouped_album_present`
- `message_url`

Именно это следует из уже подтверждённой `Opus/Gemini` логики: не заставлять модель считать простые structural вещи внутри prompt.

## 7. MVP Prompt Families

### 7.1. `trail_scout.screen.v1`

### Purpose

Решает, есть ли в посте excursion signal и какой downstream extractor нужен.

### Run condition

- только для posts, прошедших deterministic prefilter.

### Input contract

```json
{
  "source": {
    "username": "tanja_from_koenigsberg",
    "source_kind": "guide_personal",
    "title": "..."
  },
  "post": {
    "message_id": 3895,
    "post_date_local": "2026-02-20",
    "text_chunks": [{"id": "T1", "text": "..."}],
    "ocr_chunks": [{"id": "O1", "text": "..."}],
    "prefilter_flags": {
      "has_date_signal": true,
      "has_time_signal": true,
      "has_price_signal": true,
      "has_booking_signal": true,
      "has_status_signal": false,
      "has_group_signal": false
    }
  }
}
```

### Output schema

```json
{
  "decision": "ignore|announce|status_update|template_only",
  "post_kind": "announce_single|announce_multi|status_update|reportage|template_signal|on_demand_offer|mixed_or_non_target",
  "extract_mode": "none|announce|status|template",
  "digest_eligible_default": "yes|no|mixed",
  "contains_future_public_signal": true,
  "contains_past_report_signal": false,
  "reasons": ["..."],
  "confidence": "high|medium|low"
}
```

### Prompt draft

```text
You classify one Telegram post from a guide-related source.

Use only the provided post text, OCR, source kind, and precomputed flags.
Do not invent missing dates, guides, routes, or booking details.
Treat public scheduled excursions, status updates, and on-request/group-only offers as different classes.
If the post mostly contains lectures, gastronomy, lifestyle, or recap material, do not classify it as a public excursion announce unless the excursion signal is explicit and future-facing.
If the post mixes excursion and non-excursion content, classify by the dominant actionable excursion signal.

Return JSON only.
```

### Deterministic validation

- `decision` and `extract_mode` must agree;
- `status_update` cannot return `digest_eligible_default=yes` without a future excursion target signal;
- `on_demand_offer` must never default to `yes`.

### 7.2. `trail_scout.announce_extract_tier1.v1`

### Purpose

Из already-screened announce post извлекает one or many future occurrences, но только в concrete `Tier 1` схеме.

### Run condition

- `trail_scout.screen.v1.extract_mode = announce`

### Output schema

```json
{
  "occurrences": [
    {
      "title_raw": "Город К. Женщины, которые вдохновляют",
      "date_local": "2026-03-07",
      "time_local": "11:00",
      "duration_text": "3 часа",
      "meeting_point": "у Матери России",
      "price_text": "2000 руб",
      "booking_method": "dm",
      "booking_target": "@Yulia_Grishanova",
      "status_hint": "planned",
      "guide_names": ["Татьяна Удовенко", "Юлия Гришанова"],
      "organizer_names": [],
      "seats_text": "количество мест ограничено",
      "raw_text_snippet": "7 марта — Тематическая пешеходная прогулка «Город К. Женщины, которые вдохновляют»...",
      "fact_refs": ["T2", "T3", "T4"],
      "uncertain_fields": []
    }
  ],
  "template_hints": [
    {
      "canonical_title_hint": "Город К. Женщины, которые вдохновляют",
      "route_theme": "женские сюжеты города К.",
      "fact_refs": ["T2", "T3"]
    }
  ],
  "guide_profile_hints": []
}
```

### Prompt draft

```text
You extract structured excursion occurrences from one already-screened Telegram post.

Rules:
- Return only future or same-day active public occurrences explicitly grounded in the post or OCR.
- Resolve relative dates only from the provided post date.
- If one post contains several different dates/routes, return several occurrences.
- If a route is mentioned without a public date/time or is clearly only a template/program, do not create an occurrence for it here.
- Extract only concrete facts that are directly present in the text or OCR.
- Do not infer audience, district, language, or polished summary here.

Return JSON only.
```

### Deterministic validation

- every occurrence must have `title_raw`;
- every emitted occurrence must have `date_local`;
- no occurrence may have `date_local < today_local`.

### 7.3. `trail_scout.status_claim_extract.v1`

### Purpose

Извлекает raw status claims и meeting-point/reminder updates без создания новых occurrences и без привязки к server-side state.

### Run condition

- `trail_scout.screen.v1.extract_mode = status`

### Output schema

```json
{
  "status_claims": [
    {
      "title_fragment": "Расширенная экскурсия по Зеленоградску",
      "date_hint": "12 марта",
      "update_type": "few_seats|waitlist|sold_out|meeting_point_update|rescheduled|reminder",
      "seats_text": "Есть одно освободившееся место",
      "new_date_local": null,
      "new_meeting_point": "у супермаркета Спар, ул. Тургенева, 1Б",
      "new_time_local": "12:00",
      "duration_text": "4+ часа",
      "booking_target": "http://t.me/gid_zelenogradsk_kotova_natalia",
      "raw_text_snippet": "Есть одно освободившееся место... Встреча у супермаркета Спар...",
      "fact_refs": ["T1", "T2", "T3"]
    }
  ]
}
```

### Prompt draft

```text
You extract operational excursion-update claims from one Telegram post.

Do not create a new excursion.
Do not assume access to any database or previously created occurrences.
Your task is to capture claims such as few seats, sold out, waitlist, reminder, meeting point change, or reschedule.
If the target route/date is unclear, still return the best grounded fragment or date hint instead of inventing certainty.

Return JSON only.
```

### Deterministic validation

- `status_claim_extract` cannot emit `occurrences`;
- `rescheduled` should carry either `new_date_local` or clear old/new date hints;
- if only `meeting_point` changed, status may stay unset at claim level.

### 7.4. `trail_scout.template_extract.v1`

### Purpose

Извлекает template-only knowledge:

- `on_request` programs;
- guide/profile claims;
- route/reportage/template evidence;
- `audience_fit`.

### Run condition

- `trail_scout.screen.v1.extract_mode = template`

### Output schema

```json
{
  "template_candidates": [
    {
      "canonical_title_hint": "Масленица в Лесном Хуторке",
      "availability_mode": "on_request_private",
      "digest_eligible_default": "no",
      "audience_fit_tags": ["school_groups", "children"],
      "audience_fit_note": "Организованные школьные группы; программа адаптируется под возраст.",
      "price_text": "стоимость зависит от количества человек в группе",
      "group_constraints_text": "выбирайте удобный день",
      "fact_refs": ["T1", "T2", "T3"]
    }
  ],
  "guide_profile_hints": [],
  "template_evidence": []
}
```

### Prompt draft

```text
You extract non-public template knowledge from a guide/operator Telegram post.

Focus on:
- on-request or private-group programs,
- for-whom signals,
- adaptability,
- stable route/program framing,
- guide or operator positioning if explicitly stated.

Do not create public occurrences here unless the post explicitly contains an open public date and time.

Return JSON only.
```

### Deterministic validation

- `on_request_private` must force `digest_eligible_default=no`;
- template-only extraction cannot emit public occurrence rows;
- `audience_fit_note` must stay short and grounded.

### 7.5. `route_weaver.status_bind.v1`

### Purpose

Привязывает raw `status_claim` к конкретному active occurrence только на сервере, где уже есть DB state.

### Run condition

- deterministic status binder нашёл `2+` правдоподобных active occurrences;
- или claim пришёл от collaborator/aggregator source и title/date hints неполны.

### Input contract

- one status claim from `trail_scout.status_claim_extract.v1`;
- shortlist of active occurrences inside the same guide/source neighborhood;
- compact normalized fields:
  - `title_normalized`
  - `date`
  - `time`
  - `guide_names`
  - `meeting_point`
  - `source_quality`.

### Output schema

```json
{
  "decision": "bind_to_existing|defer_review|drop_claim",
  "matched_occurrence_id": 123,
  "confidence": "high|medium|low",
  "reason": "..."
}
```

### Prompt draft

```text
You decide whether one status-update claim should bind to an existing active excursion occurrence.

Use only the claim and the provided shortlist of active occurrences.
Do not invent a target occurrence.
Prefer defer_review over a weak bind.
Prefer original guide ownership when a collaborator or aggregator post points to the same route.

Return JSON only.
```

### Deterministic validation

- no `bind_to_existing` without `matched_occurrence_id`;
- `drop_claim` is allowed only when the claim is clearly too vague or historical;
- a bound claim cannot attach to a past occurrence.

### 7.6. `route_weaver.occurrence_match.v1`

### Purpose

LLM-assisted match/create only when deterministic shortlist is ambiguous.

### Run condition

- shortlist contains `2+` plausible active occurrences;
- or source is aggregator/operator and route title is close but not exact;
- or a reschedule/status update needs target disambiguation.

### Input contract

- one candidate occurrence fact pack;
- shortlist of active occurrence rows with compact normalized fields;
- source priority metadata.

### Output schema

```json
{
  "decision": "attach_to_existing|create_new|defer_review",
  "matched_occurrence_id": 123,
  "confidence": "high|medium|low",
  "reason": "..."
}
```

### Prompt draft

```text
You decide whether one extracted excursion occurrence should attach to an existing active occurrence or create a new one.

Use only the candidate fact pack and the provided shortlist.
Do not merge items only because they share a booking contact or a generic city.
Prefer the original guide source over aggregator duplicates when the route/date/time semantics match.
If two shortlist items are both plausible and the evidence is weak, choose defer_review.

Return JSON only.
```

### Deterministic validation

- no `attach_to_existing` without `matched_occurrence_id`;
- `defer_review` is preferred over low-confidence forced merge;
- matching is never allowed to revive a past occurrence row.

### 7.7. `route_weaver.enrich.v1`

### Purpose

Server-side semantic enrichment поверх `Tier 1` facts и коротких raw snippets.

### Run condition

- after occurrence bind/create decision;
- only for rows that passed materialization shortlist and still need semantic fields.

### Output schema

```json
{
  "canonical_title_hint": "Город К. Женщины, которые вдохновляют",
  "audience_fit_tags": ["adults", "locals", "tourists"],
  "audience_fit_note": "Подходит взрослым местным и туристам, которым интересен городской сторителлинг.",
  "excursion_type": "walking",
  "district": "Центральный район",
  "availability_mode": "scheduled_public",
  "digest_eligible_default": "yes",
  "summary_seed": "Прогулка по женским сюжетам города и скрытым биографиям проспекта Мира."
}
```

### Prompt draft

```text
You enrich one excursion occurrence from a structured Tier 1 fact pack and short grounded snippets.

Rules:
- grounded facts only;
- determine audience fit, availability mode, excursion type, district, and summary seed;
- keep audience fit short and usable in a digest card;
- if the offer is on-request or private-group only, set digest_eligible_default to no;
- do not duplicate logistics fields already stored separately.

Return JSON only.
```

### Deterministic validation

- `on_request_private` must force `digest_eligible_default=no`;
- school-group or private-group offers cannot become public digest items unless an occurrence is explicitly open and scheduled;
- empty enrichment is better than speculative enrichment.

### 7.8. `lollipop_trails.digest_batch.v1`

### Purpose

Из materialized fact packs генерирует только короткий authorial copy batch-режимом, а не по одному LLM call на карточку.

- `title`;
- `digest_blurb`;

`audience_line` для live MVP остаётся deterministic-полем из already materialized facts и не должен зависеть от отдельного writer-call.

### Run condition

- after successful materialization of one digest bundle or small batch of occurrence rows.

### Input contract

- array of materialized fact packs without raw source prose;
- explicit omission flags for unknown fields;
- deterministic shell fields are already known and passed separately.

### Output schema

```json
[
  {
    "occurrence_id": 123,
    "title": "string",
    "digest_blurb": "string"
  }
]
```

### Prompt draft

```text
You write short public digest copy for several excursion occurrences from structured fact packs.

Rules:
- grounded facts only;
- write only title + digest_blurb;
- title should stay close to the grounded route/exit name, not a creative rename;
- do not duplicate date, time, price, meeting point, or booking link in the prose line if they already live in the card fields;
- no hype, no cliches, no invented uniqueness claims;
- make the blurb readable and inviting, but without embellishment;
- choose blurb length from fact richness (`1..3` sentences), not from raw-source verbosity;
- do not rewrite the whole card layout; the deterministic shell already exists.

Return JSON only.
```

### Deterministic validation

- `title` must stay URL-free and username-free;
- no URLs or usernames in `title` or `digest_blurb`;
- no logistics duplication if the fact pack already has structured fields;
- empty/fallback deterministic shell is better than speculative line-level copy.

## 8. Deferred Prompt Families

Не блокируют MVP:

- `route_weaver.template_link.v2`
- `lollipop_trails.template_short.v1`
- `lollipop_trails.guide_summary.v1`
- `lollipop_trails.template_page.v1`
- `lollipop_trails.guide_page.v1`
- `related_material_link.v1`

Важно: они уже ожидаются будущей архитектурой, но не должны тормозить первую рабочую версию.

## 9. Non-LLM Guardrails That Must Exist From Day One

### 9.1. Past occurrences are not stored

Пока что уже прошедшие excursions не храним как `guide_occurrence`.

Правило:

- если extracted occurrence уже в прошлом относительно local date, row не создаётся;
- если active occurrence перешёл в прошлое, он удаляется cleanup job;
- если past-oriented post всё же полезен для template/profile layer, его knowledge должен быть promoted в template/profile facts before occurrence cleanup.

### 9.2. Booking link tracking future-proofing

MVP не обязан считать клики, но схема должна быть готова к этому.

Минимум:

- raw `booking_target_url` хранится как source of truth;
- `guide_occurrence.id` и `booking_target_url` считаются устойчивой опорой для будущего redirect layer;
- отдельные tracking columns и redirect endpoint осознанно откладываются до реальной реализации tracking feature, чтобы не создавать ложную готовность.

Канонический future path:

```text
digest link
-> bot/project redirect
-> raw booking target
-> click counter
```

## 10. Implementation Recommendation

Если делать MVP прямо сейчас, правильный минимальный pack такой:

1. `trail_scout.screen.v1`
2. `trail_scout.announce_extract_tier1.v1`
3. `trail_scout.status_claim_extract.v1`
4. `trail_scout.template_extract.v1`
5. `route_weaver.status_bind.v1`
6. `route_weaver.occurrence_match.v1`
7. `route_weaver.enrich.v1`
8. `lollipop_trails.digest_batch.v1`

Это уже достаточно, чтобы:

- сканировать источники;
- извлекать future public excursions;
- не тащить в digest `on_request` offers;
- аккуратно обрабатывать `few seats`, `meeting point` и переносы без доступа Kaggle к серверной БД;
- публиковать пригодный digest из fact pack без перехода в fully deterministic copy.
