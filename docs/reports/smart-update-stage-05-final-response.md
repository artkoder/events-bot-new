# Smart Update Stage 05 Final Response

Дата: 2026-03-06
Формат: финальный competitive ответ перед внедрением. Строго по запрошенным секциям.

---

## 1. Final Deterministic Sign-off

| # | Rule | Verdict | Обоснование |
|---|---|---|---|
| 1 | `same_post_exact_title` | **ship now** | Bulletproof: same source + same title + same date + no time conflict. Zero realistic false merge scenario. Самый безопасный rule в пакете |
| 2 | `same_post_longrun_exact_title` | **ship now** | 5 preconditions включая `text_same AND text_containment`. Time conflict обоснованно трактуется как extraction noise для long-running exhibitions с identical texts |
| 3 | `generic_ticket_false_friend` | **ship now** | (после cleanup `GENERIC_TITLE_TOKENS`). Rule выдаёт `different`, never merge. Same slot + generic ticket + unrelated titles + zero meaningful overlap → different events. False-merge risk = 0 by construction |
| 4 | `broken_extraction_address_title` | **ship now** | Same source + same date + same text + one title is corrupted address. 4 layers of proof. Extraction corruption is objective |
| 5 | `specific_ticket_same_slot` | **ship now** | Specific (non-generic) ticket + same slot + shared payload = same event. Guards against generic tickets, cathedral-style URLs, and time conflicts. Clean |
| 6 | `doors_start_ticket_bridge` | **ship with monitoring** | Tightened version: `both tickets non-empty OR same_source_url`. Fires only on 1 pair в casepack (gromkaya). Sound, but narrowest coverage = least validation depth. Monitor 2 weeks for fires outside casepack |
| 7 | `multi_event_source_blocker` | **ship now** | Structural blocker (different-only). `source_url_owner_pair_max ≥ 4 AND NOT title_exact → different`. False-merge risk = 0 by construction. Порог 4 validated against led_hearts (=3, safe) |
| 8 | `cross_source_exact_match` | **ship with monitoring** | 6 жёстких preconditions. На casepack clean, fires на 8 pairs (1 new + 7 already-resolved) без ошибок. Но это MERGE rule (не blocker) без shared source proof — по risk profile слабее rules 1-7. Deploy + alert-on-fire на 2 недели. Если за 2 недели ни одного ложного — promote to full ship |

### Обоснование "ship with monitoring" для doors_start и cross_source

Я **не согласен** с позицией "keep preprod only". Мои аргументы:

1. **`doors_start_ticket_bridge`** fires на 1 pair → 1 data point. Но все 5 preconditions (same_date + door_vs_start + ticket_same_non_null + title_related + venue_noise_rescuable) в сумме дают достаточный proof. Нет realistic production scenario, где ВСЕ 5 условий true, но events разные. "Keep preprod" = отложить на неопределённый срок без нового data.

2. **`cross_source_exact_match`** — coincidental exact title + exact date + exact time + exact venue from different sources = практически невозможный false positive. Waiting for shadow — это cost без corresponding risk reduction. Alert-on-fire даёт бо́льший information gain, чем shadow.

### Что я НЕ рекомендую добавлять ещё

Никаких новых deterministic rules. 8 rules — это полный narrowed subset. Всё остальное — LLM territory.

---

## 2. Residual Gray LLM Policy

### Per case class расклад

#### `hudozhnitsy_5way_cluster` (10 пар)

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`
**LLM action при `different`**: `mark_different`

**Обоснование**: все 5 events share date + time + venue (Третьяковка, 07.03, 14:00). Title'ы содержат вариации "Художниц\*". Если LLM с confidence отвечает `same_event` — это safe auto_merge. 10 пар из 5 events = standard transitive merge.

**⚠️ Конкурентное уточнение**: для `hudozhnitsy` **transitive merge** — критически важный паттерн. Если LLM говорит: A=B (same_event), B=C (same_event), A=C (uncertain) — транзитивно A=C тоже same_event. Реализация: union-find на парах с verdict `same_event`. Это снижает число необходимых LLM вызовов: вместо 10 pairwise calls, может хватить 4-5 до полного покрытия кластера.

**Prompt hints**: `participant_overlap` (если доступен из text mining), `title_keyword_overlap`, `poster_overlap`.

#### `shambala_cluster` (2 пары: [2799,2843], [2799,2844])

**LLM action при `same_event`**: `softlink` (НЕ auto_merge)
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`
**LLM action при `different`**: `mark_different`

**Обоснование**: brand vs lineup — это **спорный класс**. "Шамбала" (бренд мероприятия) vs "Влада Клепцова, Вика Козлова..." (конкретный lineup). Формально — одно и то же мероприятие. Но если один пост — анонс бренда, а другой — детальный lineup, пользователь может хотеть видеть ОБА, т.к. they carry different information.

**Recommendation**: `softlink` при `same_event`, `keep_gray` при всём остальном. Не рисковать auto_merge на brand-vs-item.

NB: [2843,2844] УЖЕ resolved deterministic (same_post_exact_title). Значит фактически в LLM идут только [2799,2843] и [2799,2844] — 2 вызова.

#### `sobakusel_default_location_conflict` (1 пара)

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: default_location conflict — venue_match=False из-за разного default_location assignment, не из-за реальной разницы venue. `title_related=True`, `same_date=True`. Это noise, не signal. LLM должен видеть titles + date + что venue отличается только из-за location policy.

**Prompt hint**: `venue_mismatch_reason: "default_location_policy"`.

#### `prazdnik_u_devchat_broken_extraction` (2 пары: [2789,2802], [2789,2803])

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: 2789 — отдельный пост о том же мероприятии, without same_source_url. LLM нужен для semantic matching: "Праздник у девчат" vs "Хоровая вечеринка «Праздник у девчат»" → очевидно одно мероприятие. `title_related=True` для [2789,2802] — strong hint.

Для [2789,2803] — `title_related=False` (title corrupted → "Октябрьская, 8"), но [2802,2803] уже resolved (broken_extraction). Если LLM resolve [2789,2802] → transitive to [2789,2803].

#### `little_women_cluster` (1 пара: [2761,2815])

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: `title_related=False` (разные framing), но `ticket_specific_same=True` и `venue_noise_rescuable=True`. Specific shared ticket — сильный signal для LLM. Если два события share the same specific ticket URL — это почти наверняка один event.

**Prompt hint**: `ticket_specific_same=true` как key evidence.

#### `makovetsky_chekhov_duplicate` (1 пара: [2758,2759])

**LLM action при `same_event`**: `softlink` (НЕ auto_merge)
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: `poster_overlap=1`, `title_related=False`. Shared poster — evidence, но weak. Titles отличаются (brand vs program framing). Аналогично shambala — brand-vs-item class. `softlink` при `same_event`, не auto_merge.

#### `oncologists_svetlogorsk_duplicate` (1 пара: [2710,2721])

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: double mismatch (title_related=False + venue_match=False), но `ticket_specific_same=True`. LLM needs to semantically evaluate: "Бесплатные приемы детских онкологов" vs "Бесплатный приём детского онколога" — grammatical variation, same event. Если LLM уверенно отвечает `same_event` — safe auto_merge.

**Prompt hint**: `ticket_specific_same=true`, `venue_mismatch_reason: "city_granularity"`.

#### `led_hearts_same_post_triple_duplicate` (3 пары)

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: overwhelming evidence — same_source_url + title_exact + text_same + poster_overlap + ticket_specific_same. Date/time mismatch = extraction bug. LLM sees: identical source, identical title, identical text, identical poster, identical ticket → same_event with max confidence.

**Prompt hints**: `same_source=true`, `title_exact=true`, `text_same=true`, `poster_overlap=true`, `date_mismatch_reason: "likely_extraction_bug"`.

**Конкурентный тезис**: led_hearts — самый ЛЁГКИЙ кейс для LLM. Если LLM не может уверенно resolve `same_event` при таком уровне evidence — у нас проблема с prompt, а не с policy.

#### `matryoshka_exhibition_duplicate` (1 пара: [2725,2726])

**LLM action при `same_event`**: `auto_merge`
**LLM action при `likely_same`**: `softlink`
**LLM action при `uncertain`**: `keep_gray`

**Обоснование**: same date + same venue + same end_date + title_related=True + same VK group. Нет shared source/ticket/poster, но semantic evidence strong: "Фабрика матрёшки" vs "Выставка матрёшки" → variations of same exhibition.

**Prompt hint**: `same_vk_owner=true`, `end_date_match=true`.

---

## 3. Verdict-to-Action Mapping

### Финальная policy

```
┌──────────────┬───────────────────────┬──────────────────────┐
│ LLM Verdict  │ Standard Action       │ Brand-vs-Item Action │
├──────────────┼───────────────────────┼──────────────────────┤
│ same_event   │ auto_merge            │ softlink             │
│ likely_same  │ softlink              │ softlink             │
│ uncertain    │ keep_gray             │ keep_gray            │
│ different    │ mark_different        │ mark_different       │
└──────────────┴───────────────────────┴──────────────────────┘
```

### Какие case classes — "brand-vs-item"?

| Case class | Brand-vs-item? |
|---|---|
| shambala [2799,*] | **ДА** — brand name vs lineup listing |
| makovetsky | **ДА** — actor brand vs specific program |
| hudozhnitsy | НЕТ — all describe same exhibition, just different framing |
| sobakusel | НЕТ — same event, venue noise |
| prazdnik | НЕТ — same event, extraction noise |
| little_women | НЕТ — same screening, naming difference |
| oncologists | НЕТ — same medical event, grammatical variation |
| led_hearts | НЕТ — extraction bug, identical content |
| matryoshka | НЕТ — same exhibition, title variation |

**Итого**: только 3 пары (shambala ×2, makovetsky ×1) из 22 попадают в brand-vs-item policy. Остальные 19 — standard policy.

### softlink implementation note

`softlink` означает:
- events НЕ merge'атся (separate DB records);
- UI показывает связь ("Возможно, это одно и то же мероприятие: [ссылка]");
- user может manually merge или dismiss.

Это гибче, чем binary merge/different, и critical для quality-first approach: пользователь видит оба варианта, не теряет information.

---

## 4. Last Blockers

### Реальные blockers для rollout: **НЕТ.**

Уточнение:

1. **Deterministic subset**: ready to ship. 8 rules, все validated. No blocker.

2. **LLM layer**: не blocker для deterministic rollout. LLM — параллельная track, может deploy'иться отдельно.

3. **Casepack size**: 35 cases — not a blocker, but a continuous improvement item. Не мешает rollout, мешает long-term confidence.

### Observations, но не blockers:

| Item | Severity | Action |
|---|---|---|
| `GENERIC_TITLE_TOKENS` overfit (5 case-specific words) | Low | Cleanup в Stage 04A deploy, уже validated |
| `doors_start_ticket_bridge` fires на 1 pair only | Low | Monitor, не block |
| `cross_source_exact_match` теоретический coincidental title risk | Very Low | Alert-on-fire, не block |
| LLM prompt for Gemma not yet designed | Medium | Parallel track, не блокирует deterministic |
| `led_hearts` extraction bug root cause | Medium | Extraction fix, не identity resolution |

**Verdict**: nothing blocks `Stage 04A` + `Stage 04B` deterministic deploy.

---

## 5. Optional Appendix: Cluster-Call Strategy

### Scope: только `hudozhnitsy`-like кластеры (≥3 events, same slot)

На текущем residual gray: только `hudozhnitsy_5way_cluster` (5 events, 10 pairwise pairs) qualifies. Все остальные case classes — isolated pairs или small groups (≤2 pairs).

### Pairwise baseline vs Cluster-call

| Метрика | Pairwise (10 calls) | Cluster (1 call) |
|---|---|---|
| RPM | 10 requests | 1 request |
| Input tokens (estimated) | 10 × ~400 = ~4000 | 1 × ~1200 = ~1200 |
| Output tokens | 10 × ~120 = ~1200 | 1 × ~300 = ~300 |
| Total TPM | ~5200 | ~1500 |
| Latency @ RPM=20 | ~30s (sequential) | ~3s |
| Context quality | Pair sees only 2 events | LLM sees all 5 simultaneously |
| Error propagation | Independent per pair | Single point of failure |

### Cluster-call prompt (Gemma-compatible)

```
Перед тобой 5 анонсов мероприятий на один и тот же день, время и площадку.
Определи: описывают ли все анонсы ОДНО мероприятие, или среди них есть РАЗНЫЕ?

Правила:
- Если названия — вариации одного и того же (alias, сокращение, расширение) → same_event
- Если среди анонсов есть реально разные программы → group them separately
- Ошибочная склейка разных мероприятий ХУЖЕ, чем оставить дубли

JSON ответ по schema:
{
  "verdict": "all_same" | "mixed",
  "groups": [[event_ids_in_same_group], ...],
  "reasoning": "краткое обоснование"
}

Данные:
{payload}
```

### Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Gemma не справляется с grouping из 5 events | Medium | Fall back to pairwise if cluster verdict = "mixed" |
| Payload > 2000 tokens | Low | Clip descriptions, send only titles + key hints |
| Inconsistent grouping (A=B, B=C, but A≠C) | Low | Union-find normalization post-LLM |

### Recommendation

**Pairwise первично, cluster — fallback optimization.**

Реализация:
1. Start with pairwise calls for all 22 gray pairs
2. If ≥3 pairs in same slot all return `same_event` → transitive merge (union-find)
3. If RPM/TPM issues arise with pairwise → switch to cluster-call for large groups

НЕ делать cluster-call default. Pairwise дает лучшую granularity per-pair. Cluster — optimization for cost/latency, не для quality.

---

## My competitive summary

1. **8 rules: ship.** 6 now, 2 with monitoring. Zero blockers. Waiting is cost without information gain.

2. **22 gray pairs: LLM with quality-first policy.** 19 standard + 3 brand-vs-item. Standard: `same_event → auto_merge`. Brand-vs-item: `same_event → softlink`. This is the key quality guarantee — not all LLM "yes" answers are equal.

3. **Transitive merge via union-find** — underutilized optimization that reduces LLM calls for clusters. Not in the brief, but critical for production efficiency.

4. **led_hearts = extraction fix, not identity resolution.** LLM will resolve it easily (overwhelming evidence), but root cause should be fixed upstream.

5. **Disagreement surface = 0.** We're done with architecture. Time to implement.
