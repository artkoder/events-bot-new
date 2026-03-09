# Smart Update Stage 04 Competitive Response

Дата: 2026-03-06
Формат: конкурентная оценка `run_03_preprod_candidate` + residual gray strategy.
Цель: максимальное стабильное качество, а не минимизация gray любой ценой.

---

## 1. Stage 04 Assessment

### 1.1. Что Stage 04 доказал

Stage 04 — **сильная инженерная работа**. Три локальных progressive dry-run, каждый добавляющий ровно одно narrowed правило, с проверкой на 35 кейсов / 72 пары — это правильный methodology.

Ключевой вывод, который Stage 04 зафиксировал явно:

> Broad heuristics не работают. Каждое правило нужно формулировать narrowed, с explicit preconditions, и cross-check'ить на полном casepack перед принятием.

Конкретно Stage 04 **корректно** опроверг 4 из моих Stage 03 proposals:

| Мой Stage 03 proposal | Что Stage 04 показал |
|---|---|
| Broad venue-alias → 9 пар | `sobakusel`, `oncologists`, часть `prazdnik` и `shambala` — не чистый venue-alias, а mix default_location + broken extraction + city/venue granularity. Таблица не решает |
| Broad `title_mismatch → different` | Ломает `hudozhnitsy [2779, 2801]` (title_related=true, но title_exact=false). Generic title overlap даёт false positive |
| Cluster-aware merge | Часть пар внутри `hudozhnitsy` НЕ держит `title_related=true`. Cluster merge unsafe |
| Relaxed source rescue | `led_hearts` без extraction-bug signature слишком опасно generalizing — exact same risk profile как double-show |

**Я принимаю эти факты.** Широкие гипотезы не прошли cross-check — значит, узкий deterministic subset + LLM для residual — правильная стратегия.

### 1.2. Что я хочу оспорить / дополнить

При этом у Stage 04 есть 3 точки, которые я считаю **недоработанными** или **улучшаемыми**. Ниже подробно.

---

## 2. Narrow Deterministic Subset: Accept / Reject

### Verdict: **ACCEPT `run_03_preprod_candidate` с двумя оговорками**

Прохожу по каждому из 6 правил:

### 2.1. `same_post_exact_title` — ✅ ACCEPT безоговорочно

```python
if same_source_url and title_exact and same_date and not time_conflict:
    return "merge"
```

**Логика bulletproof**: один пост, одно идентичное название, одна дата, нет конфликта времён → дубль. Нет realistic сценария false merge.

**Control cases**: shambala [2843,2844] — `Шамбала || Шамбала`, same post → merge ✅.

### 2.2. `same_post_longrun_exact_title` — ✅ ACCEPT

```python
if same_source_url and title_exact and same_date and time_conflict
   and end_date matches and text_same and text_containment:
    return "merge"
```

**5 preconditions**, каждый из которых сужает scope. Особенно `text_same AND text_containment` — это очень жёсткий proof. Time conflict трактуется как extraction-noise только для long-running exhibitions с доказанным identical текстом.

**Control case**: womanhood [2755,2756] — same VK post, same title, 12:00/15:00 = excursion times → merge ✅.

**Единственный edge case, который стоит мониторить**: long-running, same-post, но реально два разных мероприятия в рамках одной выставки (утренний и вечерний формат). Но `end_date matches AND text_same` делает это сценарий крайне маловероятным.

### 2.3. `generic_ticket_false_friend` — ✅ ACCEPT, но с оговоркой по GENERIC_TITLE_TOKENS

```python
if same_date and venue_match and not time_conflict and not title_related and ticket_same:
    if title_token_overlap > 0:
        return None  # bail
    if generic_ticket(left) and generic_ticket(right):
        return "different"
```

**Rule logic sound.** Если generic ticket + unrelated titles + same slot → разные события. Корректно.

**⚠️ Оговорка**: в `stage_04_consensus_dryrun.py` set `GENERIC_TITLE_TOKENS` содержит:

```python
"английская", "придворная", "века", "королева", "фей"
```

Это **case-specific tokens** из `cathedral_shared_ticket_false_friend`, а не truly generic words. Если завтра появится кейс «Выставка "Английская придворная культура XVII века" — продолжение» vs «Английская придворная культура XVII века» — эти слова не попадут в overlap, и rule ошибочно выдаст `different`.

**Рекомендация**: убрать `английская`, `придворная`, `века`, `королева`, `фей` из generic tokens. Для `cathedral` правило и так работает: titles «Английская придворная культура XVII века» vs «Королева фей» имеют **0 overlap по значимым tokens** даже без вноса этих слов в denylist. Потому что `_title_tokens` фильтрует по `len(tok) >= 3`, и оставшиеся значимые слова действительно не пересекаются:
- Left tokens (без generic): `{английская, придворная, культура, xvii, века}` → wait, `культура` и `века` уже в denylist, `xvii` < len 3... Проверим:

Фактически: titles `🎶 Английская придворная культура XVII века` и `🎭 Королева фей`:
- Left after normalize: `английская придворная культура xvii века`
- Left tokens (≥3 chars, minus GENERIC): удаляем `английская`, `придворная`, `культура`, `века` → остаётся `{xvii}`. Но `len("xvii")=4 ≥ 3` → `{xvii}`.
- Right after normalize: `королева фей`
- Right tokens: удаляем `королева`, `фей` → остаётся `{}`.
- Overlap = `{xvii} ∩ {} = 0` → rule fires → different ✅.

**Но**: если убрать `английская`, `придворная`, `века`, `королева`, `фей` из denylist:
- Left tokens: `{английская, придворная, культура, xvii, века}` — `культура` is in GENERIC → remove. Остаётся `{английская, придворная, xvii, века}`.
- Right tokens: `{королева, фей}` — оба ≥ 3 chars, ни один не в generic stopwords... wait, `фей` = 3 chars, ≥ 3 → included.
- Overlap = `{английская, придворная, xvii, века} ∩ {королева, фей} = 0` → **rule still fires**.

**Значит мой тезис ВЕРЕН**: убрать case-specific words из `GENERIC_TITLE_TOKENS` не сломает `cathedral`, но увеличит robustness. Эти слова в denylist — overfit к одному кейсу.

**Production action**: удалить `английская`, `придворная`, `века`, `королева`, `фей` из `GENERIC_TITLE_TOKENS`. Re-run dry-run для validation.

### 2.4. `broken_extraction_address_title` — ✅ ACCEPT

```python
if same_source_url and same_date and not time_conflict:
    if text_same and text_containment:
        if title_looks_address(left) XOR title_looks_address(right):
            return "merge"
```

**4 layered preconditions.** `same_source + same_date + same_text + one title is corrupted address` — bulletproof. If the text is identical and the source is the same, the "address title" is clearly extraction corruption.

**Control case**: prazdnik [2802,2803] — «Хоровая вечеринка 'Праздник у девчат'» vs «Октябрьская, 8, 4 этаж» → one title is address → merge ✅.

**Risk surface**: `_title_looks_address` regex could false-positive on legitimate venue-titled events (e.g., event literally titled "Леонова 22"). But combined with `same_source + same_text`, this is acceptable risk — if same post produces "Real Event Title" and "Леонова 22", one is clearly extraction garbage.

### 2.5. `specific_ticket_same_slot` — ✅ ACCEPT, с уточнением scope

```python
if same_date and venue_match and not time_conflict and ticket_same:
    if generic_ticket(left) or generic_ticket(right):
        return None  # bail
    if same_source_url or text_same or text_containment:
        return "merge"
```

**Логика**: specific (не generic) ticket + same slot + shared payload = один и тот же event.

**Control cases**: little_women [2815,2816] и [2815,2817] — merge ✅.

**Уточнение scope**: rule корректно НЕ сработает на `cathedral` (generic ticket), НЕ на `nutcracker_two_shows` (time_conflict), и НЕ на `backstage_tour_weekly_run` (date_mismatch).

**⚠️ Потенциальный edge case**: два РАЗНЫХ спектакля на одной площадке, оба со specific ticket URLs, которые случайно совпали. Это маловероятно (specific = event-specific path), но стоит мониторить. Guard `same_source_url or text_same` дополнительно снижает risk.

### 2.6. `doors_start_ticket_bridge` — ⚠️ CONDITIONAL ACCEPT

```python
if same_date and door_vs_start_pair and ticket_same and title_related and venue_noise_rescuable:
    return "merge"
```

**5 preconditions.** Sound, но я хочу уточнить, что check `ticket_same` здесь ВКЛЮЧАЕТ generic tickets.

**Вопрос**: у `gromkaya` какой ticket? Если generic — то правило merge'ит пару, которая share generic ticket. Это слабее, чем specific ticket proof.

Проверяю: gromkaya [2667,2792] — `vk.com/wall-214027639_10783` source vs `t.me/locostandup/3171` source. Ticket: нужно проверить... из casepack: gromkaya events don't seem to have explicit ticket links in the case window.

**Моя рекомендация**: IF ticket_same is based on non-null specific ticket → ACCEPT. IF ticket_same fires on both-null tickets → rule is merging purely on doors_vs_start + title_related + venue_noise — что менее надёжно.

**Минимально safe формулировка**:
```python
# ONLY if BOTH tickets exist and match (not both null)
if ticket_same and (left.ticket_link and right.ticket_link):
    ...
```

Если оба ticket null → `ticket_same=True` technically, но это weak signal. С `venue_noise_rescuable + doors_vs_start + title_related` в сумме мне достаточно для conditional accept, но я хочу, чтобы это было explicit.

**Verdict**: ACCEPT conditionally — добавить guard `both tickets non-null OR same_source_url` для дополнительной безопасности. Или: принять as-is, но мониторить этот rule в shadow mode первые 2 недели.

---

## 3. Rules I Still Dispute

### 3.1. `museum_holiday_program_multi_child` — всё ещё в gray, но это fixable

**Факт**: 3 must-not-merge пары всё ещё в gray. Dry-run signals показывают:

```
title_related=True, same_source_url=True, ticket_same=True,
venue_match=True, text_same=True, text_containment=True,
source_url_owner_pair_max=6
```

Все сигналы кричат "merge!" — и всё же это РАЗНЫЕ мероприятия.

**Почему `generic_ticket_false_friend` не работает**: `title_related=True` → bail condition в rule.

**Мой конкурентный proposal**: `source_url_owner_pair_max` — это **неиспользованный сильный сигнал**.

Если из одного source URL пришло ≥ 4 events (max=6 в данном случае), это reliable indicator `multi_event` source. Для multi_event sources → default to gray/different, а не merge.

```python
def _rule_multi_event_source_blocker(pair, left, right):
    """
    If the source produced many events (≥4), titles must match exactly
    for deterministic merge. Otherwise → stay gray.
    """
    if pair["source_url_owner_pair_max"] >= 4:
        if pair["title_exact"]:
            return None  # let other rules handle
        else:
            return "different"  # многоезентный source, разные titles → разные события
    return None
```

**Control check**:
- museum_holiday (`source_url_owner_pair_max=6`, `title_exact=False`) → different ✅
- led_hearts (`source_url_owner_pair_max=3`, bypass) → не affected ✅
- shambala (`source_url_owner_pair_max` low?) → не affected
- hudozhnitsy (`same_source_url=False` для cross-source пар) → не affected

**⚠️ Risk**: false different на легитимный dupe из prolific source. Но: если source produces 4+ events, title_exact=false → it's genuinely different programs.

**Я предлагаю добавить это как `run_04` в следующий dry-run**. Ожидаемый результат: `must_not_merge_gray: 3 → 0` без new false merges.

### 3.2. `led_hearts` — НЕ dispute, а recognized LLM territory

Data показывает, что `led_hearts` все 3 пары имеют `deterministic_decision=different` (NOT gray!) — Stage 03 blockers **активно** блокируют их:
- [2845,2846]: `same_date=False` → date blocker → different
- [2845,2847]: `time_conflict=True` → time blocker → different  
- [2846,2847]: `same_date=False` → date blocker → different

Это кейс, где **extraction bug создал wrong date/time** (2846 получил дату 2026-03-08 вместо 2026-03-07, и все 3 получили time 07:03 вместо 11:00). Deterministic rescue здесь требует explicit extraction-bug signature detection, что Stage 04 правильно не пытался делать.

**Мой verdict**: LLM-only. Deterministic rescue для bad-extraction без source-parsing fix слишком dangerous. Пусть LLM видит `same_source + same_post + title_exact + text_same + poster_same` и решает.

### 3.3. `oncologists_svetlogorsk` — interesting edge case

Data: `venue_match=False`, `title_related=False`, `ticket_specific_same=True`, `venue_noise_rescuable=True`.

Это кейс, где два VK поста описывают **один и тот же** приём онколога в Светлогорске, но:
- title фреймят по-разному ("Бесплатные приемы детских онкологов" vs "Бесплатный приём детского онколога")
- venue, видимо, извлечена из разных частей постов

`title_related=False` + `venue_match=False` — **двойной mismatch** при `same_date`. Deterministic merge здесь ОПАСЕН, т.к. title_mismatch + venue_mismatch мог бы быть реально разными events.

**Verdict**: LLM-only. Semantic title matching ("приемы/приём", "детских/детского") — это задача для LLM, не для regex.

---

## 4. Residual Gray: LLM-only vs Keep-gray

### 4.1. Классификация residual gray после `run_03_preprod_candidate`

#### Must-merge gray (23 пары)

| Корзина | Кейсы/пары | Обоснование |
|---|---|---|
| **LLM-only** | `hudozhnitsy_5way_cluster` (10 пар) | Title alias + cross-source. Никакой deterministic rule не покроет все 10 пар без false merge risk. LLM видит semantic title overlap ("Художниц\*" во всех) + same date/time/venue |
| **LLM-only** | `shambala_cluster` [2799,2843] и [2799,2844] (2 пары) | Brand vs lineup (title_mismatch). LLM нужен для context/participant check |
| **LLM-only** | `prazdnik_u_devchat` [2789,2802] и [2789,2803] (2 пары) | Broken extraction, but `same_source_url=False` и `text_same=False` — нет safe deterministic proof |
| **LLM-only** | `little_women_cluster` [2761,2815] (1 пара) | Cross-source, `title_related=False`, `venue_noise_rescuable=True`. LLM needs context |
| **LLM-only** | `makovetsky_chekhov_duplicate` [2758,2759] (1 пара) | `poster_overlap=1`, `title_related=False`, cross-source. Shared poster = hint for LLM, not proof for deterministic |
| **LLM-only** | `oncologists_svetlogorsk` [2710,2721] (1 пара) | `title_related=False` + `venue_match=False` — double mismatch. Only LLM can disambiguate |
| **LLM-only** | `led_hearts` (3 пары) | Extraction bug. `same_source + title_exact + text_same + poster_same` = overwhelming evidence for LLM, but deterministic blocked by date/time mismatch |
| **LLM-only** | `matryoshka_exhibition_duplicate` [2725,2726] (1 пара) | Cross-source, no shared ticket/source/poster. Pure semantic. LLM → same_event с высокой confidence |
| **LLM-only** | `plastic_nutcracker_cross_source` [1603,1622] (1 пара) | Cross-source, `title_exact=True`, `same_date`, `venue_match`. Сильнейшие сигналы, но ZERO shared proof (no source/ticket/poster/text) |

**Итого**: все 23 must-merge gray пары → **LLM-only**. Ни одну НЕ рекомендую добивать deterministic'ом — risk/reward не оправдан.

**НО**: `plastic_nutcracker` — интересный edge case. `title_exact + same_date + venue_match + not time_conflict + no source/ticket/poster overlap` — по сути **cross-source exact duplicate without proof**. Это чистый кейс для нового narrow deterministic rule:

```python
def _rule_cross_source_exact_match(pair, left, right):
    """
    Different sources, but EXACT title, date, time, venue match.
    No conflicts. → merge.
    Only if BOTH time values are known (not default).
    """
    if (pair["title_exact"]
        and pair["same_date"]
        and pair["venue_match"]
        and not pair["time_conflict"]
        # BOTH must have actual time (not default/empty):
        and left.time and right.time and left.time == right.time
        # NOT same source (that's same_post_exact_title already)
        and not pair["same_source_url"]):
        return "merge"
    return None
```

**Control check на must-not-merge кейсах**:
- `cathedral` (1979/2278): `title_exact=False` → bail ✅
- `museum_holiday`: `title_exact=False` → bail ✅
- `museum_overlap_exhibitions`: `title_exact=False` → bail ✅
- `dramteatr_same_slot_cross_title` (1428/1677): `title_exact=False` → bail ✅
- `nutcracker_two_shows` (1619/1620): `left.time != right.time` → bail ✅
- `historical_museum_overlap`: `title_exact=False` → bail ✅
- `buratino_double_show`: `left.time != right.time` → bail ✅
- `severnoe_siyanie`: `left.time != right.time` → bail ✅
- все recurring/date-mismatch: `same_date=False` → bail ✅

**Закрывает**: plastic_nutcracker [1603,1622] — `title_exact=True`, same date, same time (15:00), venue_match → merge ✅.

**Risk**: два реально разных события с IDENTICAL title + date + time + venue. Это возможно чисто теоретически (одно и то же название, случайно?) но **практически невероятно** для real-world events.

**Verdict**: рекомендую добавить как `run_04` candidate вместе с `multi_event_source_blocker`. Expected result: `must_merge_resolved: 15 → 16`, `must_merge_gray: 23 → 22`, `must_not_merge_gray: 3 → 0`.

#### Must-not-merge gray (3 пары)

| Корзина | Кейсы/пары | Решение |
|---|---|---|
| **Deterministic-upgradable** | `museum_holiday_program_multi_child` × 3 | `multi_event_source_blocker` (§3.1): `source_url_owner_pair_max ≥ 4` + `title_exact=False` → different |

### 4.2. LLM Prompt Design для residual gray

Для 22 оставшихся must-merge gray pairs, LLM получает compact pairwise payload. Ключевые hints для каждого класса:

**Тип 1: Cross-source title alias** (hudozhnitsy, shambala brand-vs-lineup)
```json
{
  "hints": {
    "date_time_venue_match": true,
    "title_exact": false,
    "shared_keyword": "художниц",
    "sources_differ": true,
    "deterministic_confirmed": ["date", "time", "venue"],
    "gray_reason": "title_alias_cross_source"
  }
}
```

**Тип 2: Extraction bug** (led_hearts, prazdnik)
```json
{
  "hints": {
    "same_source": true,
    "title_exact": true,
    "text_same": true,
    "poster_overlap": true,
    "date_or_time_mismatch": true,
    "likely_extraction_bug": true,
    "gray_reason": "same_source_date_mismatch"
  }
}
```

**Тип 3: Semantic duplicate** (matryoshka, makovetsky)
```json
{
  "hints": {
    "date_time_venue_match": true,
    "no_shared_proof": true,
    "gray_reason": "semantic_no_source_proof"
  }
}
```

**Prompt** (короткий, ~500 chars):

```
Ты судья по identity событий.
Deterministic проверка уже подтвердила: совпадение по [hints.deterministic_confirmed].
Остаётся вопрос: одно и то же это мероприятие?

Оцени ТОЛЬКО:
1. Семантику названий — одно событие разными словами?
2. Контекст — overlap по участникам/программе/описанию?

Вердикт: same_event / likely_same / different / uncertain
Ошибочная склейка ХУЖЕ дубля. При сомнении → uncertain.
```

---

## 5. Stage 04A / 04B Proposal

### Stage 04A: Deploy `run_03_preprod_candidate` as-is (1-2 дня)

**Action**: внедрить 6 rules из run_03 в production runtime.

**Pre-deploy checks**:
1. Re-run dry-run script на FRESH production snapshot (не на старом) → confirm 0 false merges
2. Verify `GENERIC_TITLE_TOKENS` cleanup (убрать case-specific words) не ломает cathedral
3. Deploy с monitoring: alert if any rule fires on a pair not in casepack

**Expected impact**:
```
must_merge_resolved:     9 → 15  (+6)
must_not_merge_resolved: 30 → 31 (+1)
false_merges:            0       (maintain)
```

### Stage 04B: Two narrow addition candidates (2-3 дня, после Stage 04A validation)

**Candidate 1**: `multi_event_source_blocker`
```python
if source_url_owner_pair_max >= 4 and not title_exact:
    return "different"
```
Expected: `museum_holiday_program_multi_child` × 3 → different.

**Candidate 2**: `cross_source_exact_match`
```python
if title_exact and same_date and venue_match and not time_conflict
   and left.time and right.time and left.time == right.time
   and not same_source_url:
    return "merge"
```
Expected: `plastic_nutcracker` → merge.

**Validation**: dry-run BOTH candidates on full casepack before any deploy.

### Stage 04C: LLM Gray Resolution (5-10 дней, параллельно)

**Action**: implement compact pairwise LLM triage для remaining ~21 gray pairs.

**Shadow mode first**: run LLM on gray pairs silently, log verdicts, compare with gold labels.

**Success criteria for production promotion**:
- 0 false merges on must_not_merge cases
- ≥80% correct on must_merge cases
- uncertain rate < 30%

### Stage 04D: Fresh Casepack Expansion (ongoing)

**Action**: add ≥6 new gold cases targeting uncovered patterns:
- Two exhibitions with SIMILAR titles in same museum
- Event rename (title change after first publication)
- Long-running exhibition with program update
- Cross-source duplicate with different emoji prefixes (more plastic_nutcracker-like)
- Multi-event source with >6 children
- Follow-up post that changes date (not just time)

---

## 6. Residual Risks

### 6.1. `GENERIC_TITLE_TOKENS` overfit

**Risk**: current denylist contains case-specific words (`английская`, `придворная`, etc.). If similar words appear in a legitimate duplicate pair, `generic_ticket_false_friend` will fire incorrectly → false different.

**Severity**: Medium.
**Mitigation**: remove case-specific words (§2.3 analysis proves cathedral still works without them).

### 6.2. `cross_source_exact_match` — "coincidental exact title" risk

**Risk**: two genuinely different events with identical title + date + time + venue. Example: "Мастер-класс" twice at same venue on same day, different organizers.

**Severity**: Low — precondition requires `title_exact` which means full normalized title match, not just keyword. "Мастер-класс: making candles" ≠ "Мастер-класс: pottery".

**Mitigation**: monitor first 2 weeks in shadow, auto-alert on any cross_source_exact_match fire.

### 6.3. `multi_event_source_blocker` false negative

**Risk**: legit same-source duplicate from prolific source (source produces 4+ events, but one IS a duplicate). Rule would block the rescue.

**Severity**: Low — in practice, 4+ event sources are digest/program pages where each event is distinct. If duplicate exists there, it's extraction bug → needs extraction fix, not merge rule.

**Mitigation**: fall through to LLM layer for these pairs.

### 6.4. LLM hallucination на residual gray

**Risk**: Gemma sees sparse evidence (only titles + date, no shared proof) and hallucinates "same_event" on actually different events.

**Severity**: Medium — especially for `hudozhnitsy`-like clusters where title_related=true but titles describe different aspects.

**Mitigation**: 
- Default verdicts `likely_same` and `uncertain` → gray_create_softlink, NOT merge
- Only `same_event` with explicit reasoning → auto-merge
- Temperature 0.0 for triage

### 6.5. Casepack size vs production diversity

**Risk**: 35 cases represent a narrow slice of production. Rules calibrated on 35 cases may not generalize.

**Severity**: High — most important long-term risk.

**Mitigation**: 
- Fresh casepack expansion (§5 Stage 04D)
- Weekly automated dry-run on growing casepack
- Production monitoring: alert on any rule fire not seen in casepack

### 6.6. `led_hearts`-class: extraction bugs without good rescue

**Risk**: extraction bugs will continue producing false date/time → date/time blockers will continue creating false `different` on these pairs → visible duplicates in production.

**Severity**: Medium — but acceptable. LLM layer should catch these in gray triage. The alternative (relaxed source rescue) carries higher false-merge risk.

**Mitigation**: fix extraction layer to not produce bad dates from single-source posts. This is source-parsing fix, not identity-resolution fix.

---

## 7. Мои конкурентные тезисы (не дублируя Opus)

1. **Stage 04 `run_03_preprod_candidate` — ACCEPT.** Все 6 rules хорошо scope'ены. Единственное уточнение: `GENERIC_TITLE_TOKENS` cleanup и `doors_start_ticket_bridge` guard на null tickets.

2. **Broad heuristics проиграли — и это ХОРОШО.** Stage 04 доказал, что quality-first = narrow-first. Широкие venue-alias/title-mismatch/cluster-merge proposals (включая мои из Stage 03) не выдержали cross-check. Это повышает доверие к тому, что осталось.

3. **`museum_holiday` → fixable через `multi_event_source_blocker`.** `source_url_owner_pair_max ≥ 4` — это объективный сигнал multi-event source. Не нужен LLM, чтобы определить, что 6 events из одного поста — это program page, а не duplicates.

4. **`plastic_nutcracker` → fixable через `cross_source_exact_match`.** Exact title + exact date + exact time + venue match + NO time conflict — если это НЕ один event, то что это? Единственный edge case — "coincidental exact title" — практически не встречается в production.

5. **Остальные 21 gray пара — честный LLM territory.** Нет смысла пытаться вытащить их deterministic. Risk/reward не оправдан. Лучше вложить усилия в качественный LLM prompt с compact payload.

6. **Extraction fix > merge rule.** `led_hearts` — не identity resolution problem, а source parsing problem. Fix extraction → проблема исчезает upstream. Workaround через merge rule dangerous.
