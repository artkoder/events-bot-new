# Smart Update Stage 06 Live Validation Response

Дата: 2026-03-07
Формат: конкурентный ответ на 9 вопросов Stage 06 prompt после первого live VK auto-import.

---

## 1. Runtime Finding Classification

| Case class | Primary factor | Secondary factor | Safe next action |
|---|---|---|---|
| `matryoshka_exhibition_duplicate` | **Upstream matching bug** — event 23524 была merge'ена в WRONG sibling 2746 вместо correct 2849 | LLM/shortlist confusion при museum multi-event source | Deterministic guard: prevent cross-sibling merge within multi-event source family |
| `oncologists_svetlogorsk_duplicate` | **Upstream extraction / venue-noise** — campaign post получил `Научная библиотека` вместо multi-city routing | LLM layer miss: campaign-to-city duplicate не resolve'ен | Fix venue extraction for campaign posts; LLM policy: same-registration-link + same-date → `likely_same` |
| `makovetsky_chekhov_duplicate` | **LLM layer miss** — acceptable duplicate territory, not false merge | Weak title overlap (brand vs program framing) | LLM policy upgrade: same-poster + same-slot → bias toward `same_event` |
| `hudozhnitsy_5way_cluster` | **Queue / operational** — partial coverage due to TPM deferrals and rejections | Title alias miss across 5 variants | Cannot assess until full cluster processed; priority = TPM resolution |
| `little_women_cluster` | **Title alias miss** — brand `westside movieclub` vs film `Маленькие женщины` | Source-ownership pollution (one TG post → 2 active events) | Two fixes needed: (1) single-source-URL owner guard, (2) LLM title-alias resolution with shared ticket proof |
| `vistynets_fair_duplicate` | **City extraction noise** — `Гусев` vs `Калининград` for same venue `Gumbinnen` | Cross-source exact-match not applied (different source entry order?) | `cross_source_exact_match` should already cover this; verify rule ordering |
| `zoo_reptile_vs_generic_excursion_false_friend` | **Correctly separated** — not a failure | N/A | Use as regression control; no action needed |

### Конкурентный тезис: matryoshka — это НЕ "LLM layer miss", это matching pipeline bug

Я категорически не согласен с classification "LLM layer miss" для matryoshka. Вот что произошло:

1. Source `23524` содержит текст про "Путешествие Матрёшки"
2. Event `2849` уже существует с title "Путешествие матрешки"
3. НО matching pipeline вместо этого merge'нул `23524` в `2746` ("Акция «Музей. Музы и творцы»")

Это значит: pipeline видел `2849` в shortlist, но ПРЕДПОЧЁЛ `2746`. Или: `2849` вообще НЕ попал в shortlist, а `2746` попал из-за shared source/venue signals.

**Root cause hypothesis**: `23524` и `2746` share the same source owner (`-9118984`). Если matching weighted `same_source_url_owner` как strong merge signal без проверки title compatibility → pipeline выбирает wrong sibling.

Это **pipeline bug в shortlist/scoring**, не в LLM, не в deterministic rules.

---

## 2. `matryoshka` False-Merge Containment

### Почему `23524` мог уйти в `2746`, а не в `2849`

**Гипотеза 1 (наиболее вероятная)**: timing + source_url_owner bias.

- `2746` уже существовал в БД ДО создания `2849`
- `2849` был создан из `23492` (первый matryoshka post)
- Когда `23524` (второй matryoshka post) обрабатывался, shortlist мог содержать ОБА `2746` и `2849`
- Если scorer weighted `source_url_owner_match` (оба от `-9118984`) без strict title check — `2746` мог получить higher score, потому что у него БОЛЬШЕ accumulated source history

**Гипотеза 2 (менее вероятная)**: `2849` ещё не был committed в БД к моменту обработки `23524` (race condition в batch processing).

### Самый узкий safe mitigation

**Не нужен новый deterministic rule.** Нужен pipeline guard:

```python
def _is_cross_sibling_false_merge(candidate_event, new_event_data):
    """
    Guard: if candidate came from multi-event source
    AND new event title doesn't match candidate title
    AND new event title DOES match another event from same source
    → block merge into this candidate.
    """
    if not candidate_event.sources:
        return False
    
    # Check if candidate is from multi-event source
    source_urls = [s.source_url for s in candidate_event.sources]
    sibling_events = get_events_from_same_source_urls(source_urls)
    
    if len(sibling_events) < 2:
        return False  # Not multi-event
    
    # Check title compatibility
    candidate_title_match = titles_look_related(
        candidate_event.title, new_event_data.title
    )
    
    if candidate_title_match:
        return False  # Title matches → legitimate merge
    
    # Check if any sibling has better title match
    for sibling in sibling_events:
        if sibling.id == candidate_event.id:
            continue
        if titles_look_related(sibling.title, new_event_data.title):
            return True  # Better sibling exists → block this merge
    
    return False
```

**Где это живёт**: в LLM action policy, НЕ в deterministic identity rules. Это post-shortlist, pre-merge guard.

**Почему не deterministic rule**: потому что это не identity resolution. Это "правильный target selection" — отдельная задача от "это тот же event?".

**Не ломает must_merge**: matryoshka [2725,2726] (dry-run pair) оба имеют title_related=True ("Фабрика матрёшки" / "Выставка матрёшки" / "Путешествие матрёшки"). Guard fires только при title_related=False + better sibling exists.

### Более radical alternative: source_url_owner shortlist boost

Вместо guard'а — изменить shortlist scoring:

- Если `source_url_owner` match, BUT `title_related=False` → не BOOST shortlist, а проверить siblings
- Если sibling с `title_related=True` существует → redirect candidate to that sibling

Это решает проблему раньше (на этапе shortlist, не post-merge), но требует более глубокого рефакторинга.

**Рекомендация**: implement cross-sibling guard first (post-merge, narrow). Если works → consider shortlist redirect later.

---

## 3. `oncologists` Policy

### Какие пары должны merge

На дату `2026-03-23` (Светлогорск):
- `2850` (campaign post, venue="Научная библиотека") ← venue extraction bug
- `2853` (city-specific post, venue="Светлогорск")
- `2855` (admin post, venue="Детская поликлиника, Светлогорск")

**Merge targets**: `2850 + 2853 + 2855` → одно событие. Все три описывают один и тот же приём онколога в Светлогорске 23 марта.

### Какие должны оставаться separate

- `2851` (campaign post, Калининград, 24 марта) — ОТДЕЛЬНОЕ событие
- `2852` (campaign post, Зеленоградск, 26 марта) — ОТДЕЛЬНОЕ событие
- `2854` (city-specific, Зеленоградск, 26 марта) — merge с `2852`

**Принцип**: campaign multi-city post порождает N child events. Child events с РАЗНЫМИ dates/cities — SEPARATE. Child events на ОДНУ date+city — MERGE.

### Какие сигналы можно доверять

| Signal | Trustworthy? | Comment |
|---|---|---|
| Same registration link | **ДА, strong** | Если один и тот же doctor/campaign link → same event |
| Same doctors listed | **Средне** | Campaign может менять состав. Useful as hint, not proof |
| Same date | **ДА, necessary** | Different date = different event, always |
| City-level match | **ДА, necessary** | "Светлогорск" vs "Калининград" = DIFFERENT events |
| City vs venue mismatch | **Noise** | "Научная библиотека" from campaign post = extraction bug |
| Same campaign text | **Weak** | Campaign text covers ALL cities; not specific to one |

### Что safe для deterministic vs LLM

**Deterministic**: `same_date + same_city + same_registration_link → merge` — это safe и narrow enough.

**LLM only**: `same_date + same_city + different_registration_link + similar_title` — requires semantic judgment.

**Neither**: `different_date OR different_city → always separate` — hard rule, no LLM override.

### Venue leakage `Научная библиотека`

Это **upstream extraction noise**, не identity signal. Campaign post text mentions multiple cities and venues. Extractor picked "Научная библиотека" because it was the first concrete venue name in text.

**Policy**: when `venue_mismatch` correlates with `campaign_style_source` (single source → multiple dates/cities), default to `uncertain`, NOT `different`. The venue is unreliable for this source class.

**But**: не делать из этого broad rule "venue can be ignored for campaign posts". Only relax venue weight when extraction clearly pulled from multi-city text.

---

## 4. `makovetsky` Policy After Live Run

### Verdict: это acceptable duplicate territory, но с upgrade path

**Не blocker для rollout.** Но live confirmation повышает priority.

### Upgrade: same-poster LLM policy

**Текущее**: `makovetsky` → LLM → `softlink` при `same_event` (brand-vs-item class).

**Upgrade proposal**: если LLM видит `same_event` AND poster_overlap=True AND same_slot → `auto_merge` вместо `softlink`.

**Обоснование**: shared SPECIFIC poster (не generic venue banner) — сильный physical evidence. Два автора одного and того же content'а не делят poster случайно. В комбинации с `same_event` от LLM — safe auto_merge.

**Guard**: poster_overlap must be TRUE (hash-based), not inferred. И оба events должны быть на same date+venue.

**Что НЕ менять**: если poster_overlap=False — стандартный brand-vs-item = `softlink`. Не возвращать broad same-poster merge rule.

---

## 5. `little_women` Policy After Prod `/daily`

### Primary failure: combination of title alias miss + source-ownership pollution

1. **Title alias miss** (primary): `Сто семнадцатый показ киноклуба westside movieclub` vs `Маленькие женщины` → `title_related=False`. Deterministic cannot solve this — it's semantic.

2. **Source-ownership pollution** (compounding): `t.me/signalkld/9892` owns BOTH `2815` and `2816`. Single source → 2 active events. This is a pipeline bug, not identity resolution.

### Какие сигналы identity-level

| Signal | Identity strength |
|---|---|
| Same slot (date+time) | **Necessary but not sufficient** |
| Same venue | **Necessary but not sufficient** |
| Same ticket link | **STRONG identity proof** — `signalcommunity.timepad.ru/event/3858105` у всех 4 events |
| Film-specific facts in text | **Strong LLM hint** — "Little Women", "Грета Гервиг", "Даня Ященко" |
| Brand-title vs film-title | **LLM-only** — deterministic cannot resolve |

### Что допустимо поднимать в deterministic

**ДА**: `same_specific_ticket_link + same_slot + same_venue → merge`. Это уже покрыто `specific_ticket_same_slot`. Если этот rule не fires — значит ticket normalization не работает (проверить: `signalcommunity.timepad.ru/event/3858105` vs `signalcommunity.timepad.ru/event/3858105/` — trailing slash!).

**НЕТ**: brand-title alias merge. `westside movieclub` → `Маленькие женщины` — это за пределами any safe deterministic rule.

### Рекомендация

1. **Immediate**: fix single-source-URL owner guard. One `t.me/signalkld/9892` → one event, not two.
2. **Deterministic**: verify `specific_ticket_same_slot` handles trailing slash normalization. Если нет → fix.
3. **LLM**: для пар с `title_related=False` + `same_ticket + same_slot` → LLM с strong hint `shared_ticket_proof`.
4. **Do NOT**: create broad movie-club alias rule.

---

## 6. `vistynets_fair_duplicate` Short Verdict

### Deterministic-safe merge candidate: **ДА.**

Этот кейс ДОЛЖЕН быть resolved by `cross_source_exact_match`:
- `title_exact=True` ("Ярмарка «Вкусов Виштынецкой возвышенности»")
- `same_date=True` (2026-03-07)
- `same_time=True` (12:00)
- `venue_match` — нужно проверить: "Дизайн-резиденция Gumbinnen, Ленина 29, Гусев" vs "Дизайн-резиденция Gumbinnen"

**Если `venue_match=True`** (containment match): rule fires → merge. Done.

**Если `venue_match=False`** (full string mismatch): это venue normalization bug — `Gumbinnen` с адресом vs `Gumbinnen` без адреса. Fix venue containment matching.

### City mismatch `Гусев` vs `Калининград`

Это **extraction noise**, не evidence of separate events. Venue "Дизайн-резиденция Gumbinnen" находится в Гусеве. Extractor для одного source достал city из адреса (Гусев), для другого — из broader region (Калининград).

**НЕ делать broad rule** "city mismatch can be ignored". Но: если `title_exact + same_slot + venue_containment_match` — city mismatch не должен block merge.

Это уже так работает в `cross_source_exact_match`: rule не проверяет city, только venue. ✅

---

## 7. `zoo_reptile_vs_generic_excursion_false_friend` Short Verdict

### Deterministic `different`: **ДА, уже достаточно.**

На deterministic уровне эта пара уже `different`:
- `title_exact=False` ("Экскурсия «Тайны панциря и чешуи, или О тех, кого не любят»" ≠ "Экскурсии в зоопарке Калининграда")
- `title_related` — possibly True (оба содержат "зоопарк" / "экскурсия"), но titles describe different activities

**Если даже title_related=True**: different ticket links (`cUYxJb` ≠ `cVbnez`) + different title specificity → `gray`, not `merge`. Ни один existing rule fires merge на этой паре.

### Anti-merge сигналы

| Signal | Anti-merge strength |
|---|---|
| Different registration links | **Strong** — different ticket = different event |
| Specific guided topic vs generic visit | **Strong for LLM** — deterministic can't assess |
| Same venue family | **Weak pro-merge** — zoo hosts many parallel activities |

### Как не сломать legit zoo duplicates

Ключевой различитель: **ticket link**. Legit zoo duplicate = same ticket link. False friend = different ticket link.

Уже safe: `specific_ticket_same_slot` catches legit duplicates with shared ticket. Zoo false-friend has different tickets → not caught. ✅

**НЕ нужен новый rule.** Existing rules + LLM already handle this correctly.

---

## 8. TPM-Aware Next Step

### Один приоритетный шаг: **deterministic pre-filter с early exit**

**Проблема**: 25 processed rows, 11 deferred. 44% failure rate на TPM. Wall-clock 6267s для 20 rows = ~5 min per row.

**Root cause**: каждый VK row проходит через LLM ВНЕ ЗАВИСИМОСТИ от того, нужен ли LLM. Для rows, где deterministic уже достаточен (same_post_exact_title, cross_source_exact_match, etc.) — LLM вызов = waste of TPM.

**Proposed next step**: deterministic pre-filter BEFORE LLM.

```python
def should_skip_llm_for_matching(pair_signals: dict) -> tuple[bool, str]:
    """
    If deterministic verdict is confident (merge or different),
    skip LLM pairwise call entirely.
    Returns (skip_llm, deterministic_verdict).
    """
    # Rules 1-8 from Stage 04B
    verdict = run_deterministic_rules(pair_signals)
    
    if verdict in ("merge", "different"):
        return True, verdict
    
    # Gray → needs LLM
    return False, "gray"
```

**TPM savings estimate**:

На текущем casepack: 16 must-merge resolved + 34 must-not-merge resolved = 50 out of 72 pairs. Если ~70% of real-world pairs are deterministic-resolvable → ~70% TPM saving on matching phase.

**Risk**: zero. Deterministic rules are already validated at 0 false merge / 0 false different. If deterministic says "merge" — LLM would agree anyway. If deterministic says "different" — LLM should not override.

**Что это НЕ меняет**: extraction still needs LLM, generation still needs LLM. Но matching is the most TPM-wasteful phase because it's per-pair, not per-event.

**Relationship to compact pairwise baseline**: fully compatible. For gray pairs, compact pairwise LLM runs exactly as before. For resolved pairs, it doesn't run at all.

---

## 9. Last Blockers

### Реальный blocker для rollout: **ОДИН.**

**`matryoshka` cross-sibling false merge** — это live proven false merge at pipeline level. Не hypothetical, не dry-run, а real runtime event.

Нельзя ship deterministic subset + LLM layer без cross-sibling guard. If unguarded, multi-event museum sources (which are THE most common source type in current data) will produce false merges at pipeline level.

### Что НЕ blocker

| Item | Why not blocker |
|---|---|
| `oncologists` duplicate miss | Acceptable: duplicate ≤ false merge |
| `makovetsky` duplicate miss | Acceptable: same reason |
| `hudozhnitsy` partial coverage | Operational, not design flaw |
| `little_women` cluster | Source-ownership bug + title alias → separate tracks, не blocking deterministic |
| `vistynets` duplicate | Should be caught by existing rule; verify |
| `zoo` false friend | Already handled correctly |
| TPM pressure | Operational, reduced by pre-filter |

### Unblock path

1. Implement cross-sibling guard (post-merge, narrow)
2. Verify `vistynets` is caught by `cross_source_exact_match`
3. Ship

**Timeline estimate**: cross-sibling guard = ~2-3 hours implementation + 1 hour validation.

---

## Конкурентные дополнения (не запрошены, но считаю важными)

### A. `single_source_url_owner_guard` — нужен ПАРАЛЛЕЛЬНО с cross-sibling

Live run показал ДВА случая, где один source URL создал >1 active event:
- `little_women`: `t.me/signalkld/9892` → events 2815 + 2816
- `prazdnik`: `t.me/terkatalk/4529` → events 2802 + 2803

Это **pipeline bug**, не identity resolution. Fix:

```python
# Before creating new event from single-event source:
existing = get_active_events_by_source_url(source_url)
if existing and not is_multi_event_source(source_url):
    # Merge into existing, do not create new
    return merge_or_update(existing[0], new_data)
```

Это отдельный fix от cross-sibling guard, но equally important. Оба — pipeline-level, не rule-level.

### B. `matryoshka` ≠ `museum_holiday` — важно не путать

`museum_holiday_program_multi_child` уже resolved by `multi_event_source_blocker` (Rule 7). Но `matryoshka` — ДРУГОЙ класс:

- `museum_holiday`: 1 source URL, 3 genuinely different events → must_not_merge → blocker fires ✅
- `matryoshka`: 2 source URLs, 2 genuinely same events → must_merge → blocker must NOT fire

Verify: `multi_event_source_blocker` does NOT fire on matryoshka pair [2725,2726]:
- `source_url_owner_pair_max` для matryoshka pair < 4 (different source posts)
- → rule doesn't fire → safe ✅

Cross-sibling guard and multi_event_source_blocker are COMPLEMENTARY, not conflicting.

### C. Ticket link normalization audit

Three cases in casebook involve trailing slash or subdomain variations:
- `signalcommunity.timepad.ru/event/3858105` vs `signalcommunity.timepad.ru/event/3858105/`
- `filippvoronin.ru/` vs `ippvoronin.ru` (typo/truncation)
- `filarmonia39.ru/?event=2735` vs `filarmonia39.ru`

If `specific_ticket_same_slot` doesn't normalize trailing slashes → it misses `little_women`. Quick fix, high ROI.
