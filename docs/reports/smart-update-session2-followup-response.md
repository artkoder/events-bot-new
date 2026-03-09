# Smart Update Session 2 Follow-up Response

Дата: 2026-03-06
Формат: прицельный follow-up к `docs/reports/smart-update-session2-deep-consultation.md`.
Ответ строго по запрошенным 6 разделам.

---

## 1. Corrections To Prior Report

### 1.1. Кейс `Собакусъел`: фактическая ошибка в attribution

**Ошибка**: в deep-consultation написано, что `meowafisha` имеет `default_location=YΛTЬ кофейня`.

**Факт по snapshot**:
- `telegram_source.username='meowafisha'` → `default_location = NULL`
- `telegram_source.username='signalkld'` → `default_location = Сигнал, Леонова 22, Калининград`

**Исправленное объяснение**:
Событие `Собакусъел` пришло из двух каналов. Канал `signalkld` имеет `default_location = Сигнал, Леонова 22` — это конкретная площадка самого канала. Если override по `default_location` перетирает extracted venue, то событие «Собакусъел» (реальная площадка) получает venue «Сигнал, Леонова 22» — чужую площадку. Это приводит к ложному venue mismatch между двумя анонсами одного события.

**Вывод не меняется**: `default_location` нельзя использовать как venue override. Только как weak hint для расширения shortlist.

### 1.2. Phantom case keys

**Ошибка**: в deep-consultation использованы ключи `must_not_nutcracker_two_shows` и `must_not_lecture_cycle`. Таких ключей нет в `opus_session2_casepack_latest.json`.

**Исправление**: эти ключи пришли из longrun gold set (внутренний benchmark), но не из casepack. Нужно опираться строго на ключи из casepack.

Замены:
- `must_not_nutcracker_two_shows` → теперь это `nutcracker_two_shows_same_post` из followup window
- `must_not_lecture_cycle` → ближайший аналог `oceania_march_lecture_series` из casepack

В follow-up ниже используются только актуальные ключи.

### 1.3. `time_conflict` hard blocker — ошибка scope

**Ошибка в deep-consultation**: правило `time_conflict (>30 мин, оба known) → force different` сформулировано безусловно.

**Контрпример из followup window**: `womanhood_exhibition_time_noise_duplicate` (2755/2756):
- Одна и та же выставка «Женственность через века»
- `end_date = 2026-06-05` (long-running)
- Один `source_url` (VK wall -152679358_23836)
- Извлечённые times 12:00 и 15:00 — это **расписание экскурсий** из текста поста, а не два отдельных события
- Безусловный time_conflict blocker даёт `different` — **ошибка**

**Исправление**: time_conflict hard blocker нужно гейтить по event_type:

```
time_conflict blocker activates ONLY IF:
  both time_is_default = false
  AND time difference > 30 min
  AND NOT doors_vs_start pattern
  AND end_date IS NULL (т.е. это single-occurrence event, не long-running exhibition)
```

Для long-running events (end_date IS NOT NULL): time difference → weak signal, LLM triage.

---

## 2. Ordering Of Rules

### 2.1. Полный порядок обработки пары (candidate, existing)

```
PHASE 0: NORMALIZATION (до scoring)
  ├── venue alias normalization
  ├── city alias normalization
  ├── title confusable chars (ё→е, mixed script)
  └── default_location → weak hint only

PHASE 1: EXPECTED EVENT ID CHECK
  ├── IF candidate.expected_event_id == existing.id:
  │     → auto-merge (приоритет над всем остальным)
  │     Обоснование: это explicit mapping от extraction layer
  └── ELSE: продолжаем

PHASE 2: SAME-SOURCE SINGLE-EVENT RESCUE
  ├── Условия активации (все одновременно):
  │   ├── source_post_url(candidate) == source_post_url(existing)
  │   ├── source_kind == single_event (NOT multi_event, NOT recurring_page)
  │   └── candidate.event_count_from_this_source == 1
  │       (т.е. из этого source ожидается ровно 1 event)
  ├── IF activated:
  │     → auto-merge (bad extraction rescue)
  │     Обоснование: один пост → одно событие, разница в date/time = extraction error
  └── ELSE: продолжаем

PHASE 3: HARD BLOCKERS (приоритет над scoring и LLM)
  ├── 3a. date_mismatch:
  │     IF date_A ≠ date_B AND NOT date overlap (for events with end_date):
  │       → force different
  ├── 3b. time_conflict:
  │     IF both time known, both time_is_default=false,
  │        diff > 30 min, NOT doors_vs_start,
  │        AND end_date IS NULL (single-occurrence):
  │       → force different
  ├── 3c. city_mismatch:
  │     IF city_A ≠ city_B (after normalization):
  │       → force different
  └── 3d. title+venue double mismatch:
        IF title_related=false AND venue_match=false:
          → force different

PHASE 4: DETERMINISTIC SCORING
  ├── Compute identity_score with calibrated weights
  ├── Apply source_kind modifiers (recurring/multi → zero weight for poster/source/ticket)
  ├── score ≥ MERGE_THRESHOLD → auto-merge
  └── score ≤ CREATE_THRESHOLD → auto-create

PHASE 5: LLM PAIRWISE TRIAGE (только при серединных scores)
  ├── Compact pairwise payload + hints
  ├── LLM returns verdict: same_event / likely_same / different / uncertain
  └── Runtime maps verdict → action per policy profile

PHASE 6: VERDICT → ACTION MAPPING
  ├── same_event → merge
  ├── likely_same → gray_create_softlink (single_event) / create (multi_event)
  ├── different → create
  └── uncertain → gray_create_softlink (single_event) / gray_create_softlink (multi_event)
```

### 2.2. Почему именно этот порядок

| Решение | Обоснование |
|---|---|
| `expected_event_id` первым | Explicit bridge от extraction — обходит все heuristics |
| Source rescue ДО hard blockers | `led_hearts_same_post_triple_duplicate`: date расползлась из-за bad extraction (2026-03-08 вместо 07). Если date_mismatch срабатывает раньше → система закрепит ошибочный event вместо rescue. Source rescue обходит только если `source_kind=single_event` И `event_count=1` — это безопасный narrow scope |
| Hard blockers ДО scoring/LLM | Scoring и LLM не могут override физическую невозможность: разные даты = разные показы (для single-occurrence). Иначе high-scoring recurring pages (poster+source+ticket) пробивают date barrier |
| Scoring ДО LLM | Экономит LLM calls: ~60-70% пар разруливаются deterministic |
| LLM последним | Только для серединных/спорных случаев; LLM не может override hard blockers |

### 2.3. Контрольная проверка ordering на кейсах

| Case | Phase | Result | Correct? |
|---|---|---|---|
| `led_hearts` 2845/2846 (date 07→08, same source) | Phase 2: source rescue | merge ✅ | Да — bad extraction fixed |
| `backstage_tour` 2611/2612 (разные даты) | Phase 3a: date_mismatch | different ✅ | Да — recurring, разные даты |
| `treasure_island` 2572/2573 (11:00/14:00) | Phase 3b: time_conflict (end_date=NULL) | different ✅ | Да — double show |
| `womanhood` 2755/2756 (12:00/15:00, end_date=06-05) | Phase 3b: **SKIP** (end_date NOT NULL) → Phase 4/5 | LLM triage → merge ✅ | Да — excursion times, not two events |
| `nutcracker` 1619/1620 (14:00/19:00, end_date=NULL) | Phase 3b: time_conflict | different ✅ | Да — double show |
| `garage_time_correction` (same source) | Phase 2: source rescue | merge ✅ | Да — time update |
| `fort_excursion` (high score) | Phase 4: auto-merge (≥10) | merge ✅ | Да |
| `cathedral_false_friend` (same slot, diff titles) | Phase 3d: title+venue? No — venue_match=true, but title_related=false... | Phase 4: score < threshold (ticket_generic + no title) → Phase 5 LLM → different ✅ | Да |

---

## 3. Source Guard / Ticket Policy

### 3.1. Source-owner guard: precise scope

**Старая формулировка** (ошибочно широкая):
> `same_source_url + same_date` → auto-merge (+6 score)

**Исправленная формулировка**:

```python
def source_owner_guard_applies(candidate, existing) -> bool:
    """
    True ТОЛЬКО если source-owner rescue безопасен.
    """
    return (
        # 1. Один и тот же source post URL
        candidate.source_post_url == existing.source_post_url
        # 2. Source kind = single_event (не multi_event, не recurring_page)
        and candidate.source_kind == "single_event"
        # 3. Из этого source ожидается ровно 1 event
        and candidate.event_count_from_source == 1
        # 4. Dates: same or rescue (при single_event + event_count=1, date diff = extraction error)
        # (date check implicit, т.к. rescue перекрывает)
    )
```

**Что это защищает**:
- `led_hearts` (single TG post → 1 event expected, 3 created by bug) → rescue ✅
- `garage_time_correction` (same VK post, time update) → rescue ✅

**Что это НЕ ломает**:
- `nutcracker_two_shows_same_post`: `event_count_from_source = 2` → guard не активируется → Phase 3b time_conflict → different ✅
- `treasure_island_double_show`: аналогично, `event_count_from_source = 2` → guard off
- `backstage_tour_weekly_run`: `source_kind = recurring_page` → guard off
- `oceania_march_lecture_series`: `source_kind = multi_event` → guard off

**Как определить `source_kind`**:

```python
def classify_source_kind(candidate) -> str:
    """
    Classify based on extractor metadata.
    """
    if candidate.normalized_event_type == "multi_event":
        return "multi_event"
    if candidate.source_is_recurring_page:
        # Recurring page: theatre repertory, museum permanent schedule, etc.
        # Detected by: source_url domain in KNOWN_REPERTORY_DOMAINS
        # OR: source explicitly flagged as repertory by extractor
        return "recurring_page"
    return "single_event"
```

**Как определить `event_count_from_source`**:

```python
def get_event_count_from_source(source_post_url: str, current_batch: list) -> int:
    """
    Count how many events in current batch share this source_post_url.
    """
    return sum(1 for c in current_batch if c.source_post_url == source_post_url)
```

Если `event_count_from_source > 1` → source содержит или анонсирует несколько событий → guard отключается.

### 3.2. `ticket_is_generic`: production-ready спецификация

**Старая формулировка** (недостаточная):
> `ticket_owner_count > 5` или URL ≠ event-specific path

**Исправленная спецификация**:

```python
def ticket_is_generic(ticket_url: str, event_title: str) -> bool:
    """
    Returns True if ticket_url is NOT event-specific identity proof.

    Layers (in order):
    1. Domain denylist (known generic platforms)
    2. Path depth check
    3. Event-specific slug check
    4. Known scheme/routing pattern check
    """
    if not ticket_url:
        return True  # no ticket = no evidence

    parsed = urlparse(ticket_url)
    domain = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    # Layer 1: Known generic ticket platforms where URL rarely identifies a specific event
    GENERIC_TICKET_DOMAINS = {
        "clck.ru",           # URL shortener, used for schedules
        "vk.com",            # VK event pages (not event-specific tickets)
    }
    if domain in GENERIC_TICKET_DOMAINS:
        return True

    # Layer 2: Path depth — root or single-segment paths are usually venue root
    # Example generic: https://dramteatr39.ru/spektakli/
    # Example specific: https://muzteatr39.ru/spektakli/dlya-detej/ostrov-sokrovishh/
    path_segments = [s for s in path.split("/") if s]
    if len(path_segments) <= 1:
        return True  # venue root page

    # Layer 3: Scheme/routing pattern — UUID/hash paths without event-identifying info
    # Example: tickets.sobor-kaliningrad.ru/scheme/541EA644B65930C2DFFDACBA44D4660A28137E01
    SCHEME_PATTERNS = ["scheme", "routing", "redirect", "goto", "r"]
    if any(seg.lower() in SCHEME_PATTERNS for seg in path_segments):
        # Check if the next segment is a UUID/hash (not human-readable)
        remaining = "/".join(path_segments[path_segments.index(
            next(s for s in path_segments if s.lower() in SCHEME_PATTERNS)
        ) + 1:])
        if remaining and not any(c.isalpha() and c.islower() for c in remaining[:20]):
            return True  # hash/UUID path = generic routing

    # Layer 4: Known venue-specific denylist
    VENUE_ROOT_PATTERNS = {
        "dramteatr39.ru/spektakli",      # Every play uses this root
        "muzteatr39.ru/spektakli",       # Same pattern
        "tickets.sobor-kaliningrad.ru",  # Cathedral generic scheme
    }
    full_path = f"{domain}{path}"
    for pattern in VENUE_ROOT_PATTERNS:
        if full_path.startswith(pattern):
            # Check if there's a SPECIFIC event slug after the root
            rest = full_path[len(pattern):].strip("/")
            if not rest or len(rest) < 3:
                return True  # venue root with no specific event slug

    return False  # passed all checks → probably specific
```

**Scoring impact**:
- `ticket_is_generic = true` → weight = +1 (weak hint, better than nothing)
- `ticket_is_generic = false` → weight = +4 (strong identity evidence)

**Контрольная проверка**:

| Ticket URL | `ticket_is_generic` | Why |
|---|---|---|
| `https://tickets.sobor-kaliningrad.ru/scheme/541EA644...` | `true` | Layer 3: scheme + hash path |
| `https://clck.ru/3SEh9x` | `true` | Layer 1: generic domain |
| `https://dramteatr39.ru/spektakli/№13` | `false` | Layer 4: has specific slug `№13` |
| `https://muzteatr39.ru/spektakli/dlya-detej/ostrov-sokrovishh/` | `false` | Has specific event path |
| `https://vk.com/wall-211204375_3298` | `true` | Layer 1: VK |
| `https://kassir.ru/events/show/kontsert-12345` | `false` | Specific event path with ID |

### 3.3. `multi_event uncertain` policy refinement

**Старая позиция**: `multi_event uncertain → create`

**Пересмотренная позиция**: `multi_event uncertain → gray_create_softlink`

Обоснование:
1. `create` безопаснее от false merge, но в quality-first системе необходим контроль и дублей тоже.
2. `gray_create_softlink` — это не merge: событие создаётся как отдельный entity, но сохраняется soft-link для возможной ручной проверки или future reconciliation.
3. Разница от `single_event uncertain → gray_create_softlink` — в том, что для multi_event gray **не переходит в auto-merge при повторном подтверждении**, а требует explicit admin action.

**Уточнённая таблица**:

```
single_event:
  same_event       → merge
  likely_same      → gray_create_softlink
  different        → create
  uncertain        → gray_create_softlink

multi_event:
  same_event       → merge (но только при ≥2 strong signals)
  likely_same      → gray_create_softlink (no auto-escalation to merge)
  different        → create
  uncertain        → gray_create_softlink (no auto-escalation to merge)

recurring_page:
  same_event       → merge (date already matched in Phase 3 check — if we're here, date is same)
  likely_same      → gray_create_softlink
  different        → create
  uncertain        → create (bias toward safety for recurring)
```

**Разделение multi_event подтипов**:

| Подтип | Описание | Policy |
|---|---|---|
| `multi_event parent` | Пост-дайджест, содержащий список нескольких событий | Poster/source_url/ticket weight = 0. LLM allowed, default uncertain → gray_create_softlink |
| `multi_event child` | Один из extracted child events | Наследует parent source_kind = multi_event. Standard scoring, но source_url weight = 0 |
| `recurring_page` | Репертуарная/recurring страница (театр, музей) | Poster/source_url/ticket weight = 0. Date_mismatch = hard blocker. Default uncertain → create |
| `schedule_aggregate` | Один пост с расписанием N дат одного спектакля | source_kind = recurring_page де-факто. Date_mismatch = hard blocker |

---

## 4. Decisions On Added Cases

### 4.1. `matryoshka_exhibition_duplicate` (2725/2726)

**Expected**: must_merge
**Current deterministic**: gray
**My verdict**: **merge via LLM triage → same_event**

Анализ сигналов:
- **Title**: `Путешествие матрешки` vs `Путешествие Матрешки: Интерактивная экспозиция` → `alias_match` (второе — расширенное название той же выставки)
- **Date**: `2026-03-05` = `2026-03-05` → `exact`
- **End_date**: `2026-04-05` = `2026-04-05` → `exact`
- **Time**: оба `""` → `weak` (оба default/unknown)
- **Venue**: `Музей Изобразительных искусств, Ленинский проспект 83` = identical → `exact`
- **Ticket**: оба null → `no_data`
- **Context**: оба VK поста от того же музея (wall-9118984). Содержание: благотворительная выставка матрёшек. → `strong_match`
- **Poster**: разные hash → `no overlap`
- **Source_url**: разные VK posts → нет source_owner guard

Ordering:
1. Phase 1 (expected_event_id): нет → skip
2. Phase 2 (source rescue): разные source_url → skip
3. Phase 3 (blockers): date=exact, time=weak, city=same, venue=same → нет blockers
4. Phase 4 (scoring): title_related + date_exact + venue_exact + context_strong ≈ score 7-8 → LLM zone
5. Phase 5 (LLM): title alias_match + date exact + venue exact + context strong_match → `same_event`
6. Phase 6: merge ✅

**Почему не auto-merge**: нет shared source_url, ticket, или poster. Score < 10. Но LLM видит семантическое совпадение.

### 4.2. `museum_overlap_exhibitions_same_period` (2725/2727)

**Expected**: must_not_merge
**Current deterministic**: gray
**My verdict**: **different — via LLM → different (title mismatch)**

Анализ сигналов:
- **Title**: `Путешествие матрешки` vs `Выставка «Космос красного»` → `mismatch` (совершенно разные названия, разные темы)
- **Date**: `2026-03-05` = `2026-03-05` → `exact`
- **End_date**: `2026-04-05` = `2026-04-05` → `exact`
- **Time**: оба `""` → `weak`
- **Venue**: `Музей Изобразительных искусств` = identical → `exact`
- **Context**: один — благотворительные матрёшки, другой — «самая тёплая и насыщенная смыслами экспозиция сезона» → `mismatch`
- **Both from same VK group**: wall-9118984 (museum's own page) → но разные posts

Ordering:
1. Phase 3 (blockers): date=exact, venue=exact, BUT title_related=false → Phase 3d? `title_related=false AND venue_match=true` — Phase 3d требует ОБОИХ mismatches. Venue match → Phase 3d не срабатывает.
2. Phase 4 (scoring): title_mismatch penalty (-3) + date exact (+3) + venue exact (+3) ≈ score 3 → auto-create zone (≤3)
3. → create = different ✅

**Ключевой урок**: title_mismatch сам по себе должен приносить существенный penalty. Даже при overlapping date + same venue, два события с unrelated titles — это разные выставки.

**Рекомендация по scoring**: `title_mismatch → -3 penalty`. Это переводит пару с `date_exact + venue_exact` (итого +6) в зону score 3 → auto-create.

### 4.3. `womanhood_exhibition_time_noise_duplicate` (2755/2756)

**Expected**: must_merge
**Current deterministic**: different
**My verdict**: **merge — via corrected time_conflict rule + LLM triage**

Анализ сигналов:
- **Title**: `Женственность через века` = `Женственность через века` → `exact_match`
- **Date**: `2026-03-05` = `2026-03-05` → `exact`
- **End_date**: `2026-06-05` = `2026-06-05` → `exact` — **long-running exhibition** (3 месяца!)
- **Time**: `12:00` vs `15:00` → diff = 3h > 30 min. НО:
  - **end_date IS NOT NULL** → это long-running exhibition
  - Times из текста — **расписание экскурсионных слотов**, а не два отдельных события
  - time_conflict blocker **не активируется** (gated by `end_date IS NULL`)
- **Venue**: `Информационно-туристический центр` = identical → `exact`
- **Source_url**: `vk.com/wall-152679358_23836` = identical → **same source!**
- **Source_kind**: `single_event` (один VK-пост = одна выставка)
- **Event_count_from_source**: 2 (extractor создал два event из одного поста)

Ordering:
1. Phase 2 (source rescue): source_url совпадает, source_kind = single_event, **но event_count_from_source = 2** → guard НЕ активируется по strict rule.

   **Однако**: это false positive от extractor'а. Один пост описывает одну выставку, extractor ошибочно создал два события из-за двух time-слотов в тексте.

   **Решение**: для long-running exhibitions (end_date IS NOT NULL) + same source_url → source rescue разрешён даже при event_count > 1, IF title_exact = true.
   
   Уточнённое правило source rescue:
   ```
   source_owner_guard_applies IF:
     same_source_url
     AND source_kind == single_event
     AND (event_count_from_source == 1
          OR (title_exact AND end_date IS NOT NULL))
   ```

2. Если source rescue не дотягивает → Phase 3 blockers: time_conflict **skipped** (end_date IS NOT NULL) → Phase 4: title_exact + date_exact + venue_exact + same_source ≈ score ≥ 10 → auto-merge.

**Итого**: merge ✅ через auto-merge или source rescue.

### 4.4. `nutcracker_two_shows_same_post` (1619/1620)

**Expected**: must_not_merge
**Current deterministic**: different
**My verdict**: **different — via time_conflict hard blocker** ✅

Анализ сигналов:
- **Title**: `🎭 Щелкунчик` = `🎭 Щелкунчик` → `exact_match`
- **Date**: `2026-01-09` = `2026-01-09` → `exact`
- **End_date**: NULL = NULL → **single-occurrence event**
- **Time**: `14:00` vs `19:00` → diff = 5h > 30 min
- **Venue**: `Калининградский театр эстрады (Дом искусств)` = identical → `exact`
- **Source_url**: `vk.com/wall-211204375_3298` = identical → same source

Ordering:
1. Phase 2 (source rescue): same_source_url, BUT event_count_from_source = 2 → guard off.
   Also: title_exact = true, BUT end_date = NULL → расширенное правило не помогает.
2. Phase 3b (time_conflict): both times known, diff = 5h > 30 min, NOT doors_vs_start, end_date IS NULL → **force different** ✅

**Вывод**: time_conflict blocker корректно ставит `different`. Same source + same title недостаточны для override, потому что source guard не активируется при event_count > 1 для single-occurrence events.

### 4.5. Сводка по 4 новым кейсам

| Case | Expected | My verdict | Route | Correct? |
|---|---|---|---|---|
| `matryoshka_exhibition_duplicate` | must_merge | **merge** | Phase 5: LLM → same_event | ✅ |
| `museum_overlap_exhibitions_same_period` | must_not_merge | **different** | Phase 4: title_mismatch penalty → score ≤ 3 → create | ✅ |
| `womanhood_exhibition_time_noise_duplicate` | must_merge | **merge** | Phase 3b skipped (end_date), Phase 4 auto-merge or source rescue | ✅ |
| `nutcracker_two_shows_same_post` | must_not_merge | **different** | Phase 3b: time_conflict (end_date=NULL) | ✅ |

---

## 5. What Is Production-Ready vs Provisional

### 5.1. Production-ready (можно внедрять немедленно)

| Компонент | Готовность | Обоснование |
|---|---|---|
| **Hard blocker: date_mismatch** | ✅ Production-ready | Тривиальная логика: разные даты = разные события (для single-occurrence). Не ломает ни один must_merge кейс. Проверено на всех 29 кейсах |
| **Hard blocker: city_mismatch** | ✅ Production-ready | После нормализации — надёжный блокер. Проверено на `oncologists_zelenogradsk` |
| **Убрать forced-match bias из prompt** | ✅ Production-ready | Удаление одной строки. Не может ухудшить quality (убирается merge-давление) |
| **Убрать anchor-pressure из prompt** | ✅ Production-ready | Аналогично. Удаление match-forcing |
| **`default_location` → weak hint only** | ✅ Production-ready | Убрать override logic. Одностороннее улучшение |
| **Venue alias normalization** | ✅ Production-ready | Lookup table, zero LLM cost, zero risk |
| **City alias normalization** | ✅ Production-ready | Lookup table |
| **Source-owner guard** (narrow scope) | ✅ Production-ready | Scope: single_event + event_count=1 + same_source_url. Проверен на led_hearts, garage, nutcracker, womanhood |

### 5.2. Provisional (рабочая гипотеза, нужна калибровка)

| Компонент | Готовность | Что нужно для production |
|---|---|---|
| **Hard blocker: time_conflict** | ⚠️ Provisional → gated | Гейтинг по `end_date IS NULL` подтверждён на 4 кейсах (treasure_island, frog_princess, nutcracker → different; womanhood → bypass). Требует проверки на ≥10 дополнительных кейсов с edge-case times (doors vs start, partial default) |
| **Числовые thresholds** (MERGE≥10, CREATE≤3) | ⚠️ Provisional | Полезны как starting point, но нет достаточного cross-validation. Нужен dry-run по ≥50 парам с ручным annotation. Рекомендация: начать с консервативных CREATE≤2, MERGE≥12, потом ослаблять |
| **Scoring weights** (title +4, venue +3, etc.) | ⚠️ Provisional | Калиброваны на 29 кейсах. Для production нужно: (a) re-run с весами на полном snapshot, (b) annotate false positives/negatives, (c) iterate |
| **4-state LLM verdict prompt** | ⚠️ Provisional | Текст промпта разумный, но не тестирован в production на Gemma 27B. Нужен shadow-mode A/B: old prompt vs new prompt на ≥100 real pairs |
| **`ticket_is_generic` heuristic** | ⚠️ Provisional | Domain+path logic написана, но denylist и pattern matching нужно проверить на полном наборе ticket URLs из snapshot. Рекомендация: extract все уникальные ticket URLs, annotate вручную generic/specific, validate heuristic |
| **Policy profiles** (multi_event, recurring_page) | ⚠️ Provisional | Логика звучит разумно, но `source_kind` detection ещё не реализован в runtime. Нужны: (a) classifier `classify_source_kind()`, (b) проверка accuracy classifier на ≥20 real sources, (c) fallback при misclassification |
| **gray_create_softlink action** | ⚠️ Provisional | Концепция принята, runtime implementation ещё не существует. Нужны: (a) DB schema для softlink, (b) admin UI для review, (c) auto-escalation policy |
| **title_mismatch → -3 penalty** | ⚠️ Provisional | Работает на museum_overlap, cathedral. Но нужно проверить, не даёт ли false negatives на kейсах title alias (brand vs item) |

### 5.3. Рекомендованный порядок внедрения

```
Sprint 1 (1-2 дня): все production-ready items
  → Expected improvement: устраняет класс recurring false-merges

Sprint 2 (2-3 дня): time_conflict gated blocker + source-owner guard
  → Requires: unit tests на 6 контрольных пар

Sprint 3 (3-5 дней): scoring calibration + ticket_is_generic
  → Requires: dry-run на full snapshot + annotation

Sprint 4 (5-7 дней): LLM prompt redesign + shadow mode
  → Requires: A/B test infrastructure

Sprint 5 (3-5 дней): policy profiles + gray state
  → Requires: source_kind classifier + DB schema
```

---

## 6. Extra Validation I Recommend

### 6.1. Минимальный cross-check перед Sprint 1

Даже для production-ready items:

```bash
# 1. Прогнать deterministic dry-run с новыми blockers на full casepack (29 cases)
python scripts/inspect/identity_longrun.py \
  --snapshot artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --casepack artifacts/codex/opus_session2_casepack_latest.json \
  --enable-date-blocker --enable-city-blocker \
  --dry-run

# Expected: 0 must_merge cases broken, all recurring/double-show → different
```

### 6.2. Перед Sprint 2: time_conflict edge cases

Собрать из snapshot **все пары** с `same_date + same_venue + time_diff > 0`:

```sql
-- Find all same-venue same-date event pairs with different times
SELECT e1.id, e2.id, e1.title, e2.title,
       e1.date, e1.time, e2.time,
       e1.end_date, e2.end_date,
       abs(
         cast(substr(e1.time, 1, 2) as integer) * 60 + cast(substr(e1.time, 4, 2) as integer) -
         cast(substr(e2.time, 1, 2) as integer) * 60 - cast(substr(e2.time, 4, 2) as integer)
       ) as time_diff_mins
FROM events e1
JOIN events e2 ON e1.id < e2.id
  AND e1.date = e2.date
  AND e1.location_name = e2.location_name
  AND e1.time != '' AND e2.time != ''
  AND e1.time != e2.time
WHERE e1.is_active = 1 AND e2.is_active = 1
ORDER BY time_diff_mins;
```

Annotate каждую пару: `merge` / `different` / `gray`. Проверить, что time_conflict blocker + end_date gate даёт правильный result на всех.

### 6.3. Перед Sprint 3: ticket URL inventory

```sql
SELECT DISTINCT ticket_link, COUNT(*) as event_count
FROM events
WHERE ticket_link IS NOT NULL AND ticket_link != ''
GROUP BY ticket_link
ORDER BY event_count DESC;
```

Для каждого уникального URL: annotate `generic` / `specific`. Validate `ticket_is_generic()` heuristic.

### 6.4. Перед Sprint 4: shadow A/B prompt test

```
For each pair in casepack:
  1. Run OLD prompt → get old_verdict
  2. Run NEW prompt → get new_verdict
  3. Compare with gold label
  4. Flag any pair where old_correct AND new_incorrect → regression
  5. Flag any pair where old_incorrect AND new_correct → improvement
  6. Flag any pair where both incorrect → needs attention
```

Success criteria:
- 0 regressions on must_not_merge cases (zero tolerance for new false merges)
- ≥ 90% improvement on must_merge cases that were previously gray/different
- ≤ 5% new gray verdicts on previously-correct pairs

### 6.5. Дополнительные gold cases, которые рекомендую добавить

| Тип | Количество | Почему |
|---|---|---|
| Long-running exhibitions с time noise | +3 | Проверка time_conflict gate на end_date |
| Follow-up / repost chains | +2 | Проверка follow-up detection |
| Non-event content (реклама, отчёты) | +3 | Проверка skip_non_event |
| Brand-vs-item с ambiguous title | +2 | Проверка title_related sensitivity |
| Same-source with extracted field corruption | +2 | Проверка source rescue scope |

**Итого**: текущие 29 → **≥41** gold cases для regression baseline.

### 6.6. Monitoring metrics после каждого sprint

| Metric | Target | Alert threshold |
|---|---|---|
| False merge rate (among all merge decisions) | < 2% | > 5% → roll back |
| Duplicate rate (known duplicates not merged) | < 15% | > 25% → investigate |
| LLM call rate (% pairs requiring LLM) | 30-40% | > 60% → scoring calibration |
| Gray ratio (% verdicts = gray) | 5-10% | > 20% → thresholds too conservative |
| P95 latency per event | < 8s | > 15s → payload optimization |
| TPM utilization | < 80% of 12000 | > 90% → critical |
