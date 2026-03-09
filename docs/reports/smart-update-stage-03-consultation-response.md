# Smart Update Stage 03 → Stage 04 Consultation Response

Дата: 2026-03-06
Формат: конкурентный ответ. Не соглашаюсь автоматически, аргументирую и предлагаю альтернативы.

---

## 1. Stage 03 Assessment

### 1.1. Что сделано хорошо

Stage 03 baseline — **правильная инженерная позиция**. 0 false merges / 0 false differents на 35 кейсах / 72 парах — сильная отправная точка. Консервативный подход с gray-хвостом вместо рискованных merge-ов — именно то, что нужно для quality-first.

Особенно показательно:
- `buratino_double_show_same_source`, `severnoe_siyanie_same_day_triple_show` — time_conflict blocker работает чисто
- `organ_booklet_exact_ticket_duplicate` — specific ticket merge работает даже при Cathedral-style URLs
- `zoo_free_range_schedule_repeat` — date_blocker корректно разводит same-source schedule

### 1.2. Где baseline слишком консервативен — и это СТОИТ денег

29 must-merge пар в gray — это **40% от всех must-merge пар**. Каждая серая пара в production:
- создаёт видимый дубль для пользователя;
- требует ручной или LLM-driven resolution;
- если не разрулена, ухудшает UX и доверие к системе.

При этом эти 29 пар распределены неравномерно:

| Класс gray | Кол-во пар | % от gray | Сложность resolution |
|---|---|---|---|
| `hudozhnitsy_5way_cluster` | 10 | 34% | Высокая — 5 sources, title alias |
| `venue_noise` | 9 | 31% | Средняя — venue alias + context check |
| `needs_llm` (semantic) | 5 | 17% | Высокая — нет strict proof |
| `same-source anomaly` | 3 | 10% | Средняя — source rescued, но date/time mismatch |
| `title_mismatch_same_slot` | 2 | 7% | Средняя — title framing alias |

**Мой тезис**: 12-15 из 29 пар можно перевести в deterministic merge без открытия false merges. Остальные 14-17 действительно нуждаются в LLM.

### 1.3. Где baseline может быть ещё рискован (несмотря на 0 false merges)

⚠️ **Замечание, которого не было в Stage 03 материалах**: текущий casepack — 35 кейсов. Это хороший набор, но **не exhaustive**. Нулевой false-merge rate на 35 кейсах не гарантирует нулевой rate на production потоке.

Конкретные не-покрытые паттерны, которые я рекомендую добавить:
1. **Two exhibitions in same museum with SIMILAR titles** (e.g. "Русское искусство XIX века" vs "Русское искусство XX века")
2. **Same performance in different venues** (гастроли — один спектакль, разные площадки)
3. **Event rename** (площадка/организатор переименовал событие после первой публикации)
4. **Long-running exhibition update** (обновлённый пост о той же выставке с новыми деталями программы)

### 1.4. Контраргумент к текущему разделению на 4 gray-класса

Текущая классификация (`venue_noise`, `same-source anomaly`, `semantic duplicate`, `safe-not-merge-but-similar`) не совсем оптимальна для принятия решений о Stage 04 actions.

**Мое предложение**: делить по **типу доказательства, которого не хватает**, а не по симптому:

| Реальная проблема | Пары | Решение |
|---|---|---|
| **Venue string не нормализован** | sobakusel, gromkaya, prazdnik, little_women, oncologists, часть shambala | Venue alias table → deterministic |
| **Title alias не распознан** | shambala (2799/2844), prazdnik (2802/2803), makovetsky, hudozhnitsy кластер | LLM или NLP title matching |
| **Source rescue заблокирован** | led_hearts, womanhood | Relaxed source rescue policy → deterministic |
| **Cross-source semantic match** | matryoshka, plastic_nutcracker, часть hudozhnitsy | LLM only |
| **Must-not-merge слишком похожие** | museum_holiday, cathedral | LLM → different (или deterministic via title_mismatch) |

---

## 2. Deterministic Upgrades Worth Adding

### 2.1. Venue Alias Table (высший приоритет, ~9 пар)

**Проблема**: 9 из 29 gray пар имеют `venue_noise_needs_llm` — venue строки отличаются только formatting/aliasing.

**Наблюдение**: venue noise — полностью детерминистически решаемая проблема. Не нужен LLM.

**Конкретные aliases из текущего casepack**:

```python
VENUE_ALIASES = {
    # Шаблон: canonical_name -> set of known aliases
    "Бар «Место Силы»": {"Место Силы", "Бар Место Силы", "бар место силы"},
    "TerkaCityHall": {"Терка", "Terka City Hall", "ТЕРКА", "TerkaСityHall"},
    "Собакусъел": {"Собакусъел", "Собакус%ел"}, # confusable chars
    "Филиал Третьяковской галереи": {
        "Третьяковка Калининград",
        "Филиал Третьяковской галереи",
        "Третьяковская галерея (филиал)",
        "Калининградский филиал Третьяковской галереи",
    },
    "Сигнал": {"Сигнал", "Сигнал, Леонова 22, Калининград"},
    # ... расширять по мере обнаружения
}
```

**Но я НЕ согласен с чисто-табличным подходом**. Таблица быстро устаревает. Лучше **комбо**:

```python
def venues_match(v1: str, v2: str) -> bool:
    """3-layer venue matching."""
    # Layer 1: exact match after normalization
    n1 = normalize_venue(v1)  # lowercase, strip whitespace, ё→е, translit
    n2 = normalize_venue(v2)
    if n1 == n2:
        return True

    # Layer 2: alias table lookup
    c1 = VENUE_ALIASES.get(n1, {n1})
    c2 = VENUE_ALIASES.get(n2, {n2})
    if c1 & c2:  # intersection
        return True

    # Layer 3: address-based matching
    a1 = extract_address(v1)  # "Леонова 22" from "Сигнал, Леонова 22, Калининград"
    a2 = extract_address(v2)
    if a1 and a2 and a1 == a2:
        return True

    # Layer 4: containment check
    if n1 in n2 or n2 in n1:
        return True  # "Бар «Место Силы»" contains "Место Силы"

    return False
```

**Пары, которые это разблокирует**: sobakusel, gromkaya, часть shambala (2843/2844), prazdnik (2789/2802), little_women (2761/2815, 2815/2816-2817), oncologists_svetlogorsk.

**Не разблокирует**: пары где venue в принципе не упоминается или было overridden (нужен LLM).

**Do-not-apply**: НИКОГДА не считать venue_match автоматическим merge. Venue_match — это **снятие venue_noise блокера**. Merge всё ещё требует title_related + date_match.

### 2.2. Source-Owner Rescue Extension for Same-Post Triple/Double (3 пары)

**Проблема**: `led_hearts` (3 пары) и `womanhood` (1 пара) — same source URL, но date/time mismatch мешает deterministic merge.

**Текущее правило**: source rescue только при `event_count_from_source == 1`.

**Предложение по расширению** (осторожное, с preconditions):

```python
def extended_source_rescue_applies(candidate, existing) -> bool:
    """
    Relaxed source rescue for same-source events with extraction errors.
    Only for KNOWN safe patterns.
    """
    if candidate.source_post_url != existing.source_post_url:
        return False
    if candidate.source_kind != "single_event":
        return False

    # Key addition: allow rescue even with event_count > 1
    # IF title is exact AND the time/date difference looks like extraction error
    if not titles_exact(candidate.title, existing.title):
        return False

    # For long-running exhibitions: time difference is noise
    if candidate.end_date and existing.end_date:
        return True  # womanhood case

    # For same-date single-events: bad extraction from same post
    if candidate.date == existing.date:
        return True  # deduplicates within same date

    # For different dates: only if TG source and telegra.ph pages differ only by suffix
    # This catches led_hearts: 3 telegra.ph pages from 1 TG post
    if (candidate.source_type == "telegram"
        and candidate.source_message_id == existing.source_message_id):
        return True

    return False
```

**Do-not-apply boundaries**:
- ❌ НЕ применять если `title_exact = false` — иначе `nutcracker_two_shows_same_post`-подобные кейсы с emoji-разницей проскочат
- ❌ НЕ применять если `source_kind != single_event` — иначе repertory pages ломаются
- ❌ НЕ применять при `venue_mismatch` И `title_mismatch` одновременно

**Пары, которые это разблокирует**: led_hearts (2845/2846, 2845/2847, 2846/2847), womanhood (2755/2756).

**Контрольная проверка safety**: `nutcracker_two_shows_same_post` — same source, title exact, BUT `end_date=NULL` + different dates? Нет — same date. НО event_count=2 + time diff = 5h. Нужен дополнительный guard:

```python
    # SAFETY: if same date but both times are known and differ significantly → NOT rescue
    if (candidate.date == existing.date
        and candidate.time and existing.time
        and not candidate.time_is_default and not existing.time_is_default
        and abs(time_diff(candidate.time, existing.time)) > 30
        and not candidate.end_date):  # single-occurrence
        return False  # This is likely a double-show, not extraction error
```

С этим guard'ом: nutcracker (14:00/19:00, end_date=NULL) → rescue off → time_conflict blocker → different ✅.

### 2.3. Title Mismatch Deterministic Blocker (4 must-not-merge gray пары)

**Проблема**: 4 must-not-merge пары в gray (`museum_holiday` × 3 + `cathedral` × 1) — все имеют title_mismatch, но текущий baseline не конвертирует их в deterministic `different`.

**Наблюдение**: я **не согласен**, что эти 4 кейса нужно оставлять на LLM. Они детерминистически решаемы.

Правило:

```python
def title_mismatch_forces_different(pair) -> bool:
    """
    At same slot (date+time+venue), completely unrelated titles → different.
    """
    if not pair.title_related:  # no word overlap, no alias, no containment
        if pair.date_match and pair.venue_match:
            return True  # same slot, different content → different events
    return False
```

**Кейсы, которые это разблокирует**:
- `cathedral_shared_ticket_false_friend` (1979/2278): «Английская придворная культура» vs «Королева фей» → title_related=false → different ✅
- `museum_holiday_program_multi_child` (2743/2744, 2743/2745, 2744/2745): «8 Марта в Музее» vs «Акция 'Вам, любимые!'» vs «Бесплатная экскурсия» → все три пары title_related=false → different ✅

**Do-not-apply** (критически важно):
- ❌ НЕ применять если `title_related=true` (хотя бы 1 общее значимое слово) — иначе brand-vs-item кейсы типа shambala (lineup vs brand) ломаются
- ❌ НЕ применять без `date_match AND venue_match` — при разных датах/venues и так работает date/city blocker

**Risk assessment**: низкий. Title_related=false при совпадающем date+venue — это однозначно разные события. Единственный edge case: event rename → но ренейм обычно сохраняет хотя бы 1 ключевое слово.

**Результат**: перевод 4 must-not-merge gray → deterministic different. Gray хвост must-not-merge = 0.

### 2.4. Что я НЕ рекомендую добавлять в deterministic layer

| Идея | Почему нет |
|---|---|
| **Participant overlap merge** | Участники могут совпадать у разных событий одного фестиваля. Нужен LLM контекст |
| **Poster hash auto-merge** | Один poster используется для recurring спектаклей. Poster = weak signal без LLM |
| **Text containment auto-merge** | Один текст может содержать другой при follow-up, но это не всегда identity |
| **Fuzzy title match auto-merge** | `titles_related` может давать false positives на generic words ("Лекция", "Мастер-класс") |

---

## 3. LLM Layer For Residual Gray

### 3.1. После deterministic upgrades: что остаётся на LLM

Если добавить venue alias (§2.1), source rescue extension (§2.2), и title_mismatch blocker (§2.3):

| Класс | Кейсы | Пары | Нужен LLM? |
|---|---|---|---|
| `hudozhnitsy_5way_cluster` | 1 cluster | 10 пар | **Да** — title alias + cross-source semantic |
| `shambala_cluster` (2799/2844) | 1 pair | 1 | **Да** — title mismatch (lineup vs brand) same-slot |
| `prazdnik_u_devchat` (2802/2803) | 1 pair | 1 | **Да** — title vs address-in-title |
| `makovetsky_chekhov_duplicate` | 1 pair | 1 | **Да** — brand name vs program title |
| `matryoshka_exhibition_duplicate` | 1 pair | 1 | **Да** — no source proof, semantic only |
| `plastic_nutcracker_cross_source_duplicate` | 1 pair | 1 | **Да** — cross-source, emoji-only diff |

**Итого LLM-only: ~15 пар** (из 29 must-merge gray). Остальные ~14 переведены в deterministic.

### 3.2. Compact Pairwise Payload для LLM gray разрешения

```python
def build_llm_triage_payload(candidate_ev, existing_ev, hints):
    """
    ~1000-1200 input tokens для Gemma.
    """
    return {
        "candidate": {
            "title": candidate_ev.title,
            "date": candidate_ev.date,
            "time": candidate_ev.time or "unknown",
            "end_date": candidate_ev.end_date,
            "venue": candidate_ev.location_name,
            "city": candidate_ev.city,
            "ticket": candidate_ev.ticket_link,
            "text": clip(candidate_ev.source_text, 500),
            "poster_title": candidate_ev.poster_title,
        },
        "existing": {
            "id": existing_ev.id,
            "title": existing_ev.title,
            "date": existing_ev.date,
            "time": existing_ev.time or "unknown",
            "end_date": existing_ev.end_date,
            "venue": existing_ev.location_name,
            "ticket": existing_ev.ticket_link,
            "description": clip(existing_ev.description, 400),
        },
        "hints": hints,  # pre-computed booleans
    }
```

### 3.3. Hints для каждого gray-класса

#### Для `hudozhnitsy`-like (title alias + cross-source):

```python
hints = {
    "title_related": True,              # содержит общее слово "Художницы"
    "title_exact": False,               # формулировки разные
    "date_exact": True,                 # 2026-03-07
    "time_exact": True,                 # 14:00
    "venue_match": True,                # Третьяковка (after alias)
    "source_cross": True,               # разные sources
    "ticket_specific_match": False,     # ticket может быть generic
    "poster_overlap": False,            # разные posters
    "participant_overlap_count": 3,     # количество общих имён в тексте
    "source_kind": "single_event",
    "gray_reason": "title_alias_cross_source",
}
```

#### Для `makovetsky`/`shambala` (brand vs item):

```python
hints = {
    "title_related": True,
    "title_exact": False,
    "brand_vs_item_suspected": True,    # NEW: один title ⊂ другого
    "date_exact": True,
    "time_exact": True,
    "venue_match": True,
    "poster_overlap": True,             # makovetsky: same poster
    "source_kind": "single_event",
    "gray_reason": "brand_vs_item",
}
```

#### Для `matryoshka` (pure semantic):

```python
hints = {
    "title_related": True,              # "матрешки" in both
    "title_exact": False,
    "date_exact": True,
    "end_date_match": True,             # both end 2026-04-05
    "time_exact": True,                 # both unknown
    "venue_match": True,                # same museum
    "source_cross": True,               # different VK posts
    "same_vk_group": True,              # same wall owner
    "source_kind": "single_event",
    "gray_reason": "semantic_no_source_proof",
}
```

### 3.4. Prompt для Gemma triage на gray residual

**Мой спор с предыдущим промптом**: предыдущий prompt из Stage 02 содержит 6 evidence dimensions (title/date/time/venue/context/ticket). Для gray residual (где deterministic layer уже обработал date/time/venue) нужен **упрощённый prompt** — LLM не должен переоценивать signals, которые deterministic уже проверил.

**Мое предложение: "тонкий" LLM prompt specifically для gray resolution**:

```python
GRAY_TRIAGE_PROMPT = (
    "Ты — судья по идентификации событий.\n"
    "Deterministic система уже проверила: дату, время, площадку, город.\n"
    "Результат: пара попала в серую зону — формального доказательства identity недостаточно.\n\n"

    "Твоя задача: определить, описывают ли два анонса ОДНО И ТО ЖЕ мероприятие.\n\n"

    "Что тебе нужно оценить:\n"
    "1) TITLE: описывают ли названия одно событие, пусть и разными словами?\n"
    "   - Пример alias: 'Шамбала' и 'Влада Клепцова, Вика Козлова...' — если участники + venue + date совпадают\n"
    "   - Пример mismatch: '8 Марта в Музее' vs 'Бесплатная экскурсия' — разные программы\n"
    "2) CONTENT: пересекаются ли участники / программа / описание?\n"
    "3) CONTEXT: если hints показывают same_vk_group или participant_overlap — учитывай, но не переоценивай\n\n"

    "ВЕРДИКТ (строго одно):\n"
    "- same_event: однозначно одно событие, разная формулировка/источник\n"
    "- likely_same: скорее всего одно, но нет 100% уверенности\n"
    "- different: разные мероприятия, несмотря на похожий slot\n"
    "- uncertain: недостаточно данных\n\n"

    "КРИТИЧЕСКИ ВАЖНО:\n"
    "- Ошибочная склейка ХУЖЕ дубля. При сомнении → uncertain.\n"
    "- Совпадение площадки + даты САМО ПО СЕБЕ не доказывает identity. В одном музее может быть несколько выставок.\n"
    "- Совпадение VK group owner САМО ПО СЕБЕ не доказывает identity. Один VK автор публикует разные события.\n"
    "- Используй ТОЛЬКО данные ниже.\n\n"

    "JSON ответ по schema.\n\n"
    f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
)
```

**Ключевые отличия от Stage 02 prompt**:
1. **Убраны DATE/TIME/VENUE evidence sections** — они уже deterministic. LLM не тратит tokens на пересчёт того, что уже проверено.
2. **Добавлен explicit anti-pattern**: "В одном музее может быть несколько выставок" — защита от museum_overlap-подобных кейсов.
3. **Prompt короче** (~600 chars vs ~2200 chars) → экономия ~45% на prompt tokens.

### 3.5. TPM impact assessment

```
Per gray pair LLM call:
  Prompt: ~600 chars = ~150 tokens
  Payload: ~1000 chars = ~250 tokens
  Response: ~100-150 tokens
  Total: ~500-550 tokens per call

At 15 LLM-only pairs per batch:
  Total: ~8000 tokens per batch
  At TPM=12000: fits in ~40 seconds
  At RPM=20: 15 calls = ~45 seconds at 1 call/3sec
  
Verdict: fits comfortably в текущие лимиты.
```

---

## 4. Case-Class Decisions

### 4.1. Triaging all 29 must-merge gray pairs

#### Deterministic-upgradable (14 пар → merge по правилам из §2)

| Pair | Case | Механизм | Precondition |
|---|---|---|---|
| sobakusel [2793,2810] | venue_noise | **Venue alias** (TerkaCityHall / Собакусъел canonical) | title_related + date_match |
| gromkaya [2667,2792] | venue_noise | **Venue alias** (Бар «Место Силы» alias) + doors_vs_start уже handled | title_related + date_match |
| shambala [2799,2843] | venue_noise | **Venue alias** | title_related + date_match |
| shambala [2843,2844] | venue_noise | **Venue alias** | title_exact + date_match |
| prazdnik [2789,2802] | venue_noise | **Venue alias** (TerkaCityHall variants) | title_related + date_match |
| little_women [2761,2815] | venue_noise | **Venue alias** (Сигнал alias chain) | title_related + date_match |
| little_women [2815,2816] | venue_noise | **Venue alias** | title_related + date_match |
| little_women [2815,2817] | venue_noise | **Venue alias** | title_related + date_match |
| oncologists [2710,2721] | venue_noise | **Venue alias** (Светлогорск variants) | title_related + date_match |
| led_hearts [2845,2846] | same-source | **Source rescue extension** (same TG post, title exact) | source_kind=single, same message_id |
| led_hearts [2845,2847] | same-source | **Source rescue extension** | source_kind=single, same message_id |
| led_hearts [2846,2847] | same-source | **Source rescue extension** | source_kind=single, same message_id |
| womanhood [2755,2756] | same-source | **Source rescue extension** (end_date not null, title exact) | long-running exhibition |
| prazdnik [2802,2803] | title_mismatch_same_slot | ⚠️ **СПОРНО** — title "Хоровая вечеринка" vs "Октябрьская, 8, 4 этаж" — это broken extraction. `title_related` может быть false, но same source (t.me/terkatalk/4529)... | Source rescue, NOT venue alias |

**Correction on prazdnik [2802,2803]**: это фактически same-source (оба из `t.me/terkatalk/4529`) с title corruption. Подходит под source rescue extension → merge. **Итого: 14 пар.**

#### LLM-only (15 пар)

| Pair | Case | Почему LLM |
|---|---|---|
| shambala [2799,2844] | title_mismatch_same_slot | "Влада Клепцова, Вика Козлова..." vs "Шамбала" — brand vs lineup. Title_related по word overlap слабый. Нужен LLM для context/participant check |
| hudozhnitsy [2541,2675] | needs_llm | Cross-source, title alias. Нужен LLM |
| hudozhnitsy [2541,2779] | needs_llm | "Художницы" vs "Художницы — Филиал Третьяковской" — alias but cross-source |
| hudozhnitsy [2541,2801] | needs_llm | Аналогично |
| hudozhnitsy [2541,2838] | needs_llm | Аналогично |
| hudozhnitsy [2675,2779] | needs_llm | Cross-source semantic |
| hudozhnitsy [2675,2801] | needs_llm | Cross-source semantic |
| hudozhnitsy [2675,2838] | needs_llm | Cross-source semantic |
| hudozhnitsy [2779,2801] | needs_llm | Cross-source semantic |
| hudozhnitsy [2779,2838] | needs_llm | Cross-source semantic |
| hudozhnitsy [2801,2838] | needs_llm | Cross-source semantic |
| makovetsky [2758,2759] | needs_llm | Brand vs program, same poster but title framing differs |
| matryoshka [2725,2726] | needs_llm | No shared source, purely semantic match |
| plastic_nutcracker [1603,1622] | needs_llm | Cross-source (VK vs TG digest), title nearly identical but from different ecosystems |

**Observation**: hudozhnitsy alone is 10/15 LLM-only pairs. Если venue alias table + title_related улучшить, некоторые hudozhnitsy пары могут стать deterministic. Но это рискованнее.

**Мое конкурентное предложение**: для `hudozhnitsy`-подобных кластеров (N≥3 events, all same date/time/venue, all title_related) можно добавить **cluster-aware deterministic rule**:

```python
def cluster_merge_shortcut(cluster: list[Event]) -> bool:
    """
    If ≥3 events share exact date+time+venue, and all titles
    contain the same root keyword, merge deterministically.
    """
    if len(cluster) < 3:
        return False
    # All must share date, time, venue
    if not all_same(e.date for e in cluster):
        return False
    if not all_same(e.time for e in cluster):
        return False
    if not venues_match_all(cluster):
        return False
    # Extract common keyword from titles
    common = common_significant_word(t.title for t in cluster)
    if not common:
        return False
    return True  # merge into canonical
```

Это бы покрыло hudozhnitsy (все содержат "Художниц"/"Худож") и shambala (все содержат venue "Место Силы" + дату). **НО** это рискованно — может ломать `museum_holiday_program_multi_child` (3 events, same date, same venue, но РАЗНЫЕ titles).

**Решение**: добавить guard `title_related_all_pairs=true` И `title_mismatch_any_pair=false`. Для museum_holiday: "8 Марта" vs "Акция" vs "Бесплатная экскурсия" → title_related ≠ true для всех пар → guard не пропускает. ✅

Это может перевести **10 hudozhnitsy пар в deterministic**, оставив на LLM только 5 пар.

### 4.2. Must-not-merge gray → deterministic different

Все 4 must-not-merge gray пары → переводятся title_mismatch blocker (§2.3):

| Pair | Case | Механизм |
|---|---|---|
| museum_holiday [2743,2744] | title_mismatch_same_slot | «8 Марта в Музее» vs «Акция 'Вам, любимые!'» → different |
| museum_holiday [2743,2745] | title_mismatch_same_slot | «8 Марта в Музее» vs «Бесплатная экскурсия» → different |
| museum_holiday [2744,2745] | title_mismatch_same_slot | «Акция 'Вам, любимые!'» vs «Бесплатная экскурсия» → different |
| cathedral [1979,2278] | title_mismatch_same_slot | «Английская придворная культура» vs «Королева фей» → different |

---

## 5. Stage 04 Rollout Proposal

### Phase 4A: Deterministic (2-3 дня, low risk)

**Что добавить:**
1. **Venue alias normalization** (§2.1) — 3-layer matching
2. **Title mismatch blocker** (§2.3) — must-not-merge gray → different
3. **Source rescue extension** (§2.2) — с safety guards

**Validation:**
```bash
# Re-run dry-run with new rules
# Expected: 
#   must_merge_resolved: 9 → 23 (+14)
#   must_merge_gray: 29 → 15 (-14)
#   must_not_merge_resolved: 30 → 34 (+4)
#   must_not_merge_gray: 4 → 0 (-4)
#   false_merges: 0
#   false_differents: 0
```

**Что НЕ трогать в Phase 4A:**
- LLM prompt changes
- Score thresholds
- Multi-event policy
- Cluster merge shortcut (§4.1 competitive proposal — только после Phase 4B validation)

### Phase 4B: Cluster-aware Merge (2 дня, medium risk, optional)

**Что добавить (если Phase 4A succeeds):**
1. **Cluster merge shortcut** (§4.1) — для ≥3 events with all-pairs title_related
2. Re-run dry-run

**Expected:**
```
must_merge_resolved: 23 → 33 (+10 hudozhnitsy)
must_merge_gray: 15 → 5
false_merges: 0 (IF title_related guard holds)
```

### Phase 4C: LLM Gray Resolution (5-7 дней, medium risk)

**Что добавить:**
1. New `_llm_pairwise_triage` function with gray-specific prompt (§3.4)
2. Compact payload builder (§3.2)
3. Hint computation for each gray class (§3.3)
4. Shadow mode: run LLM on gray pairs, log but don't act

**Shadow mode validation:**
- Run on remaining ~5 gray pairs
- Compare LLM verdict with gold label
- Accept if: 0 false merges on must_not_merge, ≥80% correct on must_merge

### Phase 4D: LLM Production (2-3 дня)

**Что добавить:**
1. Enable LLM gray resolution in production
2. Monitor metrics
3. Auto-rollback if false_merge_rate > threshold

### Phase 4E: Continuous (ongoing)

**Что добавить:**
1. Venue alias table expansion (crowdsourced from gray reports)
2. Additional gold cases from production
3. Regular regression runs

### Timeline

```
Phase 4A: Days 1-3     (deterministic upgrades)
Phase 4B: Days 4-5     (cluster merge, optional)
Phase 4C: Days 6-12    (LLM shadow mode)
Phase 4D: Days 13-15   (LLM production)
Phase 4E: Continuous    (monitoring + expansion)

Total to production-ready LLM: ~2-3 weeks
```

---

## 6. Residual Risks

### 6.1. Venue alias table completeness

**Risk**: новые venues не покрыты таблицей → gray stays gray.
**Mitigation**: (a) адресное matching (Layer 3 в §2.1), (b) containment check (Layer 4), (c) automated alias discovery из source texts.
**Severity**: Low — graceful degradation (остаётся в gray, не false merge).

### 6.2. Title_related false positives

**Risk**: generic words ("Мастер-класс", "Лекция", "Выставка") дают title_related=true для unrelated events.
**Mitigation**: stopword list для title_related computation. "Мастер-класс" alone → not significant. "Мастер-класс: Зажигаем сердца" vs "Мастер-класс: Керамика" → "Зажигаем" ≠ "Керамика" → title_related=false.
**Severity**: Medium — could cause false gray → needs attention.

### 6.3. Cluster merge false positive on festival lineups

**Risk**: фестиваль с 5 events, same date/venue, all titles contain "Festival_name" → cluster merge collapses them.
**Mitigation**: cluster merge requires **all-pairs** title_related, not just common keyword. Festival child events typically have unique sub-titles.
**Severity**: Medium — add festival test cases before Phase 4B.

### 6.4. Source rescue on multi-account TG reposts

**Risk**: один event reshared в 5 TG каналов. Source rescue не знает, что event_count > 1 across channels.
**Mitigation**: source rescue scoped to same message_id from same channel. Cross-channel dedup handled separately.
**Severity**: Low.

### 6.5. LLM hallucination on sparse payloads

**Risk**: Gemma видит sparse payload (no source_text, no poster) и фантазирует совпадения.
**Mitigation**: explicit prompt rule "Используй ТОЛЬКО данные ниже" + anti-hallucination. Default to `uncertain` при sparse data.
**Severity**: Medium.

### 6.6. Time blocker edge case: "doors + start" с >90 мин разницей

**Risk**: некоторые мероприятия имеют сбор за 2+ часа до начала (загородные экскурсии).
**Mitigation**: doors_vs_start pattern detector расширить до 120 мин. Add test cases.
**Severity**: Low — affects few events.

### 6.7. Самый опасный непокрытый gap

**Gap**: `plastic_nutcracker_cross_source_duplicate` (1603/1622) — cross-source, exact date/time/venue, title nearly identical (emoji prefix difference), no shared source/ticket/poster. Это **чистый LLM кейс**. Без LLM layer он навсегда остаётся gray.

Если LLM layer не будет внедрён → этот класс дублей (cross-source exact slots) будет система creating forever. Это ~5-10% от всех дублей in production.

---

## Мои конкурентные тезисы (не копируя Opus)

1. **Venue alias — это самый дешёвый и эффективный upgrade.** 9/29 = 31% всего gray хвоста решается таблицей + containment check. Это нужно делать ПЕРВЫМ, а не LLM.

2. **Title mismatch blocker для must-not-merge gray — обязателен.** Странно, что 4 пары с completely unrelated titles сидят в gray. Это детерминистически очевидный `different`.

3. **Cluster-aware merge — рискованнее, чем кажется**, но с правильным guard (all-pairs title_related) может покрыть hudozhnitsy целиком. Это мой самый амбициозный proposal, за пределами того, что предлагал Opus.

4. **LLM prompt для gray должен быть КОРОЧЕ, а не длиннее.** Deterministic layer уже проверил date/time/venue/city — LLM не должен тратить tokens на пересчёт. Промпт из Stage 02 слишком verbose для gray resolution.

5. **Раздельные метрики для deterministic и LLM layers.** Нельзя мешать в одну кучу. Deterministic layer с 0 false merges — production-ready. LLM layer — needs shadow mode. Разные уровни trust = разные rollout timelines.
