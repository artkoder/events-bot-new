# Smart Update Session 2: Глубокий Re-check на реальных данных

Дата: 2026-03-06
Формат: инженерная рекомендация, без общих рассуждений.
Каждая рекомендация привязана к конкретным кейсам из casepack и longrun benchmark.

---

## 1. Re-check Of Opus V1 Recommendations

### 1.1. Что подтверждаю без изменений

| # | Рекомендация v1 | Статус | Доказательство на реальных данных |
|---|---|---|---|
| 1 | Убрать forced-match bias `«Не возвращай null, если есть правдоподобный матч»` | **Подтверждаю** | Это строка L6441 в `_llm_match_event`. На кейсах `cathedral_shared_ticket_false_friend` и `dramteatr_same_slot_cross_title` модель притягивает match при generic ticket + same slot, хотя события явно разные |
| 2 | Убрать anchor-pressure `«это дубль — выбирай action=match»` | **Подтверждаю** | Строка L6556-6557 в `_llm_match_or_create_bundle`. Кейс `treasure_island_double_show` (11:00 + 14:00): time отличается, но давление forcing match через «якоря» перебивает time_conflict |
| 3 | Pairwise evidence judge вместо fat-shortlist | **Подтверждаю** | Longrun benchmark работает pairwise — 32/32 acceptable. Текущий fat-shortlist прод-промпт даёт merge на must_not_merge кейсах |
| 4 | 4-state verdict (`same_event / likely_same / different / uncertain`) | **Подтверждаю** | Longrun: 11 из 32 кейсов корректно получили `gray` (т.е. `likely_same` или `uncertain`). Без gray-состояния они бы стали false merges |
| 5 | Разделить judge и decider | **Подтверждаю** | LLM оценивает evidence → runtime маппит в action. Это позволяет менять policy без re-prompting |

### 1.2. Что корректирую после анализа реальных данных

| # | Рекомендация v1 | Коррекция | Почему |
|---|---|---|---|
| 6 | `default_location` при extraction failure — использовать | **Отклоняю полностью** | Кейс `sobakusel_default_location_conflict`: пост анонсировал `Собакусъел` (реальная площадка), канал `meowafisha` имеет `default_location=YΛTЬ кофейня`. Override перетирал правильную venue → ложный дубль. В реальности extracted venue всегда сильнее channel default. `default_location` → **только metadata-hint** для shortlist расширения, никогда не override |
| 7 | Полный запрет LLM для `multi_event` | **Смягчаю → отдельный policy profile** | Кейс `hudozhnitsy_5way_cluster`: digest-пост (`multi_event`) создал separate event 2779 «Художницы — Филиал Третьяковской», который должен мержиться с 2801 (official TG-пост). Полный запрет LLM потеряет этот merge. Решение: `multi_event` → LLM разрешён, но default verdict `gray_create` вместо `merge`, и нужен ≥ 2 strong сигнала для `merge` |
| 8 | Фиксированные scoring weights | **Заменяю калиброванными** | `ticket_link` у нас часто generic (Собор `tickets.sobor-kaliningrad.ru/scheme/...` — один URL на два разных концерта: кейс `cathedral_shared_ticket_false_friend`). `poster_hash` полностью совпадает у всех дат recurring спектакля (`backstage_tour_weekly_run`, `dramteatr_number13_recurring`). Нужен `ticket_owner_count` discriminator и `date_blocker` для recurring poster |

### 1.3. Новые рекомендации, которых не было в v1

| # | Рекомендация | Обоснование |
|---|---|---|
| 9 | **Hard date-blocker**: если `date_A ≠ date_B` → force `different`, LLM не может override | `backstage_tour_weekly_run` (разные даты, same title/poster/source), `actopus_three_day_run` (14-15-16 января), `dramteatr_number13_recurring` (январь/февраль/март). Longrun корректно ставит `different` на эти пары — подтверждение, что date-mismatch должен быть hard blocker |
| 10 | **Same-date time-conflict guard**: если `time_A ≠ time_B` и оба non-default и разница > 30 мин → force `different` | `treasure_island_double_show` (11:00 vs 14:00), `frog_princess_double_show` (11:00 vs 14:00). Longrun ставит `gray` — это **неправильно**, должно быть `different`. Guard исправит это |
| 11 | **Source-kind propagation в scoring**: `source_kind=recurring_page` → poster/source_url/ticket не пробрасывают identity weight | Recurring репертуарные страницы (`dramteatr39.ru/spektakli/...`) пере-используют один URL, одну афишу, один ticket root для всех дат. Без kind-awareness система видит «poster_same + source_url_same + ticket_same» и думает «дубль» |
| 12 | **Same-source triple-deduplicate с extraction rescue** | `led_hearts_same_post_triple_duplicate`: один TG-пост породил три event с разными extracted dates/times (07:03, 08.03, 11:00). Source-owner guard + rescue по canonical occurrence |

---

## 2. Prompt Rewrite

### 2.1. `_llm_match_event` → `_llm_pairwise_triage`

#### Что убрать (текущие строки L6428-6443):

```diff
- "Найди наиболее вероятное совпадение или верни null.\n"
- "Не возвращай null, если есть правдоподобный матч: лучше выбрать наиболее вероятное и снизить confidence.\n"
```

#### Полный новый промпт:

```python
prompt = (
    "Ты — судья по идентификации событий.\n"
    "Тебе дана пара: новый анонс (candidate) и уже существующее событие (existing).\n"
    "Определи, одно ли это и то же событие.\n\n"

    "Оцени каждый сигнал ОТДЕЛЬНО:\n\n"

    "1) TITLE:\n"
    "- exact_match: названия идентичны (кроме эмодзи/регистра/пробелов)\n"
    "- alias_match: одно — бренд/формат, другое — конкретная программа, но явно про то же\n"
    "- mismatch: названия про разное\n\n"

    "2) DATE:\n"
    "- exact: даты совпадают\n"
    "- overlap: периоды пересекаются (для выставок с end_date)\n"
    "- mismatch: разные даты\n\n"

    "3) TIME:\n"
    "- exact: времена совпадают\n"
    "- weak: time=00:00 или time_is_default=true — неизвестно, не конфликт\n"
    "- doors_vs_start: расхождение ≤ 90 мин, одно похоже на 'сбор/двери', другое на 'начало' — не конфликт\n"
    "- mismatch: разные времена (оба known), разница > 30 мин\n\n"

    "4) VENUE:\n"
    "- exact: площадки идентичны\n"
    "- alias_match: одно сокращение/транслит другого, или hall-level vs building-level одной площадки\n"
    "- mismatch: явно разные площадки\n\n"

    "5) CONTEXT:\n"
    "- strong_match: совпадают участники / программа / OCR афиши / описание\n"
    "- moderate_match: частичное совпадение (часть участников, похожий текст)\n"
    "- mismatch: явно разный контент\n"
    "- no_data: недостаточно данных для оценки контекста\n\n"

    "6) TICKET:\n"
    "- strong_match: один и тот же specific URL (не root сайта площадки)\n"
    "- weak: оба ведут на корневой сайт площадки (generic) — НЕ доказательство\n"
    "- no_data: нет ticket или разные\n\n"

    "ВЕРДИКТ (строго одно из четырёх):\n"
    "- same_event: ≥2 strong/exact сигнала, 0 mismatch на DATE или VENUE\n"
    "- likely_same: ≥2 moderate, нет hard mismatch (date или venue)\n"
    "- different: mismatch на DATE, ИЛИ (mismatch TIME при обоих known), ИЛИ (mismatch VENUE + mismatch TITLE)\n"
    "- uncertain: недостаточно сигналов или противоречивые данные\n\n"

    "КРИТИЧЕСКИ ВАЖНО:\n"
    "- Ошибочная склейка ХУЖЕ дубля. При сомнении → uncertain.\n"
    "- НЕ форсируй same_event при schedule/series/recurring: один source_url / один poster / один ticket "
    "могут быть общими для разных дат/показов того же спектакля. Это нормально, не доказательство identity.\n"
    "- Generic ticket link (root сайта площадки) — НЕ доказательство identity.\n"
    "- Для длинных событий (выставка/ярмарки) пересечение периодов + площадка НЕ означает дубль: "
    "в одном музее может идти несколько разных выставок.\n"
    "- Используй ТОЛЬКО представленные факты. НЕ додумывай совпадения.\n\n"

    "Ответь строго JSON по schema.\n\n"
    f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
)
```

#### JSON schema для ответа:

```python
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "enum": ["exact_match", "alias_match", "mismatch"]},
                "date": {"type": "string", "enum": ["exact", "overlap", "mismatch"]},
                "time": {"type": "string", "enum": ["exact", "weak", "doors_vs_start", "mismatch"]},
                "venue": {"type": "string", "enum": ["exact", "alias_match", "mismatch"]},
                "context": {"type": "string", "enum": ["strong_match", "moderate_match", "mismatch", "no_data"]},
                "ticket": {"type": "string", "enum": ["strong_match", "weak", "no_data"]},
            },
            "required": ["title", "date", "time", "venue", "context", "ticket"],
        },
        "verdict": {"type": "string", "enum": ["same_event", "likely_same", "different", "uncertain"]},
        "reason_short": {"type": "string"},
    },
    "required": ["evidence", "verdict", "reason_short"],
}
```

#### Ожидаемый размер payload:

- Compact candidate: title + date + time + location_name + city + ticket_link + source_text (600 chars) + poster_title + poster_ocr (400 chars) ≈ **1200 chars**
- Compact existing: id + title + date + time + location_name + ticket_link + description (400 chars) ≈ **600 chars**
- Pre-computed hints: 8 boolean/enum полей ≈ **200 chars**
- Промпт: ≈ **2200 chars**
- **Итого input ≈ 4200 chars ≈ 1050 tokens**
- Output: structured JSON ≈ **150-250 tokens**
- **Total per call ≈ 1200-1300 tokens** → при tpm=12000 можно ~9 triage calls/min, при rpm=20 — запас есть

### 2.2. `_llm_match_or_create_bundle` — redesign

**Архитектурная рекомендация**: разделить на два последовательных вызова:
1. `_llm_pairwise_triage` (тот же промпт из 2.1) — только identity verdict
2. `_llm_create_description_facts_and_digest` (уже есть, L6030) — только если verdict ≠ same_event

**Если бизнес-ограничения требуют сохранить single-call** (для экономии latency на create-path):

#### Что убрать (текущие строки L6541-6557):

```diff
- "- Если хотя бы одно событие в `events` совпадает по якорям (дата + начало времени/пустое время + площадка) "
- "и названию/участникам, это дубль — выбирай `action=match` и ставь `confidence` заметно выше `threshold`.\n\n"
```

#### Замена в Шаге 1 match-or-create:

```python
    "Шаг 1) IDENTITY TRIAGE:\n"
    "- Оцени каждого кандидата из `events` как пару с новым анонсом.\n"
    "- Для лучшего кандидата верни evidence по сигналам: title, date, time, venue, context, ticket.\n"
    "- ВЕРДИКТ:\n"
    "  - same_event → action=match, confidence ≥ 0.75\n"
    "  - likely_same → action=match, confidence 0.50..0.65\n"
    "  - different → action=create\n"
    "  - uncertain → action=create, reason_short начни с 'uncertain:'\n\n"

    "КРИТИЧЕСКИ ВАЖНО:\n"
    "- Ошибочная склейка ХУЖЕ дубля.\n"
    "- НЕ форсируй match без ≥2 независимых strong/moderate сигналов.\n"
    "- Один source_url / poster / ticket может быть общим для разных дат recurring спектакля — "
    "это НЕ доказательство identity.\n"
    "- Generic ticket link (root сайта площадки) НЕ считай identity proof.\n"
    "- time=00:00 / time_is_default=true → неизвестное время (слабый сигнал, не конфликт).\n"
    "- Для schedule/multi-event: общий источник + похожий текст САМИ ПО СЕБЕ не доказывают identity.\n"
    "- Используй ТОЛЬКО представленные факты. НЕ додумывай.\n\n"
```

### 2.3. Compact pairwise payload для Gemma

```python
payload = {
    "candidate": {
        "title": clean_title,                    # обязательно
        "date": candidate.date,                  # обязательно
        "time": candidate.time,                  # обязательно
        "time_is_default": bool(...),            # обязательно
        "location_name": normalized_location,    # уже нормализованное
        "city": candidate.city,                  # обязательно
        "ticket_link": candidate.ticket_link,    # обязательно
        "source_kind": source_kind,              # NEW: single_event | multi_event | recurring_page
        "text": _clip(clean_source_text, 600),   # clip до 600
        "poster_title": best_poster_title,       # одна строка или null
        "poster_ocr": _clip(best_ocr, 400),      # clip до 400
    },
    "existing": {
        "id": ev.id,                             # обязательно
        "title": ev.title,                       # обязательно
        "date": ev.date,                         # обязательно
        "time": ev.time,                         # обязательно
        "time_is_default": bool(...),            # обязательно
        "location_name": ev.location_name,       # обязательно
        "ticket_link": ev.ticket_link,           # обязательно
        "description": _clip(ev.description, 400), # clip до 400
        "poster_title": existing_poster_title,   # одна строка или null
    },
    "hints": {
        "title_exact": bool,        # pre-computed
        "title_related": bool,      # pre-computed
        "venue_match": bool,        # pre-computed (after normalization)
        "ticket_same": bool,        # pre-computed
        "ticket_is_generic": bool,  # NEW: true если URL ведёт на root площадки
        "poster_overlap": bool,     # pre-computed
        "source_kind_risk": str,    # "none" | "recurring" | "multi_event" | "schedule"
        "time_note": str,           # "exact" | "both_default" | "one_default" | "doors_vs_start" | "conflict"
    }
}
```

**Что убрано vs текущий payload:**
- `description` existing: 600→400 chars (identity не требует длинного описания)
- `source_text` existing: **убрано полностью** (identity не требует; есть `description`)
- `poster_texts` массив: заменён на один `poster_ocr` (best, 400 chars)
- `poster_titles` массив: заменён на один `poster_title` (best)
- `raw_excerpt` candidate: **убрано** (есть `text`)
- Добавлены `hints` — deterministic-слой уже посчитал часть сигналов

**Экономия**: ~40% reduction input tokens vs текущий payload.

### 2.4. Gemma-специфичные prompt rules

| Правило | Зачем | Как реализовать |
|---|---|---|
| `enum`-значения в schema | Gemma точнее на enum чем на free text | `TRIAGE_SCHEMA` с enum для каждого evidence field |
| `temperature=0.0` | Стабильность verdict | `temperature=0.0` в `_ask_gemma_json` для triage |
| `max_tokens=300` | Structured JSON ≈ 150-250 tokens | Меньше budget → быстрее, меньше TPM |
| `"Используй ТОЛЬКО представленные факты. НЕ додумывай."` | Gemma 27B склонна фантазировать о совпадениях | Explicit anti-hallucination rule |
| `hints` в payload | Уменьшает нагрузку на reasoning Gemma | LLM не пересчитывает то, что уже посчитано |
| Один `existing` вместо массива | Pairwise → один объект, не массив | Убирает confusion при выборе из нескольких |

---

## 3. Decision Policy

### 3.1. Verdict → Runtime Action

```
Deterministic layer:
  score ≥ MERGE_THRESHOLD (10) → merge (без LLM)
  score ≤ CREATE_THRESHOLD (3) → create (без LLM)
  hard blocker triggered → force different/create (без LLM)
  date_mismatch → force different (без LLM)
  same_date + time_conflict (>30 мин, оба known, не doors/start) → force different (без LLM)

LLM triage layer (только при 4 ≤ score ≤ 9):
  same_event → merge
  likely_same → gray_create_softlink
  different → create
  uncertain → gray_create_softlink
```

### 3.2. Deterministic blockers (приоритет НАД LLM)

| Blocker | Условие | Действие | Кейсы |
|---|---|---|---|
| `date_mismatch` | `date_A ≠ date_B` и не overlap (для events без end_date) | **force different** | `backstage_tour_weekly_run`, `actopus_three_day_run`, `dramteatr_number13_recurring` |
| `time_conflict` | оба `time` known, оба `time_is_default=false`, разница > 30 мин, не doors_vs_start | **force different** | `treasure_island_double_show` (11:00/14:00), `frog_princess_double_show` (11:00/14:00) |
| `city_mismatch` | `city_A ≠ city_B` (after normalization) | **force different** | `oncologists_zelenogradsk_separate` (Светлогорск ≠ Зеленоградск) |
| `title_mismatch + venue_mismatch` | `title_related=false` И `venue_match=false` | **force different** | Разные события в разных местах |

### 3.3. Deterministic auto-merge (без LLM)

| Условие | Score | Действие | Кейсы |
|---|---|---|---|
| `same_source_url + same_date` | +6 | auto-merge base | `led_hearts_same_post_triple_duplicate` |
| `title_exact + ticket_specific_same + same_date` | +8 → ≥10 | auto-merge | `fort_excursion_duplicate`, `fort_night_duplicate` |
| `poster_overlap + participant_overlap ≥ 3 + same_date + venue_match` | +7-8 → ≥10 | auto-merge | `shambala_cluster` |
| `text_containment + ticket_same + same_date` | +8 → ≥10 | auto-merge | `little_women_cluster` |

### 3.4. Policy profiles

#### Profile: `single_event` (default)

```
Shortlist: ±1 day, fuzzy venue, city_match
Deterministic: standard scoring
LLM triage: если 4 ≤ score ≤ 9
Default verdict при uncertain: gray_create_softlink
Max shortlist for LLM: top-3 by score
```

#### Profile: `multi_event`

```
Shortlist: same_date only, exact venue (после normalization)
Deterministic: standard scoring, но poster/source_url/ticket weight = 0 (т.к. общие для всех child events)
LLM triage: разрешён, но:
  - default verdict при uncertain: create (не gray)
  - merge только при LLM verdict = same_event (не likely_same)
  - нужен ≥ 2 strong evidence signals для merge
Max shortlist for LLM: top-2
```

#### Profile: `recurring_page`

```
Shortlist: exact_date only (не ±1 day)
Deterministic:
  - poster_overlap weight = 0 (same poster на все даты)
  - source_url weight = 0 (same page на все даты)
  - ticket_link weight = 0 если generic (same ticket root на все даты)
  - date_mismatch = hard blocker (разные даты = разные показы)
LLM triage: минимальный, только если exact date + exact venue + title_related
Default verdict: different
```

#### Profile: `follow_up`

```
Shortlist: same_date, same venue, source_url owner check
Deterministic: standard + bonus за same_source_url author
LLM triage: стандартный pairwise
Default verdict при uncertain: gray_create_softlink
Note: follow-up обычно explicit (пост-дополнение с пометкой)
```

---

## 4. Case-Class Decisions

### 4.1. must_merge — confirmed duplicate clusters

| Case | Verdict | Routing | Mechanism | Ожидаемый score |
|---|---|---|---|---|
| `fort_excursion_duplicate` (2729/2732) | same_event | **auto-merge** | title_exact + ticket_specific + text_same | ≥12 |
| `fort_night_duplicate` (2730/2733) | same_event | **auto-merge** | title_exact + ticket_specific + text_same | ≥12 |
| `shambala_cluster` (2799/2843/2844) | same_event | **auto-merge** | poster_overlap + participant_overlap=7 + venue + date | ≥10 |
| `sobakusel_default_location_conflict` (2793/2810) | same_event | **LLM triage** | title_related + date + context strong. Venue mismatch **rescued** by normalization (оба «Собакусъел», после fix default_loc override) | 7-9 → LLM → same_event |
| `gromkaya_doors_vs_start` (2667/2792) | same_event | **LLM triage** | title_related + date + venue + doors_vs_start time. LLM видит `time_note: doors_vs_start` → не считает conflict | 6-8 → LLM → same_event |
| `hudozhnitsy_5way_cluster` (2541/2675/2779/2801/2838) | same_event | **LLM triage (pairwise)** | title alias + date + venue + participant overlap. Пятиэлементный cluster — каждая пара проверяется отдельно | 6-9 → LLM → same_event |
| `prazdnik_u_devchat_broken_extraction` (2789/2802/2803) | same_event | **LLM triage + extraction rescue** | Broken extracted fields (address в title). После normalization → title_related + date + venue | 5-7 → LLM → same_event |
| `little_women_cluster` (2761/2815/2816/2817) | same_event | **auto-merge** (некоторые пары) / **LLM** (2761/2815) | ticket_same + text_containment для 2815/2817. Для 2761/2815 — participant_overlap=10 | ≥10 или 7-9 |
| `garage_time_correction` (2546/2554) | same_event | **auto-merge** | same_source_url + same_date + title_exact | ≥10 |
| `makovetsky_chekhov_duplicate` (2758/2759) | same_event | **LLM triage** | title alias (brand vs item) + date + venue + participant overlap | 6-8 → LLM → same_event |
| `oncologists_svetlogorsk_duplicate` (2710/2721) | same_event | **LLM triage** | title alias + date + city_match + context | 5-7 → LLM → same_event |
| `sisters_followup_post` (2676/2677) | same_event | **LLM triage** | follow-up source from same author, same date, title alias | 6-8 → LLM → same_event |
| `led_hearts_same_post_triple_duplicate` (2845/2846/2847) | same_event | **auto-merge** via source owner guard | same_source_url → все три из одного поста. Rescue: canonical occurrence 2847 (11:00) — merge siblings into canonical | ≥10 |
| `zoikina_missing_time_duplicate` (282/360) | same_event | **LLM triage** | title_exact + date_exact + venue (если нормализовано) + missing time = weak (не conflict) | 7-9 → LLM → merge |

### 4.2. must_not_merge — danger cases

| Case | Verdict | Routing | Key blocker | Почему не merge |
|---|---|---|---|---|
| `oncologists_zelenogradsk_separate` (2710/2712) | different | **hard blocker: city_mismatch** | Светлогорск ≠ Зеленоградск | Regional campaign → разные города = разные events |
| `museum_holiday_program_multi_child` (2743/2744/2745) | different | **multi_event policy** | Одна VK-стена, один пост → три child events. `source_kind=multi_event` → poster/source_url не несут identity weight | Umbrella holiday post, child events разные |
| `oceania_march_lecture_series` (2660-2663) | different | **hard blocker: date_mismatch** | 06.03, 13.03, 20.03, 27.03 — разные даты. Same ticket (generic clck.ru) не помогает | Серия лекций из одного VK-поста |
| `backstage_tour_weekly_run` (2611-2614) | different | **hard blocker: date_mismatch** | 07.03, 14.03, 21.03, 28.03 — разные даты. Same poster/source_url = recurring pattern | Еженедельная экскурсия |
| `actopus_three_day_run` (2193-2195) | different | **hard blocker: date_mismatch** | 14.01, 15.01, 16.01 — разные даты. Same poster/source = multi-day run | Три подряд спектакля |
| `treasure_island_double_show` (2572/2573) | different | **hard blocker: time_conflict** | 11:00 ≠ 14:00, same date, same everything else. Два показа в один день — normal theatre practice | Legal double-show |
| `frog_princess_double_show` (2576/2577) | different | **hard blocker: time_conflict** | 11:00 ≠ 14:00, аналогично | Legal double-show |
| `dramteatr_number13_recurring` (2049/2161/2501) | different | **hard blocker: date_mismatch** | Январь/февраль/март — разные даты. Same poster/source = recurring repertory | Повторяющийся репертуарный спектакль |
| `cathedral_shared_ticket_false_friend` (1979/2278) | different | **LLM → different** + ticket_is_generic=true | Same ticket URL, same date, same time, same venue — но РАЗНЫЕ концерты! `ticket_is_generic=true` → Gemma не считает это identity proof | Два разных концерта с одним generic ticket URL |
| `dramteatr_same_slot_cross_title` (1428/1677) | different | **LLM → different** + title_mismatch | Same slot (venue+date+time) из разных источников, но completely unrelated titles: «Коралина в стране кошмаров» vs «Северное сияние» | Разные спектакли в одном слоте (разные залы?) |

### 4.3. must_skip — non-events

| Case | Verdict | Routing |
|---|---|---|
| `giveaway_non_event` (2701) | skip_non_event | **hard guard** в pre-LLM layer. Розыгрыш билетов = не событие |

### 4.4. Позиция по классам кейсов

#### Time correction
**Кейс**: `garage_time_correction` (2546/2554) — один source_url, исправление времени.
**Позиция**: `same_source_url + same_date` → auto-merge. Time update идёт через merge logic (canonical source priority).

#### Doors vs start
**Кейс**: `gromkaya_doors_vs_start` (2667/2792) — «сбор 19:30» vs «начало 20:00».
**Позиция**: расхождение ≤ 90 мин при паттерне doors/start → `time_note: doors_vs_start` в hints → LLM не считает конфликтом. Промпт явно: `"doors_vs_start: расхождение ≤ 90 мин, одно похоже на 'сбор/двери', другое на 'начало' — не конфликт"`.

#### Follow-up update post
**Кейс**: `sisters_followup_post` (2676/2677) — дополнение от того же автора VK.
**Позиция**: follow-up определяется по `same_source_url author` + `same_date`. В scoring → бонус. Default verdict: gray_create → с accumulation hints → LLM → merge.

#### Brand vs item (title framing)
**Кейсы**: `shambala_cluster` (бренд vs line-up), `makovetsky_chekhov_duplicate` (артист vs программа).
**Позиция**: title_related (через `_titles_look_related`) + participant_overlap + venue + date → LLM triage. Промпт объясняет: `"alias_match: одно — бренд/формат, другое — конкретная программа, но явно про то же"`.

#### Same-source duplicate
**Кейсы**: `garage_time_correction`, `led_hearts_same_post_triple_duplicate`.
**Позиция**: `same_source_url + same_date` = strong identity signal (+6 score). При triple-duplicate — source owner guard: один active event per source_url per date.

#### Same-source multi-child schedule risk
**Кейсы**: `oceania_march_lecture_series`, `museum_holiday_program_multi_child`.
**Позиция**: `source_kind=multi_event` → poster/source_url/ticket weight = 0. Date_mismatch = hard blocker. Merge только при exact date + venue + title_related + LLM = same_event.

#### Recurring repertory page
**Кейсы**: `backstage_tour_weekly_run`, `dramteatr_number13_recurring`.
**Позиция**: `source_kind=recurring_page` → все shared signals (poster, source_url, ticket) weight = 0. Date_mismatch = hard blocker. Промпт: `"один source_url / один poster / один ticket могут быть общими для разных дат/показов того же спектакля. Это нормально, не доказательство identity."`.

#### Same-day double show
**Кейсы**: `treasure_island_double_show`, `frog_princess_double_show`.
**Позиция**: **hard blocker time_conflict**. Если оба time known, time_is_default=false, разница > 30 мин → force different. Это закрывает кейс без LLM. Longrun сейчас даёт gray — **это баг**, должен быть different.

#### Generic ticket false-friend
**Кейс**: `cathedral_shared_ticket_false_friend` (1979/2278).
**Позиция**: добавить `ticket_is_generic: true` в hints. Промпт: `"weak: оба ведут на корневой сайт площадки (generic) — НЕ доказательство"`. В scoring: generic ticket → +1 (weak) вместо +4 (strong). Определение generic: `ticket_owner_count > 5` или URL ≠ event-specific path.

#### Corrupted extraction / venue noise
**Кейсы**: `prazdnik_u_devchat_broken_extraction` (address в title), `sobakusel_default_location_conflict`.
**Позиция**: normalization layer ДО scoring: venue aliasing, confusable chars. Для extraction corruption → LLM triage с текстовым контекстом позволяет увидеть identity через content, несмотря на broken fields.

#### `default_location` conflict
**Кейс**: `sobakusel_default_location_conflict`.
**Позиция**: **Отклоняю v1 рекомендацию целиком**. `default_location` НЕ используется для venue resolution. Остаётся только metadata-hint для расширения shortlist. В handlers.py уже найдена строка override (L2878-2894) — её нужно убрать или заменить на weak-hint propagation.

---

## 5. Dry-Run / Validation Feedback

### 5.1. Достаточность данных

| Аспект | Статус | Что добавить |
|---|---|---|
| must_merge cases | ✅ 14 кейсов, покрывают все классы | Достаточно |
| must_not_merge cases | ✅ 11 кейсов (включая 10 sample refresh) | Достаточно |
| must_skip cases | ⚠️ 1 кейс (giveaway) | Добавить 2-3: реклама, пост-отчёт, поздравление |
| Recurring repertory | ✅ 3 cases (backstage, number13, actopus) | Достаточно |
| Same-day double-show | ✅ 2 cases (treasure_island, frog_princess) | Достаточно |
| Generic ticket false-friend | ✅ 1 case (cathedral) | Добавить ещё 1 если есть |
| Follow-up posts | ⚠️ 1 case (sisters) | Добавить 1-2 с VK repost pattern |

### 5.2. Проблемы в longrun, выявленные на sample refresh

| Case | Longrun verdict | Ожидаемый | Проблема | Исправление |
|---|---|---|---|---|
| `led_hearts` (2845/2846, 2845/2847, 2846/2847) | gray | **merge** | Source owner guard не сработал (разные telegra.ph pages, но один TG source) | Добавить source_url matching по TG post URL, не только по telegra.ph |
| `actopus` (2193/2194, 2194/2195) | gray | **different** | Date_mismatch не стал hard blocker | Усилить date_mismatch → hard different |
| `treasure_island` (2572/2573) | gray | **different** | Time_conflict не стал hard blocker | Добавить time_conflict hard blocker (>30 мин, оба known) |
| `frog_princess` (2576/2577) | gray | **different** | Аналогично | Аналогично |
| `oceania` (2660/2663) | gray | **different** | Same generic ticket тянет в gray | ticket_is_generic → weight 0 + date_mismatch → hard |

### 5.3. Что из expanded sample — true gray vs переквалификация

| Case | Текущий longrun | Моя оценка | Обоснование |
|---|---|---|---|
| `cathedral_shared_ticket_false_friend` (1979/2278) | gray (dryrun) | **different** → должен стать different после добавления `ticket_is_generic` hint и title_mismatch blocker | Два совершенно разных концерта. «Английская придворная культура» ≠ «Королева фей» |
| `dramteatr_same_slot_cross_title` (1428/1677) | gray (dryrun) | **different** → title_mismatch = hard signal | «Коралина в стране кошмаров» ≠ «Северное сияние» |
| `treasure_island_double_show` | gray | **different** → time_conflict hard blocker | 11:00 ≠ 14:00, legal double-show |
| `frog_princess_double_show` | gray | **different** → time_conflict hard blocker | 11:00 ≠ 14:00 |
| `actopus` 2193/2194, 2194/2195 | gray | **different** → date_mismatch hard blocker | 14≠15, 15≠16 января |
| `must_not_nutcracker_two_shows` | gray | **legitimate gray** → в идеале different, но без hard time_conflict info это gray нормально | Возможно two-show pattern, нужно verify times |
| `must_not_lecture_cycle` | gray | **legitimate gray** → без title_mismatch hard signal, при same venue + same date, gray корректен | Лекции цикла, нужен LLM context для different |

### 5.4. Prompt-фразы для Gemma против переоценки recurring signals

Конкретные формулировки, которые нужно включить в промпт:

```
"- Один source_url / один poster / один ticket могут быть общими для РАЗНЫХ дат/показов
  того же спектакля или серии лекций. Это нормально для репертуарных/recurring страниц.\n"
"- Совпадение poster + source_url + ticket при РАЗНЫХ датах → different, не merge.\n"
"- Generic ticket (ведёт на корневой сайт площадки, не на конкретное событие) → слабый сигнал, НЕ доказательство.\n"
"- Same slot (venue + date + time) при unrelated title → НЕ merge. Это могут быть разные события в разных залах площадки.\n"
```

---

## 6. Rollout Guidance

### Phase 0: Hard Blockers (1 день, zero risk)

**Изменения:**
1. `date_mismatch` → force different (без LLM)
2. `time_conflict` (>30 мин, оба known, не doors/start) → force different
3. `city_mismatch` (после alias normalization) → force different

**Проверка:**
- Прогон longrun с новыми blockers
- Все `backstage_tour`, `actopus`, `number13`, `treasure_island`, `frog_princess` должны стать `different`
- Все must_merge кейсы должны остаться `merge` или `gray` (не затронуты)

### Phase 1: Normalization Layer (2 дня, zero risk)

**Изменения:**
1. Venue alias normalization в `_normalize_location`
2. City alias map (`Гурьевский городской округ` → `Гурьевск`)
3. Title confusable chars normalization (mixed script, ё→е)
4. `default_location` → weak hint only (убрать override в `handlers.py`)

**Проверка:**
- `fort_*` кейсы → venue_match=true после normalization
- `sobakusel` → extracted venue preserved, not overridden
- `city_alias_guryevsk` → city normalized

### Phase 2: Deterministic Scoring Calibration (2 дня, low risk)

**Изменения:**
1. Новая `_identity_score()` с калиброванными весами
2. `ticket_is_generic` discriminator
3. `source_kind` awareness (recurring_page → poster/source_url/ticket weight = 0)
4. Auto-merge при score ≥ 10, auto-create при score ≤ 3

**Проверка:**
- fort_*, shambala → auto-merge (score ≥ 10)
- cathedral_false_friend → no auto-merge (ticket_is_generic → score < 10)
- backstage_tour, number13 → different (date_mismatch blocker from Phase 0)

### Phase 3: LLM Prompt Redesign (3 дня, medium risk)

**Изменения:**
1. Замена `_llm_match_event` → `_llm_pairwise_triage` с новым промптом
2. Новая schema `TRIAGE_SCHEMA` с structured evidence
3. `temperature=0.0`, `max_tokens=300`
4. Compact pairwise payload с hints

**Проверка:**
- **Shadow mode**: параллельный прогон old vs new промпта на тех же парах
- Регрессионный тест: все 25 casepack кейсов
- Longrun должен дать ≥ 32/32 acceptable

### Phase 4: Gray State + Source Guard (3 дня, low risk)

**Изменения:**
1. `gray_create_softlink` runtime action
2. Source owner guard (один active event per source_url per date)
3. `source_kind` propagation из extractor

**Проверка:**
- `led_hearts_same_post_triple_duplicate` → source guard triggers merge to canonical
- Uncertain LLM verdicts → gray state (не merge, не create)

### Phase 5: Multi-event Policy (2 дня, low risk)

**Изменения:**
1. `multi_event` policy profile
2. `recurring_page` policy profile
3. `skip_non_event` в hard guards

**Проверка:**
- `museum_holiday_program` → create (multi_event policy)
- `oceania_lecture_series` → different (date_mismatch + recurring policy)
- `giveaway_non_event` → skip

### Regression test suite

Минимальный набор для CI: 25 casepack кейсов + 12 longrun must_not_merge = **37 regression tests**.

Каждый тест: пара event_ids → ожидаемый final_decision (`merge / gray / different / skip`).

---

## 7. Residual Risks

| Риск | Severity | Mitigation | Статус |
|---|---|---|---|
| **Same-source triple+ duplicate с bad extraction** | High | Source owner guard + canonical occurrence rescue | Addressed by Phase 4 |
| **Generic ticket false-friend при same slot** | High | `ticket_is_generic` discriminator + title_mismatch blocker | Addressed by Phase 2 |
| **Hall-level venue confusion** | Medium | Venue hierarchy (building vs hall) в normalization; `allow_parallel_events` flag | Partially addressed |
| **Title brand vs item без ticket/poster overlap** | Medium | LLM triage с `uncertain` default → gray | Addressed by design |
| **Campaign-wide post vs city-specific child** | Low | City-level anchor matching + city_mismatch blocker | Addressed by Phase 0 |
| **Non-event content (реклама, розыгрыши, отчёты)** | Medium | `skip_non_event` в hard guards | Addressed by Phase 5, but needs more gold cases |
| **Extraction corruption** (address в title, description в location) | Medium | Normalization layer + LLM context rescue | Partially addressed; long-tail risk remains |
| **VK repost / forward chain не распознаются как follow-up** | Low | Same-author detection в source metadata | Not yet addressed |
| **Gemma instability на edge-case payloads** | Low | Schema enforcement + retry + fallback to 4o | Already in runtime |
| **TPM exhaustion при массовых schedule imports** | Medium | Deterministic auto-merge/create обходит LLM на 60-70% пар; compact payload на остальных | Addressed by design |
| **Double-show при time_is_default=true на одном из events** | Low | time_conflict blocker only activates when both times are known. Fallback: LLM triage | By design |
| **Same poster text (OCR) при разных визуальных афишах** | Low | Hash-based poster_overlap vs OCR-based poster_text_overlap — разные signals | Need separate scoring |

### Ключевой остаточный gap

Самый опасный кейс из нерешённых: **две разных выставки в одном музее с пересекающимися периодами**. Для них:
- date = overlap (not mismatch)
- venue = exact
- poster = разные
- title = разные

Решение: `time_conflict` blocker не поможет (обе без времени). Title_mismatch + venue_exact → LLM triage → different. Промпт уже содержит: `"Для длинных событий (выставка/ярмарки) пересечение периодов + площадка НЕ означает дубль"`.
