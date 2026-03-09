# Opus → Gemma Event Copy: Pattern Redesign Narrow Follow-up

Дата: 2026-03-07

---

## 1. `copy_assets v1`: что обязательно, что опционально, что отложить

### v1 required

| Field | Зачем в v1 | Extraction burden |
|-------|-----------|------------------|
| `core_angle` (string) | Без него routing и lead variety невозможны. Всё остальное опирается на наличие dominant angle. | Низкий: 1 строка, ≤15 слов |
| `format_signal` (enum) | Deterministic routing primitive. Уже почти есть в виде `event_type`, но нужен нормализованный enum. | Нулевой: классификация |
| `program_highlights` (string[], 2-6) | Нужен для `program_led` routing gate и для generation: даёт generation знание, что показывать в списке. | Низкий: уже частично извлекаются как facts |
| `experience_signals` (string[], grounded) | Входит в `compact_fact_led` gate (отсутствие = бедный source). Помогает generation писать от experience. | Средний: grounding rule нужен |
| `why_go_candidates` ({reason, strength}[]) | Why-go gate. Без него `value_led` и inline why-go невозможны. | Средний: strength classification |
| `credibility_signals` (string[]) | Поддерживает why-go и `person_led`. Уже частично извлекаются как facts. | Низкий |

### v1 optional (извлекать, но не давать routing dependency)

| Field | Зачем optional | Что делать в v1 |
|-------|---------------|----------------|
| `voice_fragments` ({text, speaker, ...}[]) | Quote_led отложен, но inline quote и epigraph selection выигрывают от наличия. | Извлекать, передавать в generation, **но НЕ использовать для routing**. Epigraph picker может использовать. |
| `subformat` (string?) | Уточняет format для рендера, но не влияет на routing. | Извлекать, передавать, не routing dependency. |

### Defer to later

| Field | Почему defer |
|-------|-------------|
| `tone_hint` | Tone-adaptive generation — research-only. Поле без payoff в v1. Убрать из v1 schema целиком. |
| `routing_features` (6 booleans) | См. Section 4: лучше derive routing в runtime из primitive fields. Нет нужды тащить 6 boolean в LLM output. |
| `contrast_or_tension` | Editorializing risk. Полезно для scene_led lead, но scene_led тоже deferred (см. Section 2). |
| `scene_cues` | Deferred вместе с `scene_led` (см. Section 2). |

### Итого: v1 schema

```json
{
  "type": "object",
  "properties": {
    "facts": { "type": "array", "items": { "type": "string" } },
    "copy_assets": {
      "type": "object",
      "properties": {
        "core_angle":          { "type": "string" },
        "format_signal":       { "type": "string", "enum": ["спектакль","лекция","концерт","показ","мастерская","экскурсия","игра","фестиваль","встреча"] },
        "subformat":           { "type": ["string", "null"] },
        "program_highlights":  { "type": "array", "items": { "type": "string" } },
        "experience_signals":  { "type": "array", "items": { "type": "string" } },
        "why_go_candidates":   { "type": "array", "items": {
          "type": "object",
          "properties": {
            "reason":   { "type": "string" },
            "strength": { "type": "string", "enum": ["strong", "regular"] }
          },
          "required": ["reason", "strength"]
        }},
        "voice_fragments":     { "type": "array", "items": {
          "type": "object",
          "properties": {
            "text":              { "type": "string" },
            "speaker":           { "type": ["string", "null"] },
            "speaker_role":      { "type": ["string", "null"] },
            "is_direct_quote":   { "type": "boolean" },
            "opens_event_theme": { "type": "boolean" }
          },
          "required": ["text", "speaker", "is_direct_quote", "opens_event_theme"]
        }},
        "credibility_signals": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["core_angle", "format_signal", "program_highlights",
                    "experience_signals", "why_go_candidates", "credibility_signals"]
    }
  },
  "required": ["facts", "copy_assets"],
  "additionalProperties": false
}
```

Это **10 полей** вместо 14. Убраны: `tone_hint`, `routing_features`, `contrast_or_tension`, `scene_cues`. Добавлены как optional (не в `required`): `voice_fragments`, `subformat`.

---

## 2. `scene_led`: решение

**Решение: `scene_led` целиком откладывается и НЕ входит в first medium-risk branch.**

Обоснование:

1. `scene_cues` — самое хрупкое extraction поле. Gemma может выдумывать «атмосферу» там, где в source её нет. Grounding rule (`ТОЛЬКО если в source есть конкретная сенсорная деталь`) трудно верифицировать автоматически.

2. `scene_led` без reliable `scene_cues` будет misfire. В лучшем случае — generic «представьте себе...» opening. В худшем — hallucinated atmosphere.

3. Удаление `scene_led` из v1 routing **не обедняет** v1 набор критически:
   - `topic_led` (default) уже перекрывает большинство событий, где scene_led мог бы включиться;
   - `person_led` покрывает speaker/artist events;
   - `value_led` покрывает events с сильным credibility signal;
   - `program_led` покрывает participatory formats.

4. Добавить `scene_led` позже (Phase 3) проще, чем убрать его из v1 после misfire: это additive change, не breaking change.

**Практический эффект на routing:**

```python
# v1 routing (без scene_led)
def determine_pattern_v1(copy_assets: dict, facts_text_clean: list[str]) -> str:
    # 1. Program-led
    if _is_program_led(copy_assets, facts_text_clean):
        return "program_led"

    # 2. Compact (poor source)
    if _is_poor_source(copy_assets, facts_text_clean):
        return "compact_fact_led"

    # 3. Person-led
    if _is_person_led(copy_assets):
        return "person_led"

    # 4. Value-led
    if _is_value_led(copy_assets):
        return "value_led"

    # 5. Default
    return "topic_led"
```

5 patterns вместо 7. `quote_led` и `scene_led` deferred.

---

## 3. Anti-template heuristics: hard bans vs soft preferences

### Hard bans (deterministic, coverage check enforces)

| Rule | Scope | Rationale |
|------|-------|-----------|
| Lead не начинается с `«{title}» — это ...` | All patterns | Самый частый template marker. Механическая связка title-is-definition. |
| Lead не содержит дословную копию title целиком | All patterns | Пересказ title = нулевая информативность lead. |
| Нет двух подзаголовков из heading stop-list в одном тексте | All patterns | «Подробности» + «О мероприятии» = бухгалтерский текст. |
| Нет duplicate facts: один факт не повторяется в lead и в ### секции | All patterns | Redundancy. |

### Soft preferences (generation prompt suggests, coverage может flag, но revise не ломает natural lead)

| Rule | Scope | Nuance |
|------|-------|--------|
| Lead предпочтительно начинается с core_angle, а не со слова из title | `topic_led`, `value_led`, `compact_fact_led` | **Исключение**: для `person_led` lead МОЖЕТ начинаться с имени из title, если это proper noun и credibility signal. |
| Нет двух подряд предложений с одинаковым началом | All patterns | Soft: иногда anaphora уместна. Coverage может flag, но revise решает. |
| Heading не повторяет слова из lead | All patterns | Soft: если lead упоминает performer и heading = «Кто на сцене», это не дубль. |

### Конкретная формулировка для generation prompt

Было (слишком жёстко):
```
- Lead не начинается со слова из title
```

Стало (точнее):
```
Hard ban:
- НЕ начинай lead с конструкции «{title} — это ...» или «{title} представляет собой ...».
- НЕ копируй title целиком в lead.

Soft preference:
- Предпочитай начинать lead с core_angle или с самого яркого факта.
- Исключение для person_led: если lead начинается с имени человека из title и это natural proper-noun opening, это допустимо.
- Исключение для format-specific openers: «Мастерская по ...», «Лекция о ...» допустимы, если дальше идёт angle, а не определение.
```

### Конкретная формулировка для coverage check

```
quality_flags.weak_lead:
- true, если lead дословно начинается с «{title} — это» или «{title} представляет собой».
- true, если lead не содержит ни одного слова из core_angle.
- false для person_led, если lead начинается с имени из title.
```

---

## 4. Runtime vs LLM: чёткое разделение

### Что только runtime

| Decision | Почему runtime, не LLM |
|---------|----------------------|
| Pattern selection (`determine_pattern_v1`) | Deterministic, дешёвый, debuggable, A/B testable. LLM routing создаёт black box. |
| Why-go gate (`should_include_why_go`) | Count/strength logic — trivial Python. |
| Compact detection (`is_poor_source`) | Len check — trivial Python. |
| Budget estimation | Already runtime (`_estimate_fact_first_description_budget_chars`). |
| Epigraph selection | Already runtime (`_pick_epigraph_fact`). |

### Что LLM извлекает (primitive signals)

| Signal | Тип | Почему LLM |
|--------|-----|-----------|
| `core_angle` | string | Требует semantic understanding source |
| `format_signal` | enum | Classification, одно значение |
| `program_highlights` | string[] | Selection + compression |
| `experience_signals` | string[] | Requires source understanding + grounding judgment |
| `why_go_candidates` | [{reason, strength}] | Strength classification требует judgment |
| `credibility_signals` | string[] | Selection from source |
| `voice_fragments` | [{...}] (optional) | Quote detection + speaker attribution |

### Что НЕ нужно от LLM: `routing_features` убраны

`routing_features` (6 booleans) в текущем дизайне — это LLM-produced routing decisions, замаскированные под features. Проблема:

1. **Drift risk**: LLM может вернуть `is_participatory = true` для лекции с Q&A (потому что «участники задают вопросы»). Runtime не сможет это оспорить.

2. **Redundancy**: все 6 booleans можно вывести из primitive fields, которые LLM уже возвращает:

```python
def derive_routing_signals(copy_assets: dict, facts_text_clean: list[str]) -> dict:
    fs = copy_assets.get("format_signal", "")
    ph = copy_assets.get("program_highlights", [])
    es = copy_assets.get("experience_signals", [])
    cs = copy_assets.get("credibility_signals", [])

    is_participatory = fs in ("мастерская", "экскурсия", "игра")
    has_hands_on = any(
        re.search(r"(кисти|краски|материал|инструмент|гуашь|бумаг|ножниц|глин)", h, re.I)
        for h in ph
    )
    has_stepwise = len(ph) >= 3 and is_participatory
    is_performance = fs in ("спектакль", "концерт", "показ")
    is_speaker_led = fs in ("лекция", "встреча") and any(
        re.search(r"(лектор|спикер|автор|куратор|исследователь|режиссёр|основатель)", c, re.I)
        for c in cs
    )
    has_true_program_list = (
        len(ph) >= 4
        and is_participatory
        and not is_performance
    )

    return {
        "is_participatory": is_participatory,
        "has_hands_on_materials": has_hands_on,
        "has_stepwise_action": has_stepwise,
        "is_performance": is_performance,
        "is_speaker_led": is_speaker_led,
        "has_true_program_list": has_true_program_list,
    }
```

3. **Debuggability**: если routing ошибся, я вижу primitive fields и могу проследить, какой `format_signal` привёл к какому pattern. С LLM-booleans я не могу понять, почему LLM решил `is_participatory = true`.

### Решение

- `routing_features` убраны из LLM extraction schema (и из v1, и из later).
- Routing целиком вычисляется runtime из primitive fields.
- LLM отвечает только за semantic extraction (что это, кто, что делать, зачем).
- Runtime отвечает за structural decisions (какой pattern, включать ли why-go, какой budget).

### Итого: разделение ответственностей

```
LLM extraction:
  facts[]              ← what happened / what will happen
  copy_assets:
    core_angle         ← semantic center
    format_signal      ← classification
    program_highlights ← content selection
    experience_signals ← grounded experience
    why_go_candidates  ← value evidence + strength
    credibility_signals← reputation/rarity markers
    voice_fragments    ← quotes (optional)

Runtime:
  routing_signals      ← derived from format_signal + highlights + credibility
  pattern              ← determined from routing_signals + copy_assets
  why_go_gate          ← 1 strong OR 2 regular
  budget               ← char estimate
  epigraph             ← picked from facts/voice_fragments
```

---

## 5. Честные возражения: где review ошибается, и я готов отстоять позицию

Перечитав свой ответ и review, я считаю, что в трёх местах я согласился слишком легко, и это повредит качеству итогового текста.

### 5.1. `scene_led` нужно оставить в v1 — но с жёстким conservative gate

**Позиция review**: `scene_led` отложить целиком, потому что `scene_cues` хрупкое поле.

**Моё возражение**: это самая крупная ошибка в текущем narrowing.

Вот почему. Все оставшиеся 4 active patterns (topic, program, value, person) производят **объясняющий** текст:
- `topic_led`: «Это лекция о ...»
- `program_led`: «Участники попробуют ...»
- `value_led`: «Единственный в России показ ...»
- `person_led`: «Режиссёр Иванов, лауреат ..., покажет ...»

Ни один из них не даёт **живой** opening. Они все начинают с пересказа/определения — просто разными способами. Без `scene_led` мы получим v1, который структурно разнообразнее, но всё ещё **ощущается как текст, написанный LLM**, потому что все 4 patterns работают в одном регистре: от общего к частному, через explaining.

`scene_led` — это единственный pattern, который начинает с **конкретной детали** и строит текст от частного к общему. Это именно тот тип вариативности, за который стоит весь redesign.

**Как снять risk**: не через отказ от pattern, а через conservative gate:

```python
def _has_safe_scene_cue(copy_assets: dict, source_text: str) -> bool:
    """Scene cue is safe only if it's traceable to a specific phrase in source."""
    cues = copy_assets.get("scene_cues", [])
    if not cues:
        return False
    # At least one cue must appear (fuzzy) in source
    for cue in cues:
        # Extract key content words (3+)
        words = [w for w in cue.lower().split() if len(w) > 3][:3]
        if len(words) < 2:
            continue
        if all(w in source_text.lower() for w in words):
            return True
    return False
```

Это **runtime grounding check**: если `scene_cues` вернулись из extraction, но их ключевые слова не traceable к source_text — pattern не включается, fallback → `topic_led`.

Hallucination risk устраняется не отказом от фичи, а runtime validation.

**Итого**: `scene_cues` остаётся в v1 schema (optional), `scene_led` включается с runtime grounding gate. Если gate не проходит — fallback.

### 5.2. Runtime-only routing через regex — хрупче, чем кажется

**Позиция review**: убрать `routing_features` из LLM, derive всё runtime из primitive fields.

**Мой первый ответ**: полностью согласился. Но это неполная правда.

Проблема: предложенный `derive_routing_signals()` использует **regex по русским словам** для определения `is_speaker_led` и `has_hands_on_materials`:

```python
is_speaker_led = fs in ("лекция", "встреча") and any(
    re.search(r"(лектор|спикер|автор|куратор|исследователь|режиссёр|основатель)", c, re.I)
    for c in cs
)
```

Это regex по `credibility_signals`, которые сами — LLM output. Если LLM напишет «победитель международного конкурса дирижёров» вместо «дирижёр-лауреат», regex не сработает. RuntimeONLY routing на таком уровне — это **иллюзия детерминизма**: формально runtime, фактически зависит от формулировок LLM.

**Практичнее**: hybrid approach. Оставить 2 targeted booleans в LLM extraction, но ТОЛЬКО те, которые semantic по природе и не derivable из structure:

| Boolean | Оставить в LLM? | Почему |
|---------|-----------------|--------|
| `is_participatory` | **Нет** | Derivable: `format_signal in (мастерская, экскурсия, игра)` |
| `has_hands_on_materials` | **Нет** | Derivable: regex по `program_highlights` (приемлемо) |
| `has_stepwise_action` | **Нет** | Derivable: `len(program_highlights) >= 3 and is_participatory` |
| `is_performance` | **Нет** | Derivable: `format_signal in (спектакль, концерт, показ)` |
| `is_speaker_led` | **Да, оставить** | Semantic judgment: «режиссёр участвует в Q&A» ≠ «режиссёр — главный фокус повествования». Runtime regex не может это различить. |
| `has_true_program_list` | **Да, оставить** | Semantic judgment: «5 тем лекции» vs «5 шагов мастер-класса» — структурно одинаково, семантически разное. |

Итого: 2 boolean вместо 6. Только те, где LLM judgment реально не заменим regex.

**Как устранить drift risk**: validation в runtime:

```python
# Sanity check: if LLM says is_speaker_led but no credibility_signals mention a person, override
if routing_signals["is_speaker_led"] and not copy_assets.get("credibility_signals"):
    routing_signals["is_speaker_led"] = False
```

### 5.3. `contrast_or_tension` стоит оставить как optional в v1

**Позиция review**: defer, editorializing risk.

**Моё возражение**: это самый дешёвый способ сделать lead интересным, и risk addressable.

Пример: для кинопоказа фильм описан как «не хроника войны, а история взросления на фоне оккупации». Это contrast — и это **лучший possible lead** для этого события. Без `contrast_or_tension` generation начнёт с generic «На экране — фильм режиссёра X ...».

Editorializing risk реален, но конкретен:
- *Опасно*: LLM сам конструирует контраст, которого нет в source.
- *Безопасно*: LLM находит контраст, который явно сформулирован в source.

Grounding rule уже есть в prompt: `ТОЛЬКО если в source есть явное противопоставление`. Это достаточно, если добавить runtime check аналогично `scene_cues`:

```python
def _has_safe_contrast(copy_assets: dict, source_text: str) -> bool:
    contrast = copy_assets.get("contrast_or_tension")
    if not contrast:
        return False
    words = [w for w in contrast.lower().split() if len(w) > 3][:4]
    return len(words) >= 2 and sum(1 for w in words if w in source_text.lower()) >= 2
```

Добавлять: да, как optional поле. Использовать: generation может использовать в lead, если оно есть. Routing dependency: нулевая.

---

## 6. Итоговый medium-risk v1 contract (с учётом возражений)

После 4 уточнений + 3 возражений, первый medium-risk branch выглядит так:

### Extraction
- Merged single-call: `facts + copy_assets`
- Schema: **11 полей** (6 required + 5 optional)
- Required: `core_angle`, `format_signal`, `program_highlights`, `experience_signals`, `why_go_candidates`, `credibility_signals`
- Optional: `voice_fragments`, `subformat`, `scene_cues`, `contrast_or_tension`, `is_speaker_led` + `has_true_program_list` (2 targeted booleans)
- Убраны навсегда: `tone_hint`, оставшиеся 4 `routing_features`

### Patterns (v1)
- **6 patterns**: `topic_led`, `program_led`, `compact_fact_led`, `person_led`, `value_led`, `scene_led`
- `scene_led` с runtime grounding gate (fallback → topic_led если gate не проходит)
- Без: `quote_led` (research-only)

### Routing
- Hybrid: 4 signals derived runtime + 2 LLM booleans (`is_speaker_led`, `has_true_program_list`)
- Runtime sanity checks для LLM booleans
- Decision tree: program → compact → person → value → scene (gated) → topic

### Anti-template
- 4 hard bans (template lead, title-as-definition, heading stop-list, redundancy)
- 3 soft preferences (lead from angle, no anaphora, heading ≠ lead)
- Explicit exemption for proper-noun leads in person_led

### Call budget
- 3 calls (same as current): enriched extraction + generation + coverage/revise
- +0 extra calls
- +~500-700 tokens per event (slightly more than stripped v1, but buys scene_led and contrast)
