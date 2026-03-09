# Opus → Gemma Event Copy: Quality-First Calibration (v2)

Дата: 2026-03-07

---

## 0. Главный принцип: полнота фактов — P0

> **Patterns определяют КАК организован текст, но НЕ сокращают ЧТО в нём сказано.**
>
> Все извлечённые факты из `facts_text_clean` должны быть включены в итоговый текст.
> Если источник богат фактами — описание большое и подробное.
> Если беден — компактное. Budget масштабируется от объёма фактов.

Это не новое требование — оно уже работает в текущей системе:

| Механизм | Где | Как работает |
|---------|-----|-------------|
| Generation rule | `_fact_first_description_prompt` L1739 | «Цель: связный текст, где **каждая деталь из фактов упомянута**» |
| Budget scaling | `_estimate_fact_first_description_budget_chars` | `facts_chars * 1.10 + 420` → больше фактов = больше бюджет |
| Coverage check | `_fact_first_coverage_prompt` | Возвращает `missing[]` (пропущенные) и `extra[]` (додуманные) |
| Repair layer | `_llm_integrate_missing_facts_into_description` | Встраивает пропущенные факты в текст |

**Риск pattern-driven redesign**: новые pattern instructions (scene opening, why-go angle, value lead) могут конкурировать с полнотой, отъедая budget на narrative framing вместо фактов.

**Как этот риск устраняется** (в каждом разделе ниже это встроено):

1. Every pattern instruction начинается с «ВСЕ факты включены»
2. Budget formula остаётся fact-proportional
3. Coverage check остаётся обязательным
4. `_llm_integrate_missing_facts_into_description` **НЕ помечается как remove-later** — это safety net
5. Acceptance criteria включают coverage stability как hard gate

---

## 1. `scene_led`: exact activation contract

### 1.1. Extraction contract для `scene_cues`

```
scene_cues (string[], ≤2):
Конкретная сенсорная деталь из source: что видно, слышно, ощущается в момент события.

ПРАВИЛА:
- ТОЛЬКО фрагменты, traceable к конкретным словам source_text / raw_excerpt / poster_texts.
- НЕ выдумывай «атмосферу». НЕ добавляй общих слов вроде «уютная обстановка» или «тёплая атмосфера».
- Хорошие cues: «звук виолончели в каменном зале», «запах кофе и масляных красок», «проекция на фасад здания».
- Плохие cues (запрещены): «душевная атмосфера», «незабываемые впечатления», «яркая энергия».
- Если в source нет конкретных сенсорных деталей — возвращай пустой список [].
- Не извлекай cues из логистики (адрес, время) — только из описания того, что происходит.
```

### 1.2. Runtime gate

```python
def _should_activate_scene_led(
    copy_assets: dict,
    source_text: str,
    facts_text_clean: list[str],
) -> bool:
    """Scene-led activates only when scene_cues pass traceability + quality checks."""
    cues = copy_assets.get("scene_cues", [])
    if not cues:
        return False

    # Gate 1: At least one cue must be traceable to source
    source_lower = (source_text or "").lower()
    has_traceable = False
    for cue in cues:
        content_words = [w for w in cue.lower().split() if len(w) > 3]
        if len(content_words) < 2:
            continue
        matches = sum(1 for w in content_words if w in source_lower)
        if matches >= 2:
            has_traceable = True
            break
    if not has_traceable:
        return False

    # Gate 2: Cue must not be a generic atmosphere phrase
    GENERIC_BLOCKLIST = [
        "атмосфер", "незабываем", "уникальн", "яркая энерги",
        "тёплая обстановк", "душевн", "особая атмосфер",
    ]
    for cue in cues:
        cue_lower = cue.lower()
        if any(g in cue_lower for g in GENERIC_BLOCKLIST):
            return False

    # Gate 3: Source must be rich enough (scene-led on poor source = disaster)
    if len(facts_text_clean) < 6:
        return False

    return True
```

### 1.3. Fallback rules

| Ситуация | Fallback |
|---------|----------|
| `scene_cues` пусты | → routing tree (topic/value/person) |
| `scene_cues` не traceable | → `topic_led` |
| `scene_cues` generic phrase | → `topic_led` |
| Source бедный (< 6 facts) | → `compact_fact_led` (раньше в routing) |
| Scene gate passed, но strong why-go | → `value_led` (scene_cue передаётся как inline aid) |

### 1.4. P0 constraint: scene opening НЕ сокращает факты

`scene_led` pattern instruction явно включает:

```
ОБЯЗАТЕЛЬНО: ВСЕ факты из facts_text_clean должны быть включены.
Scene cue — это lead opening (1-2 предложения), а не замена фактического содержания.
Остальная структура (2-3 ### секции) раскрывает все факты полностью.
```

### 1.5. Failure modes

| # | Ситуация | `scene_cues` | Activation | Outcome |
|---|---------|-------------|-----------|---------|
| 1 | Лекция, нет сенсорных деталей | `[]` | ❌ | `topic_led` / `person_led` |
| 2 | Концерт, LLM выдумал: `["зал наполнится звуками"]` | Hallucinated | ❌ Gate 1 fails | `topic_led` |
| 3 | Мастерская, `["руки в глине на гончарном круге"]` | Traceable ✅ | ✅ Но... | `program_led` wins (participatory) |
| 4 | Выставка + strong why-go + `["проекция на стены"]` | Traceable ✅ | ✅ Но... | `value_led` wins; scene inline |
| 5 | Летний кинопоказ, `["кресла под звёздным небом"]` | Traceable ✅ | ✅ | `scene_led` ✓ |

---

## 2. `contrast_or_tension`: оставить как optional lead aid

### 2.1. Decision

| Aspect | Answer |
|--------|--------|
| В v1? | **Да**, optional |
| Routing dependency? | **Нулевая** |
| Где используется? | Lead aid в generation, любой pattern кроме `compact_fact_led` |

### 2.2. Extraction contract

```
contrast_or_tension (string | null):
Противопоставление, ЯВНО сформулированное в source.
- ТОЛЬКО «не X, а Y» / «X на фоне Y» / «вопреки X, ...» из source.
- НЕ конструируй. Если нет — null.
- Формат: ≤20 слов.
ХОРОШО: «не военная хроника, а история взросления» (если source так формулирует)
ПЛОХО: «классика vs современность» (constructed frame)
```

### 2.3. Traceability gate

```python
def _has_safe_contrast(copy_assets: dict, source_text: str) -> bool:
    contrast = copy_assets.get("contrast_or_tension")
    if not contrast:
        return False
    source_lower = (source_text or "").lower()
    content_words = [w for w in contrast.lower().split() if len(w) > 3]
    if len(content_words) < 3:
        return False
    matches = sum(1 for w in content_words if w in source_lower)
    return matches / len(content_words) >= 0.6
```

### 2.4. P0 constraint: contrast НЕ заменяет факты

```
contrast_or_tension — это framing для lead (1 предложение), а не замена фактического содержания.
Все факты из facts_text_clean по-прежнему должны быть включены в текст полностью.
```

---

## 3. Helper set: 1 LLM boolean

### 3.1. Final set

| Helper | В v1? | Обоснование |
|--------|-------|-------------|
| `is_speaker_led` | **Да** | Runtime regex по credibility_signals хрупок: пропускает нестандартные формулировки. LLM judgment нужен. |
| `has_true_program_list` | **Нет** | Derivable: `format_signal in (мастерская, экскурсия, игра) AND len(program_highlights) >= 4` |

### 3.2. Runtime sanity check

```python
if copy_assets.get("is_speaker_led") and not copy_assets.get("credibility_signals"):
    is_speaker_led = False  # no credibility → cannot be speaker-led
```

### 3.3. Почему `is_speaker_led` не derivable

| Source | credibility_signals | Runtime regex? | LLM? |
|--------|---------------------|----------------|------|
| Лекция историка, автор монографии | `["автор монографии"]` | ✅ | ✅ |
| Дискуссия с приглашённым гостем (не фокус) | `["участник дискуссии"]` | ❌ miss | ✅ `false` |
| Q&A с режиссёром после показа (фокус на фильме) | `["режиссёр фильма"]` | ✅ false positive | ✅ knows context |

---

## 4. Pattern precedence: naturalness + completeness

### 4.1. Precedence matrix

```
                      scene_led  value_led  person_led  program_led
    scene_led            —        context    person ✓    program ✓
    value_led          context      —        person ✓    program ✓
    person_led         person ✓   person ✓     —         program ✓
    program_led        program ✓  program ✓  program ✓      —
```

**Hardest conflict — `scene_led` vs `value_led`:**

```python
def _resolve_scene_vs_value(copy_assets: dict, scene_gate_passed: bool) -> str:
    if not scene_gate_passed:
        return "value_led"
    why_go = copy_assets.get("why_go_candidates", [])
    strong_count = sum(1 for c in why_go if c.get("strength") == "strong")
    if strong_count >= 1:
        return "value_led"  # strong value wins, scene_cue goes inline
    return "scene_led"      # regular-only value: scene gives more natural opening
```

### 4.2. Full routing tree (v1)

```python
def determine_pattern_v1(copy_assets, facts_text_clean, source_text) -> str:
    # 1. Program-led (participatory → structure MUST show steps)
    if _is_program_led(copy_assets):
        return "program_led"
    # 2. Compact (poor source)
    if _is_poor_source(copy_assets, facts_text_clean):
        return "compact_fact_led"
    # 3. Person-led
    if _is_person_led(copy_assets):
        return "person_led"
    # 4. Value vs Scene (context-dependent)
    scene_gate = _should_activate_scene_led(copy_assets, source_text, facts_text_clean)
    value_gate = _should_include_why_go(copy_assets)
    if value_gate and scene_gate:
        return _resolve_scene_vs_value(copy_assets, scene_gate)
    if value_gate:
        return "value_led"
    if scene_gate:
        return "scene_led"
    # 5. Default
    return "topic_led"
```

### 4.3. P0 constraint в routing: pattern ≠ filter

Routing выбирает **structure**, не фильтрует факты. Все patterns получают полный `facts_text_clean` и обязаны включить их все. Для events с 15+ фактами budget автоматически вырастет ≥ 2000 chars — это нормально.

---

## 5. Prompt family for quality-first v1

### 5.1. Extraction prompt additions

К текущему `_llm_extract_candidate_facts`:

```
Дополнительно верни объект copy_assets:

1. core_angle (string, ≤15 слов): о чём событие, одна dominant формулировка.
2. format_signal (enum): спектакль | лекция | концерт | показ | мастерская | экскурсия | игра | фестиваль | встреча.
3. subformat (string | null): уточнение формата.
4. program_highlights (string[], 2-6): самые конкретные пункты программы/содержания.
5. experience_signals (string[]): что участник реально увидит/услышит/попробует. ТОЛЬКО grounded.
6. why_go_candidates ([{reason, strength}]): grounded основания ценности. Max 3.
7. credibility_signals (string[]): премии, уникальные форматы, «впервые»/«единственный».
8. voice_fragments ([{text, speaker, speaker_role, is_direct_quote, opens_event_theme}]): прямые цитаты. Max 2.
9. scene_cues (string[], ≤2): конкретные сенсорные детали из source. ТОЛЬКО из source. Пустой список если нет.
10. contrast_or_tension (string | null): ЯВНОЕ противопоставление из source. null если нет.
11. is_speaker_led (bool): человек — главный сюжетный центр.

ВАЖНО: copy_assets — это вспомогательные сигналы для оформления текста. 
Они НЕ заменяют и НЕ сокращают основной список facts.
Все значимые факты из источника должны быть в facts[], независимо от copy_assets.
```

### 5.2. Generation prompt → `SHARED_DESCRIPTION_RULES`

Вынести в shared constant, подставляемый во все generation/revise prompts:

```python
SHARED_DESCRIPTION_RULES = (
    "ПРИОРИТЕТ P0 — ПОЛНОТА ФАКТОВ:\n"
    "- ВСЕ факты из facts_text_clean должны быть упомянуты в тексте.\n"
    "- Pattern определяет КАК организован текст (lead, structure, headings),\n"
    "  но НЕ сокращает ЧТО в нём сказано.\n"
    "- Если фактов много — описание длиннее. Если мало — короче.\n"
    "- Не жертвуй фактами ради narrative framing.\n"
    "- description_budget_chars масштабируется от числа фактов.\n\n"
    f"{SHARED_LOGISTICS_BAN}\n"
    f"{SHARED_HALLUCINATION_BAN}\n"
    f"{SHARED_QUOTE_POLICY}\n"
    f"{SHARED_LIST_POLICY}\n"
    f"{SHARED_HEADING_PALETTE}\n"
    f"{SMART_UPDATE_YO_RULE}\n"
)
```

### 5.3. Generation prompt → pattern preamble

```python
GENERATION_PREAMBLE = (
    "Ты пишешь Markdown-анонс события для Telegram.\n"
    "Источник истины: ТОЛЬКО facts_text_clean. Нельзя добавлять новые сведения.\n\n"
    "Выбранный narrative pattern: {pattern_name}.\n"
    "core_angle: {core_angle}\n"
    "{contrast_line}"
    "{scene_cue_line}\n"
    "PATTERN INSTRUCTIONS:\n"
    "{pattern_instructions}\n\n"
    "{SHARED_DESCRIPTION_RULES}\n\n"
    "Anti-template checks:\n"
    "HARD BAN: НЕ начинай lead с «{title} — это ...»; НЕ копируй title целиком в lead.\n"
    "SOFT: Предпочитай lead с core_angle; для person_led допустим proper-noun lead.\n\n"
    "Данные:\n{payload_json}"
)
```

### 5.4. Pattern instructions (все начинаются с P0 constraint)

```python
_FACT_COMPLETENESS_PREAMBLE = (
    "P0: ВСЕ факты из facts_text_clean включены в текст. "
    "Pattern определяет structure, не фильтрует содержание.\n"
)

PATTERN_INSTRUCTIONS_V1 = {
    "topic_led": _FACT_COMPLETENESS_PREAMBLE + (
        "Начни lead с core_angle. 2-3 ### секции по разным граням темы.\n"
        "Variation: начни с самого яркого факта, не с обобщения."
    ),
    "scene_led": _FACT_COMPLETENESS_PREAMBLE + (
        "Начни lead с микросцены из scene_cues (1-2 предложения).\n"
        "Второе предложение — что за событие и формат.\n"
        "НЕ придумывай сцену: ТОЛЬКО scene_cues и facts.\n"
        "Остальная структура (2-3 ### секции) раскрывает все факты.\n"
        "Variation: один чувственный канал (зрение / слух / тактильность)."
    ),
    "program_led": _FACT_COMPLETENESS_PREAMBLE + (
        "Начни lead с формата + главной цели.\n"
        "Обязательно дай список главных пунктов программы.\n"
        "Условия участия — в отдельной ### секции.\n"
        "Variation: список до или после 1 абзаца контекста."
    ),
    "value_led": _FACT_COMPLETENESS_PREAMBLE + (
        "Начни lead с конкретного grounded факта ценности.\n"
        "Назови конкретную premию / формат / исполнителя.\n"
        "Variation: value statement в lead или в финальном абзаце."
    ),
    "compact_fact_led": _FACT_COMPLETENESS_PREAMBLE + (
        "Source бедный. Описание пропорционально: все факты включены,\n"
        "но без раздувания. 0-1 ### секция.\n"
        "НЕ добавляй «атмосферных» расширений."
    ),
    "person_led": _FACT_COMPLETENESS_PREAMBLE + (
        "Начни lead с имени + credibility signal.\n"
        "Затем — что покажет/расскажет.\n"
        "НЕ превращай lead в биографию.\n"
        "Variation: начни с темы, которую speaker раскрывает."
    ),
}
```

### 5.5. Coverage prompt → расширение

Добавить к текущему `missing` / `extra`:

```json
{
  "missing": ["..."],
  "extra": ["..."],
  "template_issues": ["..."],
  "quality_flags": {
    "template_feel": false,
    "weak_lead": false,
    "weak_heading": false,
    "redundancy": false
  }
}
```

Coverage prompt additions:

```
ПРИОРИТЕТ P0: проверь полноту фактов ПЕРВЫМ ДЕЛОМ.
Каждый факт из facts_text_clean должен быть либо упомянут в описании,
либо обоснованно поглощён более общей формулировкой (но смысл сохранён).

Дополнительная проверка (quality_flags):
- template_feel: одинаковый ритм, lead = пересказ title, generic headings.
- weak_lead: lead начинается с «{title} — это», lead без core_angle.
- weak_heading: heading из стоп-листа.
- redundancy: один факт в lead и в ### секции.
```

### 5.6. Revise prompt → pattern-aware

```
Pattern: {pattern_name}
core_angle: {core_angle}

P0: если coverage report нашёл missing facts — встрой их ВСЕ. Не жертвуй фактами.

Дополнительные исправления (по quality_flags):
- template_feel → переписать lead с core_angle
- weak_lead → начать с core_angle или contrast
- weak_heading → заменить из heading palette
- redundancy → убрать повтор (предпочтительно из body)
```

### 5.7. `_llm_integrate_missing_facts` — сохраняется

**Не помечается как `remove-later`.** Это safety net. Даже после pattern-driven redesign generation может пропускать факты, особенно когда lead тратится на scene/value framing. Repair layer остаётся.

### 5.8. Token budget summary

| Component | Current | After v1 | Delta |
|-----------|---------|---------|-------|
| Extraction prompt | ~600 | ~800 | +200 |
| Extraction output | ~200 | ~500 | +300 |
| Generation prompt | ~900 | ~950 | +50 |
| Coverage prompt | ~400 | ~480 | +80 |
| Revise prompt | ~700 | ~650 | -50 (deduplicated) |
| **Total per event** | **~2900** | **~3530** | **+630 (~22%)** |

Calls: 3 (same). +0 extra calls.

---

## 6. Acceptance criteria

### 6.1. P0: полнота фактов (hard gate)

| # | Criterion | Measure | Threshold | Rollback trigger |
|---|----------|---------|-----------|-----------------|
| **1** | **Coverage stability** | Missing facts count (automated) | **Median ≤ 1 per event** | Median > 2 → rollback enriched extraction |
| **2** | **No fact loss from patterns** | A/B: missing facts v1 vs current | **v1 ≤ current** | v1 > current + 0.5 median → disable pattern routing |
| **3** | **Budget scales with facts** | Automated: description_chars vs facts_chars | **Correlation ≥ 0.7** | Flat budget regardless of facts → fix budget formula |

### 6.2. Quality uplift (positive signals)

| # | Criterion | Measure | Threshold |
|---|----------|---------|-----------|
| 4 | Lead variety | Manual: distinct lead types in 20 events | ≥ 3 types |
| 5 | No generic headings | Automated: stop-list grep | 0 in 20 events |
| 6 | No lead-as-title | Automated: `«{title}» — это` check | 0 in 20 events |
| 7 | Reduced redundancy | Manual: same fact in lead + body | ≤ 1 in 20 events |
| 8 | Scene-led naturalness | Manual: scene_led vs topic_led equivalent | ≥ 70% rated "better" |
| 9 | Pattern distribution | Automated: patterns across 50 events | ≥ 3 used, no one > 60% |

### 6.3. Rollback triggers

| # | Signal | Action |
|---|--------|--------|
| 1 | Missing facts median > 2 | Remove `copy_assets`, revert extraction |
| 2 | `scene_led` > 30% hallucinated openings | Disable `scene_led` |
| 3 | `contrast_or_tension` editorializes > 20% | Remove from generation |
| 4 | `is_speaker_led` disagrees > 40% | Remove, improve runtime |
| 5 | Token overhead > +1000 median | Trim optional fields |
| 6 | **Pattern-driven generation loses more facts than current** | **Disable pattern routing, keep only shared rules** |

### 6.4. Допустимый overhead

| Metric | Acceptable | Unacceptable |
|--------|-----------|-------------|
| Token per event | +400–800 median | > +1000 median |
| p95 tokens | ≤ +1200 | > +1500 |
| Extra calls | 0 | ≥ 1 |
| Latency | ≤ 300ms | > 500ms |
| Parsing failures | ≤ 3% | > 5% |

---

## 7. Summary: Quality-First v1 final spec

| Dimension | Spec | Priority |
|-----------|------|----------|
| **Fact completeness** | ALL facts from `facts_text_clean` must appear. Budget scales with facts. | **P0** |
| **Missing facts repair** | `_llm_integrate_missing_facts` stays active (not remove-later) | **P0** |
| **Extraction** | 11 fields (6 req + 5 opt). 1 LLM boolean: `is_speaker_led`. | P1 |
| **Patterns** | 6: topic, program, compact, person, value, scene | P1 |
| **Routing** | Deterministic (+ 1 LLM boolean). program → compact → person → value/scene → topic | P1 |
| **Scene gate** | Traceability + generic blocklist + min 6 facts + fallback | P1 |
| **Contrast** | Optional lead aid, no routing dependency, traceability gate | P2 |
| **Anti-template** | 4 hard bans + 3 soft prefs + proper-noun exemption | P1 |
| **Coverage** | Extended: +4 quality flags. Fact completeness checked FIRST. | P0 |
| **Calls** | 3 (same as current) | — |
| **Token Δ** | +~630 median, controllable | — |
| **Acceptance** | 3 P0 coverage criteria + 6 quality criteria + 6 rollback triggers | — |
