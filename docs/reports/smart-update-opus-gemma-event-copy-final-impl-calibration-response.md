# Opus → Gemma Event Copy: Final Implementation Calibration

Дата: 2026-03-07

---

## 1. Dense fact sets: concrete structure rules

### 1.1. Core problem

«Все факты включены» + 14 фактов + pattern-driven lead = risk of wall-of-text. Решение не в ослаблении P0, а в **структурных rules для high-density events**.

### 1.2. Density tiers

| Tier | facts_text_clean count | Structure rule |
|------|----------------------|----------------|
| **Sparse** (≤5) | `compact_fact_led`: 40-80 слов, 0-1 ### | Всё в prose. Нет списков. |
| **Normal** (6-12) | 2-3 ### секции | Mix: prose lead + 2-3 prose paragraphs under headings |
| **Dense** (13-20) | 2-3 ### секции + **≤1 structured list** | Prose lead + prose sections + 1 list for groupable items |
| **Very dense** (21+) | 3 ### секции + **≤2 structured lists** | Prose lead + sections + lists. Maximum detail retention. |

### 1.3. When to use list vs prose

Списки для:
- **Program items**: пункты программы, темы лекции, имена участников
- **Visitor conditions**: где, что взять, что входит/не входит
- **Lineup / performer list**: если ≥3 исполнителей/спикеров

Prose для:
- **Lead**: всегда prose (1-2 предложения)
- **Context/background**: зачем событие существует, его значимость
- **Experience**: что участник увидит / почувствует

### 1.4. Grouping strategy для dense events

```
Generation instruction (для dense tier):

У тебя много фактов. Все должны войти. Стратегия:
1. Lead: core_angle + 1-2 самых ярких факта (prose, 2 предложения).
2. ### секция 1: основное содержание / программа.
   — Если пунктов >3 → оформи списком.
   — Если пунктов ≤3 → prose.
3. ### секция 2: контекст, участники, credibility.
4. ### секция 3 (если нужна): условия участия / visitor details.
   — Оформи списком, если >2 condition-like фактов.

НЕ делай отдельный ### для каждого факта.
НЕ превращай текст в plain bullet dump.
Group related facts: если 3 факта про спикера — это один абзац, не 3 пункта.
```

### 1.5. Visitor conditions — отдельный вопрос

Факты типа «длительность 2 часа», «возраст 12+», «принести кисти и передник», «стоимость материалов отдельно» — это conditions. Их много, они все нужны (P0), но ломают ритм prose.

Правило:

```
Если в facts_text_clean ≥3 фактов-условий (длительность/возраст/формат/что взять/что входит) →
оформи их как ОДИН ### блок «Что нужно знать» с компактным списком.
Не разбрасывай условия по разным секциям. Не вставляй условие в lead.
```

Это уже частично работает — текущий prompt говорит «Включай условия участия». Но при переходе к patterns нужно **явно указать**, что conditions group в отдельную секцию при dense events.

### 1.6. Budget formula — никаких изменений

```python
budget = int(facts_chars * 1.10 + 420)  # уже fact-proportional
return max(800, min(SMART_UPDATE_DESCRIPTION_MAX_CHARS, budget))
```

Это уже правильно. Для 20 facts (~1500 chars facts) budget = ~2070 chars. Для 8 facts (~600 chars) budget = ~1080. Пропорциональность встроена.

### 1.7. Критическое замечание: не переусердствовать с grouping instructions

Здесь стоит быть осторожным. Gemma — не Claude. Слишком детальные grouping instructions в prompt (>5-6 правил) создадут **instruction noise**: модель начнёт следовать букве grouping rules вместо того, чтобы писать естественный текст.

Practical limit для Gemma: **≤4 structural rules** в generation prompt. Остальное — на уровне coverage check / revise, не на уровне initial generation.

Рекомендуемые 4 rules для dense tier:

1. Lead = 1-2 предложения (core_angle + яркий факт).
2. 2-3 ### секции, не больше.
3. Если ≥4 однородных item → список, иначе prose.
4. Conditions group: объединяй visitor details в один блок.

---

## 2. Missing-facts safety net: exact placement

### 2.1. Current runtime placement

В текущем коде `_llm_integrate_missing_facts_into_description` вызывается в **merge flow** (L11481-11520): после `clean_description` готов (от `_llm_merge_event`), перед финальной чисткой.

В **fact-first flow** (L2064+) coverage и revise встроены внутрь `_llm_fact_first_description_md` — 3 LLM calls (generation + coverage + revise). Missing facts handling НЕ является отдельным 4-м call в fact-first flow.

### 2.2. Recommendation for quality-first v1

**Placement**: после `_llm_fact_first_description_md` returns, в runtime, **conditional**.

```python
# Inside the fact-first block, after ff_desc is produced:
if ff_desc:
    cleaned_ff = _cleanup(ff_desc)
    # Post-generation missing-facts check
    missing = _find_missing_facts_in_description(
        description=cleaned_ff,
        facts=facts_text_clean,
        max_items=5,
    )
    if len(missing) >= MISSING_FACTS_REPAIR_THRESHOLD:
        enriched = await _llm_integrate_missing_facts_into_description(
            description=cleaned_ff,
            missing_facts=missing,
            source_text=source_text,
            label="ff_fact_coverage",
        )
        if enriched:
            cleaned_ff = _cleanup(enriched)
```

### 2.3. Threshold

| Threshold | Rationale |
|-----------|-----------|
| `missing >= 2` | 1 missing fact может быть acceptable (поглощено обобщением). 2+ = generation gap. |
| `missing includes quoted fact` | Quoted facts — direct characterizations from source. Losing them = information loss. Trigger repair even if only 1. |

```python
MISSING_FACTS_REPAIR_THRESHOLD = 2

def _should_repair_missing_facts(missing: list[str]) -> bool:
    if len(missing) >= MISSING_FACTS_REPAIR_THRESHOLD:
        return True
    # Single missing fact: repair only if it's a quoted slogan/characterization
    if len(missing) == 1:
        f = missing[0]
        return bool(re.search(r'[«"]', f))  # quoted → high-value
    return False
```

### 2.4. When NOT to do 4th call

| Condition | Don't repair |
|-----------|-------------|
| `missing == 0` | Ничего не пропущено |
| `missing == 1` and not quoted | 1 non-quoted fact может быть absorbed in generalization |
| `compact_fact_led` pattern | Source бедный, missing fact = probably anchor-like or noise |
| Repair call already failed once | Don't retry — accept residual gap |

### 2.5. TPM impact

Repair call = ~1500-2500 tokens (description + missing_facts + source_text + response).

Expected trigger rate: **15-25% of events** (based on current `_find_missing_facts_in_description` behavior — it's already tuned to catch real misses, not noise).

Mean extra tokens per batch of 20 events: 20 × 0.20 × 2000 = **~8,000 tokens**. This is < 3% of typical batch TPM consumption. Negligible.

### 2.6. Критическое замечание: repair call — не панацея

Есть risk, что pattern-driven generation будет регулярно пропускать факты (потому что lead и structure забирают budget), и repair call станет **де-факто обязательным для большинства events**. Если trigger rate превысит 40%, это сигнал, что generation prompt нужно переписать, а не что repair layer спасает.

**Hard metric**: если repair trigger rate > 40% на golden set → проблема в generation, не в repair. Действие: усилить P0 instructions в generation prompt.

---

## 3. TPM-aware acceptance criteria

### 3.1. Реальные constraints

Текущая система работает в batch mode: import → extract facts → generate descriptions. Нет real-time user-facing latency. Ключевые constraints:

| Constraint | Current value | Source |
|-----------|--------------|--------|
| Gemma TPM limit | Model-dependent (Vertex AI) | Provider |
| Batch size | ~10-30 events per import cycle | Runtime |
| Calls per event (fact-first) | 3 (generation + coverage + revise) | Current code |
| Calls per event (merge) | 1-2 (merge + conditional repair) | Current code |

### 3.2. Revised acceptance criteria (TPM-aware)

| # | Metric | Acceptable | Unacceptable | How to measure |
|---|--------|-----------|-------------|---------------|
| 1 | **Token overhead per event** (median) | +400–800 | > +1200 | Log input+output tokens, compare v1 vs current |
| 2 | **Token overhead per event** (p95) | ≤ +1500 | > +2000 | Same log |
| 3 | **Extra repair calls** (% of events) | ≤ 25% | > 40% | Count repair triggers in batch |
| 4 | **Total calls per event** (median) | 3 | > 4 | 3 core + conditional repair |
| 5 | **Total calls per event** (p95) | 4 | > 4 | Maximum 1 repair per event |
| 6 | **Batch completion** (20-event import) | Completes without TPM throttle | TPM rate-limit errors on standard batch | Monitor provider errors |
| 7 | **Extraction parse failure rate** | ≤ 3% | > 5% | Count JSON parse failures for copy_assets |
| 8 | **end-to-end batch time** | ≤ 1.5× current | > 2× current | Wall-clock time for standard batch |

### 3.3. Что убрано из acceptance criteria

- ~~`Latency ≤ 300ms`~~ — не релевантно для batch processing, нет user-facing latency.
- ~~`Latency > 500ms = unacceptable`~~ — same.

### 3.4. Что добавлено

- **Batch completion without throttle** — если дизайн проходит per-event tests, но ломается на batch из 20 events из-за TPM burst, это operationally broken.
- **Repair trigger rate** — если >40%, это не performance issue, это quality issue в generation prompt.

### 3.5. Критическое замечание: enriched extraction увеличивает input/output

Текущий extraction call: ~600 tokens prompt + ~200 tokens response.
Quality-first v1: ~800 tokens prompt + ~500 tokens response.

Это +500 tokens per event **гарантированно** (не conditional). Для batch из 20 events = +10,000 tokens. Нужно убедиться, что этот overhead помещается в TPM window.

Но. Реальный вопрос — не токены по отдельности, а **do we hit TPM rate limit within a batch?** Это нужно измерить на реальном batch, а не теоретически рассчитать. Поэтому acceptance criterion #6 (batch completion) — самый важный performance metric.

---

## 4. Stronger traceability for `scene_cues` and `contrast_or_tension`

### 4.1. Проблема с текущим word-overlap approach

Текущий gate:
```python
content_words = [w for w in cue.lower().split() if len(w) > 3]
matches = sum(1 for w in content_words if w in source_lower)
```

Проблемы для русского текста:

| Problem | Example | Result |
|---------|---------|--------|
| **Morphology** | cue: «гончарного круга» vs source: «гончарным кругом» | ❌ Miss: «круга» ≠ «кругом» |
| **OCR noise** | source: «гoнчарным» (латинская 'o') vs cue: «гончарным» | ❌ Miss |
| **Paraphrasing** | cue: «звуки скрипки» vs source: «скрипичная музыка» | ❌ Miss |

### 4.2. Solution: evidence_span extraction (не evidence_snippet)

Вместо runtime word-overlap: потребовать от LLM **вернуть evidence_span** — точный фрагмент source, который обосновывает cue.

Extraction contract:

```
scene_cues ([{cue: string, evidence_span: string}], ≤2):
- cue: конкретная сенсорная деталь (что видно/слышно).
- evidence_span: ДОСЛОВНАЯ цитата из source_text (5-20 слов), из которой следует cue.

Пример:
source: «...зрители рассядутся в складных креслах летнего кинотеатра и будут смотреть фильм под звёздным небом...»
→ cue: «складные кресла летнего кинотеатра под звёздным небом»
→ evidence_span: «рассядутся в складных креслах летнего кинотеатра и будут смотреть фильм под звёздным небом»

Если не можешь найти дословный фрагмент в source — НЕ возвращай cue.
```

Аналогично для `contrast_or_tension`:

```
contrast_or_tension: {text: string, evidence_span: string} | null
- text: противопоставление (≤20 слов)
- evidence_span: ДОСЛОВНАЯ цитата из source
```

### 4.3. Runtime gate becomes trivial

```python
def _is_evidence_span_traceable(evidence_span: str, source_text: str) -> bool:
    """Check that the evidence span literally appears in source."""
    if not evidence_span or not source_text:
        return False
    span = evidence_span.strip()
    if len(span) < 10:
        return False
    # Exact substring check (case-insensitive, whitespace-normalized)
    source_norm = re.sub(r"\s+", " ", source_text.lower().strip())
    span_norm = re.sub(r"\s+", " ", span.lower().strip())
    return span_norm in source_norm
```

Это решает все 3 проблемы:

| Problem | With evidence_span |
|---------|-------------------|
| Morphology | ✅ span is verbatim from source → exact match |
| OCR noise | ✅ span copied from source → same encoding |
| Paraphrasing | ✅ if LLM paraphrased, span won't match source → gate fails → safe fallback |

### 4.4. Fallback tolerance: fuzzy match для OCR

Для events с poster OCR (часто с noise) — strict exact match может быть too harsh. Добавляем один fallback уровень:

```python
def _is_evidence_span_traceable(evidence_span: str, source_text: str) -> bool:
    if not evidence_span or not source_text:
        return False
    span = evidence_span.strip()
    if len(span) < 10:
        return False
    source_norm = re.sub(r"\s+", " ", source_text.lower().strip())
    span_norm = re.sub(r"\s+", " ", span.lower().strip())
    # Level 1: exact substring
    if span_norm in source_norm:
        return True
    # Level 2: fuzzy (for OCR noise) — at least 80% of 5-char windows match
    if len(span_norm) >= 20:
        windows = [span_norm[i:i+5] for i in range(0, len(span_norm) - 4)]
        hits = sum(1 for w in windows if w in source_norm)
        if hits / len(windows) >= 0.8:
            return True
    return False
```

### 4.5. Token overhead от evidence_span

Каждый `scene_cue` добавляет ~10-30 tokens (evidence_span). Максимум 2 cues = +20-60 tokens. `contrast_or_tension` = +10-30 tokens.

Total: +30-90 tokens per event. Negligible.

### 4.6. Критическое замечание: Gemma compliance с evidence_span

Это самый realistic risk. Gemma может:

1. **Hallucinate evidence_span** — вернуть строку, которой нет в source. → Gate fails, safe fallback. Acceptable.
2. **Return empty evidence_span** — cue без evidence. → Gate fails, safe fallback. Acceptable.
3. **Return verbatim quote that's too long** (>50 words). → Clip to first 30 words, check traceability.
4. **Refuse to return scene_cues at all** (instruction following failure). → Empty list, no scene_led. Acceptable.

Все failure modes → safe fallback. Нет catastrophic outcomes.

Но. Если Gemma compliance rate с evidence_span окажется <50% (она просто не умеет reproducibly copy exact substrings), то evidence_span становится useless и нужно fallback к improved word-overlap.

**Staged approach**:
1. Implementation: add evidence_span to schema.
2. Measurement: check compliance rate on 20-event golden set.
3. If compliance < 60%: fallback to word-overlap with morphological stem matching.
4. If compliance ≥ 60%: keep evidence_span.

---

## 5. Summary: implementation-ready contract

| Component | Decision | Status |
|-----------|---------|--------|
| Dense facts | 4 density tiers, ≤4 structural rules in prompt, conditions group | Ready |
| Missing-facts repair | After fact-first generation, threshold = 2 OR 1 quoted, ≤25% trigger rate | Ready |
| TPM acceptance | 8 operational metrics, batch completion = key, no micro-latency | Ready |
| Traceability | evidence_span extraction, exact substring gate, fuzzy OCR fallback | Ready (staged) |
| P0 (unchanged) | All facts included, budget scales with facts, repair layer retained | Confirmed |

### Не требует отдельного Opus-раунда

Всё в этом ответе — это implementation-level decisions, не architectural changes. После review этого ответа следующий шаг — local implementation branch.
