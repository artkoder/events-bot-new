# Opus → Gemma Event Copy: Quality Consultation Follow-up (Recalibration)

Дата: 2026-03-07

---

## 1. Recalibrated event reading

Принимаю критику. Пересматриваю каждый кейс по 7 осям, не перевешивая structural signals.

### Event 2660 — Дуальность (выставка, 3→5 facts)

| Axis | Baseline | Prototype | Who wins |
|------|----------|-----------|----------|
| Factual coverage | 2 missing | 3 missing | Baseline |
| Unsupported prose | «многолетних поисков» — ungrounded | «напоминая о противоречивой природе» — ungrounded | Draw (оба грешат) |
| Service leakage | None | None | Draw |
| Duplication | None | None | Draw |
| Readability | Readable, 3 headings (overkill for 3 facts, but readable) | Flat, no headings — **acceptable if sparse** | Draw (разные styles, оба workable) |
| Structural fit | 3 headings for 3 facts = over-structured | 1 generic heading «О событии» = under-structured | Neither ideal |
| Professional tone | Средний | Средний | Draw |

**Corrected verdict**: Draw with slight baseline advantage on coverage. Мой прошлый verdict «baseline лучше» был overweighted by structural preference. На самом деле оба текста mediocre, prototype format вполне acceptable для sparse event. Реальная проблема: **оба** имеют unsupported embellishment, **оба** недостаточно профессиональны. Pattern choice (value_led) — неправильный, но это routing failure, не pattern concept failure.

---

### Event 2745 — Сёстры (спектакль, 5 identical facts)

| Axis | Baseline | Prototype | Who wins |
|------|----------|-----------|----------|
| Factual coverage | All 5 present (metric ложноположительный) | All 5 present | Draw |
| Unsupported prose | «обещает стать глубоким и эмоциональным опытом» — CTA/promo | Чище, no promo | Prototype |
| Service leakage | None | None | Draw |
| Duplication | None | None | Draw |
| Readability | 2 headings + bullet list → structured | 1 paragraph, no headings → compact | Baseline slightly (structure helps on Telegraph) |
| Structural fit | Headings + list хорошо для 5 тематических facts | Compact paragraph — acceptable for same-theme facts | Draw |
| Professional tone | «заставляющим задуматься о ценности родственных связей» — pompous | Нейтральнее | Prototype |

**Corrected verdict**: Closer to draw than I said. Baseline structural advantage real on Telegraph, но prototype cleaner by tone (no promo filler). Мой прошлый verdict «baseline заметно лучше» overcounted structure. **Honest assessment**: baseline лучше визуально, prototype лучше по чистоте тона. Оба mediocre из-за abstract source facts.

---

### Event 2734 — Концерт Гудожникова (8→4 facts)

| Axis | Baseline | Prototype | Who wins |
|------|----------|-----------|----------|
| Factual coverage | 8 facts → all present | 4 facts → extraction loss | **Baseline** |
| Unsupported prose | Minimal | «выпускник СПб консерватории, обладатель редкого тенора» — from copy_assets, not facts | Baseline |
| Service leakage | «Возрастное ограничение: 12+. Продолжительность не уточняется.» | None | **Prototype** |
| Duplication | None | None | Draw |
| Readability | Blockquote + track list + clear headings | No blockquote, broken grammar («в центре внимания великой любви») | **Baseline** |
| Structural fit | Good: Репертуар/Образ/Условия | Acceptable but understructured | Baseline |
| Professional tone | Good | Broken sentence harms professionalism | **Baseline** |

**Corrected verdict**: Baseline clearly better. Но мне нужно признать что prototype hygiene win (removal of service leakage) — genuine and valuable. Ошибка в prototype — extraction cut (8→4) и grammar defect, не pattern concept.

**Recalibration on credibility_signals**: «выпускник СПб консерватории, обладатель редкого тенора» — я сказал «editorialized». Пересматриваю: эти данные **могут быть в source_text/raw_excerpt**, просто не прошли в `facts_text_clean`. Если они source-backed, это не editorialized — это enrichment. Я был слишком жёсткий тут. Correct judgment: допустимы, **если** traceable к source. Нужен evidence contract, а не blanket ban.

---

### Event 2687 — Лекция «Художницы» (9→7 facts)

| Axis | Baseline | Prototype | Who wins |
|------|----------|-----------|----------|
| Factual coverage | All 9 present | All 7 present (+ severe repeats) | Baseline (fewer facts, but clean) |
| Unsupported prose | «позволит глубже погрузиться в тему» — filler | None visible | Prototype (less filler) |
| Service leakage | «Формат мероприятия — лекция» — meta-noise | None | Prototype |
| **Duplication** | **None** | **5 instances of 2-3x repetition** | **Baseline** (catastophic prototype failure) |
| Readability | Clean list + headings | Unreadable due to duplication | **Baseline** |
| Structural fit | Good: named artist list → heading → separate themes | Broken by duplication | Baseline |
| Professional tone | Good (minus filler) | Unacceptable due to duplication | Baseline |

**Corrected verdict**: Baseline clearly better **but the cause is harness failure (revise/dedup loop), not pattern failure**. `program_led` routing was correct. The duplicate catastrophe was caused by revise loop blindly reinserting facts without checking they already exist. If revise loop had dedup → prototype text quality would likely be comparable or better (coverage 1 missing vs 5 missing).

**This is the key recalibration**: мой прошлый verdict правильно идентифицировал duplication problem, но неправильно bundled pattern assessment. `program_led` для лекции с 6 именами — correct pattern. Failure = revise implementation.

---

### Event 2673 — Собакусъел (11→7 facts)

| Axis | Baseline | Prototype | Who wins |
|------|----------|-----------|----------|
| Factual coverage | ~10/11 present | ~6/7 present + CTA leak | Baseline |
| Unsupported prose | Minimal | «приглашает на презентацию» — CTA | Baseline |
| Service leakage | None | CTA word | Baseline |
| Duplication | None | 2 instances (печенье, задачи платформы) | Baseline |
| Readability | Blockquote + 3 clear headings | 1 heading «О событии», long paragraphs | Baseline |
| Structural fit | Good: Программа/Для кого/Запуск | Weak | Baseline |
| Professional tone | Good | Acceptable minus CTA+duplication | Baseline |

**Corrected verdict**: Baseline better, confirmed. But again: duplication (harness) + CTA leak (prompt hygiene) are fixable. `program_led` routing was correct — presentation with program items suits this pattern. If fixes applied, prototype could win on coverage (5→1 missing).

---

## 2. Pattern vs harness failure map

| Event | Pattern choice | Pattern correct? | Failure source |
|-------|---------------|------------------|---------------|
| 2660 | `value_led` | ❌ Wrong: sparse event (3 facts) needs `compact_fact_led` | **Routing** |
| 2745 | `value_led` | ❌ Wrong: sparse same-theme event needs `compact_fact_led` | **Routing** |
| 2734 | `value_led` | ⚠️ Acceptable, but `person_led` (about performer) could be better | Routing (secondary); **Extraction** (primary: 8→4 facts lost) |
| 2687 | `program_led` | ✅ Correct: lecture about 6 named artists = program-structured | **Revise/repair** (duplication); prompt (no dedup rule) |
| 2673 | `program_led` | ✅ Correct: presentation with program items | **Revise/repair** (duplication); **prompt** (CTA leak) |

**Conclusion**: pattern concepts are mostly sound. 0 failures caused by pattern idea itself. All failures are implementation-level:
- 2/5 = routing (wrong pattern selection for sparse events)
- 2/5 = revise/repair (duplication)
- 1/5 = extraction (too aggressive fact reduction)
- 1/5 = prompt hygiene (CTA not blocked)

---

## 3. Copy_assets boundary

### Was I too strict?

Да. Мой прошлый verdict «всё, чего нет в facts_text_clean = editorialized» — **слишком грубый**.

Пересмотренная позиция:

| copy_assets field | Previous verdict | Corrected verdict | Rule |
|------------------|-----------------|-------------------|------|
| `core_angle` | OK | **Accept**: useful generation anchor | Must reference only facts_text_clean content |
| `format_signal` | LLM-derived = risky | **Accept with constraint**: derive from `event_type` field, not LLM | Deterministic derivation preferred |
| `program_highlights` | OK | **Accept**: structured list is valuable for program_led | Items must be traceable to facts or source |
| `why_go_candidates` | OK | **Accept**: useful for value_led lead | Must be derivable from facts, not invented |
| `experience_signals` | Editorialized | **Accept with evidence contract** | Must cite source fragment. If no source fragment → drop. |
| `credibility_signals` | Editorialized | **Recalibrated: accept if source-backed** | Allowed if in source_text/raw_excerpt, even if not in facts_text_clean |
| `scene_cues` | OK with evidence_span | **Accept with evidence_span** | evidence_span must match source. No span = no cue. |
| `tone_hint` | Not discussed | **Accept as soft guidance** | Never appears in output text, only guides style |

### Evidence-backed copy_assets contract

Правильная граница — не «in facts_text_clean or not», а:

```
Level 1 (strict): content IS in facts_text_clean → always allowed
Level 2 (evidence): content IS in source_text/raw_excerpt, traceable → allowed as generation aid
Level 3 (inferred): content is LLM inference from source → allowed ONLY for soft aids (tone_hint, core_angle)
Level 4 (invented): content has no source basis → forbidden
```

`credibility_signals` типа «выпускник СПб консерватории» — Level 2, если это есть в source_text. Это не editorialized, это enrichment from source that didn't pass the bucket filter into facts_text_clean. Допустимо.

`experience_signals` типа «эпоха золотого времени советской эстрады» — Level 3: inferred from tone of source, not verbatim. Допустимо как soft aid, но generation не должен копировать verbatim.

---

## 4. Baseline coverage floor rule

### Previous (too rigid)

> «Prototype extraction must ≥ baseline facts, not fewer»

### Corrected: content-preservation floor

```
Prototype extraction MUST preserve publishable content coverage ≥ baseline.
A fact may be dropped ONLY if:
  - duplicate of another retained fact (same information, different wording)
  - service-like (logistics, CTA, meta-noise like "длительность не уточняется")
  - policy-forbidden (price, address, phone, age restriction without context)
  - anchor-like (title repetition, format restating "Формат: лекция")

A fact MUST NOT be dropped if:
  - it's a concrete detail (person name, program item, technique, track title)
  - it's a quoted characterization or slogan
  - it's a visitor condition (what to bring, format, group size)
```

**Кейс 2734**: baseline had track «Лучший город земли» + «и другие композиции». Prototype dropped both. By content-preservation floor, «Лучший город земли» is a concrete detail → must not be dropped. «Продолжительность не уточняется» — service-like meta-noise → may be dropped. This is the correct granularity.

---

## 5. Corrected v2 patch set

| # | Proposal | Verdict | Modification (if any) |
|---|---------|---------|----------------------|
| 1 | Anti-duplication rule in generation prompt | **Accept now** | Add: «Каждый факт упоминается ровно один раз. Не повторяй одну деталь в разных секциях.» |
| 2 | Anti-duplication runtime check after revise/repair | **Accept now** | Sentence-level N-gram overlap ≥ 60% → remove duplicate. This is the critical fix for 2687/2673. |
| 3 | Restore blockquote/epigraph as cross-pattern rule | **Accept with modification** | Not «always blockquote». Rule: if `_pick_epigraph_fact` returns a quote → blockquote. If it returns null → no blockquote, pattern lead stands alone. Blockquote = enhancement, not obligation. Compact sparse events should NOT have blockquote (it over-structures thin content). |
| 4 | Sparse routing → `compact_fact_led` | **Accept now** | Threshold: `len(facts_text_clean) ≤ 5` → `compact_fact_led`. No `value_led` or `scene_led` on sparse events. |
| 5 | Ban generic headings | **Accept with modification** | Ban list: «О событии», «О лекции», «О концерте», «О спектакле», «Подробности». But NOT a hard runtime block — a prompt instruction + `_collect_policy_issues` flag. Runtime doesn't reject, it guides revise. |
| 6 | CTA detection | **Accept now** | Add CTA words to forbidden marker check: «приглашаем», «приходите», «ждём вас», «не пропустите», «приглашает». Already partially exists, needs to cover prototype outputs too. |
| 7 | Derive `format_signal` from `event_type` | **Accept with modification** | Primary: `event_type` field → `format_signal`. Fallback: if `event_type` is null/generic, use LLM-derived value. Не full LLM replacement, а fallback chain. |
| 8 | Constrain `credibility_signals` | **Accept with modification (recalibrated)** | Previous: «only from facts_text_clean». Corrected: allowed from Level 1 (facts) or Level 2 (source_text traceable). Evidence contract: must be in source material. Generation can use, but must not present as new fact — use as background context. |
| 9 | Drop or keep `experience_signals` | **Keep (recalibrated)** | Previous: «drop, mostly editorialized». Corrected: keep as Level 3 soft aid. Generation uses for tone/style guidance, never copies verbatim. If no source basis detectable → skip for this event. |
| 10 | Merge `value_led` + `topic_led` | **Defer** | Not enough evidence from 5 events. Both have distinct roles: value_led = «why attend» lead, topic_led = «what this is about» lead. Merging without more test cases is premature. Revisit after 20-event dry-run. |
| 11 | Reduce pattern set from 6 to 4 | **Defer** | Same reasoning as #10. Current dry-run tested only 2 patterns (value_led, program_led). Can't eliminate untested patterns based on untested data. Revisit after broader test coverage. |

### Additional v2 patches (new proposals)

| # | Proposal | Verdict | Rationale |
|---|---------|---------|-----------|
| 12 | **Revise loop: add «already mentioned» check** | **Accept now** | Before inserting a «missing» fact, check if its core content (key nouns) already appears in text. This directly prevents the 2687 catastrophe. |
| 13 | **Generation prompt: add section-level content budget** | **Accept now** | Rule: «Под каждым ### должно быть 2-4 предложения уникального контента. Если не хватает фактов для секции — не создавай её.» This prevents micro-sections and forces content density. |
| 14 | **Extraction: content-preservation floor** (§4 above) | **Accept now** | Replace rigid «≥ baseline count» with semantic preservation rule. |

### v2 dry-run scope

Same 5 events + **patches 1, 2, 3 (modified), 4, 5 (modified), 6, 12, 13, 14** applied.

Expected outcomes:
- 2660: `compact_fact_led` → shorter, cleaner, no over-structuring
- 2745: `compact_fact_led` → compact paragraph, no bloated value_led
- 2734: Full 8 facts preserved, no service leakage (cleanup handles it), blockquote restored
- 2687: Zero duplication, `program_led` with clean artist list
- 2673: Zero duplication, `program_led` with program list, no CTA leak

If v2 achieves these → transfer to code. If 2+ events still have critical issues → one more iteration.

---

## Summary of recalibration

| What I corrected | Previous position | New position |
|-----------------|-------------------|-------------|
| Structural signals weight | Blockquote/headings absence = regression | Absence ≠ regression for sparse events; blockquote = enhancement not obligation |
| Pattern concept assessment | Patterns failed in v1 | Patterns mostly correct; failures are routing/harness/revise |
| copy_assets boundary | Not in facts = editorialized | 4-level evidence contract; source-backed enrichment is valid |
| Fact count floor | Rigid numeric ≥ baseline count | Content-preservation floor with explicit drop reasons |
| experience_signals | Drop | Keep as Level 3 soft aid |
| credibility_signals | Drop if not in facts | Accept if source-traceable (Level 2) |
