# Opus → Gemma Event Copy: Preservation / Migration Matrix

Дата: 2026-03-07

---

## 1. Preservation Matrix

### 1.1. Fact pipeline (upstream)

| # | Mechanism | Verdict | Why | Where in new architecture |
|---|-----------|---------|-----|--------------------------|
| 8 | `_facts_text_clean_from_facts` | **Keep as-is** | Тонкая pipeline: bucket classification, anchor filtering, dedupe, chat-URL stripping. Всё это — prerequisite для P0 fact completeness. Нечего менять. | Upstream, до extraction и generation. Без изменений. |
| 3 | `_sanitize_fact_text_clean_for_prompt` | **Keep as-is** | Rewrite «посвящ...» → «Тема: ...» предотвращает LLM copy-paste запрещённого корня. Работает на input, не на output. Проверена. | Upstream, применяется к `facts_text_clean` перед подачей в generation prompt. |
| 10 | `FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE` | **Adapt → move to extraction** | Сейчас живёт только в extraction prompt. В quality-first v1 это правило нужно дополнить: extraction должен также возвращать `program_highlights` в `copy_assets`. Но исходное правило для facts extraction остаётся. | Extraction prompt (facts). Дополнительно: copy_assets extraction получает аналогичное rule для `program_highlights`. |

### 1.2. Epigraph / opening logic

| # | Mechanism | Verdict | Why | Where in new architecture |
|---|-----------|---------|-----|--------------------------|
| 1 | `_pick_epigraph_fact` | **Adapt** | Сильный механизм: превращает цитату/слоган в blockquote opening. Это даёт «живой вход», который pattern-driven opening может потерять. Нужно интегрировать с `voice_fragments` из `copy_assets`. | Pattern-aware layer: если `voice_fragments` содержит direct quote, `_pick_epigraph_fact` переносится. Если нет — epigraph picking из facts остаётся как fallback. |
| 2 | Epigraph / blockquote opening в `_fact_first_description_prompt` | **Adapt** | Текущая логика: если `epigraph_fact` не null → blockquote opening; если null → plain lead. Это нужно **адаптировать к patterns**: `person_led` и `scene_led` получают разные opening rules. Но blockquote opening — это не pattern, это **cross-pattern enhancement**, applicable к любому pattern когда есть хорошая цитата. | Generation prompt: cross-pattern rule. Если `epigraph_fact` / `voice_fragments[0]` доступен → blockquote перед lead, regardless of pattern. Pattern instructions определяют lead ПОСЛЕ blockquote. |
| 15 | `Style C` logic | **Replace** | Style C (Сцена → смысл → детали) — это прототип `scene_led`. В quality-first v1 `scene_led` заменяет Style C с более сильным contract (traceability gate, evidence_span, safe fallback). | Заменён на 6 named patterns. Style C instructions больше не нужны — всё покрыто patterns + `SHARED_DESCRIPTION_RULES`. |

### 1.3. Quality enforcement (coverage / cleanup / repair)

| # | Mechanism | Verdict | Why | Where in new architecture |
|---|-----------|---------|-----|--------------------------|
| 6 | `_collect_policy_issues` | **Adapt** | Ценный механизм: headings count (2-3), lead paragraph validation, orphan heading detection, duplicate headings. Нужно **расширить** quality_flags из нашего redesign (template_feel, weak_lead, weak_heading, redundancy). | Coverage / policy layer: после generation. Объединить текущие `_collect_policy_issues` checks с новыми quality_flags в единый coverage step. |
| 11 | `_find_missing_facts_in_description` | **Keep as-is** | Deterministic, tested, tuned. Handles quoted facts, ignores anchor-like/service facts, respects length limits. Ничего менять не нужно. | Post-generation, pre-repair. Unchanged. |
| 12 | `_llm_integrate_missing_facts_into_description` | **Keep as-is** | P0 safety net. Prompt уже правильный: не adds new facts, preserves blockquotes, respects logistics ban. В new architecture — conditional после coverage check (threshold ≥ 2 or 1 quoted). | Post-coverage repair. Условный 4-й call если coverage gap found. |
| 4 | `_fact_first_remove_posv_prompt` | **Keep as-is** | Last-mile «посвящ...» fix. Gemma упорно копирует это слово. Upstream sanitize (mechanism #3) ловит в input, этот ловит в output. Дублирования нет — это defence in depth. | Post-revise cleanup. Triggered by regex check на output. |

### 1.4. Formatting / cleanup

| # | Mechanism | Verdict | Why | Where in new architecture |
|---|-----------|---------|-----|--------------------------|
| 5 | `_cleanup_description` (internal) | **Keep as-is** | Format-only: strip private-use chars, fix bullet lists, normalize blockquotes, limit emojis, sanitize, dedupe, normalize paragraphs, ensure headings. Все operations — deterministic, meaning-preserving. | Post-generation cleanup. Applied to raw LLM output before coverage check. Same placement. |
| 7 | `_ensure_minimal_description_headings` | **Adapt (minor)** | Полезный fallback: если LLM вернул 2+ paragraphs без headings → inject «### О событии». Но «О событии» — generic heading, попадает под anti-template weak_heading check. Нужно: заменить fallback heading на что-то из heading palette или derive from `core_angle`. | Cleanup layer. Fallback heading: derive from `format_signal` (e.g. «### Программа» для мастерской, «### О спектакле» для спектакля) вместо generic «О событии». |
| 13 | `_llm_reflow_description_paragraphs` | **Keep as-is** | Safety valve для overlong paragraphs (>850 chars). Happens rarely but when it does, it's critical for Telegraph readability. | Late cleanup. Same placement: triggered by `_has_overlong_paragraph`. |
| 14 | `_llm_enforce_blockquote` | **Adapt** | В new architecture `voice_fragments` из `copy_assets` делает blockquote-управление более systematic. Но `_llm_enforce_blockquote` — fallback для merge flow. Пока merge flow жив, механизм остаётся. | Merge flow: after description ready. В fact-first flow: replaced by epigraph logic + `voice_fragments`. |

### 1.5. Rules / constants

| # | Mechanism | Verdict | Why | Where in new architecture |
|---|-----------|---------|-----|--------------------------|
| 9 | `VISITOR_CONDITIONS_RULE` | **Adapt → merge into SHARED_DESCRIPTION_RULES** | Хорошее правило: conditions = facts о событии, должны попадать в description. Нужно объединить с density tier logic: при dense events conditions group в отдельный ### блок. | `SHARED_DESCRIPTION_RULES` — conditions section. Дополнительно: dense tier rule для grouping. |

---

## 2. Do Not Lose: 8 strongest elements

| # | Element | What it gives | Risk при redesign |
|---|---------|--------------|-------------------|
| **1** | **`_pick_epigraph_fact` + blockquote opening** | Живой opening через цитату/слоган. Telegraph анонсы с blockquote opening выглядят профессиональнее. | Pattern-driven lead может заменить blockquote opening на prose lead → потеря визуального якоря. **Fix**: blockquote = cross-pattern enhancement, не часть pattern. |
| **2** | **`_facts_text_clean_from_facts` pipeline** | Чистый, deduplicated, bucket-filtered fact inventory. Garbage in → garbage out без этого. | Redesign может попытаться bypass fact pipeline и работать напрямую с `copy_assets`. **Fix**: `copy_assets` — дополнение к facts, не замена. Generation получает оба. |
| **3** | **`_find_missing_facts_in_description` + `_llm_integrate_missing_facts`** | P0 safety net: ловит пропущенные факты после generation. | Redesign может решить «patterns уже обеспечивают coverage» и убрать repair layer. **Fix**: repair layer остаётся, threshold = 2+. |
| **4** | **`_sanitize_fact_text_clean_for_prompt`** | Предотвращает «посвящ...» leak на input level. | Если перестроить extraction → generation pipeline, этот sanitizer может выпасть из chain. **Fix**: sanitizer применяется к `facts_text_clean` ПЕРЕД любым prompt, не внутри одного конкретного prompt. |
| **5** | **`_collect_policy_issues`** | Headings count (2-3), lead validation, duplicate heading detection, orphan heading check. Это structural quality gate. | Quality-first redesign фокусируется на lead/pattern quality, может забыть structural checks. **Fix**: extend `_collect_policy_issues`, не заменять. |
| **6** | **`VISITOR_CONDITIONS_RULE`** | Conditions = facts, не logistics. Без этого LLM иногда выбрасывает conditions из description. | Новые pattern instructions могут не включать conditions правило. **Fix**: visitor conditions → `SHARED_DESCRIPTION_RULES`, applies to all patterns. |
| **7** | **Budget scaling** (`_estimate_fact_first_description_budget_chars`) | Description length ∝ facts count. Это P0 enabler: без пропорционального budget P0 невозможен. | Redesign может ввести fixed budget для patterns (e.g. «scene_led = 200 words»). **Fix**: budget ALWAYS fact-proportional, patterns don't override budget. |
| **8** | **`_fact_first_remove_posv_prompt`** | Last-mile fix для stubborn «посвящ...» copies. Gemma-specific. | Redesign может решить «sanitizer на input достаточен». Но Gemma reproducibly leaks этот корень. **Fix**: keep both upstream sanitizer and output posv fix. |

---

## 3. Migration Plan

### Phase 1: Transfer without breaking (before pattern branch)

| What | Action | Risk |
|------|--------|------|
| `_facts_text_clean_from_facts` | No changes. Verify still called in new pipeline. | Zero |
| `_sanitize_fact_text_clean_for_prompt` | No changes. Apply to facts before ANY prompt. | Zero |
| `_find_missing_facts_in_description` | No changes. | Zero |
| `_cleanup_description` (internal) | No changes. | Zero |
| `VISITOR_CONDITIONS_RULE` | Move into `SHARED_DESCRIPTION_RULES`. | Low |
| Budget formula | Verify unchanged. | Zero |

### Phase 2: Transfer with pattern-driven branch

| What | Action | Risk |
|------|--------|------|
| `_pick_epigraph_fact` | Adapt: если `voice_fragments` есть → use `voice_fragments[0]`; иначе → fallback to `_pick_epigraph_fact`. | Medium: нужен fallback path |
| Epigraph blockquote logic | Move to cross-pattern rule в generation prompt. | Medium: prompt complexity |
| `_collect_policy_issues` | Extend: добавить quality_flags. Keep existing checks. | Low |
| `_ensure_minimal_description_headings` | Replace fallback heading «О событии» → derive from `format_signal`. | Low |
| `Style C` | Replace with 6 named patterns. | Medium: style loss risk |
| `FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE` | Keep for facts extraction. Add `program_highlights` to copy_assets. | Low |
| `_llm_enforce_blockquote` | Keep for merge flow. Adapt for fact-first: blockquote via epigraph/voice_fragments. | Low |

### Phase 3: Validate then remove (only after real A/B)

| What | Action | Condition |
|------|--------|-----------|
| `Style C` prompt text | Remove residual code | Only after patterns produce equivalent+ quality |
| `_llm_integrate_missing_facts` trigger rate | Evaluate: if <10%, consider simplifying | Only after 50-event validation shows low trigger rate |
| `_fact_first_remove_posv_prompt` | Keep indefinitely | Gemma-specific. Remove only if model changes |

---

## 4. Prompt vs Runtime Boundary

| Mechanism | Prompt? | Runtime check? | Repair layer? | Cleanup layer? | Rationale |
|-----------|---------|----------------|---------------|----------------|-----------|
| Fact completeness (P0) | ✅ Prompt (shared rule) | ✅ Coverage check (`_find_missing_facts`) | ✅ Repair (`_llm_integrate_missing_facts`) | — | Defence in depth: 3 layers |
| Pattern selection | — | ✅ Deterministic routing | — | — | No LLM judgment needed |
| Epigraph blockquote | ✅ Prompt (cross-pattern rule) | ✅ `_pick_epigraph_fact` (deterministic) | — | ✅ Cleanup (blockquote normalization) | Runtime picks fact, prompt says format it |
| Headings (2-3 count) | ✅ Prompt (structural rule) | ✅ `_collect_policy_issues` (count check) | ✅ Revise (если count wrong) | ✅ `_ensure_minimal_description_headings` | Prompt says 2-3; runtime verifies; cleanup adds if 0 |
| «посвящ...» ban | ✅ Prompt (запрет) | ✅ `_sanitize_fact_text_clean_for_prompt` (input) | ✅ `_fact_first_remove_posv_prompt` (output) | — | 3-layered: input sanitize → prompt ban → output fix |
| Visitor conditions | ✅ Prompt (shared rule) | — | — | — | Prompt-only; no runtime check needed. Coverage check catches missing facts including conditions. |
| Scene cue traceability | — | ✅ `_is_evidence_span_traceable` | — | — | Runtime-only. Prompt asks for evidence_span, runtime verifies it. |
| Contrast traceability | — | ✅ `_has_safe_contrast` | — | — | Same as scene cues. |
| Generic blocklist for scene_cues | — | ✅ `GENERIC_BLOCKLIST` check | — | — | Runtime-only. Content check. |
| Anti-template (lead variety) | ✅ Prompt (hard ban + soft pref) | ✅ Coverage (`quality_flags.template_feel`) | ✅ Revise (если flag triggered) | — | Prompt says don't; coverage checks; revise fixes. |
| Overlong paragraphs | — | ✅ `_has_overlong_paragraph` (>850 chars) | ✅ `_llm_reflow_description_paragraphs` | — | Runtime detects, repair fixes. Not worth prompt space. |
| Budget enforcement | ✅ Prompt (`description_budget_chars`) | — | — | ✅ `_clip(DESCRIPTION_MAX_CHARS)` | Prompt guides, Gemma's adherence variance, cleanup clips. |
| Dense tier grouping | ✅ Prompt (≤4 structural rules) | ✅ `_collect_policy_issues` (headings + list structure) | — | — | Prompt-driven, verified by policy. |

### Критическое замечание: conflict resolution

Есть одно место, где prompt и runtime могут конфликтовать:

**Prompt** говорит «pattern_name = scene_led, начни с микросцены».
**Runtime** `_pick_epigraph_fact` нашёл хорошую цитату и хочет blockquote opening.

Кто побеждает? **Blockquote epigraph всегда выше pattern lead.** Rationale: blockquote — это визуально сильный элемент Telegraph, и он grounded (из facts). Pattern lead идёт ПОСЛЕ blockquote.

Prompt формулировка:

```
Если epigraph_fact не null:
  → Blockquote opening: `> {epigraph_fact}` перед lead.
  → Lead строится по pattern ({pattern_name}) ПОСЛЕ blockquote.
  → В lead НЕ повторяй epigraph_fact.
Если epigraph_fact null:
  → Lead строится по pattern ({pattern_name}).
```

Это сохраняет strengths обоих механизмов: epigraph даёт «крючок», pattern даёт структуру.

---

## 5. Summary

| Category | Keep as-is | Adapt | Replace | Remove-later |
|----------|-----------|-------|---------|-------------|
| Fact pipeline | `_facts_text_clean_from_facts`, `_sanitize_fact_text_clean_for_prompt` | `FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE` (extend for copy_assets) | — | — |
| Opening | — | `_pick_epigraph_fact` (integrate with voice_fragments), epigraph logic (cross-pattern) | Style C → 6 patterns | — |
| Quality | `_find_missing_facts`, `_fact_first_remove_posv_prompt` | `_collect_policy_issues` (extend quality_flags), `_ensure_minimal_description_headings` (better fallback heading) | — | — |
| Repair | `_llm_integrate_missing_facts`, `_llm_reflow_description_paragraphs` | `_llm_enforce_blockquote` (merge-only, adapt for fact-first) | — | — |
| Cleanup | `_cleanup_description` | — | — | — |
| Rules | — | `VISITOR_CONDITIONS_RULE` → `SHARED_DESCRIPTION_RULES` | — | — |

**Zero mechanisms removed.** Всё либо stays, либо adapts, либо replaced by stronger equivalent. Safety net intact.
