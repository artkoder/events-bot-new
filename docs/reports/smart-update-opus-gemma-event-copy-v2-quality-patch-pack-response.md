# Opus → Gemma Event Copy: V2 Quality Patch Pack

Дата: 2026-03-07

---

## 1. Prioritization

| # | Proposal | Verdict | Rationale |
|---|---------|---------|-----------|
| 1 | Anti-duplication rule in prompt | `must_include_v2` | Production blocker: 3/5 events. One line in prompt. |
| 2 | Anti-duplication runtime guard | `must_include_v2` | Safety net when prompt fails. ~15 LOC. |
| 3 | Anti-embellishment rule | `must_include_v2` | Highest quality signal after dedup. One line. |
| 4 | Anti-metatext rule | `must_include_v2` | Eliminates «лекция расскажет о...» pattern. One line. |
| 5 | Filler phrase ban | `must_include_v2` | Gemma follows explicit ban lists well. 6-8 phrases. |
| 6 | Blockquote/epigraph recovery | `must_include_v2` | Conditional: only if `_pick_epigraph_fact` returns value. 0 new prompt tokens — it's runtime logic. |
| 7 | Sparse routing → compact | `must_include_v2` | Fixes 2/5 events (2660, 2745). Routing change, not prompt. |
| 8 | Generic heading ban | `include_if_compact` | Useful but lower priority than anti-metatext. Compact enough. |
| 9 | Lead engineering | `include_if_compact` | 3 patterns (concrete/quote/contrast). Drop question-led. ~40 tokens. |
| 10 | Sentence variety rule | `include_if_compact` | Keep ONLY: «не начинай 2 предложения подряд с одного слова». 1 line. |
| 11 | Few-shot example | `defer` | High-risk (structure copy, token overhead). After v2 dry-run. |
| 12 | Density-aware prompts (3 tiers) | `defer` | Sparse/non-sparse split is enough for v2. 3 tiers premature. |
| 13 | Paragraph quality gate (runtime) | `defer` | Too fuzzy for v2. False positives > benefit. |
| 14 | Sentence-level prose quality gate | `defer` | Same. Only anti-repeated-start worth taking (goes into #10). |

**Summary**: 7 `must_include`, 3 `include_if_compact`, 4 `defer`. Total prompt overhead: ~100-120 tokens.

---

## 2. Generation blocks

### 2.1. `generation_quality_block`

Вставить **перед** блоком «Запреты» в `_llm_fact_first_description_md`:

```
Качество текста:
- Каждый факт — ровно ОДИН раз. Не повторяй деталь в разных секциях.
- Не достраивай смысл: можно перестроить фразу факта, нельзя добавить то, чего нет.
- Не начинай 2 предложения подряд с одного слова.
- Запрещены филлеры: «обещает стать», «позволит погрузиться», «заставит задуматься»,
  «не оставит равнодушным», «подарит эмоции», «даёт возможность».
- Описывай содержание, не формат. Не пиши «лекция расскажет о...» / «спектакль рассказывает...» /
  «выставка представляет собой...» — пиши сразу содержание.
```

**~80 tokens. 5 high-signal rules. Covers proposals #1, #3, #4, #5, #10.**

### 2.2. `lead_block`

Вставить **внутрь** блока «Структура», заменив текущий Style C section:

```
Лид (1-2 предложения, без заголовка):
- Начни с самого яркого/необычного факта. Не с определения события.
- Если есть цитата (epigraph_fact) — blockquote уже стоит; лид идёт после него, не повторяя цитату.
- Если нет яркого факта — начни с конкретной детали, контраста или ключевой фигуры.
- ЗАПРЕТ: не начинай с «Это...», «Данное мероприятие...», «Лекция расскажет о...».
```

**~50 tokens. Covers #9 (lead engineering). No question-led. No gimmicks.**

### 2.3. `heading_quality_rule`

Добавить **внутрь** блока «Структура», после правил про `### ...`:

```
- Запрещённые headings: «О событии», «О лекции», «О концерте», «О спектакле»,
  «Подробности», «Основная идея», «Формат мероприятия», «Ключевые мотивы».
  Heading должен называть СОДЕРЖАНИЕ секции, а не тип.
```

**~35 tokens. Covers #8.**

### 2.4. `revise_quality_block`

Добавить в `_fact_first_revise_prompt` после «Цели (строго)»:

```
Редакторские правила:
- Каждый факт упоминается ровно ОДИН раз. Если факт уже есть — не добавляй снова.
- Удали филлеры: «обещает стать», «позволит погрузиться», «заставит задуматься»,
  «не оставит равнодушным».
- Если лид начинается с «Лекция расскажет...» / «Это...» — перепиши содержательно.
- Heading «О событии» / «Подробности» / «Основная идея» — замени на конкретный.
```

**~55 tokens. Mirrors generation rules for revise consistency.**

### Total prompt growth

| Block | Tokens | Where |
|-------|--------|-------|
| generation_quality_block | ~80 | generation prompt |
| lead_block | ~50 | generation prompt (replaces ~40 tokens of Style C) |
| heading_quality_rule | ~35 | generation prompt |
| revise_quality_block | ~55 | revise prompt |
| **Net new** | **~180** | across 2 prompts |

Net growth in generation prompt: ~125 tokens (lead_block replaces ~40 tokens of Style C).
This is within Gemma instruction budget.

---

## 3. Runtime gates

### 3.1. `must_include_v2` runtime gates

**Gate 1: Duplicate sentence guard** (after revise/repair)

```python
def _dedup_description_sentences(text: str) -> str:
    """Remove near-duplicate sentences. Run AFTER revise."""
    paragraphs = re.split(r'\n{2,}', text)
    seen_cores = []  # list of word-sets from kept sentences
    cleaned_paragraphs = []
    for para in paragraphs:
        if para.strip().startswith('###') or para.strip().startswith('>'):
            cleaned_paragraphs.append(para)
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        kept = []
        for s in sentences:
            words = set(re.findall(r'[а-яёa-z]{4,}', s.lower()))
            if len(words) < 4:
                kept.append(s)
                continue
            is_dup = any(
                len(words & prev) / max(len(words), 1) > 0.6
                for prev in seen_cores
            )
            if not is_dup:
                kept.append(s)
                seen_cores.append(words)
        if kept:
            cleaned_paragraphs.append(' '.join(kept))
    return '\n\n'.join(cleaned_paragraphs)
```

**Gate 2: CTA detection** (extend existing forbidden marker check)

```python
CTA_BANNED = [
    'приглашаем', 'приходите', 'ждём вас', 'ждем вас',
    'не пропустите', 'успейте', 'присоединяйтесь',
    'предлагаем', 'встречайте', 'приглашает',
]
# Add to _collect_policy_issues or forbidden marker check
for cta in CTA_BANNED:
    if cta in description.lower():
        issues.append(f'CTA-слово «{cta}» — запрещено, удали.')
```

**Gate 3: Metatext lead detection** (add to `_collect_policy_issues`)

```python
def _detect_metatext_lead(description: str) -> str | None:
    lead = description.split('###')[0].strip().lstrip('> ').strip()
    if re.match(
        r'(?i)(Это|Лекция|Спектакль|Концерт|Выставка|Мероприятие)\s+'
        r'(рассказыва|расскаж|представля|явля|это\s)',
        lead
    ):
        return 'Лид начинается с метатекста. Перепиши: опиши содержание напрямую.'
    return None
```

**Gate 4: Weak heading detection** (add to `_collect_policy_issues`)

```python
WEAK_HEADINGS = {
    'о событии', 'о лекции', 'о концерте', 'о спектакле', 'о выставке',
    'подробности', 'основная идея', 'формат мероприятия', 'ключевые мотивы',
    'описание', 'детали',
}

def _detect_weak_headings(description: str) -> list[str]:
    issues = []
    for line in description.splitlines():
        m = re.match(r'###\s+(.+)', line)
        if m and m.group(1).strip().lower() in WEAK_HEADINGS:
            issues.append(f'Heading «{m.group(1).strip()}» слишком generic — замени на содержательный.')
    return issues
```

### 3.2. Runtime gate summary

| Gate | Lines of code | Runs when | False positive risk |
|------|--------------|-----------|-------------------|
| Dedup sentences | ~20 | After revise | Low (60% threshold is conservative) |
| CTA detection | ~5 | In `_collect_policy_issues` | Very low (exact match) |
| Metatext lead | ~8 | In `_collect_policy_issues` | Low (regex pattern specific) |
| Weak heading | ~8 | In `_collect_policy_issues` | Very low (exact set match) |
| **Total** | **~40 LOC** | | |

Gates 2-4 feed into revise loop; they don't block or reject, they inform revise prompt.

---

## 4. Compact sparse contract

### When: `len(facts_text_clean) ≤ 5`

### Format:

| Parameter | Value | Why |
|-----------|-------|-----|
| Headings | **0** headings (`###`) | ≤5 facts don't have enough content for sections |
| Blockquote | **Conditional**: if `_pick_epigraph_fact` returns a quote → yes; otherwise no | Quote-worthy facts exist even in sparse events; but don't force |
| Target length | **150-400 chars** (2-4 предложения prose) | Не растягивать. Budget formula: `max(150, len(facts) * 80)` |
| Emoji | 0-1 | Sparse text with emoji looks clownish |
| Lists | Allowed if facts contain enumeration (e.g., 3 items) | Short list ≠ heading structure |

### What to avoid:

| Avoid | Why |
|-------|-----|
| Multiple `###` sections | Over-structures thin content |
| Opening with metatext «Это...» | Generic for any event |
| Expanding 3 facts into 3 paragraphs | Creates repetitive padding |
| «Обещает стать... / не оставит равнодушным...» filler | Fills budget without content |

### Sparse prompt variant:

```
Ты пишешь короткий Markdown-анонс. Факты мало — не растягивай.

Правила:
- НИКАКИХ заголовков ###.
- 2-4 предложения prose, максимум.
- Если epigraph_fact не null — одна строка `> epigraph_fact`, затем пустая строка, затем prose.
- Если epigraph_fact null — сразу prose.
- Не дублируй факт. Не добавляй то, чего нет.
- Не пиши «обещает стать», «позволит погрузиться», «не оставит равнодушным».
- Не начинай с «Это...» или «Лекция расскажет о...».
- 0-1 эмодзи.

{SMART_UPDATE_YO_RULE}
{SMART_UPDATE_VISITOR_CONDITIONS_RULE}

Контекст:
- title: {title}
- event_type: {event_type}
- epigraph_fact: {epigraph_fact or 'null'}
- description_budget_chars: {budget_chars}

Факты:
{facts_block}

Верни только Markdown-текст (без JSON).
```

**~100 tokens. Clean. Unambiguous.**

---

## 5. Final v2 patch pack

### Prompt changes

| # | What | Where | Tokens |
|---|------|-------|--------|
| P1 | `generation_quality_block` | `_llm_fact_first_description_md`, before «Запреты» | +80 |
| P2 | `lead_block` (replaces Style C) | `_llm_fact_first_description_md`, «Структура» section | +10 net |
| P3 | `heading_quality_rule` | `_llm_fact_first_description_md`, «Структура» section | +35 |
| P4 | `revise_quality_block` | `_fact_first_revise_prompt`, after «Цели» | +55 |
| P5 | Sparse prompt variant | New function `_llm_compact_fact_led_prompt` | +100 (new) |

### Runtime changes

| # | What | Where | LOC |
|---|------|-------|-----|
| R1 | `_dedup_description_sentences` | After revise/repair, before final cleanup | ~20 |
| R2 | CTA banned words check | `_collect_policy_issues` | ~5 |
| R3 | Metatext lead detection | `_collect_policy_issues` | ~8 |
| R4 | Weak heading detection | `_collect_policy_issues` | ~8 |
| R5 | Epigraph/blockquote recovery | Route `_pick_epigraph_fact` result into prototype prompt | ~3 |

### Routing changes

| # | What | Where |
|---|------|-------|
| RT1 | `len(facts_text_clean) ≤ 5` → sparse prompt variant (P5) | Pattern routing |
| RT2 | `len(facts_text_clean) > 5` → standard prompt (P1-P3 applied) | Pattern routing |

### Do not include in v2

| # | What | Why |
|---|------|-----|
| ❌ | Question-led openings | Gimmick risk, synthetic engagement |
| ❌ | Few-shot examples in prompt | Token overhead, structure copy risk |
| ❌ | 3-tier density prompts | sparse/non-sparse split is enough |
| ❌ | Paragraph quality gate | Too fuzzy, false positives |
| ❌ | Sentence-level prose gate | Only anti-repeated-start taken (#10), rest deferred |
| ❌ | Pattern set reduction (6→4) | Not enough dry-run data |
| ❌ | Merge value_led + topic_led | Not enough dry-run data |
| ❌ | experience_signals in generation | Tone drift risk too high for v2 |

---

## Expected v2 outcomes (same 5 events)

| Event | v1 problem | v2 fix | Expected result |
|-------|-----------|--------|----------------|
| 2660 | Over-structured sparse, wrong routing | RT1 → sparse prompt P5 | 2-4 sentences, no headings, no embellishment |
| 2745 | Wall of text, value_led wrong | RT1 → sparse prompt P5 | Compact prose, clean tone |
| 2734 | Extraction loss, broken grammar, service leak | R1 (dedup), content-preservation floor, P1 (anti-metatext) | 8 facts retained, clean grammar, no service block |
| 2687 | Catastrophic duplication (5×) | P1+P4 (anti-dup prompt), R1 (dedup runtime) | Each fact once. program_led structure intact. |
| 2673 | Duplication, CTA leak | P1+P4 (anti-dup), R1 (dedup), R2 (CTA check) | Clean text, no «приглашает», no repeats |

**If these outcomes hold → transfer to production code.**
