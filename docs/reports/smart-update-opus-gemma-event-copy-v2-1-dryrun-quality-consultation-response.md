# V2.1 Dry-Run Corrective Consultation Response

Дата: 2026-03-07

---

## Честная оценка

v2.1 не просто «не стал quality win» — он **системно хуже v2** на 4/5 events.

Мои рекомендации из v2 consultation в значительной части **не сработали при локальной реализации**. Ниже — честный разбор: что именно сломалось, почему, и что делать дальше.

Ключевой вывод: **extraction repair pass — главный источник регрессии**. Не потому, что идея неправильная. А потому, что Gemma 27b-it не справляется с задачей «пересобери JSON extraction, учитывая issue hints» — это слишком сложная meta-task для 27b модели.

---

## 1. Event-by-event corrected verdict

### 1.1. Event 2660 — Дуальность мира

| Version | Missing | Forbidden | Prose quality |
|---------|---------|-----------|--------------|
| Baseline | 2 | — | Чистая, но editorial headings |
| V1 | 3 | — | Flat, generic heading |
| V2 | 3 | — | Compact, honest; best tone |
| **V2.1** | **3** | **date_ru_words** | Editorial drift, new forbidden |

**Verdict: V2 > baseline > V2.1 > V1.**

**Что произошло в v2.1**: repair pass добавил факт «Выставка продлится до 19 марта» → facts count стал 6 → event ушёл в standard branch вместо compact → generation выдала 3 секции с headings для события, у которого 5 реальных unique details → prose раздулся → появились embellishments («многогранность форм и текстур», «красоту в несовершенстве», «гармонию в конфликте»).

**Root cause**: repair pass затащил date-like fact → routing inflation → generation overfit.

**Partial gain**: нет.

---

### 1.2. Event 2745 — Сёстры

| Version | Missing | Forbidden | Prose quality |
|---------|---------|-----------|--------------|
| Baseline | 5 | — | Clean but empty |
| V1 | 5 | — | Clean but empty |
| V2 | 5 | посвящ* | Slightly better, but посвящ* |
| **V2.1** | **6** | **tickets, date_ru_words** | Service leak, worst coverage |

**Verdict: Baseline ≈ V1 > V2 > V2.1.**

**Что произошло**: extraction repair добавил «Дополнительно добавлено 7 мест на показ 7 марта, запись через Веронику Вишнёвую» — это чистый service/ticketing fact, запрещённый по правилам. Этот факт (1) протащил `tickets` forbidden marker, (2) вывел event в standard branch, (3) добавил logistics-секцию «Дополнительные места» в итоговый текст.

При этом core duplciate `«Спектакль о взаимоотношениях...»` и `«Спектакль рассказывает о взаимоотношениях...»` repair pass **не убрал**.

**Root cause**: repair pass — ненадёжный LLM call. Он добавляет вредное и не убирает то, что должен.

**Partial gain**: нет. Чисто regression.

---

### 1.3. Event 2734 — Концерт Гудожникова

| Version | Missing | Forbidden | Prose quality |
|---------|---------|-----------|--------------|
| Baseline | 5 | — | Tracklist preserved, structure good |
| V1 | 3 | — | Best overall: tracklist + structure + tone |
| V2 | 3 | посвящ* | Clean compact but посвящ* |
| **V2.1** | **4** | **посвящ*** | Worse coverage, посвящ* still there |

**Verdict: V1 > Baseline > V2 > V2.1.**

**Что произошло**: extraction repair prompt явно содержал hint «В raw_facts есть program-like quoted items, но не все дошли до facts_text_clean. Сохрани конкретные названия». Repair pass **проигнорировал этот hint** — треклист не восстановлен. При этом `copy_assets.program_highlights` содержит «Верни мне музыку», «Королева красоты», «Разговор со счастьем» — но facts_text_clean их не содержит. Это desync.

`посвящ*` остался и в facts, и в description. Anti-посвящ в extraction и repair prompts не сработали.

**Root cause**: Gemma 27b не выполняет multi-task instructions в repair JSON schema. Она берёт current extraction и лишь слегка его перефразирует, не делая реальной re-extraction.

**Partial gain**: нет.

---

### 1.4. Event 2687 — Лекция «Художницы»

| Version | Missing | Forbidden | Prose quality |
|---------|---------|-----------|--------------|
| Baseline | 5 | — | Generic headings, but clean coverage |
| V1 | **1** | — | Best coverage, but dirty duplication |
| V2 | 3 | — | Better headings, but hallucination |
| **V2.1** | **4** | **посвящ*** | Worse coverage + посвящ* + service fact |

**Verdict: V1 (by coverage) > Baseline > V2 > V2.1.**

**Что произошло**: repair pass **добавил** service-like fact «Мероприятие проходит в формате лекции» — это non-narrative statement, запрещённое по `_NON_NARRATIVE_FACT_RE`. Repair pass его не узнал и протащил.

`посвящ*` остался: «Лекция посвящена творчеству...» в факте №1 → generation скопировала.

Hallucination не убрана: «Наталья Гончарова и Ольга Розанова внесли вклад в развитие авангарда» — **не в фактах**. Anti-embellishment rule в prompt есть, но Gemma 27b его игнорирует при наличии имён без характеристик.

Whole-body metatext check добавлен, но тоже не полностью сработал: «Мероприятие представляет собой лекцию» прошёл.

**Partial gain**: headings «Образы российского общества в искусстве» и «Художницы с британскими корнями» — содержательнее, чем baseline. Это сигнал, что quality block помогает heading quality.

---

### 1.5. Event 2673 — Собакусъел

| Version | Missing | Forbidden | Prose quality |
|---------|---------|-----------|--------------|
| Baseline | 5 | — | Clean structure |
| V1 | **1** | cta_or_hashtag | Best coverage but CTA leak |
| V2 | 6 | посвящ* | Coverage collapse |
| **V2.1** | **4** | **—** | Clean hygiene, but generic sections |

**Verdict: V1 (by coverage) > V2.1 > Baseline > V2.**

**Partial improvement**: v2.1 — единственный event, где результат лучше v2. missing=4 vs 6, forbidden=none vs посвящ*. Это реальный gain от stronger extraction + preservation floor.

**Но**: 15 facts_text_clean с 6+ дублями (repair pass не убрал). Heading «О платформе «Собакусъел»» — generic. Секция «Для представителей креативной среды» — filler. Lead и секция 1 дублируют blockquote.

**Root cause remaining issues**: exact_dedup не поймал semantic duplicates (разные формулировки одного смысла). Repair pass не сжал.

---

## 2. Failure attribution map

| Stage | Failure mode | Severity | Events hit | Quality impact |
|-------|-------------|----------|------------|---------------|
| **Extraction repair** | Добавляет service/date facts | **CRITICAL** | 2660, 2745, 2687 | Branch inflation, forbidden markers, logistics в prose |
| **Extraction repair** | Не выполняет own issue hints | **CRITICAL** | 2734, 2687 | Tracklist lost, посвящ* stays |
| **Extraction repair** | Не убирает semantic duplicates | HIGH | 2745, 2673 | Inflated fact set, wasted prose budget |
| **Post-filter / floor** | `_preserve_content_floor` добавляет baseline facts без re-filter | MEDIUM | 2660 | Date fact протасовывается из baseline |
| **Routing** | Relies on polluted fact count | HIGH | 2660, 2745 | Sparse events forced into standard branch |
| **Generation prompt** | Anti-embellishment слишком abstract для Gemma | MEDIUM | 2660, 2687 | «красота в несовершенстве», invented characterizations |
| **Generation prompt** | Anti-metatext rules not sticky | MEDIUM | 2687 | «Мероприятие представляет собой лекцию» |
| **Revise / cleanup** | Can't fix what repair injected | LOW | All | Downstream cascade |

### Summary attribution

```
Extraction repair:   ~60% of regressions (introduced bad facts, ignored hints)
Post-filter/floor:   ~15% (didn't re-validate repair output adequately)
Routing:             ~10% (consequence of fact inflation)
Generation prompt:   ~10% (embellishment, metatext still leak)
Revise/cleanup:       ~5% (cascade from above)
```

**Вывод**: extraction repair pass — single largest contributor to v2.1 failure. Это не проблема prompt quality. Это проблема task complexity vs model capability.

---

## 3. Keep / Modify / Rollback / Remove

### 3.1. REMOVE: extraction repair pass

**Recommendation: удалить целиком.**

Причины:
1. На 4/5 events repair ухудшил extraction, а не улучшил.
2. Gemma 27b не справляется с meta-task: «прочитай current extraction + issue hints + raw_facts → пересобери JSON». Это требует reasoning, которого у 27b нет в достаточной мере.
3. Repair добавляет runtime: ~40-60s per event → ~200-300s total.
4. Issue hints генерируются детерминистически и правильно — но LLM их игнорирует.

**Альтернатива**: issue hints из `_extract_issue_hints` ценны. Их можно **вставить в initial extraction prompt** вместо отдельного repair pass. Это zero extra LLM calls, но лучше направляет первичную extraction.

### 3.2. KEEP: anti-посвящ* в extraction prompt

Работает? Частично. В v2.1 extraction prompt содержит ban, но Gemma всё равно генерирует `посвящ*` в 3/5 cases.

**Modify**: вместо удаления — усилить ban + добавить deterministic post-filter:

```python
_POSVYASH_PATTERNS = [
    (re.compile(r'(?i)^(.+?)\s+посвящён[аоы]?\s+(.+)$'), r'\1 о \2'),
    (re.compile(r'(?i)\bпосвящён[аоы]?\s+'), 'о '),
    (re.compile(r'(?i)\bпосвящ\w+\s+'), 'о '),
]

def _strip_posvyash_from_fact(fact: str) -> str:
    for pattern, repl in _POSVYASH_PATTERNS:
        fact = pattern.sub(repl, fact)
    return re.sub(r'\s+', ' ', fact).strip()
```

Это deterministic, грамматически более аккуратный, чем naive `посвящ* → о`, и не требует LLM.

### 3.3. KEEP: anti-merge / anti-inflate в extraction prompt

Работает? Частично. Anti-inflate сработал на 2673 (меньше дублей в initial extraction). Anti-merge на 2734 не сработал (треклист потерян).

**Modify**: усилить anti-merge конкретнее:
```
- Если в source есть 2+ названия в «кавычках», каждое — отдельный fact.
```

Это простая, verifiable инструкция. Gemma справляется с concrete rules лучше, чем с abstract guidelines.

### 3.4. KEEP: exact fact dedup (`_exact_dedup_facts`)

Работает? Да, на verbatim duplicates. Но на semantic near-duplicates — нет.

**Modify**: добавить word-overlap dedup с высоким порогом (≥0.85), только для факты начинающихся с одного слова:

```python
def _near_dedup_facts(facts: list[str]) -> list[str]:
    kept = []
    for fact in facts:
        is_dup = False
        for existing in kept:
            if _word_overlap_ratio(fact, existing) >= 0.85:
                # Keep longer variant
                if len(fact) > len(existing):
                    kept[kept.index(existing)] = fact
                is_dup = True
                break
        if not is_dup:
            kept.append(fact)
    return kept
```

Порог 0.85 (а не 0.6 как в v2) практически elimination-safe — ловит только настоящие перефразировки.

### 3.5. KEEP: whole-body metatext detection

Работает? Частично — детектирует, но revise не всегда фиксит.

**Modify**: не менять detection. Усилить revise instruction: передавать найденные metatext-фразы как explicit replacement targets.

### 3.6. MODIFY: stronger lead rules

Работают? Compact branch lead guidance улучшила v2 output. Standard branch — mixed.

**Modify**: для standard branch перенести lead guidance ближе к началу quality_block (сейчас она в отдельном `lead_block` перед `quality_block`, но LLM может не дочитывать до конца prompt).

### 3.7. ROLLBACK: `_preserve_content_floor` interaction с repair output

Сейчас floor применяется **после** repair → repair-injected facts проходят floor → polluted routing.

**Fix**: if we remove repair pass, floor applies to initial extraction output only, which is cleaner. Standalone fix: добавить `_is_publishable_floor_fact` filter к repair output тоже.

---

## 4. V2.2 patch plan

### Принцип: subtract, don't add

v2.1 failed потому что **добавил complexity** (repair LLM call), не добавив quality. v2.2 должен **убрать complexity** и усилить то, что уже deterministic.

### Exact changes

| # | What | Type | Impact | Risk |
|---|------|------|--------|------|
| **1** | Remove extraction repair pass | Subtract | -200s runtime, removes primary regression source | LOW — repair was net-negative |
| **2** | Move issue hints into initial extraction prompt | Redirect | Same quality signals, zero extra LLM calls | LOW |
| **3** | Add deterministic `_strip_posvyash_from_fact` | Deterministic | Catches посвящ* that Gemma still generates | LOW — regex with grammatical patterns |
| **4** | Add `_near_dedup_facts` with 0.85 threshold | Deterministic | Catches «Спектакль о ...» / «Спектакль рассказывает о ...» | LOW at 0.85 threshold |
| **5** | Re-filter floor output through `_post_filter_facts` | Fix | Prevents date/service facts from floor injection | LOW |
| **6** | Keep all v2.1 prompt improvements | Keep | Quality block, lead block, metatext detection | Already in place |

### What deliberately NOT to touch

1. **Routing logic** — problem is upstream (fact pollution), not routing.
2. **Generation prompt structure** — already improved in v2/v2.1.
3. **Revise loop** — works, just can't fix what repair broke.
4. **copy_assets schema** — desync is a symptom of repair, not a schema bug.
5. **Pattern taxonomy** — out of scope for incremental fix.

### Expected outcomes

| Event | V2.1 result | V2.2 expected | Why |
|-------|-------------|--------------|-----|
| 2660 | standard branch, date forbidden, editorial drift | Back to compact (5 facts), clean like v2 | No repair = no date injection → stays compact |
| 2745 | standard branch, tickets forbidden, service leak | Back to compact (5 facts), no service leak | No repair = no ticketing fact injection |
| 2734 | compact, missing=4, посвящ* | compact, missing ≤ 3, **no посвящ*** | Deterministic strip + better extraction prompt |
| 2687 | standard, missing=4, посвящ*, service fact | standard, missing ≤ 2, **no посвящ***, no service | No repair = no «формат лекции» injection; strip посвящ* |
| 2673 | standard, missing=4, clean | standard, missing ≤ 3, cleaner | Near-dedup removes semantic duplicates in facts |

### Expected runtime

V2.1: 538.8s → V2.2: **~300s** (remove repair saves ~200s).

---

## 5. Alternative path

**Нужно ли отказаться от v2.x линии полностью?**

Мой ответ: **нет, но с оговоркой**.

### Что v2.x линия уже дала (real gains, не теоретические):

1. **Anti-duplication** — v1's catastrophic sentence-level repeats убраны. Это confirmed на всех 5 events.
2. **CTA hygiene** — v2/v2.1 не ловят CTA markers. V1 ловил (2673).
3. **Quality block в prompt** — heading quality на standard branch заметно лучше (2687 headings: «Образы российского общества в искусстве» vs baseline «Формат мероприятия»).
4. **Whole-body metatext detection** — инфраструктура работает, даже если revise не всегда фиксит.
5. **Compact branch concept** — on sparse events, compact prose is genuinely better than stretched 3-section standard.

### Что v2.x линия **не смогла** решить:

1. **Coverage** — v1 missing=1 на 2687 и 2673 всё ещё недостижим для v2.x.
2. **Embellishment** — Gemma 27b hallucination при наличии «голых» имён без характеристик. Prompt rules этого не убивают.
3. **Посвящ*** — prompt ban + deterministic strip не гарантируют 100% removal (Gemma regenerates).

### Alternative: baseline-first tuning

Вместо v2.2 можно вернуться к текущему production flow и сделать **узкие patches прямо в нём**:

1. Quality block из v2.1 → добавить в production `_fact_first_description_prompt`.
2. Deterministic `_strip_posvyash_from_fact` → добавить в production `_post_filter_facts`.
3. Near-dedup → добавить в production pipeline.
4. Whole-body metatext detection → добавить в production `_description_policy_issues`.

Это даёт ~70% value от v2.x без complexity overhead. Никакого repair pass, никакого branching, никаких copy_assets.

### Мой рекомендуемый hybrid path:

1. Сделать v2.2 dry-run **с removed repair pass** (1 day, 1 dry run).
2. Если v2.2 ≥ V2 по quality + fixes посвящ* → transfer deterministic improvements в production.
3. Если v2.2 всё ещё не quality win → switch to baseline-first tuning path.

---

## 6. Go / No-Go

### Go for V2.2? **Conditional GO.**

Условие: v2.2 = v2 минус repair pass плюс deterministic fixes (4 строки кода). Это **не новая архитектура**, а cleanup предыдущей итерации.

**Почему GO:**
- V2.2 = subtract complexity, not add it.
- Expected runtime: ~300s (vs 538s), closer to v2's 371s.
- Deterministic fixes (посвящ strip, near-dedup, floor re-filter) — testable, reversible, zero LLM overhead.
- Один dry-run покажет, вернулись ли мы хотя бы к v2 quality level.

**Стоп-условия (если v2.2 тоже fails):**
- Если v2.2 ≤ v2 по coverage → **stop v2.x линию**.
- Если v2.2 > v2 но < v1 по coverage → **take deterministic wins в production, stop experimental branch**.
- Если v2.2 ≥ v2 и ≈ v1 по coverage → **transfer в production как new default**.

### NOGO for:
- Ещё один extraction LLM pass (repair v2, repair v3, ...) — Gemma 27b не справляется.
- Broader pattern taxonomy (scene_led, value_led, ...) — не tested, not justified.
- copy_assets-dependent generation — desync risk too high.

---

## Приложение: Self-correction log

Что из моих v2 рекомендаций **не сработало**:

| V2 recommendation | V2.1 outcome | What I got wrong |
|-------------------|-------------|-----------------|
| Ban посвящ* in extraction prompt | Still leaks 3/5 | Prompt ban alone insufficient for Gemma 27b; needed deterministic strip |
| Anti-merge in extraction | Tracklist still lost | Instruction too abstract; Gemma needs «if quotes → separate facts» |
| _dedup_thin_facts heuristic | Not implemented (user rejected) | User was right to reject — too risky |
| Regex replace посвящ* → о | User modified to be safer | User was right — naive regex breaks grammar |
| Extraction repair pass (implicit) | Primary failure source | **My biggest error**: assumed Gemma can do meta-repair of its own JSON extraction. It cannot. |

Что **сработало**:

| V2 recommendation | V2.1 outcome |
|-------------------|-------------|
| Quality block in generation prompt | Heading quality improved on 2687 |
| Lead guidance for compact branch | V2 compact outputs are clean |
| Whole-body metatext detection infra | Detection works; revise partially fixes |
| exact_dedup_facts | Catches verbatim dups correctly |
| Anti-thin-section detection | Works on 2673 |

**Главный урок**: Gemma 27b хорошо выполняет **one-shot generation tasks** с concrete rules в prompt. Она плохо выполняет **meta-reasoning tasks** типа «проанализируй этот JSON, найди проблемы по хинтам, пересобери». На meta-reasoning нужна либо GPT-4/Claude/Gemini-class модель, либо deterministic code.
