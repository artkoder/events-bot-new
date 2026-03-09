# Opus → Gemma Event Copy: Quality Consultation Response

Дата: 2026-03-07

> Эта консультация строится не на концепциях, а на 5 реальных side-by-side outputs.
> Каждый verdict — по прочитанному тексту, не по метрикам.

---

## 1. Event-by-event review

### Event 2660 — Дуальность этого мира (выставка, sparse: 3 → 5 facts)

**Verdict: Baseline лучше. Pattern prototype ухудшил текст.**

| Dimension | Baseline | Prototype |
|-----------|----------|-----------|
| Coverage | Missing 2 — приемлемо для 3 facts | Missing 3 — хуже |
| Opening | ✅ Blockquote `> Анна Полтавец — автор выставки` — сухо, но structural | ❌ Flat prose `Анна Полтавец — автор выставки. Это её дебютная выставка.` — два предложения, zero rhythm |
| Prose quality | Средне: «размышления о хрупкости и прочности, о границе между реальностью и иллюзией» — editorialized, но не ужасно | Плохо: «Работы Анны Полтавец создают ощущение, что образы выходят за пределы плоскости, напоминая о противоречивой природе окружающего мира» — тяжёлая фраза, пустой смысл |
| Structure | 3 секции (Техника, Основная идея, Первый проект). Слишком дробно для 3 facts, но headings осмысленные | 1 секция «О событии» → generic heading, проваленная структура |
| Embellishment | «результат её многолетних поисков и экспериментов» — ungrounded. Откуда «многолетних»? | «напоминая о противоречивой природе окружающего мира» — purple prose not from facts |

**Root cause**: sparse event (3 facts) не выиграл от pattern-driven approach. `value_led` forced a «why-go» angle, which doesn't work when facts are minimalistic. Baseline at least structured the content into readable sections, though it over-expanded (3 headings for 3 facts → too many).

**Better approach**: `compact_fact_led` — 40-80 слов prose without headings. This event is too sparse for any structural pattern.

**Качество текста**: оба текста mediocre. Baseline лучше структурирован, но оба имеют embellishment problem. Ни один не читается как профессиональный культурный анонс — больше как студенческий пересказ.

---

### Event 2745 — Сёстры (спектакль, sparse: 5 facts identical)

**Verdict: Baseline заметно лучше. Prototype — regression.**

| Dimension | Baseline | Prototype |
|-----------|----------|-----------|
| Coverage | Missing 5 (deterministic metric). Но reading shows all 5 facts ARE present — metric ложно-положительный. | Missing 5. Same metric issue. But text shorter and structurally worse. |
| Opening | ✅ «Спектакль «Сёстры» рассказывает о взаимоотношениях двух сестёр, исследуя темы...» — professional, direct | ❌ Почти тот же текст, но без heading structure → wall of text |
| Prose quality | List «Любовь и её проявления / Прощение как путь к освобождению / Принятие себя и других» — хоть и слегка press-release, но читаемо | Один абзац без structure, скомканно |
| Structure | 2 headings + list → визуально ясно на Telegraph | Zero headings → flat paragraph on Telegraph |
| Grounding | «Спектакль обещает стать глубоким и эмоциональным опытом» — CTA / promo tone, ungrounded | Чище, нет прямых CTA |

**Root cause**: 5 facts, все тематически одинаковые (все про «темы» и «атмосферу»). Это не dense event, это semi-sparse event с repetitive facts. Prototype не добавил ничего нового, потому что copy_assets просто повторили те же facts в другом формате. value_led пытался найти «why-go», а единственный why-go = пересказ тех же тем.

**Better approach**: `compact_fact_led`. 5 фактов про одно и то же → short cohesive paragraph, без headings. Не пытаться растянуть.

**Quality issue**: baseline тоже далёк от идеала. «Спектакль обещает стать глубоким и эмоциональным опытом» — это promo. «заставляющим задуматься о ценности родственных связей» — CTA-adjacent. Но структурно baseline readable, а prototype — нет.

**Критическое замечание по facts quality**: все 5 facts — это абстрактные описания тем, а не конкретные факты ('Спектакль рассказывает о...', 'Постановка исследует темы...'). Это meta-facts, а не content-facts. Текст из таких фактов неизбежно будет generic, потому что input generic. Проблема не в generation, а в **extraction**: из source posts извлечены не конкретные детали (кто играет, камерный зал, формат, длительность), а пересказы пересказов. Для sparse events это critical bottleneck.

---

### Event 2734 — Концерт Гудожникова (concert, medium: 8 → 4 facts)

**Verdict: Mixed. Baseline лучше по structure и полноте, prototype лучше по hygiene.**

| Dimension | Baseline | Prototype |
|-----------|----------|-----------|
| Coverage | Missing 5. Но baseline включает 8 facts in text (track list, Муза, credibility). | Missing 3. Но prototype имеет только 4 facts → extraction потеряла 4 факта! |
| Opening | ✅ Blockquote → professional |  ❌ Flat sentence, no blockquote |
| Track list | ✅ «Верни мне музыку / Лучший город земли / Королева красоты / Разговор со счастьем» — полный, list format | ⚠️ 3 из 4 tracks, пропущен «Лучший город земли» |
| Prose | «музыкальное повествование о великой любви» — чуть press-release, но acceptable | «в центре внимания великой любви» — broken syntax, not a valid Russian sentence |
| Hygiene | ❌ «Возрастное ограничение: 12+. Продолжительность не уточняется.» — service leakage | ✅ Чисто, без service leakage |
| Credibility | Blockquote: «лауреат... включая Янтарный соловей» | Добавлены «выпускник СПб консерватории, обладатель редкого тенора» — from copy_assets, not from facts_text_clean → editorialized |

**Root cause**: extraction prototype слишком агрессивно: из 8 baseline facts оставил 4. Потерялся «Лучший город земли», возрастное ограничение, пометка про продолжительность. Hygiene win (убран возраст) перечёркивается потерей track list полноты.

**Grammar issue**: «Концерт Владимира Гудожникова в центре внимания великой любви Муслима Магомаева» — это не грамматически корректное предложение. «в центре внимания» требует топика, а не события. Правильно: «В центре программы — великая любовь...» или «Концерт ... посвящён великой любви...». Это серьёзный quality defect в prototype.

**Better approach**: baseline structure + prototype hygiene. Keep 8 facts. Remove service leakage through cleanup, not through extraction cut.

---

### Event 2687 — Лекция «Художницы» (лекция, medium-dense: 9 → 7 facts)

**Verdict: Baseline значительно лучше. Prototype — catastrophic regression по quality.**

| Dimension | Baseline | Prototype |
|-----------|----------|-----------|
| Coverage | Missing 5 (deterministic). But reading: all 9 facts present in baseline text. | Missing 1. But... |
| Prose quality | ✅ Лид ёмкий, список художниц отформатирован | ❌ **Катастрофические повторы**: одни и те же фразы повторяются 2-3 раза дословно |
| Structure | 3 headings, logical flow | 3 headings, but content duplicated across sections |
| Readability | Хорошо | Нечитабельно |

**Prototype text verbatim problems** (цитирую):

1. «Лекция сосредоточена на творчестве Елены Поленовой...» — повторено в paragraph 2
2. «Лекция расскажет о вкладе художниц в историю русского искусства» — повторено **3 раза**: lead, body, секция «Творчество художниц»
3. «Эмилия и Мария Шанкс запечатлевали жизнь российского общества» — повторено 2 раза
4. «Лекция посвящена жизни и творчеству русских художниц с британскими корнями» — повторено 2 раза
5. «Елена Поленова руководила мастерской резьбы в Абрамцеве» — повторено 2 раза

Это **production-blocking quality failure**. Missing = 1 выглядит хорошо, но текст unreadable. Coverage metric misleading: все факты «присутствуют», потому что каждый повторён 2-3 раза.

**Root cause**: prototype generation + revise loop. Вероятно, coverage check нашёл «пропуски» и revise loop вбил все факты ещё раз, не заметив что они уже есть. Revise loop не проверяет dedupe — только adds.

**Baseline** при этом отлично справился: список художниц в list format, каждая с 1 detail, «Русские художницы с британскими корнями» как отдельная секция. Чисто, читабельно, professional. Единственный weak spot baseline: «Формат мероприятия — лекция. Она позволит глубже погрузиться в тему» — generic + пустая фраза.

---

### Event 2673 — Собакусъел (presentation, dense: 11 → 7 facts)

**Verdict: Baseline лучше, но ненамного. Prototype показывает obещание, но ещё broken.**

| Dimension | Baseline | Prototype |
|-----------|----------|-----------|
| Coverage | Missing 5 (det.). Reading: 10 of 11 facts present (пропущена «предрегистрация»). | Missing 1. But text has duplicates and CTA leak. |
| Opening | ✅ Blockquote → lead → clean | ❌ «приглашает» — CTA word banned |
| Structure | 3 headings, each with content | 1 heading «О событии» — generic + understructured |
| Prose quality | Good: «это пространство для поиска единомышленников...» → smooth transition from blockquote | Mixed: «Рекомендуем принести с собой печенье» → immediate repeat «Посетителям рекомендуется принести с собой печенье» |

**Prototype duplication**: «Посетителям рекомендуется принести с собой печенье» → said twice in consecutive sentences. Same with «расскажут о задачах, устройстве и возможностях платформы» — repeated almost verbatim.

**CTA leak**: «приглашает на презентацию» — banned word «приглашает». Baseline avoided it.

**Baseline strength**: blockquote opening with project definition → clear lead → 3 well-titled sections (Программа мероприятия / Для кого интересно / Запуск проекта). This is the closest to target quality in the entire set.

---

## 2. Cross-case quality diagnosis

### Recurring strengths of prototype

1. **Coverage number improvement** on 2687 and 2673 (5→1 missing).
2. **Service leakage elimination** on 2734 (removed «Условия посещения» section).
3. **copy_assets extraction** produces useful metadata (core_angle, program_highlights, why_go).

### Recurring weaknesses of prototype

| # | Problem | Frequency | Severity |
|---|---------|-----------|----------|
| 1 | **Verbatim duplication** within text (same fact repeated 2-3x) | 3/5 events (2687, 2673, partially 2734) | **Critical** |
| 2 | **Broken grammar** (ungrammatical sentences) | 1/5 (2734: «в центре внимания великой любви») | High |
| 3 | **CTA leak** despite explicit ban | 1/5 (2673: «приглашает») | High |
| 4 | **Lost blockquote opening** | 4/5 events (only 2734 had a line that could serve as one, but no `>` markup) | Medium |
| 5 | **Generic heading «О событии»** | 3/5 events | Medium |
| 6 | **Over-aggressive extraction** (fewer facts than baseline) | 2/5 (2734: 8→4, 2660: 3→5 but copy_assets editorialized) | High |

### Where baseline consistently wins

1. **Blockquote opening** — baseline has it in 3/5 events, prototype in 0/5.
2. **Heading quality** — baseline headings are specific («Репертуар», «Программа мероприятия», «Творческий путь художниц»). Prototype falls to generic «О событии» / «О концерте».
3. **No duplication** — baseline never repeats a fact twice. Prototype does it severely.
4. **Telegraph readability** — baseline texts look better on Telegraph due to headings + structure.

### Where prototype shows genuine promise

1. **Coverage improvement** — when it works (2687: 5→1), it works dramatically.
2. **Hygiene** — cleaner on service leakage (2734).
3. **Metadata enrichment** — copy_assets contain genuinely useful signals that could improve text if used properly.

---

## 3. Pipeline failure map

### 3.1. Facts extraction / filtering

**Verdict: TUNE**

| What works | What breaks quality |
|-----------|-------------------|
| Baseline `_facts_text_clean_from_facts` pipeline is solid | Experimental extraction sometimes loses facts (2734: 8→4) |
| Bucket classification correctly separates text_clean vs infoblock | Sparse events get meta-facts instead of concrete details (2745: all 5 facts are abstract) |
| Anchor filtering works | — |

**Action**: Don't change baseline extraction. Prototype extraction is too aggressive — it must preserve at minimum the same facts as baseline. If enrichment adds new facts, fine. But it must never remove facts that baseline had.

### 3.2. copy_assets extraction

**Verdict: TUNE (promising but noisy)**

| What works | What breaks quality |
|-----------|-------------------|
| `core_angle` produces useful anchor | `format_signal` sometimes wrong (2660: «встреча» instead of «выставка») |
| `program_highlights` useful for structured events | `experience_signals` often editorialized or ungrounded |
| `why_go_candidates` decent | `credibility_signals` sometimes pulled from nowhere (2734: «выпускник СПб консерватории» — not in facts_text_clean) |

**Action**: keep copy_assets extraction but constrain `credibility_signals` and `experience_signals` to verbatim or near-verbatim from source. `format_signal` should be derived from `event_type` field, not from LLM.

### 3.3. Routing

**Verdict: REWRITE routing rules**

| What works | What breaks quality |
|-----------|-------------------|
| `program_led` routing on 2687 and 2673 correct | `value_led` on 2660 and 2745 wrong — sparse events shouldn't get value_led |
| — | Routing never selects `compact_fact_led` for sparse | 

**Root problem**: routing defaults to `value_led` whenever there are `why_go_candidates`. But `why_go` exists for almost every event (any event can have a «reason to go»). Result: `value_led` becomes default, which defeats the purpose of having multiple patterns.

**Action**: routing must use `len(facts_text_clean)` as primary signal. ≤5 → `compact_fact_led`, not value_led. Program list present → `program_led`. Only when why_go is genuinely strong AND facts >6 → `value_led`.

### 3.4. Generation prompt

**Verdict: TUNE (major)**

| What works | What breaks quality |
|-----------|-------------------|
| Fact-driven content follows facts | **Does not deduplicate** — same fact repeated 2-3x in output |
| Pattern-aware structure (when correct) | Pattern instructions compete with P0 completeness → tries to include everything everywhere |
| — | Lost blockquote/epigraph logic from baseline prompt |
| — | Generic headings produced when pattern instructions unclear |

**Root problem #1**: generation prompt doesn't include anti-duplication rule. Gemma, when given a list of facts + copy_assets + instructions, often weaves the same fact into multiple places without realizing.

**Root problem #2**: blockquote opening from baseline (`_pick_epigraph_fact`) is NOT transferred to prototype. This is a direct quality loss.

**Action**:
1. Add explicit anti-duplication rule: «Каждый факт упоминается ровно один раз. Не повторяй одну и ту же деталь в разных местах текста.»
2. Restore epigraph/blockquote opening from baseline as cross-pattern rule.
3. Add heading quality rule: «Заголовки должны быть конкретными. Не используй: О событии, О концерте, О лекции — используй содержательные заголовки.»

### 3.5. Revise / repair

**Verdict: REWRITE revise loop**

| What works | What breaks quality |
|-----------|-------------------|
| Missing-fact detection finds real gaps | Revise adds facts without checking if they're already present → creates duplicates |
| — | No deduplicate check in revise |

**Root problem**: revise/repair loop is the primary cause of duplication. It finds «missing» facts using lexical match, inserts them, but doesn't check if the same fact is already present in paraphrased form. Event 2687 is the clearest example: all facts present, but repair added them again → 2-3x repetition of each.

**Action**: After revise/repair, add deterministic dedup check. If N-gram overlap between two sentences exceeds 60%, flag and remove.

### 3.6. Cleanup / hygiene

**Verdict: KEEP baseline, extend**

| What works | What breaks quality |
|-----------|-------------------|
| `_cleanup_description` solid | Doesn't catch within-text duplication |
| `_fact_first_remove_posv_prompt` works | Doesn't catch CTA leaks in prototype output |
| Forbidden marker detection works | — |

**Action**: Add deterministic duplication detection to cleanup layer: paragraph-level n-gram check. If >60% overlap between two paragraphs → remove the second.

### 3.7. Evaluation methodology

**Verdict: TUNE**

| What works | What breaks quality |
|-----------|-------------------|
| Deterministic missing/forbidden provides signal | Missing metric is too lexical: marks facts as «missing» when they're paraphrased |
| Side-by-side reporting format good | No quality metric for duplication, grammar, heading quality |

**Action**:
1. Add duplication metric: count of unique sentences with >60% N-gram overlap with another sentence.
2. Add CTA detection: check for banned CTA words in output.
3. Add heading quality signal: «О событии» / «О лекции» = weak heading flag.
4. Don't remove deterministic missing metric — it's useful as a signal, just not as a definitive verdict.

---

## 4. Prioritized improvements

### P0 — необходимы до любого production use

| # | Improvement | Expected impact | Risk | Ownership |
|---|------------|----------------|------|-----------|
| 1 | **Anti-duplication rule in generation prompt** | Eliminates 2-3x fact repetition (3/5 events affected) | Low | Prompt |
| 2 | **Anti-duplication check in revise/repair loop** | Prevents repair from reinserting already-present facts | Low | Runtime |
| 3 | **Restore blockquote/epigraph opening** from baseline for all patterns | Recovers strongest visual element lost in prototype | Low | Prompt + `_pick_epigraph_fact` |
| 4 | **Fix routing: ≤5 facts → compact_fact_led** | Prevents sparse events from getting bloated pattern treatment | Low | Runtime routing |
| 5 | **Don't reduce fact count below baseline** | Prototype extraction must ≥ baseline facts, not fewer | Low | Extraction |

### P1 — важные, реализовать после P0

| # | Improvement | Expected impact | Risk | Ownership |
|---|------------|----------------|------|-----------|
| 6 | **Ban generic headings** in prompt: «О событии / О лекции / О концерте» → must be specific | Better Telegraph readability | Low | Prompt |
| 7 | **Derive format_signal from event_type**, not from LLM | Fixes wrong format (2660: «встреча» vs «выставка») | Low | Runtime |
| 8 | **Constrain credibility_signals to grounded content** | Prevents editorialized additions (2734: «выпускник СПб консерватории» not in facts) | Medium | Extraction prompt |
| 9 | **Add deterministic paragraph dedup to cleanup** | Last-mile safety for duplication | Low | Runtime cleanup |
| 10 | **Add CTA detection to forbidden marker check** | Catches «приглашает» in prototype output | Low | Runtime |

### P2 — позже / экспериментально

| # | Improvement | Expected impact | Risk | Ownership |
|---|------------|----------------|------|-----------|
| 11 | **Richer extraction for sparse events** | Sparse events get concrete details, not meta-facts | Medium — may require source text access | Extraction |
| 12 | **evidence_span for scene_cues** | Stronger traceability | Medium — Gemma compliance uncertain | Extraction |
| 13 | **Merge value_led + topic_led** into single pattern | Simplifies routing without quality loss | Low | Routing + prompt |
| 14 | **Reduce pattern set from 6 to 4** | Less routing complexity, fewer edge cases | Medium | Architecture |

---

## 5. Concrete next iteration

### What to do in prototype v2

1. **Add anti-duplication rule** to generation prompt: «Каждый факт упоминается ровно один раз в тексте. Не повторяй.»
2. **Add anti-duplication check** after revise step: if a sentence is >60% N-gram overlap with another → flag for removal.
3. **Restore epigraph/blockquote** from baseline as cross-pattern feature.
4. **Fix routing**: ≤5 facts → `compact_fact_led`; don't allow `value_led` on sparse events.
5. **Don't cut facts**: prototype extraction must retain ≥ baseline fact count.
6. **Ban «О событии» heading** in prompt.
7. **Derive format_signal** from `event_type` field directly.

### What not to touch

- `_facts_text_clean_from_facts` — works, don't change.
- `_sanitize_fact_text_clean_for_prompt` — works.
- `_cleanup_description` — works. Only extend (add dedup check).
- `_find_missing_facts_in_description` — works. Keep as safety net.
- Budget formula — works.

### What to remove/simplify

- **`value_led` as default fallback** — replace with `compact_fact_led` for sparse events.
- **`experience_signals`** from copy_assets — mostly editorialized, not useful for Gemma generation.
- **`credibility_signals`** that aren't in facts — they leak ungrounded info into text.

### Need another dry-run before implementation?

**Yes. One more, focused dry-run with v2 fixes.** Rationale:

1. The duplication problem is critical and must be verified as fixed before code integration.
2. Routing fix (sparse → compact) needs testing on the same 5 events.
3. Blockquote restoration needs visual verification.

**Scope**: same 5 events, v2 prototype with P0 fixes applied. If v2 shows no duplication, correct routing, and blockquote recovery → ready for code integration.

### Can we transfer a subset to production code now?

**Not yet.** The duplication problem (3/5 events) is production-blocking. Once v2 fixes it, the transfer-ready subset is:

1. Enhanced extraction (copy_assets) — yes, if fact count preserved.
2. `compact_fact_led` for sparse events — yes.
3. Blockquote opening as cross-pattern rule — yes.
4. Improved routing logic — yes, after v2 dry-run.
5. Full pattern-driven generation for all events — not yet, need v2 validation.

---

## 6. Bottom line

**Prototype v1 is not production-ready, but the direction has clear value.**

Самые сильные baseline элементы, которые prototype потерял:
- Blockquote opening
- Heading quality
- No duplication

Самые сильные prototype элементы, которые стоит перенести:
- Coverage improvement (2687, 2673: 5→1 missing)
- Service leakage cleanup (2734)
- copy_assets metadata (core_angle, program_highlights)

**Critical blocker**: duplication. Fix it first, everything else follows.
