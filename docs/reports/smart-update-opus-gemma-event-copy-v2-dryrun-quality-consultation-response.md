# V2 Dry Run Quality Consultation Response

Дата: 2026-03-07

---

## 0. Честная оценка

V2 не впечатляет. Ваш review прав: v2 patch pack в текущем исполнении не даёт quality win.

Но причины — не в том, что идеи v2 плохие.
Причины — в конкретных engineering failures, которые можно починить.

Ниже — event-by-event root cause analysis и конкретные fixes.

---

## 1. Event-by-event диагностика

### 1.1. Event 2660 — Дуальность мира (sparse, compact_fact_led)

**V2 output:**
> Анна Полтавец — автор выставки. Это её дебютная работа, в которой художница использует технику барельефа на стыке живописи и скульптуры. В экспозиции отражена борьба противоположностей: мягкого и жесткого, хрупкого и вечного, внешнего и внутреннего, прекрасного и не очень.

**Что хорошо**: нет дублей, нет filler, компактно.

**Что плохо**:
1. Начинается с «Анна Полтавец — автор выставки. Это её дебютная работа» — **это метатекст**. Prompt запрещает «Это...», но Gemma вставила «Это» во второе предложение.
2. `missing=3` — хуже baseline (2). Факт «В каждой работе застыла борьба...» присутствует, но цитата обрезана.
3. Текст плоский: нет никакого hook, это просто пересказ.

**Root cause**: prompt для compact branch слишком «минималистичный». Он говорит «2-4 предложения», но не говорит **как** писать интересно.

**Fix**:

```diff
# В compact_fact_led prompt добавить:
- 2-4 предложения prose, компактно и без пустых связок.
+ 2-4 предложения prose. Начни с самой яркой детали или образа из фактов.
+ Не пиши «Автор — ...» или «Это ...» первым предложением.
+ Если в фактах есть цитата в «кавычках» — используй её как blockquote перед prose.
```

**Лучшая версия этого текста** (как reference):
```
> В каждой работе застыла борьба мягкого и жёсткого, хрупкого и вечного, прекрасного и не очень.

Дебютная выставка Анны Полтавец — барельефы на стыке живописи и скульптуры: объём, который можно читать взглядом.
```

Это 2 предложения + blockquote. Все 5 фактов покрыты. Яркий вход.

---

### 1.2. Event 2745 — Сёстры (sparse, compact_fact_led)

**V2 output:**
> Спектакль посвящен взаимоотношениям двух сестёр. В постановке исследуются темы любви, прощения, принятия и семейных связей. Камерная атмосфера спектакля создает ощущение интимности и погружает в мир воспоминаний и тайн.

**Что плохо**:
1. **`посвящ*` leak** — prompt ban не сработал.
2. **Метатекст**: «Спектакль посвящен... В постановке исследуются... Камерная атмосфера создает...» — 3 предложения с одной структурой «подлежащее + глагол».
3. `missing=5` — нулевой выигрыш. Но это не вина generation: **все 5 facts в таблице — это тавтологии одного смысла**. Извлечение вернуло 5 фактов, которые говорят одно и то же разными словами.

**Root cause №1 — extraction failure**: Посмотрите на facts:
- «Спектакль рассказывает о взаимоотношениях двух сестёр.»
- «Постановка исследует темы любви, прощения и принятия.»
- «Спектакль камерный, создающий атмосферу интимности.»
- «Спектакль затрагивает универсальные темы семейных связей.»
- «Постановка предлагает зрителям погрузиться в мир воспоминаний и тайн.»

Это **не 5 фактов, а 1 факт**, растянутый на 5 строк. Extraction LLM «распушил» источник.

**Root cause №2 — `_post_filter_facts` недостаточно агрессивно дедуплицирует**. `_drop_subsumed_facts` ищет substring inclusion, а тут overlap семантический, а не текстовый.

**Root cause №3 — `посвящ*` в самом extraction prompt**: `_experimental_extract_prompt` нигде не запрещает `посвящ*` на уровне extraction. Gemma генерирует fact «Спектакль посвящён...» → он попадает в facts_text_clean → generation prompt получает его как source of truth → Gemma генерирует «посвящён» из своего же факта.

**Fixes**:

1. **В extraction prompt добавить явный ban `посвящ*`**:
```diff
+ - В `facts_text_clean` не используй слово «посвящён/посвящена/посвящено» —
+   перефразируй: «о чём-то», «рассказывает о...», «в центре — ...».
```

2. **В `_post_filter_facts` добавить regex strip `посвящ*`**:
```python
_POSVYASH_RE = re.compile(r'(?i)\bпосвящ\w*')
def _sanitize_posvyash(fact: str) -> str:
    return _POSVYASH_RE.sub('о', fact).replace('  ', ' ').strip()
```

3. **Для semantic dedup фактов** — пока defer (требует embedding/LLM call). Но можно добавить simple heuristic:
```python
def _dedup_thin_facts(facts: list[str]) -> list[str]:
    """If all facts share >60% words, merge into 1-2 composite facts."""
    if len(facts) <= 2:
        return facts
    all_words = [set(re.findall(r'[а-яёa-z]{4,}', f.lower())) for f in facts]
    common = set.intersection(*all_words) if all_words else set()
    if len(common) / max(max(len(w) for w in all_words), 1) > 0.4:
        # All facts are near-duplicates; keep longest + shortest
        sorted_by_len = sorted(facts, key=len, reverse=True)
        return sorted_by_len[:2]
    return facts
```

---

### 1.3. Event 2734 — Концерт Гудожникова (sparse, compact_fact_led)

**V2 output:**
> Владимир Гудожников – лауреат всероссийских и международных конкурсов, включая «Янтарный соловей».
>
> Концерт Владимира Гудожникова посвящен любви Муслима Магомаева и Тамары Синявской. В программе – произведения, писавшиеся для Муслима Магомаева, а образ Музы воплощает танцовщица.

**Что хорошо**: blockquote сохранился, структура чистая, все 4 факта покрыты.

**Что плохо**:
1. **`посвящ*` leak** — «посвящен любви». Fact №1 сам содержит «посвящен» → generation копирует.
2. `missing=3` — НО. Это спорно: из 4 фактов 4 упомянуты. Missing=3 из baseline фактов, которые были шире (они включали треклист). Это extraction loss, не generation loss.
3. **Routing**: с 4 facts этот event правильно ушёл в compact. Но baseline имел 8 фактов (включая треклист), что дало бы standard branch. Значит **проблема в extraction, которая потеряла треклист**.

**Root cause**: Extraction вернула 4 факта вместо 8. Потеряны: «Лучший город земли», конкретные песни из программы, формат «вокальное исполнение», «12+» (service, ok). Факты о программе — publishable, их нельзя было терять.

**Fix**: В extraction prompt усилить preservation:
```diff
  - Если есть program-like items, сохраняй конкретику и названия.
+ - Названия песен, пьес, фильмов, треков — ОБЯЗАТЕЛЬНО сохраняй каждое отдельным фактом.
+ - Не паковай программный список в один обобщающий факт.
```

Второй fix — runtime `_preserve_content_floor` **уже есть**, но он не спасает если baseline facts тоже не содержат треклист. Нужно проверить, есть ли треклист в `raw_facts`.

---

### 1.4. Event 2687 — Лекция «Художницы» (standard, fact_first_v2)

**V2 output:**
> 📚 Лекция рассказывает о творчестве Елены Поленовой, Марии Якунчиковой-Вебер, Зинаиды Серебряковой, Натальи Гончаровой, Ольги Розановой и Любови Поповой, а также о вкладе художниц в историю русского искусства. Особое внимание будет уделено русским художницам с британскими корнями.
>
> ### Творческий путь художниц
> В центре внимания — Елена Поленова, Мария Якунчикова-Вебер, Зинаида Серебрякова, Наталья Гончарова, Ольга Розанова и Любовь Попова. Елена Поленова руководила мастерской резьбы в Абрамцеве, а Зинаида Серебрякова известна портретами близких и полотнами на крестьянские темы.
>
> ### Разнообразие техник и сюжетов
> Мария Якунчикова-Вебер работала в станковой живописи, офорте, книжной графике и декоративно-прикладном искусстве. Наталья Гончарова и Ольга Розанова экспериментировали с формой и цветом, а Любовь Попова искала новые пути в абстракции.
>
> ### Художницы с британскими корнями
> Лекция расскажет о жизни и творчестве русских художниц с британскими корнями, а также о том, как Эмилия и Мария Шанкс запечатлевали жизнь российского общества. Лекция расскажет о влиянии британского происхождения на их творчество и вклад в культурный обмен между странами.

**Что хорошо**: headings содержательные! «Разнообразие техник и сюжетов» и «Художницы с британскими корнями» — это improvement over baseline's «Формат мероприятия». Дублирования нет (кроме секции 3). Структура 3 headings — правильная.

**Что плохо**:
1. **Metatext lead**: «Лекция рассказывает о...» — prompt ban не сработал. «Лекция расскажет...» повторяется **3 раза** в тексте.
2. **Hallucination**: «Наталья Гончарова и Ольга Розанова экспериментировали с формой и цветом, а Любовь Попова искала новые пути в абстракции» — **этого нет в фактах**. Facts говорят только имена, но не говорят что Гончарова «экспериментировала с формой и цветом». Это embellishment, которое prompt должен был убить.
3. **Hallucination №2**: «о влиянии британского происхождения на их творчество и вклад в культурный обмен между странами» — в фактах сказано только «запечатлевали жизнь российского общества».
4. `missing=3` vs v1's `missing=1` — regression.
5. `Лекция расскажет...` дважды в одном абзаце (секция 3).

**Root causes**:
- Anti-metatext rule в `generation_quality_block` есть, но формулировка мягкая: «Описывай содержание, а не формат». Gemma 27b нуждается в **explicit ban** с примером, а не абстрактном guideline.
- Anti-embellishment rule не остановил hallucination. Текущая формулировка «Не достраивай смысл» — недостаточно конкретна.
- Дублирование «Лекция расскажет» — dedup guard не поймал, потому что предложения достаточно разные по содержанию (порог 60% word overlap), хотя структурно идентичны.

**Fixes**:

1. **Ужесточить anti-metatext в generation prompt**:
```diff
- Описывай содержание, а не формат. Не начинай с «Лекция расскажет о...» / «Спектакль рассказывает...» / «Это...».
+ ЗАПРЕТ: слово «лекция/спектакль/концерт/выставка» + глагол «расскажет/рассказывает/рассказывает о» —
+ запрещено в любом месте текста, не только в начале. Вместо «лекция расскажет о Х» — пиши сразу о Х.
```

2. **Ужесточить anti-embellishment**:
```diff
- Не достраивай смысл: можно перестроить фразу факта, нельзя добавить то, чего нет.
+ Не достраивай смысл. Если факт содержит только имя (напр., «Наталья Гончарова»),
+ ты НЕ можешь добавить «экспериментировала с формой и цветом» — это не в фактах.
+ Если в фактах нет деталей о конкретной персоне, упомяни только имя.
```

3. **Добавить runtime gate для «лекция расскажет»**:
```python
def _detect_metatext_verbs(description: str) -> list[str]:
    """Catch 'лекция расскажет' / 'спектакль рассказывает' anywhere, not just in lead."""
    issues = []
    pattern = re.compile(
        r'(?i)\b(лекция|спектакль|концерт|выставка|мероприятие)\s+'
        r'(расскаж\w*|рассказыва\w*|представля\w*)'
    )
    matches = pattern.findall(description)
    if len(matches) >= 1:
        issues.append(
            f'В тексте {len(matches)}× встречается «{matches[0][0]} {matches[0][1]}...». '
            'Перепиши: опиши содержание напрямую, без глагола-посредника.'
        )
    return issues
```

---

### 1.5. Event 2673 — Собакусъел (standard, fact_first_v2)

**V2 output:**
> > Проект «Собакусъел» — это социальная сеть для профессионалов креативных индустрий.
>
> Платформа «Собакусъел» задумана как пространство для поиска единомышленников, объединения и запуска совместных проектов. На презентации расскажут о задачах, устройстве и возможностях платформы, а также о причинах её появления и решаемой проблеме. Откроется предрегистрация для участников проекта. 🐶
>
> ### О платформе «Собакусъел»
> Презентация посвящена устройству социальной сети «Собакусъел». Расскажут, зачем появился проект и какую проблему он решает.
>
> ### Программа презентации
> В программе запланированы:
> - выступления артистов
> - шоу дрессированных собак
> - чтение стихов
>
> Организаторы предоставят чай, посетителям рекомендуется принести с собой печенье.
>
> ### Для профессионалов креативных индустрий
> Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

**Что хорошо**: blockquote есть, heading «Программа презентации» — содержательный, список программы сохранён, «печенье + чай» — есть.

**Что плохо**:
1. **`missing=6`** — это **катастрофическая regression** (v1=1, baseline=5). Куда делись факты?
2. **`посвящ*`**: «Презентация посвящена» в секции 1.
3. **Heading «О платформе «Собакусъел»»** — generic (ban list не поймал, потому что формат «О платформе X» не в списке, только «О событии»).
4. **Лид пересказывает blockquote**: blockquote говорит «социальная сеть для профессионалов», лид повторяет «пространство для поиска единомышленников». Одно и то же.
5. **Дублирование смыслов**: секция 1 (лид) и секция «О платформе» по сути об одном.
6. **Секция «Для профессионалов креативных индустрий»** — filler. Одно предложение, не добавляет ничего.

**Root cause `missing=6`**: 12 facts in → 6 missing. Посмотрим на факты:
- Факт №7 и №1 — дубли (оба про «социальная сеть для проф. креат. индустрий»).
- Факт №3, №10, №11 — пересекаются (все про «расскажут о задачах/устройстве/проблеме»).
- Но даже с учётом дублей, 6 missing из 12 — слишком много. Генерация просто не покрыла.

Второй root cause: **лид слишком длинный** — 3 предложения до первого heading забирают key facts, а потом секция «О платформе» дублирует тот же смысл. Бюджет потрачен впустую.

**Fixes**:

1. **Extraction**: факты №1 и №7 — exact duplicates. `_drop_subsumed_facts` должен убрать один. Проверить: он ищет substring containment, а тут verbatim dup. Нужен exact match dedup:
```python
def _exact_dedup_facts(facts: list[str]) -> list[str]:
    seen = set()
    out = []
    for f in facts:
        key = re.sub(r'\s+', ' ', f).strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
```

2. **Anti-dup в лид vs body**: prompt rule нужен:
```
- Если blockquote уже содержит суть (кто/что проект), лид должен добавлять НОВУЮ деталь, не повторять blockquote.
```

3. **Heading ban расширить** — не только «О событии», но паттерн:
```python
_WEAK_HEADING_RE = re.compile(r'(?i)^о\s+(событии|лекции|концерте|спектакле|выставке|платформе|проекте|программе)')
```

4. **Anti-filler paragraph gate**: секция «Для профессионалов...» — 1 предложение. В revise prompt добавить:
```
- Если секция под heading содержит только 1 предложение без конкретики, слей её с другой секцией или удали.
```

---

## 2. Cross-case diagnosis: 3 системные проблемы

### 2.1. `посвящ*` — lifecycle problem, а не prompt problem

Текущий подход: ban в generation → ban в revise → `_remove_posv` LLM call → regex check.

**Почему не работает**: `посвящ*` появляется **в extraction output**, попадает в `facts_text_clean` как source of truth, и generation prompt говорит «строго по фактам» → LLM копирует «посвящён» из факта.

**Fix — нужен 3-layer defence**:

| Layer | Action | Where |
|-------|--------|-------|
| **Extraction** | Ban `посвящ*` в extraction prompt | `_experimental_extract_prompt` |
| **Post-extraction** | Regex replace в `_post_filter_facts` | Before routing |
| **Generation/Revise** | Existing bans | Already there |

```python
# Add to _post_filter_facts, before dedup:
def _sanitize_posvyash_in_facts(facts: list[str]) -> list[str]:
    pattern = re.compile(r'(?i)\bпосвящ\w*\s+')
    return [pattern.sub('о ', f).replace('  ', ' ').strip() for f in facts]
```

### 2.2. Extraction aggressiveness — coverage loss root cause

| Event | Raw facts | Extracted | Missing vs baseline |
|-------|-----------|-----------|-------------------|
| 2734 | 8+ | 4 | Потерян треклист |
| 2673 | 12+ | 12 (с дублями) | 6 missing = coverage hole |
| 2745 | 5 | 5 (все тавтологии) | Content = 0 |

Extraction LLM делает две ошибки:
1. **Merger**: пакует 3 конкретных песни в 1 обобщающий факт «произведения, писавшиеся для Магомаева».
2. **Inflation**: генерирует 5 перефразировок одного смысла (2745).

**Fix**: добавить в extraction prompt explicit anti-merge / anti-inflate:
```
- Не обобщай перечень в один факт. Если в источнике 4 песни — верни 4 отдельных факта.
- Не перефразируй один и тот же смысл в несколько фактов. Один факт = одна деталь.
```

### 2.3. Metatext verbs survive all layers

«Лекция расскажет» / «Спектакль рассказывает» прошли:
- generation prompt (ban есть, но слабый);
- revise prompt (ban есть, но слабый);
- _detect_metatext_lead (только для lead, не для body!);
- final policy revise.

**Fix**: расширить detection на WHOLE body, не только lead (fix выше в §1.4).

---

## 3. Конкретный v2.1 patch set

### Prompt changes

| # | What | Where | Impact |
|---|------|-------|--------|
| P1 | Ban `посвящ*` в extraction | `_experimental_extract_prompt` | Stops `посвящ*` at source |
| P2 | Anti-merge rule в extraction | `_experimental_extract_prompt` | Preserves track lists |
| P3 | Anti-inflate rule в extraction | `_experimental_extract_prompt` | Stops semantic duplicates |
| P4 | Expand metatext ban to whole body | `generation_quality_block` | Kills «лекция расскажет» everywhere |
| P5 | Concrete anti-embellishment with example | `generation_quality_block` | Stops hallucinated characterizations |
| P6 | Compact lead hook guidance | compact prompt | «Начни с самой яркой детали» |
| P7 | Blockquote ≠ leed repeat rule | generation prompt | Stops blockquote echo in lead |
| P8 | Anti-thin-section rule в revise | revise prompt | Merges 1-sentence sections |

### Runtime changes

| # | What | LOC | Impact |
|---|------|-----|--------|
| R1 | `_sanitize_posvyash_in_facts` | ~3 | Strip `посвящ*` from facts post-extraction |
| R2 | `_exact_dedup_facts` | ~8 | Remove verbatim duplicate facts |
| R3 | `_dedup_thin_facts` heuristic | ~12 | Merge near-identical sparse facts |
| R4 | Expand `_WEAK_HEADING_RE` to pattern `^О\s+` | ~2 | Catch «О платформе X» variants |
| R5 | `_detect_metatext_verbs` for whole body | ~8 | Feed into revise loop |

### Routing changes

None needed. Routing is actually correct: sparse events → compact, dense → standard.

---

## 4. Что НЕ сломано (чтобы не overreact)

| What | Status |
|-------|--------|
| Routing sparse/standard split | ✅ Correct on all 5 |
| Heading quality on standard branch | ✅ 2687 headings are good |
| Program list preservation | ✅ 2673 list intact |
| Blockquote conditional | ✅ Only when epigraph exists |
| Anti-duplication | ✅ v1's catastrophic repeats are gone |
| CTA detection | ✅ No CTA leaks in v2 |

---

## 5. Ожидаемый результат v2.1 по каждому event

| Event | v2 problem | v2.1 fix | Expected quality |
|-------|-----------|---------|-----------------|
| 2660 | Flat metatext lead | P6 hook guidance + blockquote from fact №5 | Quote opening + 2 sentences. missing ≤ 1 |
| 2745 | Tautological facts | P3 anti-inflate + R3 thin dedup | 2 real facts → honest compact prose |
| 2734 | Lost tracklist | P2 anti-merge + R1 posv-sanitize | 7-8 facts, tracklist preserved, no `посвящ*` |
| 2687 | Metatext + hallucination | P4+P5 expanded bans + R5 body detection | No «лекция расскажет», no invented characterizations |
| 2673 | Coverage loss + dup facts | R2 exact dedup + P7 blockquote ≠ lead + P8 thin sections | missing ≤ 2, structured content |

---

## 6. Priorities

**Если делать только 3 вещи — делать эти**:

1. **P1 + R1**: `посвящ*` elimination at extraction layer — fixes 3/5 events.
2. **P2 + P3**: extraction anti-merge + anti-inflate — fixes coverage regressions on 2734, 2745.
3. **P4 + R5**: metatext verb ban for whole body — fixes 2687 tone.

Всё остальное — quality polish. Эти три — structural fixes.
