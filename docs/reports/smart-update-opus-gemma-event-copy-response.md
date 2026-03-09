# Opus → Gemma Event Copy: Synthesis & Prompt Architecture

Дата: 2026-03-06

---

## 1. Pattern Synthesis

После просмотра всех материалов — internal Telegram patterns, 53 external references (Tate, V&A, ICA Boston, Serpentine, Art Institute of Chicago), fact-first dry-run на 5 событиях и lexicon — я выделяю **12 паттернов сильного event copy**, отранжированных по применимости к вашему Russian fact-first Telegraph context.

### 12 паттернов

| # | Паттерн | Откуда виден | Применимость к fact-first |
|---|---------|-------------|--------------------------|
| 1 | **Dominant angle first** — текст знает свой единственный смысловой центр и выносит его в первую фразу | Tate, V&A, @domkitoboya | Высокая: можно извлекать `core_angle` из фактов |
| 2 | **Format-topic lead** — лид сразу называет формат и предмет: «Лекция о …», «Показ фильма …» | ICA Boston, @klassster | Высокая: уже есть в facts, но не всегда попадает в lead |
| 3 | **Scene-led hook** — первая фраза рисует конкретную микросцену или наблюдение | @kldzoo, @kozia_gorka | Средняя: работает, если в facts есть визуальная/чувственная деталь |
| 4 | **Quote-as-opener** — сильная цитата в blockquote задаёт интонацию всему тексту | @domkitoboya, Serpentine | Средняя: зависит от наличия grounded цитаты |
| 5 | **Program evidence** — конкретные пункты программы как доказательство ценности | @dom_semii, Art Institute | Высокая: списки уже хорошо извлекаются |
| 6 | **Experience promise** — описание того, что зритель увидит/услышит/поймёт | V&A talks, ICA performances | Высокая при grounding: `experience_signals` из source |
| 7 | **Credibility signal** — one-liner про премию, ретроспективность, уникальный формат | Tate (Frida), Event 2767 (награды) | Высокая: уже извлекается, но не используется compositionally |
| 8 | **Why-go as value explanation** — не CTA, а объяснение, в чём ценность | V&A career talks, ICA tours | Средняя: только при ≥2 grounded оснований |
| 9 | **Compositional variety** — разные events → разная структура (число секций, тип lead, наличие списка) | Сравнение всех 53 примеров | ★ Критично: главная слабость текущего output |
| 10 | **Paragraph air** — короткие абзацы, 1 мысль = 1 абзац | @kulturnaya_chaika, @klassster | Высокая: нужно в prompt rules |
| 11 | **Specific nouns over adjectives** — конкретные существительные вместо общих эпитетов | Tate exhibition pages | Высокая: проблема лечится на generation level |
| 12 | **Heading as micro-label** — подзаголовок отвечает на конкретный вопрос, а не дублирует «О мероприятии» | Lexicon list, ICA pages | Высокая: нужна heading palette в prompt |

---

## 2. Problem Diagnosis

### Почему текущий Gemma output шаблонен

Разбираю на основе 5 событий из dry-run v28:

| # | Проблема | Пример из dry-run | Слой лечения |
|---|---------|-------------------|-------------|
| 1 | **Единый ритм** — все 5 событий имеют одинаковую компоновку: lead + 3-4 sections | Event 2767, 2695, 2771, 2233, 2638 | Generation: mode routing |
| 2 | **Механический lead** — первое предложение пересказывает title, а не задаёт angle | 2695: «Правополушарное рисование — это творческая мастерская…» | Generation: lead template variety |
| 3 | **Headings-бухгалтерия** — «Организация и ведущий», «Информация о спектакле», «Атмосфера и общение» | 2695, 2638 | Generation: heading palette |
| 4 | **List-as-text anti-pattern** — факты нанизываются через «.» без narrative arc | 2771: «Год: 2026. Страна: Великобритания. Жанр: …» | Generation: formatting rules |
| 5 | **Why-go отсутствует даже когда уместен** | 2767: богатейший набор наград не складывается в «зачем идти» | Extraction: `credibility_signals`, `why_go_candidates` |
| 6 | **Quote не работает как compositional element** | 2233: цитата Маккоя вставлена посреди текста | Extraction: `voice_fragments`; Generation: quote-led mode |
| 7 | **Experience не извлекается** | 2695: «создадут свою неповторимую работу» — signal, но система его не видит как `experience_signal` | Extraction: enriched schema |
| 8 | **Tone однородный** | 2638 (настолка) и 2767 (спектакль о войне) звучат в одном register | Extraction: `tone_flags`; Generation: mode routing |
| 9 | **«Это история не о войне»** проникает как механический факт | 2767: антиштамп из brief, попал в facts_text_clean | Extraction: denylist на уровне facts |
| 10 | **Redundancy** — один и тот же факт повторяется в lead и в section | 2695: «правополушарное рисование» × 4 | Generation: coverage dedup rule |

### Итог по слоям

- **Extraction** лечит: 5, 6, 7, 8, 9 → через enriched schema
- **Generation** лечит: 1, 2, 3, 4, 10 → через mode routing + improved instructions
- **Оба слоя**: ни один из них в одиночку не решит проблему

---

## 3. Extraction Schema v2

### Что добавить поверх текущего fact contract

Текущая extraction уже делит факты на `facts_infoblock / facts_text_clean / facts_drop`. Предложение: **не менять** эту базу, а добавить **второй extraction pass** — лёгкий, из `facts_text_clean`, который выделяет copy-relevant signals.

Название: **copy assets extraction**.

```
copy_assets:
  core_angle:       string    # О чём событие в 1 формулировке (≤15 слов)
  format_signal:    string    # спектакль | лекция | концерт | показ | мастерская | экскурсия | игра | фестиваль | встреча
  subformat:        string?   # правополушарное рисование | настольные игры | киноклуб | etc.
  program_highlights: string[]  # 2-6 наиболее говорящих деталей программы
  experience_signals: string[]  # что зритель увидит/услышит/попробует/разберёт (ТОЛЬКО grounded)
  why_go_candidates: string[]   # фактически подтверждённые основания ценности (≤3)
  voice_fragments:  string[]    # цитата, афористичная фраза из источника (≤2)
  credibility_signals: string[] # премии, редкие форматы, заметные имена, first/only/retro
  tone_hint:        enum        # камерное | разговорное | исследовательское | семейное | торжественное | клубное | нейтральное
  structure_hint:   enum        # compact_notice | reported_preview | program_led | quote_led
```

### Правила заполнения

1. Все поля заполняются **только** из `facts_text_clean`.
2. `why_go_candidates` заполняются только если есть ≥2 конкретных основания; иначе `[]`.
3. `voice_fragments` — только прямые цитаты/фразы, уже grounded в source.
4. `structure_hint` — предложение, а не обязательство; generation может перекрыть.
5. `tone_hint` определяется по format_signal + содержанию фактов, а не по желанию «сделать красиво».

### Пример для Event 2767

```json
{
  "core_angle": "Спектакль по повести Окуджавы о юности на войне",
  "format_signal": "спектакль",
  "subformat": null,
  "program_highlights": [
    "Постановка по повести Булата Окуджавы",
    "В ролях: Георгий Сальников, Ярослав Жалнин, Александра Власова, Тимур Орагвелидзе, Алексей Боченин, Александр Хитев",
    "Длительность 1 час 30 мин без антракта"
  ],
  "experience_signals": [
    "история юности, которая продолжает мечтать",
    "разговор на универсальном языке чувств"
  ],
  "why_go_candidates": [
    "Гран-при XXVII международного фестиваля ВГИК",
    "Приз за лучший актёрский ансамбль на МКФ ВГИК",
    "Лауреат премии «Золотой лист» за лучшую женскую и мужскую роль"
  ],
  "voice_fragments": [
    "За любой войной — не карты стратегий, а миллионы мечтающих вернуться домой"
  ],
  "credibility_signals": [
    "Гран-при XXVII международного фестиваля ВГИК",
    "III международный театральный фестиваль «Твой Шанс»"
  ],
  "tone_hint": "камерное",
  "structure_hint": "reported_preview"
}
```

### Пример для Event 2638 (Codenames)

```json
{
  "core_angle": "Командная словесная игра Codenames",
  "format_signal": "игра",
  "subformat": "настольные игры",
  "program_highlights": [
    "Поле из 20-25 карточек со словами",
    "Капитаны дают подсказки для поиска агентов",
    "Партия длится 15-20 минут"
  ],
  "experience_signals": [
    "атмосфера располагает к общению и новым знакомствам"
  ],
  "why_go_candidates": [],
  "voice_fragments": [],
  "credibility_signals": [
    "одна из самых популярных игр сообщества"
  ],
  "tone_hint": "разговорное",
  "structure_hint": "compact_notice"
}
```

---

## 4. Mode Routing Policy

### 4 режима

| Режим | Когда | Lead-тип | Секций | Why-go |
|-------|-------|----------|--------|--------|
| `compact_notice` | бедный source; ≤5 facts_text_clean; нет voice/experience | format-topic | 1-2 | никогда |
| `reported_preview` | лекция, спектакль, встреча, показ; есть angle + 3+ опор | angle-driven | 2-3 | по условию |
| `program_led` | мастер-класс, семейный формат, мероприятие с чёткой программой | format-action | 2-3 | по условию |
| `quote_led` | есть `voice_fragments` длиной ≥8 слов с конкретным смыслом | epigraph | 2-3 | по условию |

### Decision tree

```
if voice_fragments.length > 0 AND voice_fragments[0].word_count >= 8:
    mode = "quote_led"
elif format_signal in ["мастерская", "экскурсия", "игра", "фестиваль"]
     OR program_highlights.length >= 4:
    mode = "program_led"
elif facts_text_clean.length <= 5 AND experience_signals.length == 0:
    mode = "compact_notice"
else:
    mode = "reported_preview"
```

### Когда включать `зачем идти`

Включается **только** при ALL условиях:

1. `why_go_candidates.length >= 2`
2. mode ≠ `compact_notice`
3. Есть хотя бы 1 `credibility_signal` или 1 `experience_signal`

Формат: не отдельная секция `### Зачем идти`, а **встроенная формулировка** внутри одной из секций, используя конструкции из Lexicon:

- «Стоит идти ради …»
- «Этот формат особенно хорош тем, что …»
- «Для зрителя это возможность увидеть / услышать …»

### Когда НЕ включать `зачем идти`

- `why_go_candidates` пусто или содержит только 1 элемент
- source очень бедный
- все кандидаты — просто переформулировки `core_angle`
- нельзя назвать конкретную причину, кроме общей похвалы

---

## 5. Prompt Drafts

### 5.1. Copy Assets Extraction Prompt

> **System**: Ты помощник для выделения copy-relevant сигналов из уже очищенного списка фактов.

```
Вот список проверенных фактов о событии (facts_text_clean):

{facts_text_clean}

Формат события: {event_title}

Выдели из этих фактов (и ТОЛЬКО из них) следующие copy assets:

1. core_angle — о чём событие в одной строгой формулировке (≤15 слов).
2. format_signal — формат: спектакль | лекция | концерт | показ | мастерская | экскурсия | игра | фестиваль | встреча.
3. subformat — уточнение формата, если применимо (например: «настольные игры», «правополушарное рисование»). Если не применимо, оставь null.
4. program_highlights — от 2 до 6 наиболее конкретных деталей программы, состава, содержания. Не общие слова, а конкретные пункты.
5. experience_signals — что именно зритель/участник увидит, услышит, попробует, разберёт. Только то, что прямо следует из фактов. Если ничего конкретного — пустой список.
6. why_go_candidates — фактически подтверждённые основания ценности (не CTA, не продажа, не общая похвала). Максимум 3. Если нет конкретных оснований — пустой список.
7. voice_fragments — прямые цитаты или афористичные фразы из источника. Максимум 2. Только если они реально есть в фактах. Если нет — пустой список.
8. credibility_signals — премии, уникальные форматы, заметные имена, «впервые», «единственный», «ретроспектива». Если нет — пустой список.
9. tone_hint — одно из: камерное | разговорное | исследовательское | семейное | торжественное | клубное | нейтральное.
10. structure_hint — одно из: compact_notice | reported_preview | program_led | quote_led.

ПРАВИЛА:
- Не придумывай ничего нового. Все значения должны быть traceable к конкретным фактам из списка.
- Если для поля нет достаточных данных, верни пустой список [] или null.
- core_angle — это не пересказ всех фактов, а ONE dominant angle.
- structure_hint — это предложение на основе наличия цитат, длины программы и богатства фактов.

Ответь строго в JSON.
```

### 5.2. Description Generation Prompt (mode-aware)

> **System**: Ты культурный журналист. Пишешь описание события для Telegraph на основе проверенных фактов.

```
Тебе нужно написать описание события для Telegraph-страницы.

ВХОДНЫЕ ДАННЫЕ:
- Название: {event_title}
- Проверенные факты (facts_text_clean): {facts_text_clean}
- Copy assets: {copy_assets_json}

РЕЖИМ ПИСЬМА: {structure_hint}

ОБЩИЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО):

1. ЗАПРЕЩЕНО упоминать в тексте: дату, время, город, площадку, адрес, цены, билеты, регистрацию, возрастные ограничения, Пушкинскую карту, телефоны, URL, афиши. Это всё будет в infoblock.
2. ЗАПРЕЩЕНО выдумывать факты, которых нет в facts_text_clean.
3. ЗАПРЕЩЕНО использовать: «уникальная возможность», «не оставит равнодушным», «незабываемая атмосфера», «настоящий праздник», «для всех желающих», «не пропустите», «ждём вас», «приглашаем», «успейте», «обещает стать», «подарит эмоции», «погрузиться в», «это не просто …, а …».
4. Списки (программа, состав, треклист) — сохранять ПОЛНОСТЬЮ, названия не перефразировать.
5. Подзаголовки (###) — конкретные и живые: «О чём этот разговор», «На чём держится постановка», «Как устроена программа», «Что услышат зрители». НЕ использовать: «Подробности», «О мероприятии», «Описание».
6. Одна мысль — один абзац. Короткие абзацы. 2-3 предложения максимум в абзаце.
7. Не повторять один и тот же факт в lead и в секции.

ПРАВИЛА ПО РЕЖИМУ:

compact_notice:
- Lead: 1-2 предложения, format + topic + 1 конкретная деталь.
- 1 секция с ### : ключевые факты, без излишней разбивки.
- Без «зачем идти».
- Общий объём: 60-120 слов.

reported_preview:
- Lead: angle-driven. Начни с core_angle, не с пересказа названия. Вырази, о чём событие и почему это интересно, за 2-3 предложения.
- 2-3 секции с ###.
- Если why_go_candidates непустые и их ≥2, встрой «зачем это может быть интересно» как 1-2 предложения ВНУТРИ одной из секций (не отдельной секцией).
- Общий объём: 120-250 слов.

program_led:
- Lead: format + action. Объясни, как устроен формат и что будет происходить.
- 2-3 секции: содержание программы, практическая сторона.
- Если why_go_candidates непустые и их ≥2, встрой ценность через «Этот формат хорош тем, что …» или подобное.
- Общий объём: 100-200 слов.

quote_led:
- Начни с blockquote цитаты из voice_fragments[0].
- После blockquote: 1 абзац, объясняющий, что это за событие и почему эта цитата задаёт тон.
- 1-2 секции с ###.
- Общий объём: 100-200 слов.

ЦИТАТЫ (если есть voice_fragments):
- Используй > blockquote для прямой цитаты.
- Не пересказывай цитату сразу после неё.
- Атрибуция короткая: «— режиссёр», «— куратор».

WHY-GO (если включается):
- Не отдельная секция, а 1-2 предложения внутри narrative.
- Используй конструкции: «Стоит идти ради …», «Для зрителя это возможность …», «Этот формат особенно хорош тем, что …».
- Каждое утверждение traceable к конкретному why_go_candidate или experience_signal.

LEAD — ВАРИАТИВНОСТЬ:
Не начинай всегда одинаково. Выбери один из вариантов:
a) «{format} о {topic}. В центре — {angle}.» — нейтральный factual
b) «В этом {format} важны не только {X}, но и {Y}.» — мягкий образный
c) Начни с конкретной детали программы, которая сразу цепляет.
d) Если quote_led: начни с blockquote.

Формат вывода: Markdown для Telegraph (### подзаголовки, > blockquote, списки через -, **bold** для акцентов).
```

### 5.3. Critic / Revise Prompt

Предлагаю **не делать отдельный LLM-call для critic**. Вместо этого — расширить существующий coverage pass.

Текущий coverage pass уже проверяет `missing / extra / forbidden_markers`. Предлагаю добавить 4 дополнительные проверки:

```
Дополнительно к coverage проверь:

1. TEMPLATE_FEEL: Первое предложение начинается с пересказа названия? Есть ли «Информация о …» / «Подробности» / «О мероприятии» в headings? Один и тот же факт повторяется ≥2 раз? → Если да, пометь template_feel = true.

2. LEAD_QUALITY: Первое предложение содержит конкретный angle или просто пересказывает title? → Если нет angle, пометь weak_lead = true.

3. HEADING_QUALITY: Проверь каждый ### heading. Если heading расплывчатый (из стоп-листа: «Подробности», «О мероприятии», «Описание», «Информация», «Организация»), пометь weak_heading = true.

4. REDUNDANCY: Есть ли факт, который дважды (почти) дословно повторяется? → Если да, пометь redundancy = true.

Если любой из флагов = true → запроси revise с указанием конкретных проблем.
```

Это добавляет **0 дополнительных LLM-вызовов** к pipeline (расширение existig coverage pass). Revise вызывается только при обнаружении проблемы — как сейчас.

---

## 6. Why-Go Gate Rule

Формальная запись:

```python
def should_include_why_go(copy_assets: dict) -> bool:
    """Решает, включать ли why-go в narrative."""
    if copy_assets["structure_hint"] == "compact_notice":
        return False
    if len(copy_assets["why_go_candidates"]) < 2:
        return False
    has_evidence = (
        len(copy_assets["credibility_signals"]) >= 1
        or len(copy_assets["experience_signals"]) >= 1
    )
    if not has_evidence:
        return False
    return True
```

Допустимые формулировки (из Lexicon):
- «Стоит идти ради …»
- «Почему это может быть интересно: …»
- «Этот формат особенно хорош тем, что …»
- «Для зрителя это возможность увидеть / услышать …»
- «Ценность этой программы в том, что …»

Why-go **не** выносится в отдельную `### Зачем идти` секцию. Встраивается в одну из существующих секций 1-2 предложениями.

---

## 7. Evaluation Rubric

### 7 критериев с весами

| Критерий | Вес | 5 (отлично) | 3 (нормально) | 1 (плохо) |
|----------|-----|-------------|---------------|-----------|
| **Groundedness** | 25% | Все факты traceable; ничего нового | 1 minor extrapolation | Invented details |
| **Event clarity** | 20% | После прочтения lead ясно, что за событие, формат, angle | Ясно после 2-го абзаца | Не ясно до конца |
| **Naturalness** | 15% | Звучит как написанный человеком текст | Местами роботизированный | Очевидно LLM output |
| **Non-template feel** | 15% | Уникальная композиция для этого события | Похоже на другие events, но есть различия | Все events написаны одинаково |
| **Journalist tone** | 10% | Нейтрально-комплиментарный, конкретный | Нейтральный, но сухой | Рекламный или канцелярит |
| **Coverage** | 10% | Все facts_text_clean отражены; ничего лишнего | 1-2 minor gaps | Серьёзные пропуски |
| **Economy** | 5% | 1 LLM call для extraction + 1 для generation | 3 calls | 5+ calls |

### A/B Protocol

1. Выбрать 20 events из production DB с разными `format_signal`:
   - 5 спектаклей/лекций → reported_preview
   - 5 мастер-классов/игр → program_led
   - 5 бедных sources → compact_notice
   - 5 с цитатами → quote_led

2. Для каждого event сгенерировать:
   - A: current pipeline (baseline)
   - B: new pipeline (extraction v2 + mode routing + new prompt)

3. 3 независимых ревьювера оценивают по 7 критериям (blind).

4. Acceptance threshold: **медианный score B ≥ 3.5** по каждому критерию; **ни один critical event** с groundedness < 3.

---

## 8. Iteration Ladder

### Phase 0: Baseline capture (1 день)

- Прогнать 20 отобранных events через текущий pipeline.
- Зафиксировать baseline scores.
- Подготовить golden dataset с ожидаемыми copy assets.

### Phase 1: Extraction v2 только (2-3 дня)

- Добавить copy assets extraction prompt.
- Прогнать на тех же 20 events.
- Проверить: правильно ли заполняются copy_assets? Нет ли hallucination?
- **Не менять generation prompt** — просто убедиться, что extraction работает.
- Criteria: extraction quality ≥ 80% agreement с golden dataset.

### Phase 2: Generation v2 + mode routing (3-5 дней)

- Подключить новый description generation prompt с mode routing.
- Прогнать на 20 events.
- Провести A/B evaluation.
- Итерировать prompt wording: lead variety, heading palette, why-go phrasing.
- Criteria: B scores ≥ baseline по всем 7 metrics; groundedness stable.

### Phase 3: Extended coverage critic (1-2 дня)

- Добавить template_feel / lead_quality / heading_quality / redundancy checks в coverage pass.
- Прогнать → проверить, что revise срабатывает на реальных проблемах, а не шумит.
- Criteria: false positive rate < 15%.

### Phase 4: Production canary (2-3 дня)

- Включить новую pipeline для 10% events в production.
- Мониторить: latency, token cost, manual quality spot-checks.
- Criteria: latency ≤ +20% от baseline; 0 groundedness incidents; qualitative approval.

### Phase 5: Full rollout

- Включить для 100%.
- Первые 7 дней — daily manual spot-checks (5 random events/day).
- Criteria: sustained quality.

---

## 9. Как это ложится на текущий runtime

### Текущий pipeline

```
source_text → LLM extract facts → bucket(infoblock/text_clean/drop)
  → LLM generate description (from text_clean)
  → coverage check (missing/extra/forbidden)
  → optional revise
  → short_description, search_digest
```

### Предлагаемый pipeline

```
source_text → LLM extract facts → bucket(infoblock/text_clean/drop)
  → LLM extract copy_assets (from text_clean)        ← +1 light call
  → determine mode (deterministic, no LLM)
  → LLM generate description (from text_clean + copy_assets + mode)
  → extended coverage check (+ template/lead/heading/redundancy)
  → optional revise
  → short_description, search_digest
```

**Дельта по бюджету:**
- +1 LLM call (copy_assets extraction), ~300-500 tokens input, ~200-300 tokens output.
- Mode selection — **чистая логика, 0 tokens**.
- Generation prompt длиннее на ~200 tokens (mode-specific rules).
- Extended coverage — embedded в existing call, +100 tokens.

**Итого:** ~+500-800 tokens per event. При Gemma pricing это пренебрежимо.

---

## 10. Ожидаемый результат по событиям из dry-run

### Event 2767 (спектакль «Будь здоров, школяр!»)

**Текущий output** (проблемы): lead пересказывает title; «Это история не о войне» попало как механический факт; награды перечислены без compositional value; heading «Информация о спектакле» расплывчатый.

**Ожидаемый output** (reported_preview):

> Постановка по повести Булата Окуджавы — о юности, которая продолжает мечтать, даже оказавшись на фронте. В центре — не война как таковая, а человеческие чувства: мечты, любовь, способность оставаться живым внутри.
>
> ### Актёрский состав
> В ролях: Георгий Сальников, Ярослав Жалнин, Александра Власова, Тимур Орагвелидзе, Алексей Боченин, Александр Хитев.
>
> ### Чем отмечен спектакль
> Спектакль получил Гран-при XXVII международного фестиваля ВГИК, приз за лучший актёрский ансамбль на МКФ ВГИК, а также премии «Золотой лист» за лучшую женскую и мужскую роли. Стоит идти ради ансамбля, отмеченного именно за совместную игру.
>
> Длительность — 1 час 30 минут без антракта.

### Event 2638 (Codenames)

**Текущий output** (проблемы): 4 секции для простой игры; heading «Атмосфера и общение» — отдельная секция из 1 предложения.

**Ожидаемый output** (compact_notice):

> Codenames — командная словесная игра на поле из 20-25 карточек. Игроки делятся на две команды, капитаны по очереди дают подсказки, чтобы найти своих агентов среди слов.
>
> ### Как устроена игра
> Правила объяснят перед началом. Партия длится 15-20 минут, от 4 до 10 участников. Мастер игры — Тимур.

### Event 2233 (EURODANCE'90)

**Ожидаемый output** (quote_led):

> > «Россия стала для меня вторым домом! Я всегда рад быть здесь!»
> > — Кевин Маккой
>
> «EURODANCE'90» — концерт с Natasha Wright из La Bouche и Kevin McCoy из Bad Boys Blue. Вечер, собранный вокруг мировой музыки 90-х.
>
> ### Что прозвучит
> Natasha Wright исполнит «Be My Lover», «Sweet Dreams» и «Tonight is the Night». Kevin McCoy — «You're A Woman», «Pretty Young Girl» и «Come Back And Stay».

---

## 11. Резюме

| Что | Решение | Бюджет |
|-----|---------|--------|
| Недоизвлекаем copy signals | +1 light extraction call (copy_assets) | +500 tokens |
| Один register для всех | 4 mode routing (deterministic) | 0 tokens |
| Шаблонные leads | Varied lead templates в prompt | 0 tokens |
| Бухгалтерские headings | Heading palette + стоп-лист | 0 tokens |
| Why-go как ритуал | Evidence-gated, inline, не отдельная секция | 0 tokens |
| Цитата-вставка | Quote-led mode, blockquote first | 0 tokens |
| Повторы | Extended coverage check | +100 tokens |
| Качество не падает | A/B evaluation + 5-phase rollout | Process, not tokens |
