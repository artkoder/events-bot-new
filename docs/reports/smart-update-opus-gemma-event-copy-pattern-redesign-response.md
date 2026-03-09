# Opus → Gemma Event Copy: Pattern-Driven Redesign

Дата: 2026-03-07

---

## 0. Executive Summary

Текущий text flow в Smart Update — это набор **independently evolved prompts**, каждый из которых оптимизировался под свою задачу. Они работают, но:

- генерируют шаблонный output (один ритм / один lead type / одинаковые headings);
- дублируют style rules между prompts;
- не используют enriched extraction для compositional routing;
- лечат downstream (reflow, logistics removal, missing facts re-integration) то, что upstream prompt не сделал сразу.

Предлагаемый redesign строится на 3 принципах:

1. **Pattern-first**: generation prompt выбирает narrative pattern до начала текста, а не пишет по единственному шаблону.
2. **Signal-enriched extraction**: один source-backed call извлекает и `facts`, и `copy_assets`, дающие generation prompt нужные сигналы для routing.
3. **Fewer repair layers**: если upstream generation/revise уже сильнее, downstream support prompts (`reflow`, `logistics_remove`, `missing_facts_integrate`) через 2-3 фазы становятся вторичными или ненужными.

---

## 1. Pattern Library

### 1.1. Overview

| # | Pattern | Когда | Когда НЕ | Required evidence | Типичная структура | Риск шаблона | Как варьировать |
|---|---------|-------|----------|-------------------|-------------------|-------------|----------------|
| 1 | `topic_led` | Событие с ясным интеллектуальным/художественным фокусом (лекция, выставка, спектакль) | Нет dominant angle; бедный source | `core_angle` ≠ null, ≥6 facts | Lead = angle → 2–3 тематических ### | Средний | Менять порядок тем; менять, ведёт ли lead с формата или с содержания |
| 2 | `scene_led` | В source есть конкретная визуальная/чувственная деталь или атмосферная зацепка | Нет `scene_cues`; формальный/информационный контент | `scene_cues[]` ≥ 1 | Lead = микросцена → что за событие → детали | Низкий (самый нешаблонный) | Менять объект сцены (инструмент, зал, реакция, жест) |
| 3 | `program_led` | Participatory format с пошаговыми действиями или реальным перечнем программы | Performance с сетлистом; лекция со списком тем; перечень наград | `is_participatory` + (`has_stepwise_action` OR `has_hands_on_materials`) OR (`has_true_program_list` AND NOT `is_performance`) | Lead = формат + главная цель → ### с numbered/bullet list → условия | Средний | Менять объём программного списка; добавлять/убирать experience sentence |
| 4 | `value_led` | Есть сильные grounded основания ценности (why-go evidence) | Нет evidence; одни декларативные похвалы | `why_go_gate = true` (1 strong OR 2 regular) | Lead = «Зачем это интересно» через конкретный факт → ### с деталями | Высокий | Встраивать value в lead, а не в отдельную секцию; не использовать слово «уникальный» |
| 5 | `compact_fact_led` | Бедный source (≤5 facts, нет experience_signals, нет why_go) | Богатый source с множеством деталей | ≤5 `facts_text_clean` AND 0 `experience_signals` AND 0 `why_go_candidates` | Lead 1–2 предложения → 0–1 ### → минимальный body | Низкий (слишком короткий для шаблона) | Нечего варьировать, кроме порядка фактов |
| 6 | `quote_led` | Сильная прямая цитата с атрибуцией, которая открывает тему события | Нет speaker; цитата ≤8 слов; цитата = slogan/CTA; цитата не раскрывает тему | `is_valid_for_quote_led(fragment)` = true (multi-criteria gate) | `> «цитата»` + атрибуция → пояснительный абзац → ### с деталями | Низкий (rare mode) | Менять позицию пояснительного абзаца |
| 7 | `person_led` | Фигура выступающего/автора/куратора — главный смысловой центр | Когда человек лишь один из элементов; когда persona не grounded | `is_speaker_led = true` AND `credibility_signals` упоминают конкретного человека | Lead = кто + чем известен → ### что расскажет/покажет → ### контекст | Средний | Менять, начинает ли lead с имени или с проблемы, которую speaker раскрывает |

### 1.2. Anti-template rule (applies to all patterns)

Каждый pattern должен включать **variation seed** — элемент, который не позволяет двум событиям одного pattern выглядеть одинаково:

- `topic_led`: вставить первым не тот тематический блок, который «по умолчанию» идёт перед конкретикой, а самый яркий факт;
- `scene_led`: менять чувственный канал (слыш- / видимо- / тактильна-);
- `program_led`: менять, идёт ли список в начале или после 1 абзаца контекста;
- `value_led`: value statement может быть в lead или в последнем абзаце;
- `person_led`: lead может начинаться с имени ИЛИ с темы, которую speaker раскрывает.

Промпту нужно давать seed-intent: «Выбери один из следующих variation seeds и используй в этом тексте: ...».

### 1.3. Patterns we do NOT add

| Candidate | Почему нет |
|-----------|-----------|
| `chronological_led` | Линейная хронология — anti-pattern для описаний (это расписание, а не текст) |
| `comparison_led` | «Этот спектакль отличается от...» — нужен grounded referent, которого у нас нет |
| `question_led` | «Задумывались ли вы...» — risk of fake engagement, не grounded |

---

## 2. Extraction Redesign

### 2.1. Current state

`_llm_extract_candidate_facts` извлекает `facts[]` — массив atomic fact strings. Это работает, но:

- не извлекает `core_angle`, `experience_signals`, `scene_cues`, `credibility_signals`;
- не даёт generation prompt достаточно информации для informed pattern routing;
- downstream generation пишет «от общего к частному» без знания, что в этом event является самым ярким.

### 2.2. Proposed enrichment: `copy_assets` (confirmed from follow-up round)

Архитектура: **merged single-call extraction** (подтверждено в follow-up response + review).

#### Enrichment fields

| Field | Зачем | Как улучшает письмо | Риск перегруза | Required/Optional |
|-------|-------|---------------------|---------------|-------------------|
| `core_angle` (string, ≤15 слов) | Даёт generation один dominant angle для lead | Lead перестаёт быть пересказом title | Низкий | Required |
| `format_signal` (enum) | Deterministic routing feature | Позволяет выбирать pattern | Нулевой | Required |
| `subformat` (string?) | Уточнение для вариативности | «мастерская по рисованию» vs «мастерская по шоколадоварению» | Низкий | Optional |
| `program_highlights` (string[], 2-6) | Конкретные пункты программы | Generation может показать diversity of content, не перечисляя всё | Средний: модель может перечислить всё | Required |
| `experience_signals` (string[], grounded) | Что зритель реально увидит/услышит/попробует | Позволяет писать от experience, а не от абстракции | Средний: hallucination risk | Required, grounding rule |
| `scene_cues` (string[], ≤2) | Конкретные визуальные/чувственные детали из source | Делают scene_led возможным; делают lead ярче | Средний: модель может выдумать | Optional, grounding rule |
| `why_go_candidates` ({reason, strength}[]) | Основания ценности с силой (strong/regular) | Контролируемый why-go gate | Низкий | Required |
| `voice_fragments` ({text, speaker, ...}[]) | Цитаты для quote_led и inline quote | Цитата перестаёт быть случайной вставкой | Средний: хрупкое поле | Optional |
| `credibility_signals` (string[]) | Премии, headliner, уникальные форматы | Обогащает value-led и why-go | Низкий | Required |
| `contrast_or_tension` (string?) | «Не военная хроника, а...» | Даёт lead interesting angle через противопоставление | Средний: может тянуть editorializing | Optional |
| `tone_hint` (enum) | Камерное / разговорное / торжественное / ... | Mode-specific register | Низкий | Required |
| `routing_features` (6 booleans) | Deterministic signals для pattern routing | Переключает pattern без LLM | Нулевой | Required |

#### Что НЕ добавляем

| Candidate field | Почему нет |
|----------------|-----------|
| `safe_descriptors` | Слишком декоративно; не влияет на composition routing |
| `audience_value_signal` | Overlap с `why_go_candidates`; redundant |
| `named_entities_for_lead` | `core_angle` уже покрывает; дополнительное поле создаёт ambiguity |

### 2.3. Schema (final, incorporates follow-up review corrections)

Итоговая schema — та же, что предложена в follow-up response (Section 2.1), с добавлением:

```diff
 copy_assets:
+  scene_cues:          string[]  # ≤2 конкретных визуальных/чувственных деталей из source
+  contrast_or_tension: string?   # противопоставление из source, если есть (null иначе)
   core_angle:          string
   format_signal:       enum
   subformat:           string?
   program_highlights:  string[]
   experience_signals:  string[]
   why_go_candidates:   [{reason, strength}]
   voice_fragments:     [{text, speaker, speaker_role, is_direct_quote, opens_event_theme}]
   credibility_signals: string[]
   tone_hint:           enum
   routing_features:    {6 booleans}
```

### 2.4. Extraction prompt additions

К текущему prompt в `_llm_extract_candidate_facts` добавляются дополнительные instructions для `copy_assets` (см. Section 4.1 для полного prompt fragment).

Ключевые grounding rules:

```
- scene_cues: ТОЛЬКО если в source_text/raw_excerpt/poster_texts есть конкретная
  сенсорная деталь (что видно, слышно, ощущается). Не выдумывай «атмосферу».
  Если ясных деталей нет — пустой список.

- contrast_or_tension: ТОЛЬКО если в source есть явное противопоставление
  (напр. «это не про войну, а про юность»). Не конструируй контрасты.
  Если нет — null.

- experience_signals: ТОЛЬКО то, что прямо следует из source.
  «Участники создадут свою работу» ✅ (if source says so).
  «Зрители получат незабываемые впечатления» ❌ (not grounded).
```

---

## 3. Prompt Surface Matrix

### 3.1. Full matrix

| # | Function | Label | Verdict | Priority | Short reason | Relation to patterns |
|---|----------|-------|---------|----------|-------------|---------------------|
| 1 | `_llm_extract_candidate_facts` | facts_extract | **rewrite** | P0 | Extend with `copy_assets`; tighten existing rules; merge denylist items | **Gateway**: produces signals for routing |
| 2 | `_fact_first_description_prompt` | fact_first_desc | **rewrite** | P0 | Make pattern-aware; add lead variation seeds; sharpen heading palette; remove duplicated style rules that revise repeats | **Core**: must reference selected pattern |
| 3 | `_fact_first_coverage_prompt` | fact_first_cov | **tune** | P0 | Add checks: template_feel, weak_lead, weak_heading, redundancy | Quality gate after generation |
| 4 | `_fact_first_revise_prompt` | fact_first_revise | **tune** | P0 | Dedup shared rules with generation; add anti-template patch-up; reference pattern | Revise within selected pattern |
| 5 | `_fact_first_remove_posv_prompt` | fact_first_remove_posv | **keep** (transitional) | P2 | Surgical last-mile repair; keep until stronger upstream eliminates need | Unrelated to patterns |
| 6 | `_llm_integrate_missing_facts_into_description` | missing_facts | **remove-later** | P2 | With stronger generation + revise flow, missing facts should be ≤1; redundant after redesign | Superseded by coverage+revise |
| 7 | `_llm_reflow_description_paragraphs` | reflow | **remove-later** | P2 | If generation is already paragraph-disciplined, reflow is redundant; currently needed for legacy/fallback descriptions | Superseded at high water mark |
| 8 | `_llm_remove_infoblock_logistics` | remove_logistics | **tune** | P1 | Still needed for non-fact-first paths (rewrite, merge); strengthen deterministic pre-check to reduce LLM calls | Unrelated to patterns |
| 9 | `_llm_enforce_blockquote` | blockquote_enforce | **tune** | P1 | Still needed for non-fact-first paths; could become part of merge-time cleanup | Supports quote pattern |
| 10 | `_llm_shrink_description_to_budget` | shrink_desc | **keep** | P1 | Utilitarian; prompt is clean; add explicit protection for headings + lists during shrink | Unrelated to patterns |
| 11 | `_rewrite_description_journalistic` | rewrite | **tune** | P1 | Huge prompt with redundant rule overlap; extract shared rule blocks; eventually align with pattern system when fact-first covers 95%+ | Partial alignment needed |
| 12 | `_llm_create_description_facts_and_digest` | create_bundle | **rewrite** | P1 | Description sub-prompt needs pattern-awareness; facts sub-prompt should match enriched extraction; keep bundle efficiency | Must embed pattern routing |
| 13 | `_llm_match_or_create_bundle` | match_create_bundle | **tune** | P1 | Match logic stays; create sub-prompt needs same rewrite as #12; separated but consistent | Description part → pattern aware |
| 14 | `_llm_merge_event` | merge | **tune** | P1 | Heavy prompt; description sub-section needs pattern hints; factual diffing stays; extract shared rule blocks | Description part → pattern hints |
| 15 | `_llm_build_short_description` | short_description | **tune** | P2 | Works but templated; add `core_angle` as input for better focus sentence | Can use `core_angle` |
| 16 | `_llm_build_search_digest` | search_digest | **tune** | P2 | Works; align wording rules with generation; pass `core_angle` | Can use `core_angle` |

### 3.2. Shared Rule Blocks to extract

Сейчас многие rules повторяются в 3+ prompts. Нужно вынести в reusable constants:

| Block name | Content | Used in |
|-----------|---------|---------|
| `SHARED_LOGISTICS_BAN` | Нет даты/времени/адреса/города/ссылок/телефонов/цен/билетов/возраста/Пушкинская карта/афиш | facts_extract, fact_first_desc, revise, rewrite, create_bundle, merge |
| `SHARED_HALLUCINATION_BAN` | Нет нейросетевых клише, пустых обещаний, прогнозов не из источника | facts_extract, fact_first_desc, revise, rewrite, create_bundle |
| `SHARED_QUOTE_POLICY` | Прямую речь в «...» сохранять дословно, blockquote `>`, атрибуция без новых глаголов | fact_first_desc, revise, rewrite, merge, blockquote_enforce |
| `SHARED_LIST_POLICY` | Списки/треклисты не перефразировать, каждый пункт — отдельной строкой | fact_first_desc, revise, rewrite, create_bundle, merge |
| `SHARED_HEADING_PALETTE` | 2–3 `###`; без «Подробности»; ≤60 chars; информативные; stop-list | fact_first_desc, revise, create_bundle |
| `SHARED_YO_RULE` | Уже есть (`SMART_UPDATE_YO_RULE`) | All |

Это сократит total prompt volume на ~15-20% и устранит stylistic drift между prompts.

### 3.3. Prompts that heal upstream failures

| Healer prompt | What it heals | Why upstream should fix it |
|--------------|--------------|--------------------------|
| `reflow` | Wall-of-text from generation | Generation should already have paragraph discipline |
| `missing_facts_integrate` | Coverage gaps from generation | Coverage check + revise should close gaps |
| `remove_logistics` (on fact-first path) | Logistics in narrative from generation | Fact-first generation explicitly bans logistics |
| `remove_posv` | `посвящ...` leaks from generation | Should be banned in generation prompt more aggressively |

After pattern redesign, 2 из 4 (reflow, missing_facts) can be `remove-later`. Остальные 2 нужны для non-fact-first paths.

---

## 4. Rewritten Prompt Family

### 4.1. `_llm_extract_candidate_facts` → Enriched Extraction

Текущий prompt остаётся базой. Добавить в конец:

```
Дополнительно верни объект `copy_assets` (для улучшения качества текста события):

1. core_angle (string, ≤15 слов): о чём событие. Одна dominant формулировка, не перечисление.
2. format_signal (enum): спектакль | лекция | концерт | показ | мастерская | экскурсия | игра | фестиваль | встреча.
3. subformat (string | null): уточнение формата, если уместно.
4. program_highlights (string[], 2-6): самые конкретные детали программы/содержания.
5. experience_signals (string[]): что зритель/участник реально увидит/услышит/попробует. ТОЛЬКО grounded. Пустой список если неясно.
6. scene_cues (string[], ≤2): конкретные сенсорные детали (что видно/слышно/ощущается). ТОЛЬКО из source. Пустой список если нет.
7. why_go_candidates ([{reason: string, strength: "strong"|"regular"}]): grounded основания ценности. Max 3.
   - strong: самодостаточная причина (крупная премия, уникальный формат, редкий исполнитель).
   - regular: поддерживающая деталь.
   Пустой список если нет.
8. voice_fragments ([{text, speaker, speaker_role, is_direct_quote, opens_event_theme}]): прямые цитаты. Max 2. Пустой список если нет.
9. credibility_signals (string[]): премии, уникальные форматы, «впервые»/«единственный»/«ретроспектива».
10. contrast_or_tension (string | null): если в source есть contradiction/contrast («не про войну, а про юность»). null если нет.
11. tone_hint (enum): камерное | разговорное | исследовательское | семейное | торжественное | клубное | нейтральное.
12. routing_features (object):
    - is_participatory (bool): участники делают что-то руками/телом?
    - has_stepwise_action (bool): пошаговая программа?
    - has_hands_on_materials (bool): конкретные материалы/инструменты?
    - is_speaker_led (bool): фокус на конкретном спикере/авторе?
    - is_performance (bool): зрительский формат?
    - has_true_program_list (bool): реальный список пунктов программы (не награды/темы)?

ПРАВИЛА ДЛЯ copy_assets:
- Все значения traceable к фрагментам source_text / raw_excerpt / poster_texts.
- Не выдумывай.
- experience_signals и scene_cues: если данных нет — пустой список.
- contrast_or_tension: если в source нет явного противопоставления — null.
```

JSON schema расширяется (см. Section 2.3).

### 4.2. `_fact_first_description_prompt` → Pattern-Aware Generation

Ключевые изменения:

**A. Pattern-aware preamble** (заменяет текущий fixed «Стиль C»):

```
Ты пишешь Markdown-анонс события.
Выбранный narrative pattern: {pattern_name}.

PATTERN-SPECIFIC INSTRUCTIONS:
{pattern_instructions}

core_angle: {copy_assets.core_angle}
tone_hint: {copy_assets.tone_hint}
```

Где `{pattern_instructions}` — подставляемый блок по pattern:

```python
PATTERN_INSTRUCTIONS = {
    "topic_led": (
        "Начни lead с dominant angle (core_angle), а не с пересказа title.\n"
        "2-3 ### секции раскрывают разные грани темы.\n"
        "НЕ начинай lead с «{title} — это...».\n"
        "Variation seed: начни с самого яркого факта, а не с определения формата."
    ),
    "scene_led": (
        "Начни lead с конкретной микросцены из scene_cues (если дано).\n"
        "Второе предложение lead — что за событие и формат.\n"
        "НЕ придумывай новую «атмосферу»: используй ТОЛЬКО факты.\n"
        "Variation seed: выбери один чувственный канал (зрение / слух / тактильность)."
    ),
    "program_led": (
        "Начни lead с формата + главной цели.\n"
        "Обязательно дай numbered/bullet list главных пунктов программы.\n"
        "Условия участия (что взять, длительность, кол-во участников) — в отдельной ### секции.\n"
        "Variation seed: список может идти перед или после 1 абзаца контекста."
    ),
    "value_led": (
        "Начни lead с конкретного grounded факта, объясняющего ценность события.\n"
        "НЕ используй общие слова: назови конкретную премию / уникальный формат / редкого исполнителя.\n"
        "Variation seed: value statement может быть в lead или в финальном абзаце."
    ),
    "compact_fact_led": (
        "Описание 40-80 слов. Не раздувай бедный source.\n"
        "0-1 ### секция. Только суть: что, когда, формат.\n"
        "НЕ добавляй «атмосферных» расширений."
    ),
    "quote_led": (
        "Начни с blockquote лучшей цитаты: > «...»\n"
        "Добавь атрибуцию: > — {speaker} ({speaker_role})\n"
        "Следующий абзац объясняет, что за событие. НЕ пересказывай цитату.\n"
        "Variation seed: пояснительный абзац может быть 1 или 2 предложения."
    ),
    "person_led": (
        "Начни lead с имени и credibility signal: кто и чем известен.\n"
        "Затем — что покажет/расскажет/представит на этом событии.\n"
        "НЕ превращай lead в биографию.\n"
        "Variation seed: можно начать с темы, которую speaker раскрывает, а не с имени."
    ),
}
```

**B. Shared rules** (вынесены в constants, подставляются одинаково в generation и revise):

```python
SHARED_DESCRIPTION_RULES = (
    f"{SHARED_LOGISTICS_BAN}\n"
    f"{SHARED_HALLUCINATION_BAN}\n"
    f"{SHARED_QUOTE_POLICY}\n"
    f"{SHARED_LIST_POLICY}\n"
    f"{SHARED_HEADING_PALETTE}\n"
    f"{SMART_UPDATE_YO_RULE}\n"
)
```

**C. Heading stop-list** (добавить):

```
Запрещённые подзаголовки (не использовать):
- «Подробности»
- «О мероприятии»
- «Информация о событии»
- «Описание»
- «О спектакле» / «О концерте» / «О лекции» (если это единственный heading)
- «Организация и ведущий»
- «Атмосфера и общение» (как heading; атмосфера может быть в lead)

Предпочтительные heading стратегии:
- Конкретный вопрос: «Что прозвучит», «Кто на сцене», «Как устроен формат»
- Тематическая метка: «Программа вечера», «Актёрский состав», «Чем отмечен спектакль»
- Предметная: «О чём повесть», «Что попробуете», «О технике»
```

**D. Anti-template checks** (embedded in generation, not separate):

```
Самопроверка перед финальным ответом:
- Lead не начинается с «{title} — это ...»
- Lead не начинается со слова из title
- Нет двух подряд предложений, начинающихся одинаково
- Heading не повторяет lead
- Нет duplicate facts (один факт в lead + в ### секции)
```

### 4.3. `_fact_first_coverage_prompt` → Extended Coverage

Добавить к текущему `missing` / `extra`:

```json
{
  "missing": ["..."],
  "extra": ["..."],
  "template_issues": ["..."],
  "quality_flags": {
    "weak_lead": false,
    "weak_heading": false,
    "redundancy": false,
    "template_feel": false
  }
}
```

Prompt addition:

```
Дополнительно проверь:
- template_feel: описание звучит как шаблон LLM? (одинаковый ритм, lead начинается с пересказа title, generic headings)
- weak_lead: lead = пересказ title без angle/hook?
- weak_heading: есть heading из стоп-листа («Подробности», «О мероприятии», etc.)?
- redundancy: один и тот же факт повторяется в lead и в секции?

Если есть проблемы, опиши в template_issues (конкретно, что сломано).
Установи соответствующие quality_flags в true.
```

### 4.4. `_fact_first_revise_prompt` → Pattern-Aware Revise

Убрать дублированные rules (→ `SHARED_DESCRIPTION_RULES`). Добавить:

```
Pattern: {pattern_name}
core_angle: {copy_assets.core_angle}

Дополнительные исправления (если report указал):
- template_feel → переписать lead с другим angle; менять ритм предложений
- weak_lead → начать с core_angle, не с пересказа title
- weak_heading → заменить на конкретный heading из heading palette
- redundancy → убрать повторный факт (предпочтительно из body, не из lead)
```

### 4.5. `_rewrite_description_journalistic` → Partial Alignment

Этот prompt обслуживает **non-fact-first paths** (TG/VK legacy rewrite). Полный redesign не нужен сейчас, но:

- извлечь shared rules в constants (reduce duplication by ~30%);
- добавить heading stop-list;
- добавить anti-template self-check;
- если/когда fact-first path покроет 95%+ events, этот prompt станет vestigial.

### 4.6. `_llm_create_description_facts_and_digest` → Pattern-Aware Bundle

Description sub-prompt внутри bundle нужно:

- добавить `copy_assets` в payload;
- добавить pattern routing (runtime определяет pattern, передаёт в prompt);
- добавить pattern-specific instructions;
- заменить duplicated rules на shared constants.

Facts sub-prompt: если merged extraction уже даёт enriched `facts + copy_assets`, можно убрать дублирование extraction rules из bundle.

### 4.7. `_llm_merge_event` → Pattern Hints

Description sub-section в merge prompt нужно:

- передать `copy_assets` от merged event (если есть);
- добавить heading stop-list;
- добавить anti-template self-check;
- shared rules → constants.

### 4.8. `_llm_build_short_description` → Core Angle

Добавить `core_angle` в payload:

```
Данные:
{
  "title": "...",
  "event_type": "...",
  "description": "...",
  "core_angle": "..."  ← NEW
}
```

Prompt addition: «Используй `core_angle` как смысловой центр предложения, если он информативнее title.»

### 4.9. `_llm_build_search_digest` → Core Angle

Аналогично short_description: передать `core_angle`, добавить instruction использовать его как focus.

---

## 5. Routing Logic

### 5.1. Full Decision Tree

```python
def determine_pattern(copy_assets: dict, facts_text_clean: list[str]) -> str:
    """Select narrative pattern based on extracted copy_assets."""

    # 1. Quote-led (rare, highest bar)
    if should_use_quote_led(copy_assets):
        return "quote_led"

    # 2. Program-led (participatory or structured program)
    if should_use_program_led(copy_assets):
        return "program_led"

    # 3. Compact (poor source)
    if is_poor_source(copy_assets, facts_text_clean):
        return "compact_fact_led"

    # 4. Person-led (strong speaker/creator figure)
    rf = copy_assets.get("routing_features", {})
    if rf.get("is_speaker_led") and has_person_credibility(copy_assets):
        return "person_led"

    # 5. Value-led (when why-go gate passes AND value is the dominant signal)
    if should_include_why_go(copy_assets) and is_value_dominant(copy_assets):
        return "value_led"

    # 6. Scene-led (when scene cues available)
    if len(copy_assets.get("scene_cues", [])) >= 1:
        return "scene_led"

    # 7. Default: topic-led
    return "topic_led"
```

### 5.2. Helper functions

```python
def is_poor_source(copy_assets: dict, facts_text_clean: list[str]) -> bool:
    return (
        len(facts_text_clean) <= 5
        and len(copy_assets.get("experience_signals", [])) == 0
        and len(copy_assets.get("why_go_candidates", [])) == 0
        and len(copy_assets.get("scene_cues", [])) == 0
    )

def has_person_credibility(copy_assets: dict) -> bool:
    """Person is not just speaker but has specific credibility."""
    creds = copy_assets.get("credibility_signals", [])
    return any(
        re.search(r"(премия|лауреат|награда|основатель|куратор|автор книги)", c, re.I)
        for c in creds
    )

def is_value_dominant(copy_assets: dict) -> bool:
    """Value is the strongest signal, not just present."""
    candidates = copy_assets.get("why_go_candidates", [])
    strong = sum(1 for c in candidates if c.get("strength") == "strong")
    return strong >= 1 and len(copy_assets.get("program_highlights", [])) <= 3
```

### 5.3. Conflict resolution

| Конфликт | Решение |
|---------|---------|
| `quote_led` + `program_led` signals | `quote_led` wins (rarer, higher threshold) |
| `person_led` + `value_led` signals | `person_led` wins (person IS the value) |
| `scene_led` + `value_led` signals | `value_led` wins (value более конкретно) |
| `program_led` + `person_led` signals | `program_led` wins (participatory > speaker) |
| All flags false | `topic_led` (safe default) |

### 5.4. Why-go within non-value patterns

Даже если pattern ≠ `value_led`, why-go может быть встроен inline:

```python
def should_include_inline_why_go(copy_assets: dict, pattern: str) -> bool:
    if pattern == "compact_fact_led":
        return False  # too short
    if pattern == "value_led":
        return True  # already included by pattern
    return should_include_why_go(copy_assets)  # 1 strong OR 2 regular
```

Prompt получает флаг `include_why_go: true/false`. Если true, generation вставляет 1-2 grounded предложения **inline** в одну из ### секций.

---

## 6. Quality Controls

### 6.1. Controls that must remain after redesign

| Control | Where | Status |
|---------|-------|--------|
| Coverage check (missing/extra) | `_fact_first_coverage_prompt` | **Keep, extend** |
| Anti-hallucination ban | Shared constant, all prompts | **Keep** |
| Anti-logistics in narrative | Shared constant, all prompts | **Keep** |
| Heading discipline (count, quality) | Generation + coverage check | **Keep, strengthen** |
| Quote blockquote enforcement | Generation + dedicated `_llm_enforce_blockquote` | **Keep** (`enforce` for non-fact-first paths) |
| List preservation | Shared constant | **Keep** |
| `посвящ...` ban | Generation + `_fact_first_remove_posv_prompt` | **Keep both** (posv prompt as surgical backup until upstream is solid) |
| Compactness (budget) | Generation prompt + `_llm_shrink_description_to_budget` | **Keep** |

### 6.2. New controls post-redesign

| Control | Where | What |
|---------|-------|------|
| Template feel check | Coverage prompt (extended) | Flag if lead = title restatement, same-rhythm sentences, generic headings |
| Weak lead check | Coverage prompt (extended) | Flag if lead misses core_angle |
| Heading stop-list | Generation + coverage | Ban generic headings by name |
| Redundancy check | Coverage prompt (extended) | Flag if same fact in lead + body |
| Anti-template self-check | Generation prompt (embedded) | 5-point pre-output checklist |

### 6.3. Support prompts — lifecycle after redesign

| Prompt | Phase 1 (now) | Phase 2 (medium-risk) | Phase 3 (mature) |
|--------|--------------|----------------------|-----------------|
| `_llm_reflow_description_paragraphs` | Active | Active for legacy descs | **Remove** |
| `_llm_integrate_missing_facts_into_description` | Active | Monitor residual rate | **Remove** if residual < 5% |
| `_llm_remove_infoblock_logistics` | Active | Active (needed for merge/rewrite) | **Keep** (merge path needs it) |
| `_llm_enforce_blockquote` | Active | Active (needed for merge/rewrite) | **Keep** (merge path needs it) |
| `_fact_first_remove_posv_prompt` | Active | Monitor leak rate | **Remove** if leak rate < 2% |

---

## 7. Implementation Order

### Phase 1: Low-risk immediate (no extraction changes, prompt edits only)

| # | What | Effort | Impact |
|---|------|--------|--------|
| 1 | Extract `SHARED_LOGISTICS_BAN`, `SHARED_HALLUCINATION_BAN`, `SHARED_QUOTE_POLICY`, `SHARED_LIST_POLICY`, `SHARED_HEADING_PALETTE` into constants | Medium | De-duplication across 8+ prompts |
| 2 | Add heading stop-list to generation + revise | Small | Kills «Подробности», «О мероприятии» |
| 3 | Add anti-redundancy instruction to generation | Small | Kills duplicate facts in lead + body |
| 4 | Add lead variety instructions (don't start with title, don't start same as heading) | Small | Varies lead openings |
| 5 | Add compact sizing rule for ≤3 facts | Small | Prevents over-expansion of poor sources |
| 6 | Extend coverage prompt with `template_feel`, `weak_lead`, `weak_heading`, `redundancy` | Medium | Coverage check catches more issues for revise |
| 7 | Add anti-template self-check to generation prompt | Small | Pre-output quality gate |

**Expected quality lift**: moderate. Addressing most visible symptoms (generic headings, template lead, redundancy) without schema migration.

### Phase 2: Medium-risk branch (extraction enrichment + pattern routing)

| # | What | Effort | Dependency |
|---|------|--------|-----------|
| 8 | Extend extraction JSON schema with `copy_assets` | Large | Golden dataset ready |
| 9 | Implement pattern routing logic (`determine_pattern`) | Medium | #8 |
| 10 | Make generation prompt pattern-aware (PATTERN_INSTRUCTIONS dict) | Large | #8, #9 |
| 11 | Make revise prompt pattern-aware | Medium | #10 |
| 12 | Implement why-go gate (strength-based) + inline why-go | Medium | #8 |
| 13 | Pass `core_angle` to short_description + search_digest | Small | #8 |
| 14 | A/B evaluate on 20-event golden dataset | — | #8-#13 |

**Expected quality lift**: significant. Events get compositionally different descriptions based on their content profile.

### Phase 3: Refinement (after Phase 2 A/B validation)

| # | What | Effort | Condition |
|---|------|--------|-----------|
| 15 | Align `_rewrite_description_journalistic` with pattern system | Medium | If fact-first still < 90% coverage |
| 16 | Align `create_bundle` description sub-prompt with patterns | Medium | After #14 validates |
| 17 | Add `scene_cues` to extraction and scene_led | Small | After #14 validates |
| 18 | Remove `_llm_reflow_description_paragraphs` | Small | If residual need < 5% |
| 19 | Remove `_llm_integrate_missing_facts_into_description` | Small | If residual need < 5% |

### Phase 4: Research-only (not before Phase 3)

| # | What | Why later |
|---|------|----------|
| 20 | `quote_led` pattern activation | `voice_fragments` extraction quality unknown |
| 21 | Tone-adaptive generation (register varies by tone_hint) | Stylistic drift risk |
| 22 | `contrast_or_tension` as lead driver | Editorializing risk |

---

## 8. Expected Impact Summary

| Dimension | Current | After Phase 1 | After Phase 2 |
|-----------|---------|--------------|---------------|
| Lead variety | 1 type | 2-3 types | 5-6 types |
| Heading quality | Generic | Stop-list enforced | Pattern-specific |
| Compositional variety | 1 structure | 1 structure + sizing | 5+ patterns |
| Why-go | Absent | Absent | Evidence-gated |
| Template feel | High | Medium | Low |
| Redundancy | Frequent | Reduced | Minimal |
| Call count | 3 | 3 | 3 |
| Token overhead | Baseline | +~200/event | +~600-800/event |
| Prompt maintenance | 8+ duplicated rule blocks | Shared constants | Shared + pattern dict |

---

## 9. Ответ на каждый пункт brief

| Brief deliverable | Section |
|-------------------|---------|
| 1. Pattern Library | Section 1 (7 patterns + anti-template rule) |
| 2. Extraction Redesign | Section 2 (12 enrichment fields, grounding rules, merged architecture) |
| 3. Prompt Surface Matrix | Section 3 (16 surfaces × verdict/priority/reason/pattern relation) |
| 4. Rewritten Prompt Family | Section 4 (9 prompt rewrites/tunes with concrete fragments) |
| 5. Routing Logic | Section 5 (decision tree, helpers, conflict resolution, inline why-go) |
| 6. Quality Controls | Section 6 (existing controls + 5 new post-redesign controls + lifecycle) |
| 7. Implementation Order | Section 7 (4 phases, 22 items) |
