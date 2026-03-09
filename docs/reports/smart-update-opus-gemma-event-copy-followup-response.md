# Opus → Gemma Event Copy: Follow-up Response (Revised Architecture)

Дата: 2026-03-07

---

## 0. Top-line

Все 4 возражения приняты. Ниже — пересобранная architecture, revised schema, stronger gates и пересмотренные prompt fragments.

Главные изменения по сравнению с первым ответом:

1. **copy_assets extraction перенесена в первый source-backed call** — единый call возвращает `facts + copy_assets`. Вариант «два call, второй по голому `facts_text_clean`» отвергнут.
2. **why-go gate** — `1 strong OR 2 regular`, а не фиксированное `≥2`.
3. **quote_led gate** — multi-criteria (speaker attribution + semantic density + denylist), а не `≥8 слов`.
4. **program_led routing** — по semantic features (participatory, stepwise_action, hands_on_materials), а не по raw count `program_highlights ≥ 4`.

---

## 1. Revised Minimal-Call Architecture

### Отвергнутый вариант (из первого ответа)

```
source → call₁: extract facts → bucket
       → call₂: extract copy_assets from facts_text_clean only  ← LOSSY
       → deterministic mode routing
       → call₃: generate description
       → coverage/revise
```

Проблема: если `call₁` не сохранил голос, цитату, experience detail или тональный сигнал в `facts_text_clean`, `call₂` уже не восстановит их. Потери необратимы.

### Принятый вариант: Merged Single-Call Extraction

```
source → call₁: extract facts + copy_assets (from source_text + raw_excerpt + poster_texts)
       → runtime bucket (infoblock / text_clean / drop)
       → runtime attach copy_assets to event context
       → deterministic mode routing (0 tokens)
       → call₂: generate description (from text_clean + copy_assets + mode)
       → extended coverage/revise (embedded in existing call)
```

Почему это лучше:

- **Нет потерь**: copy_assets извлекаются из полного source payload, а не из уже отфильтрованного `facts_text_clean`.
- **1 call вместо 2**: merged extraction дешевле, чем отдельный second pass.
- **Source-grounded**: `voice_fragments`, `experience_signals`, `tone_hint` видят сырой текст + poster OCR.
- **Backward-compatible**: `facts[]` в JSON-ответе остаются тем же самым; bucketization не меняется. `copy_assets` — это дополнительный блок в том же JSON.

Потенциальный downside:

- Prompt длиннее → output длиннее → чуть дороже.
- Но: сейчас extraction prompt уже ~1000 tokens; добавление `copy_assets` schema добавит ~300 tokens к prompt и ~200-400 tokens к output. Это пренебрежимо по сравнению с альтернативой (целый второй call).

### Бюджет: финальный

| Шаг | Calls | Delta vs current |
|-----|-------|-----------------|
| Extraction (facts + copy_assets) | 1 | +0 calls, +~300-500 tokens |
| Mode routing | 0 (deterministic) | +0 |
| Description generation | 1 | +~200 tokens (mode-specific rules in prompt) |
| Coverage/revise | 1 (existing) | +~100 tokens (extended checks) |
| **Total** | **3** | **+0 calls, +~600-800 tokens per event** |

---

## 2. Revised Schema

### 2.1. Merged extraction JSON schema

```json
{
  "type": "object",
  "properties": {
    "facts": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Atomic facts about the event (6-18 items)"
    },
    "copy_assets": {
      "type": "object",
      "properties": {
        "core_angle": {
          "type": "string",
          "description": "О чём событие в ≤15 словах. Один dominant angle, не пересказ всех фактов."
        },
        "format_signal": {
          "type": "string",
          "enum": [
            "спектакль", "лекция", "концерт", "показ",
            "мастерская", "экскурсия", "игра", "фестиваль", "встреча"
          ]
        },
        "subformat": {
          "type": ["string", "null"],
          "description": "Уточнение формата, если применимо. Null если нет."
        },
        "program_highlights": {
          "type": "array",
          "items": { "type": "string" },
          "description": "2-6 наиболее конкретных деталей программы, состава, содержания"
        },
        "experience_signals": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Что зритель/участник увидит, услышит, попробует, разберёт (ONLY grounded in source)"
        },
        "why_go_candidates": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "reason": { "type": "string" },
              "strength": {
                "type": "string",
                "enum": ["strong", "regular"]
              }
            },
            "required": ["reason", "strength"]
          },
          "description": "Фактически подтверждённые основания ценности. Max 3. Strong = single obvious reason. Regular = supporting evidence."
        },
        "voice_fragments": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "text": { "type": "string" },
              "speaker": { "type": ["string", "null"] },
              "speaker_role": { "type": ["string", "null"] },
              "is_direct_quote": { "type": "boolean" },
              "opens_event_theme": { "type": "boolean" }
            },
            "required": ["text", "speaker", "speaker_role", "is_direct_quote", "opens_event_theme"]
          },
          "description": "Прямые цитаты или авторские формулировки из source. Max 2."
        },
        "credibility_signals": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Премии, уникальные форматы, заметные имена, first/only/retro"
        },
        "tone_hint": {
          "type": "string",
          "enum": [
            "камерное", "разговорное", "исследовательское",
            "семейное", "торжественное", "клубное", "нейтральное"
          ]
        },
        "routing_features": {
          "type": "object",
          "properties": {
            "is_participatory": {
              "type": "boolean",
              "description": "Участники делают что-то руками / телом (мастер-класс, воркшоп, тренинг)"
            },
            "has_stepwise_action": {
              "type": "boolean",
              "description": "Есть пошаговая программа или чёткая последовательность действий"
            },
            "has_hands_on_materials": {
              "type": "boolean",
              "description": "Участники приносят или используют конкретные материалы/инструменты"
            },
            "is_speaker_led": {
              "type": "boolean",
              "description": "Фокус на конкретном спикере, авторе, кураторе, режиссёре"
            },
            "is_performance": {
              "type": "boolean",
              "description": "Зрительский формат: спектакль, концерт, показ, перформанс"
            },
            "has_true_program_list": {
              "type": "boolean",
              "description": "Есть реальный перечень пунктов программы (не просто перечисление наград или тем)"
            }
          },
          "required": [
            "is_participatory", "has_stepwise_action", "has_hands_on_materials",
            "is_speaker_led", "is_performance", "has_true_program_list"
          ]
        }
      },
      "required": [
        "core_angle", "format_signal", "subformat",
        "program_highlights", "experience_signals",
        "why_go_candidates", "voice_fragments",
        "credibility_signals", "tone_hint", "routing_features"
      ]
    }
  },
  "required": ["facts", "copy_assets"],
  "additionalProperties": false
}
```

### 2.2. Ключевые изменения vs первый ответ

| Поле | Что изменилось | Зачем |
|------|---------------|-------|
| `why_go_candidates` | Теперь `[{reason, strength}]` вместо `string[]` | Поддерживает `1 strong OR 2 regular` gate |
| `voice_fragments` | Теперь `[{text, speaker, speaker_role, is_direct_quote, opens_event_theme}]` | Поддерживает stronger quote_led gate |
| `routing_features` | Новый объект из 6 boolean features | Заменяет raw count `program_highlights ≥ 4` |
| `structure_hint` | **Убран из extraction** | Mode теперь определяется runtime, не LLM |

---

## 3. Revised Gates

### 3.1. Why-Go Gate (revised)

```python
def should_include_why_go(copy_assets: dict) -> bool:
    candidates = copy_assets.get("why_go_candidates", [])
    if not candidates:
        return False

    # mode guard
    if _determine_mode(copy_assets) == "compact_notice":
        return False

    strong = sum(1 for c in candidates if c["strength"] == "strong")
    regular = sum(1 for c in candidates if c["strength"] == "regular")

    # 1 strong reason OR 2 regular reasons
    if strong >= 1:
        return True
    if regular >= 2:
        return True

    return False
```

Примеры:

| Событие | why_go_candidates | Gate result |
|---------|------------------|-------------|
| Спектакль с Гран-при ВГИК | `[{reason: "Гран-при ВГИК", strength: "strong"}]` | ✅ (1 strong) |
| Мастер-класс: не требует навыков + создадите свою работу | `[{reason: "не требует навыков", strength: "regular"}, {reason: "создадите свою работу", strength: "regular"}]` | ✅ (2 regular) |
| Настолка Codenames | `[]` | ❌ (empty) |
| Кинопоказ с интересными темами для обсуждения | `[{reason: "обсуждение броманса/экспрессии/саспенса", strength: "regular"}]` | ❌ (1 regular) |

### 3.2. Quote-Led Gate (revised)

Старое правило: `voice_fragments[0].word_count >= 8`.

Новое правило — **multi-criteria gate**:

```python
# Denylist: slogans, CTA, poster noise, empty declarations
_QUOTE_DENYLIST_PATTERNS = [
    r"приглашаем|приходите|ждём|не пропустите|успейте",
    r"уникальная возможность|незабываемая|настоящий праздник",
    r"для всех желающих|каждый найдёт",
    r"подарит эмоции|обещает стать",
    r"билеты|вход|регистрация|запись",
    r"^\d+\s*(₽|руб|р\.)",  # price lines
    r"^https?://",  # URLs
]

def is_valid_for_quote_led(fragment: dict) -> bool:
    text = fragment.get("text", "").strip()
    if not text:
        return False

    # 1. Minimum word count
    if len(text.split()) < 8:
        return False

    # 2. Must be a direct quote
    if not fragment.get("is_direct_quote", False):
        return False

    # 3. Must have speaker attribution
    if not fragment.get("speaker"):
        return False

    # 4. Should open event theme (not just a generic praise)
    if not fragment.get("opens_event_theme", False):
        return False

    # 5. Denylist check
    for pattern in _QUOTE_DENYLIST_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    return True

def should_use_quote_led(copy_assets: dict) -> bool:
    fragments = copy_assets.get("voice_fragments", [])
    return any(is_valid_for_quote_led(f) for f in fragments)
```

Таким образом, quote_led включается **только** если цитата:

1. ≥ 8 слов
2. `is_direct_quote = true`
3. Есть `speaker` (не анонимная)
4. `opens_event_theme = true` (открывает тему события, а не просто хвалит)
5. Не попадает в denylist

Примеры:

| Цитата | Speaker | Direct | Opens theme | Gate |
|--------|---------|--------|-------------|------|
| «Россия стала для меня вторым домом! Я всегда рад быть здесь!» | Кевин Маккой | ✅ | ❌ (не раскрывает тему события) | ❌ |
| «За любой войной — не карты стратегий, а миллионы мечтающих вернуться домой» | null | ❌ (нет speaker) | ✅ | ❌ |
| «Спектакль о юности, когда всё горит, а ты держишься за мечту» — режиссёр | Режиссёр | ✅ | ✅ | ✅ |

Обратите внимание: для Event 2233 (EURODANCE'90) цитата Маккоя теперь **не** включает quote_led, потому что она не открывает тему события. Это правильно: она о личном отношении к России, а не о программе концерта. В `reported_preview` она остаётся как фрагмент внутри текста.

### 3.3. Program-Led Routing (revised)

Старое правило: `format_signal in ["мастерская", "экскурсия", "игра", "фестиваль"] OR program_highlights >= 4`.

Новое правило — **feature-based**:

```python
def should_use_program_led(copy_assets: dict) -> bool:
    rf = copy_assets.get("routing_features", {})

    # Primary signal: participatory format with concrete action
    if rf.get("is_participatory") and (
        rf.get("has_stepwise_action") or rf.get("has_hands_on_materials")
    ):
        return True

    # Secondary signal: true program list in non-performance format
    if rf.get("has_true_program_list") and not rf.get("is_performance"):
        return True

    return False
```

Что это меняет:

| Событие | routing_features | Старый routing | Новый routing |
|---------|-----------------|----------------|---------------|
| Мастер-класс по рисованию | participatory=✅, stepwise=✅, hands_on=✅ | program_led ✅ | program_led ✅ |
| Codenames (настолка) | participatory=✅, stepwise=❌, hands_on=❌, has_true_program_list=❌ | compact_notice ✅ | compact_notice ✅ |
| Концерт с сетлистом | performance=✅, has_true_program_list=❌ | ❌ misroute risk | reported_preview ✅ |
| Лекция с 5 темами | speaker_led=✅, has_true_program_list=❌ | ❌ misroute risk | reported_preview ✅ |
| Фестиваль с 6 площадками | has_true_program_list=✅, performance=❌ | program_led ✅ | program_led ✅ |
| Спектакль с 5 наградами | performance=✅, has_true_program_list=❌ | ❌ misroute risk | reported_preview ✅ |

### 3.4. Full Routing Decision Tree (revised)

```python
def determine_mode(copy_assets: dict, facts_text_clean: list[str]) -> str:
    # 1. Quote-led: strongest signal, checked first
    if should_use_quote_led(copy_assets):
        return "quote_led"

    # 2. Program-led: participatory or structured program
    if should_use_program_led(copy_assets):
        return "program_led"

    # 3. Compact notice: poor source
    if (
        len(facts_text_clean) <= 5
        and len(copy_assets.get("experience_signals", [])) == 0
        and len(copy_assets.get("why_go_candidates", [])) == 0
    ):
        return "compact_notice"

    # 4. Default: reported preview
    return "reported_preview"
```

---

## 4. Revised Prompt Fragments

### 4.1. Addition to Extraction Prompt

Добавляется в конец текущего extraction prompt (после текущих правил для `facts`):

```
Дополнительно к списку фактов, верни объект `copy_assets` с полями:

1. core_angle: о чём событие в ≤15 словах. Одна точная формулировка, не перечисление.
2. format_signal: основной формат. Одно из: спектакль | лекция | концерт | показ | мастерская | экскурсия | игра | фестиваль | встреча.
3. subformat: уточнение, если применимо. null если нет.
4. program_highlights: 2-6 самых конкретных деталей программы/содержания/состава. Не общие слова.
5. experience_signals: что зритель/участник реально увидит/услышит/попробует. ТОЛЬКО то, что прямо следует из текста. Если неясно — пустой список.
6. why_go_candidates: основания ценности события. Каждое с полем `strength`:
   - "strong": очевидная, самодостаточная причина (крупная премия, редкий исполнитель, единственный показ, уникальный формат).
   - "regular": дополнительная, поддерживающая деталь.
   Максимум 3. Если нет конкретных оснований — пустой список.
7. voice_fragments: прямые цитаты из источника. Для каждой цитаты:
   - text: текст цитаты;
   - speaker: кто сказал (null если неизвестно);
   - speaker_role: роль говорящего (режиссёр, куратор, участник и т.д., null если неизвестно);
   - is_direct_quote: true если это прямая цитата с кавычками/атрибуцией в источнике;
   - opens_event_theme: true если цитата раскрывает тему/идею события, false если это общая похвала или личное мнение не по теме.
   Максимум 2. Если в тексте нет подходящих цитат — пустой список.
8. credibility_signals: премии, уникальные форматы, заметные имена, «впервые»/«единственный»/«ретроспектива».
9. tone_hint: одно из: камерное | разговорное | исследовательское | семейное | торжественное | клубное | нейтральное.
10. routing_features: объект с 6 boolean полями:
    - is_participatory: участники делают что-то руками/телом?
    - has_stepwise_action: есть пошаговая программа?
    - has_hands_on_materials: участники приносят/используют материалы?
    - is_speaker_led: фокус на конкретном спикере/авторе/режиссёре?
    - is_performance: зрительский формат (спектакль/концерт/показ)?
    - has_true_program_list: реальный перечень пунктов программы (не просто перечисление наград/тем)?

ПРАВИЛА ДЛЯ copy_assets:
- Все значения ДОЛЖНЫ быть traceable к конкретным фрагментам source_text / raw_excerpt / poster_texts.
- Не выдумывай новые сведения.
- core_angle — это ONE dominant angle, а не пересказ всех фактов.
- Если данных для поля нет — возвращай пустой список [] или null.
```

### 4.2. Description Generation Prompt — Ключевые изменения

Промпт генерации из первого ответа остаётся в силе, за исключением:

**Изменение 1: Why-go phrasing**

Было:
```
Если why_go_candidates непустые и их ≥2, встрой...
```

Стало:
```
Если system указал why_go=true, встрой 1-2 предложения о ценности события
ВНУТРИ одной из ### секций (не отдельной секцией).
Каждое предложение ДОЛЖНО опираться на конкретный why_go_candidate.
```

(Решение why_go принимается в runtime, до LLM call; prompt получает уже бинарный флаг).

**Изменение 2: Quote-led**

Было:
```
Начни с blockquote цитаты из voice_fragments[0].
```

Стало:
```
Начни с blockquote лучшей цитаты (указана в copy_assets.selected_quote).
После blockquote добавь атрибуцию: `— {speaker_role}` или `— {speaker}`.
Следующий абзац объясняет, что это за событие и почему цитата задаёт тон.
НЕ пересказывай цитату в следующем абзаце.
```

(Runtime выбирает лучшую цитату через `is_valid_for_quote_led` и передаёт её в prompt как `selected_quote`).

**Изменение 3: Compact_notice sizing**

Добавить в compact_notice rules:
```
Если facts_text_clean содержит ≤3 факта, описание может быть 40-80 слов.
Не пытайся развернуть бедный source в развёрнутый текст.
```

---

## 5. Implementation Posture

### Low-risk immediate candidates (можно внедрять на Phase 1-2)

| # | Что | Риск | Complexity |
|---|-----|------|-----------|
| 1 | Heading palette + heading stop-list в generation prompt | Очень низкий | Prompt edit only |
| 2 | Anti-redundancy rule в generation prompt | Низкий | Prompt edit only |
| 3 | Extended coverage checks (template_feel, weak_lead, weak_heading, redundancy) | Низкий | Python logic + prompt edit |
| 4 | Lead variety instructions | Низкий | Prompt edit only |
| 5 | `compact_notice` sizing rule для бедных sources | Низкий | Prompt edit + minor logic |

Эти 5 пунктов **не требуют** extraction changes вообще. Их можно внедрить прямо в `_llm_fact_first_description_md` и в coverage/revise logic.

### Medium-risk experiments (Phase 2-3, нужен A/B)

| # | Что | Риск | Complexity |
|---|-----|------|-----------|
| 6 | Merged extraction (facts + copy_assets в одном call) | Средний | Schema change, prompt extension, output parsing |
| 7 | Mode routing (deterministic, based on copy_assets) | Средний | New routing logic, depends on #6 |
| 8 | Why-go gate (strength-based) | Средний | Depends on #6 |
| 9 | Inline why-go в generation prompt | Средний | Prompt edit, depends on #7 and #8 |

### Research-only (Phase 4+, не ранее)

| # | Что | Почему research | Concern |
|---|-----|----------------|---------|
| 10 | Quote-led mode | Зависит от качества `voice_fragments` extraction | Может misfire если extraction ошибочна |
| 11 | Tone-adaptive generation (разные register для камерного vs клубного) | Нужен validated tone extraction | Может создать неожиданные стилистические скачки |

### Почему quote_led в research

- `voice_fragments` extraction — самая хрупкая часть schema.
- Нужно сначала убедиться, что Gemma стабильно различает:
  - прямую цитату vs пересказ,
  - speaker attribution vs безымянная фраза,
  - theme-opening vs generic praise.
- Если extraction ошибается в 20%+ случаев, quote_led будет производить мусор.
- Рациональнее: включить extraction на Phase 2, собрать статистику по 50+ events, и только потом решить, включать ли quote_led в routing.

---

## 6. Revised Event Examples

### Event 2767 (Будь здоров, школяр!)

**Copy assets (revised extraction):**

```json
{
  "core_angle": "Спектакль по повести Окуджавы о юности на войне",
  "format_signal": "спектакль",
  "why_go_candidates": [
    {"reason": "Гран-при XXVII международного фестиваля ВГИК + приз за лучший ансамбль", "strength": "strong"},
    {"reason": "Премии «Золотой лист» за лучшие мужскую и женскую роли", "strength": "regular"}
  ],
  "voice_fragments": [
    {
      "text": "За любой войной — не карты стратегий, а миллионы мечтающих вернуться домой",
      "speaker": null,
      "speaker_role": null,
      "is_direct_quote": false,
      "opens_event_theme": true
    }
  ],
  "routing_features": {
    "is_participatory": false,
    "has_stepwise_action": false,
    "has_hands_on_materials": false,
    "is_speaker_led": false,
    "is_performance": true,
    "has_true_program_list": false
  }
}
```

**Routing**: quote_led? ❌ (нет speaker). program_led? ❌ (performance). compact? ❌ (rich facts). → `reported_preview`.

**Why-go gate**: 1 strong → ✅.

**Expected output:**

> Постановка по повести Булата Окуджавы — о юности, которая продолжает мечтать, даже оказавшись на фронте. В центре не военная хроника, а мир чувств: мечты, первая любовь, желание вернуться домой.
>
> ### Актёрский состав
> В ролях: Георгий Сальников, Ярослав Жалнин, Александра Власова, Тимур Орагвелидзе, Алексей Боченин, Александр Хитев. Длительность — 1 час 30 минут без антракта.
>
> ### Чем отмечен спектакль
> Гран-при XXVII международного фестиваля ВГИК, приз за лучший актёрский ансамбль, премии «Золотой лист» за лучшую женскую и мужскую роль. Стоит идти ради ансамбля, отмеченного именно за совместную игру.

### Event 2695 (Творческая мастерская по рисованию)

**Routing**: is_participatory=✅, has_hands_on_materials=✅ → `program_led`.

**Why-go gate**: `[{reason: "не требует навыков", strength: "regular"}, {reason: "создание неповторимой работы", strength: "regular"}]` → 2 regular → ✅.

**Expected output:**

> Мастерская по правополушарному рисованию устроена так, чтобы участник создал свою работу без правил и навыков, доверившись интуиции. Навыки рисования не требуются — формат подойдёт для любого уровня подготовки.
>
> ### Что понадобится
> Гуашь и кисти: тонкая, плоская широкая (щетина) и круглая кисть с острым кончиком. Встречу проводит Елена Калинцева, продолжительность 1-1,5 часа.
>
> ### Что здесь особенно интересно
> Техника помогает раскрыть творческий потенциал и снять внутренние блоки. Для зрителя это возможность получить готовую работу, не обладая никакой предварительной подготовкой.

### Event 2233 (EURODANCE'90)

**Quote_led gate**: Цитата Маккоя `opens_event_theme = false` → ❌. → `reported_preview`.

**Why-go gate**: `[]` → ❌.

**Expected output:**

> «EURODANCE'90» — концерт с Natasha Wright из La Bouche и Kevin McCoy из Bad Boys Blue, выстроенный вокруг мировой музыки 90-х.
>
> ### Кто на сцене
> Natasha Wright — солистка La Bouche. Kevin McCoy — участник Bad Boys Blue. Кевин Маккой отмечает: «Россия стала для меня вторым домом! Я всегда рад быть здесь! Тут мои любимые зрители, самые лучшие на земле!»
>
> ### Что прозвучит
> Natasha Wright исполнит «Be My Lover», «Sweet Dreams» и «Tonight is the Night». Kevin McCoy — «You're A Woman», «Pretty Young Girl» и «Come Back And Stay».

---

## 7. Ответ на каждый пункт follow-up prompt

| Пункт | Ответ |
|-------|-------|
| 1. Revised architecture | Merged single-call (section 1) |
| 2. Revised schema | JSON schema v2 с strength/routing_features (section 2) |
| 3. Revised gates | why-go: 1 strong OR 2 regular. quote_led: multi-criteria. program_led: feature-based (section 3) |
| 4. Prompt adjustments | Extraction extension + 3 generation prompt changes (section 4) |
| 5. Implementation posture | 5 low-risk, 4 medium-risk, 2 research-only (section 5) |

---

## 8. Рекомендуемый следующий шаг

Начинать с **low-risk candidates (#1-5)**: heading palette, anti-redundancy, extended coverage checks, lead variety, compact sizing. Это не требует extraction changes и даст быстрое улучшение качества output.

Параллельно: подготовить golden dataset из 20 events для A/B evaluation, чтобы к моменту внедрения medium-risk (#6-9) была готова baseline.
