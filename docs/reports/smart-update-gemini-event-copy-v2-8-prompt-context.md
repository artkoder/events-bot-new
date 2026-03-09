# Smart Update Gemini Event Copy V2.8 Prompt Context

Дата: 2026-03-07

Этот документ нужен как docs-only срез ключевых prompt и algorithm changes в `v2.8`, чтобы внешний консультант мог увидеть не только outputs, но и то, какими именно правилами они были получены.

## 1. База `v2.8`

`v2.8` был собран **от `v2.6`**, не от `v2.7`.

Что сознательно НЕ менялось:

- routing logic как принцип;
- fact-first architecture;
- cleanup pipeline;
- missing-repair stage;
- deterministic support layer не добавлялся.

Что менялось:

- extraction contract;
- `_pre_extract_issue_hints`;
- post-extract hints;
- standard generation wording;
- compact generation wording;
- revise wording для `посвящ*`.

## 2. Branch logic

Branch logic в `v2.8` осталась такой:

```python
if len(facts_text_clean) <= 6 and not _program_rich_blocker(copy_assets, event_type):
    return "compact_fact_led"
return "fact_first_v2_8"
```

То есть никакого нового routing intelligence не добавлялось.

## 3. Extraction prompt changes

Ключевой rollback block:

```text
- Извлекай чистые, плотные и самодостаточные факты (data points).
- Запрещены label-style facts вроде `Тема: ...`, `Идея: ...`, `Цель: ...`, `Сюжет: ...`. Извлекай тему как самодостаточный факт без ярлыка.
- Не используй пустые обёртки вроде `Выставка носит название ...`, `Автор выставки — ...`, `Мероприятие пройдет ...`.
- Не используй корень «посвящ» в `facts_text_clean` и `copy_assets`. Заменяй его прямым фактом о предмете события без рекламной или объясняющей упаковки.
- Если в источнике есть намерения вроде «на встрече расскажут о...» или «лекция познакомит с...», извлекай только сам предмет обсуждения, отбрасывая рамку намерения.
- Один факт = одна distinct detail. Не раздувай один смысл в 3-5 перефразов.
- Не обобщай перечни в один vague fact.
```

Ключевой смысл:

- убрать `v2.7` safe-positive wrappers;
- вернуть плотные factual propositions;
- не давать sentence-level narrative shaping внутри extraction.

## 4. `_pre_extract_issue_hints`

В `v2.8` hints были deliberately simplified.

Фактически:

```text
если есть «посвящ»:
  "Критическое требование: не используй слово или корень «посвящ». Извлекай только сам предмет события напрямую."

если есть label-style raw facts:
  "Запрещены label-style факты вроде «Тема: ...» или «Идея: ...». Извлекай тему как самодостаточный факт без ярлыка."

если есть intent facts:
  "Если в raw_facts есть «расскажут о...» или похожее намерение, извлекай только сам предмет обсуждения напрямую."
```

Идея была в том, чтобы убрать sentence templates и не толкать Gemma к механическому копированию форм.

## 5. Post-extract hints

Post-extract correction hints тоже были упрощены.

Ключевые формулировки:

```text
"В facts_text_clean ещё осталось «посвящ...». Перепиши этот факт через сам предмет события напрямую."
"Не возвращай label-style facts вроде «Тема: ...». Перепиши их как плотные publishable facts без ярлыков."
"В facts_text_clean осталась бюрократическая рамка вроде «на мероприятии расскажут о...». Оставь только сам предмет обсуждения."
```

## 6. Standard generation prompt changes

В quality block были добавлены два новых акцента:

```text
- Разрешён естественный event framing: «Лекция о ...», «В центре внимания — ...», «Спектакль исследует ...».
- Не злоупотребляй одной и той же вводной конструкцией. Если уже использовано «В центре ...», не начинай так ещё одно предложение.
```

Сохранялся ban на bureaucratic metatext:

```text
- Не используй бюрократические рамки вроде «На мероприятии расскажут о...», «Проект представит...», «Будет представлен обзор...». Переходи сразу к сути события.
```

## 7. Compact generation prompt changes

Для compact branch было добавлено то же направление:

```text
- Не используй бюрократические рамки вроде «на мероприятии расскажут о...», «проект представит...», «будет представлен обзор...».
- Разрешён естественный event framing: «лекция о ...», «спектакль исследует ...», «в центре внимания — ...».
- Не злоупотребляй одной и той же вводной конструкцией. Нельзя повторять один и тот же стартовый шаблон больше одного раза на текст.
```

## 8. Revise / policy wording for `посвящ*`

В `v2.8` blocking issue для forbidden marker `посвящ*` был сокращён и сделан более механическим:

```text
КРИТИЧЕСКИЙ СБОЙ: найдено «посвящ». Удали всё предложение с этим словом и перепиши его через естественный event framing: «лекция о ...», «главная тема — ...», «в центре внимания — ...».
```

Идея:

- не делать длинную объясняющую инструкцию;
- не предлагать целый каталог шаблонов;
- заставить модель удалить bad sentence целиком.

## 9. Что именно тестировал `v2.8`

Практический hypothesis pack `v2.8` был таким:

1. rollback к `v2.6` base;
2. extraction снова делает dense data points;
3. hints больше не задают sentence templates;
4. generation следит за overuse одного opening pattern;
5. revise для `посвящ*` становится короче и жёстче.

`v2.8` НЕ тестировал:

- новый routing;
- deterministic cleanup facts;
- новую architecture;
- перенос narrative shaping в другой слой кроме generation/revise.

## 10. Existing prompt-facing sanitizer from current runtime

Есть ещё один важный algorithm interaction из `smart_event_update.py`, который действует **до generation prompt** и который не был частью самого `v2.8 patch pack`, но реально влияет на то, какие facts доходят до LLM.

Функция:

```python
def _sanitize_fact_text_clean_for_prompt(fact: str) -> str:
    s = str(fact or "").strip()
    if not s or not _FACT_FIRST_POSV_WORD_RE.search(s):
        return s

    # Common pattern: "<event> посвящена/посвящён ... <topic>" -> "Тема: <topic>."
    m = re.match(
        r"(?i)^\\s*(?:лекци\\w*|встреч\\w*|бесед\\w*|показ\\w*|концерт\\w*|спектакл\\w*|"
        r"мастер-?класс\\w*|мастерск\\w*|заняти\\w*|экскурс\\w*|презентац\\w*|выставк\\w*)\\s+"
        r"посвящ\\w+\\s+(.+?)\\s*[.!?]?\\s*$",
        s,
    )
    if m:
        topic = (m.group(1) or "").strip()
        if topic:
            return f"Тема: {topic}."
    return s
```

В experimental harness `v2.8` prompt-facing facts проходят через:

```python
def _safe_prompt_fact_sanitize(fact: str) -> str:
    cleaned = su._sanitize_fact_text_clean_for_prompt(fact)
    return _normalize_text(cleaned)
```

Это означает:

- даже если extraction вернул `Выставка посвящена теме противоречий мира.`,
- в prompt-facing `facts_text_clean` это может превратиться в `Тема: теме противоречий мира.`

Практический вывод:

- часть label-style artifacts в `v2.8` может рождаться не только в extraction prompt,
- но и в существующем prompt-facing sanitizer слое текущего runtime.
