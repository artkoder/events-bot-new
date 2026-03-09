# Smart Update Gemini Event Copy V2.9 Prompt Context

Дата: 2026-03-08

Этот документ нужен как docs-only срез ключевых prompt и algorithm changes в `v2.9`, чтобы внешний консультант видел не только outputs, но и exact contract, который эти outputs породил.

## 1. База `v2.9`

`v2.9` был собран **от `v2.8`**, не от `v2.6` и не от `v2.7`.

Что сознательно сохранялось из `v2.8`:

- fact-first architecture;
- routing logic;
- generation-side anti-bureaucracy;
- natural event framing;
- template-overuse control;
- missing-repair stage;
- policy / revise stage.

Что менялось в `v2.9`:

- prompt-facing sanitizer interaction;
- extraction wording;
- `_pre_extract_issue_hints`;
- post-extract issue hints;
- exact wording для forbidden `посвящ*`.

## 2. Branch logic не менялась

В `v2.9` не было нового routing redesign.

Логика оставалась той же:

```python
if len(facts_text_clean) <= 6 and not _program_rich_blocker(copy_assets, event_type):
    return "compact_fact_led"
return "fact_first_v2_9"
```

Практически это означает:

- `v2.9` тестировал не новый router;
- а качество fact shaping под тем же routing contract.

## 3. Prompt-facing sanitizer change

Это был главный corrective patch `v2.9`.

В текущем runtime есть prompt-facing sanitizer:

```python
def _sanitize_fact_text_clean_for_prompt(fact: str) -> str:
    ...
    # "<event> посвящена ... <topic>" -> "Тема: <topic>."
```

В experimental `v2.9` мы не отключали весь sanitize layer.
Но добавили bypass именно для synthetic label generation:

```python
def _safe_prompt_fact_sanitize(fact: str) -> str:
    original = _normalize_text(fact)
    cleaned = _normalize_text(su._sanitize_fact_text_clean_for_prompt(original))
    if _LABEL_STYLE_FACT_RE.search(cleaned) and not _LABEL_STYLE_FACT_RE.search(original):
        return original
    return cleaned
```

Идея:

- не терять прочую sanitize-полезность;
- но не допускать, чтобы runtime сам рождал `Тема: ...`.

## 4. `_pre_extract_issue_hints`

В `v2.9` hints стали error-style, но без shouting.

Ключевые формулировки:

```text
ОШИБКА: в исходных фактах есть слово «посвящ». Не используй этот корень. Замени его прямым указанием на предмет события.
ОШИБКА: найдены ярлыки вроде «Тема: ...» или «Идея: ...». Сформулируй предмет без двоеточий и префиксов.
ОШИБКА: в исходных фактах есть рамка вроде «расскажут о...». Извлеки только сам предмет обсуждения без рамки намерения.
ОШИБКА: один и тот же смысл раздут в несколько похожих facts. Объедини пересекающиеся смыслы, но не схлопывай distinct names и program items.
```

Цель:

- повысить forcing power;
- не возвращать `v2.7`-style sentence templates;
- не уходить в `КРИТИЧЕСКАЯ ОШИБКА` / CAPS framing.

## 5. Post-extract issue hints

В `v2.9` post-extract hints тоже были усилены:

```text
ОШИБКА: в facts_text_clean всё ещё есть «посвящ...». Перепиши этот факт через прямой предмет события.
ОШИБКА: facts_text_clean раздут похожими или метатекстовыми facts. Объедини перефразы, но сохрани отдельные имена, названия и program items.
ОШИБКА: в facts_text_clean остался ярлык вроде «Тема: ...». Перепиши этот факт как плотный publishable fact без ярлыка.
ОШИБКА: в facts_text_clean осталась рамка вроде «на мероприятии расскажут о...». Оставь только чистый предмет обсуждения.
```

Отдельный post-extract LLM repair stage при этом не добавлялся.

## 6. Extraction prompt changes

Главный новый блок `v2.9` был таким:

```text
- АНТИ-ДРОБЛЕНИЕ: не разбивай одну мысль на несколько похожих facts. Объединяй смежные смыслы. Но не схлопывай distinct names, program items, titles, quotes и отдельные meaningful details.
- Запрещены label-style facts вроде `Тема: ...`, `Идея: ...`, `Цель: ...`, `Сюжет: ...`. Формулируй предмет напрямую как noun-phrase fact. Примеры: `жизнь и творчество русских художниц`; `устройство платформы и её возможности`.
- Запрещены intent-style facts вроде `Лекция расскажет о ...`, `На встрече обсудят ...`, `Презентация познакомит с ...`. Оставляй только сам предмет обсуждения. Примеры: `история создания картины`; `причины появления проекта и его задачи`.
```

Смысл:

- усилить control над fact shape;
- не вернуть sentence-level narrative shaping;
- не использовать safe-positive wrappers из `v2.7`.

## 7. Generation side в `v2.9`

Generation и compact generation radical rewrite не получали.

Сохранялись `v2.8`-принципы:

```text
- Разрешён естественный event framing: «лекция о ...», «спектакль исследует ...», «в центре внимания — ...».
- Не злоупотребляй одной и той же вводной конструкцией.
- Не используй бюрократические рамки вроде «на мероприятии расскажут о...».
```

То есть `v2.9` проверял hypothesis:

- core bottleneck всё ещё в extraction/fact shaping,
- а не в generation wording itself.

## 8. Revise / policy wording for `посвящ*`

Blocking issue для forbidden marker оставался human-readable:

```text
СБОЙ ПРАВИЛА: найдено запрещённое слово «посвящ». Удали всё предложение с ним целиком и заново сформулируй суть через естественный старт: «лекция о ...», «главная тема — ...» или «в центре внимания — ...».
```

Идея:

- не делать грубый deterministic rewrite;
- не заменять `посвящ*` машинным regex;
- дать Gemma понятный repair action.

## 9. Чего `v2.9` сознательно НЕ делал

`v2.9` не тестировал:

- новый routing layer;
- новый fact-quality gate;
- новый repair pass между extraction и generation;
- перенос narrative shaping обратно в Extraction;
- deterministic смысловые rewrites.

Это было сознательное ограничение.

## 10. Что важно для чтения результатов

`v2.9` надо читать как narrow corrective round:

1. убрать synthetic `Тема: ...`;
2. жёстче задать shape фактов;
3. не трогать остальной working core без нужды.

Если `v2.9` не дал сильного quality gain, это значит не то, что sanitizer fix был бесполезен, а то, что root problem глубже:

- в том, как Gemma формирует dense fact units;
- как сохраняет program/person specifics;
- и как трансформирует raw metatext into publishable facts.
