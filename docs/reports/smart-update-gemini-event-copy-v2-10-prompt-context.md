# Smart Update Gemini Event Copy V2.10 Prompt Context

Дата: 2026-03-08

Этот документ нужен как docs-only срез точечных prompt и algorithm changes в `v2.10`, чтобы внешний консультант видел exact contract, который породил текущие outputs.

## 1. База `v2.10`

`v2.10` был собран **от `v2.9`**.

Что сознательно сохранялось:

- prompt-facing sanitizer bypass из `v2.9`;
- routing logic;
- generation-side hygiene из `v2.8/v2.9`;
- fact-first architecture;
- missing-repair stage;
- policy / revise stage.

Что менялось:

- extraction wording;
- `_pre_extract_issue_hints`;
- post-extract issue hints;
- wording blocking issue для `посвящ*`.

## 2. Routing не менялся

В `v2.10` не было нового router.

Логика осталась:

```python
if len(facts_text_clean) <= 6 and not _program_rich_blocker(copy_assets, event_type):
    return "compact_fact_led"
return "fact_first_v2_10"
```

Значит `v2.10` тестировал именно новый extraction contract, а не новую branch strategy.

## 3. Prompt-facing sanitizer change сохранялся

Из `v2.9` полностью сохранился bypass synthetic label rewrite:

```python
def _safe_prompt_fact_sanitize(fact: str) -> str:
    original = _normalize_text(fact)
    cleaned = _normalize_text(su._sanitize_fact_text_clean_for_prompt(original))
    if _LABEL_STYLE_FACT_RE.search(cleaned) and not _LABEL_STYLE_FACT_RE.search(original):
        return original
    return cleaned
```

То есть `v2.10` не возвращал ветку:

- `<event> посвящена ...` -> `Тема: ...`

## 4. `_pre_extract_issue_hints` стали action-oriented

Ключевая идея `v2.10`:

- не только диагностировать ошибку;
- но и явно писать `ТВОЯ ЗАДАЧА: ...`.

Примеры:

```text
ОШИБКА: в исходных фактах есть слово «посвящ». ТВОЯ ЗАДАЧА: убери этот корень и назови предмет события напрямую.
ПЛОХО: «лекция посвящена истории театра».
ХОРОШО: «история театра» или «лекция об истории театра».

ОШИБКА: в исходных фактах есть рамка вроде «расскажут о...». ТВОЯ ЗАДАЧА: удали глагол намерения и оставь только предмет.
ПЛОХО: «На презентации расскажут о задачах платформы».
ХОРОШО: «задачи платформы».
```

## 5. Post-extract hints тоже стали action-oriented

Ключевые формулировки:

```text
ОШИБКА: в facts_text_clean всё ещё есть «посвящ...». ТВОЯ ЗАДАЧА: перепиши этот факт через прямой предмет события.
ОШИБКА: facts_text_clean раздут похожими или метатекстовыми facts. ТВОЯ ЗАДАЧА: собери близкие пункты в 1-2 плотных facts.
ОШИБКА: в facts_text_clean осталась рамка вроде «на мероприятии расскажут о...». ТВОЯ ЗАДАЧА: удали глагол намерения и оставь только чистый предмет обсуждения.
```

Отдельный new post-extract LLM stage при этом не добавлялся.

## 6. Новый extraction rule: `LIST CONSOLIDATION`

Это была главная новая гипотеза `v2.10`.

Ключевой блок:

```text
- LIST CONSOLIDATION: если источник перечисляет несколько имён, тем, вопросов лекции, функций проекта или пунктов программы, собирай их в 1-2 плотных facts. Не делай отдельный fact на каждый однотипный подпункт.
- АНТИ-ДРОБЛЕНИЕ: не разбивай одну смысловую единицу на несколько похожих facts.
```

Идея:

- не просто запрещать splitting;
- а явно instruct Gemma, как упаковывать list-like source material.

## 7. Compact `[ПЛОХО] -> [ХОРОШО]` intent examples

В extraction prompt появились 2 очень коротких transform-примера:

```text
- [ПЛОХО] `На презентации расскажут о задачах платформы.` -> [ХОРОШО] `задачи платформы`.
- [ПЛОХО] `На встрече обсудят, как устроен проект.` -> [ХОРОШО] `устройство проекта`.
```

Это было сделано специально:

- без возврата к `v2.7` safe-positive wrappers;
- но с более явным pattern mapping для Gemma.

## 8. Program/list examples были смягчены

Для performance cases:

```text
`Программа включает «Название 1», «Название 2», «Название 3»`
```

Для lecture/person-rich cases:

```text
`Лекция охватывает творчество ...`
`Среди героинь лекции ...`
```

Специально избегались colon-style examples вроде:

- `Художницы: ...`
- `В программе: ...`

потому что они слишком близки к label-style packaging.

## 9. Revise / policy wording for `посвящ*`

Blocking issue стал чуть более action-oriented:

```text
СБОЙ ПРАВИЛА: найдено запрещённое слово «посвящ». ТВОЯ ЗАДАЧА: удали всё предложение с ним целиком и напиши новое через прямой предмет события: «лекция о ...», «главная тема — ...» или «в центре внимания — ...».
```

Это всё ещё human-readable wording, не deterministic rewrite.

## 10. Что `v2.10` сознательно НЕ делал

`v2.10` не тестировал:

- новый routing layer;
- отдельный fact-quality gate между extraction и generation;
- новый repair pass;
- новый generation redesign;
- deterministic смысловые rewrites.

Это был narrow corrective round именно по extraction contract.
