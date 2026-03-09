# Smart Update Gemma Event Copy V2.14 Prompt Context

Дата: 2026-03-08

Связанный harness:

- `artifacts/codex/experimental_pattern_dryrun_v2_14_2026_03_08.py`

## 1. Что изменилось в `v2.14`

`v2.14` сознательно ушёл от regex-heavy drift и вернулся к более LLM-first схеме.

Новая цепочка:

1. `raw_facts`
2. shape detection
3. `full-floor normalization` через LLM
4. deterministic cleanup/dedup/hygiene
5. `story outline` через отдельный LLM call для rich cases
6. exemplar-driven generation
7. deterministic validation
8. targeted repair только по policy issues, а не по любому `missing`

## 2. Shape-specific normalization contract

Ключевая логика normalization:

- `presentation_project`
  - убирать metatext frame;
  - сохранять отдельно `задачи`, `устройство`, `возможности`, `причины появления`, `проблему`;
  - не сливать эти agenda blocks в один vague fact.
- `lecture_person`
  - сохранять grouped names в одном clean fact;
  - не дробить person-block на отдельные однотипные facts;
  - убирать framing вроде `лекция посвящена`.
- `program_rich`
  - сохранять grouped titles/program items;
  - не заменять программу общими словами типа `произведения` или `номера`.

## 3. Новый outline-pass

`v2.14` добавил отдельный `story outline` только для:

- `presentation_project`
- `lecture_person`
- `program_rich`

Outline schema:

```json
{
  "blocks": [
    {
      "kind": "lead | section",
      "heading": "string | null",
      "fact_ids": [1, 2, 3],
      "focus_note": "string"
    }
  ]
}
```

Что outline должен был делать:

- сначала собирать `lead`;
- потом 2-3 смысловые секции;
- давать headings без generic templates и без question-style;
- для `presentation_project` прямо называть формат презентации проекта, а не абстрактную встречу.

## 4. Generation contract

Generation в `v2.14`:

- остаётся exemplar-driven;
- получает `facts_text_clean`;
- для rich cases получает ещё и `story outline`;
- должен follow outline как основной план, но не копировать его бюрократически;
- должен избегать:
  - `посвящ`
  - `расскажут`
  - `представят`
  - `обсудят`
  - generic headings
  - heading/content mismatch

Ключевая shape-specific добавка для `presentation_project`:

- если факты подтверждают проект/платформу, лид должен прямо назвать презентацию проекта;
- нельзя подменять такой формат словом `встреча`.

## 5. Что сломалось в реальном dry-run

Главный новый failure mode:

- outline сам начал генерировать бюрократические headings и focus notes;
- generation потом подхватывал этот язык.

На практике это дало:

- `2673`: headings вроде `Цель и задачи проекта`, `Формат мероприятия`, `Для кого это будет интересно`
- `2687`: outline/focus note с формулой `лекция посвящена`, из-за чего forbidden framing вернулся в финальный текст

То есть проблема `v2.14` не в самом split-call, а в слишком свободном outline contract.

## 6. Что важно для следующей консультации

Нужен критический разбор:

- стоит ли сохранять outline-pass;
- как сделать его более structural и менее prose-like;
- какие prompt-level изменения для Gemma помогут:
  - убрать outline-generated bureaucracy;
  - сохранить gains на `2734`;
  - улучшить `2673`;
  - не ломать `2745` и `2660`;
  - не вернуть regex-first semantics.
