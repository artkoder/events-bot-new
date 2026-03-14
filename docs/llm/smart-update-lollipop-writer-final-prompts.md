# Smart Update Lollipop Writer Final Prompts

Канонический `v1` контракт для финального public-writer stage после `writer_pack.select`.

## Stage shape

```text
writer_pack.select.v1
-> writer.final_4o.v1
-> writer.final_4o.validate
```

Где:
- `writer.final_4o.v1` — единственный финальный `4o` call;
- `writer.final_4o.validate` — детерминированный post-run validator;
- upstream `writer_pack.select.v1` в `iter1` остаётся identity/no-op.

## Goal

- превратить один selected writer pack в финальный public text;
- сохранить grounded event facts;
- не продублировать formalized `infoblock` logistics в narrative prose;
- сохранить `literal_items` в markdown list и без смысловой мутации;
- поддержать future-safe `title_strategy = enhance` без отдельного downstream title stage.

## Input

Финальный writer получает один selected pack в контракте:

```json
{
  "event_type": "string",
  "title_context": {
    "original_title": "string",
    "strategy": "keep|enhance",
    "hint_fact_id": "fact_id|null",
    "hint_fact_text": "string|null",
    "is_bare": "boolean"
  },
  "sections": [
    {
      "role": "lead|body|program",
      "style": "narrative|list",
      "heading": "string|null",
      "fact_ids": ["fact_id"],
      "facts": [
        {
          "fact_id": "fact_id",
          "text": "string",
          "priority": 3
        }
      ],
      "coverage_plan": [
        {
          "fact_id": "fact_id",
          "mode": "narrative|literal_list|narrative_plus_literal_list|absorbed_by_list"
        }
      ],
      "literal_items": ["string"],
      "literal_list_is_partial": "boolean"
    }
  ],
  "infoblock": [
    {
      "fact_id": "LG01",
      "label": "Дата|Время|Локация|Цена|Билеты|Возраст|Прочее",
      "value": "string"
    }
  ],
  "constraints": {
    "must_cover_fact_ids": ["all non-LG fact ids"],
    "infoblock_fact_ids": ["all LG fact ids"],
    "headings": ["string"],
    "list_required": true,
    "no_logistics_in_narrative": true
  }
}
```

## Output schema

```json
{
  "title": "string",
  "description_md": "string"
}
```

`_debug` поля в canonical `v1` contract нет.

## Runtime application rules

1. Если `title_context.strategy == "keep"`:
   - модель всё равно возвращает `title`, но Python runtime его не доверяет;
   - applied title принудительно равен `title_context.original_title`.
2. Если `title_context.strategy == "enhance"`:
   - applied title берётся из model output;
   - validator блокирует unchanged title и infoblock leakage в title.

## Prompt rules

1. Идти по `sections` строго по порядку.
2. Prompt должен получать явный structure plan по секциям: lead всегда без heading, все последующие headings разрешены только как exact `### heading` из pack.
3. Первый абзац всегда короткий lead на `1-2` предложения.
4. Prompt должен получать `event_type` как explicit format signal.
5. Если title bare/stylized/opaque и может читаться как название фильма, проекта или объекта, prompt должен нести explicit signal `title_needs_format_clarity = true`, а первое предложение обязано сразу объяснить формат события.
6. Если `lead_needs_format_bridge = true`, первое предложение обязано прямо назвать формат через `event_type` (`кинопоказ`, `презентация`, `лекция` и т.п.), даже когда lead facts сами описывают только фильм или проект.
7. На каждой границе `section` следующий narrative block начинается с пустой строки, даже когда `heading = null`.
8. Если `coverage_plan.mode = literal_list`, `literal_items` выводятся markdown bullets `- item`.
9. Если `coverage_plan.mode = narrative_plus_literal_list`, сначала идёт одно короткое вводное предложение, потом markdown bullets.
10. Для `narrative_plus_literal_list` вводная строка может быть либо короткой фразой с двоеточием, либо полноценным коротким предложением; для обычного `literal_list` опирайся на heading или явную строку-ввод с двоеточием.
11. Если `coverage_plan.mode = absorbed_by_list`, соответствующий fact не раскрывается отдельно в prose.
12. Если `literal_list_is_partial = true`, вводная строка перед bullet list должна явно сигнализировать, что это примеры, а не полный перечень: например `Среди композиций:` или `В программе также прозвучат:`.
13. Prompt должен держать rough target length band:
   - sparse cases: примерно `220-520` знаков;
   - standard cases: примерно `400-700`;
   - rich/list-heavy cases: примерно `500-900`.
14. В `description_md` нельзя переносить logistics из `infoblock`: дата, время, локация/адрес, цена, билеты, регистрация, возраст, URL.
15. Нельзя добавлять новые facts, CTA, бюрократический или атмосферный filler.
16. Нельзя писать choppy fact glue в ритме `одно предложение = один факт`, если facts можно связать одной естественной редакторской формулировкой.
17. Плохие opening patterns должны быть explicitly запрещены: `Режиссёр фильма — ...`, `Проект представляет собой ...`, если они не объясняют формат события.
18. Для screening/presentation bridge-case плохое opening — сразу пересказывать фильм или проект, так и не сказав, что читатель идёт на показ/презентацию.
19. Хороший opening pattern: сначала назвать событие/показ/лекцию/презентацию, потом дать контекст.
20. Register target: живой, сдержанный русский культурный дайджест, а не справочная карточка.

## Deterministic validation

### Hard errors

- invalid JSON / invalid schema;
- `title.enhance_unchanged`;
- infoblock duplication in `description_md`;
- literal item missing from output;
- literal items not rendered as markdown bullet list;
- missing intro before `narrative_plus_literal_list`;
- missing non-exhaustive intro marker for `literal_list_is_partial = true`;
- invented headings;
- URL leakage into `description_md`.

### Warnings

- `keep` title mismatch from the model (runtime overwrites it);
- missing allowed heading;
- `lead.too_long`;
- lead still lacks event-format clarity for an opaque title;
- filler phrase detected.

## Retry policy

- один initial `4o` call;
- если validator даёт hard errors, allowed один correction retry с appended validation errors;
- если второй ответ всё ещё невалиден, family run помечает event как failed; fallback/runtime wiring лежит вне текущего лабораторного retune.
