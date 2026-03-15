# Smart Update Lollipop Writer Pack Contracts

Канонический `v1` контракт для downstream assembly после `editorial.layout`.

## Stage shape

```text
writer_pack.compose.v1
-> writer_pack.select.v1
-> writer.final_4o
```

Где:
- `writer_pack.compose.v1` — детерминированный pack assembly;
- `writer_pack.select.v1` — детерминированный identity/no-op в `iter1`;
- `writer.final_4o` — единственный финальный writer call.

## `writer_pack.compose.v1`

Цель:
- превратить `editorial.layout` plan + prioritized fact pack в один canonical writer-ready pack;
- сохранить document order;
- отделить logistics infoblock от narrative;
- сохранить literal program items буквально и без double-exposure в prose.

Важно:
- upstream `facts.prioritize` теперь может помечать fact как `narrative_policy = suppress`;
- `editorial.layout` и `writer_pack.compose` не должны реанимировать такие facts в public payload;
- suppressed facts остаются audit-layer detail, но не попадают в `sections` и `constraints.must_cover_fact_ids`.

## Output schema

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
      "literal_item_source_fact_ids": ["fact_id"],
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

## Deterministic rules

1. `sections` сохраняют exact order из `editorial.layout` blocks; root-level split на `lead/body/program` не допускается.
2. `infoblock` извлекается отдельно и сортируется по canonical order:
   `Дата -> Время -> Локация -> Цена -> Билеты -> Возраст -> Прочее`.
3. `fact_ids` внутри section хранят exact upstream ids и используются как coverage ledger.
4. `facts` — это только narrative-safe text inputs для final writer. Они могут быть подмножеством `fact_ids`, если часть покрытия уходит в `literal_items`.
5. Если `program_list` fact содержит `literal_items`, compose должен:
   - вынести literal items в `literal_items`;
   - убрать их из narrative-safe `facts`, когда это возможно;
   - записать deterministic `coverage_plan`, чтобы было видно, чем покрывается каждый `fact_id`.
6. `absorbed_by_list` допустим только когда отдельный fact не добавляет нового narrative residue поверх уже вынесенного verbatim list.
7. Если source wording явно говорит, что literal list выборочный (`и другие`, `и др.`, `среди которых`, `в том числе`), compose ставит `literal_list_is_partial = true`.
8. `writer_pack.select.v1` в текущем contract не строит варианты и возвращает identity payload.

## Final `writer.final_4o` handoff

Финальный writer должен получить один selected pack и соблюдать:

- идти по `sections` строго по порядку;
- использовать `event_type` как explicit format signal, если lead facts сами звучат как film/project reference note;
- использовать `literal_items` verbatim, без перефразирования элементов списка;
- трактовать `literal_list_is_partial = true` как presentation signal для non-exhaustive framing;
- не переносить `infoblock` logistics обратно в narrative prose;
- использовать `title_context.is_bare` и `hint_fact_text` только как title/lead hint, а не как повод для нового fact selection;
- семантически покрыть все `constraints.must_cover_fact_ids`, опираясь на `coverage_plan`.
