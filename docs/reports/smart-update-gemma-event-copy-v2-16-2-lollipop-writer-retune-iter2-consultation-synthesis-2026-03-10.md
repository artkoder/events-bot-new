# Smart Update Gemma Event Copy V2.16.2 Lollipop Writer Retune Iter2 Consultation Synthesis

Дата: 2026-03-10

## 1. Inputs

- baseline comparison synthesis: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-final-vs-baseline-synthesis-2026-03-10.md`
- `Opus` retune consultation: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-retune-iter2-consultation-opus-2026-03-10.raw.json`
- `Gemini 3.1 Pro Preview` critique: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-writer-retune-iter2-consultation-gemini-3.1-pro-preview-2026-03-10.raw.json`

## 2. Consensus

- Оба консультанта сошлись на архитектурной границе: semantic triage должен рождаться в `facts.prioritize`, а не в `writer_pack.compose`.
- Для public prose нужен бинарный сигнал `narrative_policy = include|suppress`; soft-варианты вроде `avoid` признаны опасными.
- `writer_pack.compose` должен только потреблять upstream suppress и передавать presentation metadata для literal lists.
- `writer.final_4o` должен решать presentation-layer задачу: как безопасно подать partial literal list без ложной исчерпываемости.

## 3. Landed Change Set

### `facts.prioritize`

- Добавлен deterministic `narrative_policy: suppress` для cross-promo support facts с несколькими `date -> other event title` парами.
- Добавлен deterministic suppress для low-specificity support fillers, когда у события уже есть достаточный набор более сильных `high/medium` facts.
- Для exhibition/history класса добавлен узкий rescue-path: если в `raw_facts` уже есть сильный исторический контекст про Великую депрессию или Вторую мировую, он возвращается в `support_context` до weighting и не остаётся потерянным только из-за downstream selection.

### `editorial.layout` and `writer_pack.compose`

- `editorial.layout` теперь строит downstream fact pack только из `include` facts; suppressed facts не попадают в blocks и не доживают до `must_cover_fact_ids`.
- `writer_pack.compose` добавляет `literal_list_is_partial` для секций с `literal_items`, если source fact содержит маркеры вроде `и другие`, `и др.`, `среди которых`, `в том числе`.
- Literal program coverage остаётся deterministic через `coverage_plan`; новый partial-flag ничего не меняет в grounding, только в public presentation.

### `writer.final_4o`

- Prompt теперь требует non-exhaustive framing для `literal_list_is_partial = true`.
- Validator блокирует partial list без явного intro-marker и сохраняет прежние guardrails против голых списков, infoblock leakage и literal item loss.

## 4. Event-Level Intent

- `2498`: убрать cross-promo leakage, но сохранить безопасный residue вроде `средняя сцена`, если он нужен и есть отдельно.
- `2657`: немедленно убрать слабые filler facts; strongest win возможен только если исторический контекст уже существует upstream и может быть rescued/prioritized.
- `2734`: оставить grounded literal list, но явно маркировать её как примеры, а не полный репертуар.

## 5. Scope Limit

- Этот retune закрывает writer-tail regressions, которые реально можно исправить без полного reopen downstream architecture.
- Если после нового comparison round `2657` всё ещё останется плоским, следующий reopen должен идти не в `writer.final_4o`, а обратно в `facts.extract` / `facts.merge` для exhibition-history fact recovery.

## 6. Follow-Up Run Status

Тот же downstream rerun уже выполнен:

1. `editorial.layout iter2` на `facts.prioritize iter3`
2. `writer_pack.compose iter2`
3. `writer.final_4o iter2`

Канонические lab reports:

- `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-editorial-layout-lab-iter2-2026-03-10.md`
- `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-writer-pack-compose-lab-iter2-2026-03-10.md`
- `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-writer-final-4o-lab-iter2-2026-03-10.md`

Итог rerun:

- `editorial.layout iter2`: `events_with_flags = 0`
- `writer_pack.compose iter2`: `events_with_flags = 0`
- `writer.final_4o iter2`: `events_with_errors = 0`, `events_with_warnings = 0`

Следующий ещё не закрытый шаг: повторить тот же `12`-event baseline comparison round поверх новых `iter2` outputs.
