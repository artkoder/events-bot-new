# Smart Update Event Copy V2.16.1 Ice-Cream Duel Iteration 2 Consultation Synthesis

Дата: 2026-03-09

Основание:

- [duel iter2 report](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-ice-cream-duel-iter2-2026-03-09.md)
- [Opus review](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-16-1-ice-cream-duel-iter2-opus-2026-03-09.md)
- [Gemini failure note](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-16-1-ice-cream-duel-iter2-gemini-3.1-pro-preview-2026-03-09.md)

## 1. Что показал сам duel

Итерация `ice-cream iter2` стала заметно лучше первого duel-раунда:

- aggregate missing: `29 -> 22` в пользу `ice-cream`;
- aggregate enhanced missing: `31 -> 21` в пользу `ice-cream`;
- исчез structural collapse после assembly;
- profiling теперь реально показывает точку сбоя по шагам, а не только плохой финальный текст.

Но runtime-ready веткой это всё ещё считать нельзя:

- `2731` (`party_theme_program`) сохранил coverage-win, но дал `2` unsupported claims;
- `2498` (`theater_history`) сохранил coverage-win, но дал `1` unsupported claim;
- `2659` (`screening_card`) остался слабым и хуже baseline по missing;
- `2687` не дал убедимого выигрыша относительно baseline.

## 2. Что подтвердил Opus

Главный полезный вывод `Opus`: improvement real, но aggregate-цифры нельзя читать как “ветка почти готова”.

Сильные выводы review:

- profiling действительно стал полезным и позволяет fine-tune по stage-contracts;
- две блокирующие ошибки на `2731` и `2498` действительно сидят в `generate_narrative_core`, а не в assembly;
- `repair_narrative_core` слишком мягкий, потому что разрешает `удали или перефразируй`, а Gemma использует перефраз как loophole;
- в prompts не хватает прямого запрета на `fact expansion`, когда модель делает утверждение шире исходного факта;
- `screening_card` всё ещё overconstrained и поэтому теряет валидные facts;
- логика `primary_failure_stage` местами mislabels кейс как `normalize_fact_floor`, даже если downstream уже компенсировал ambiguity.

Самые сильные конкретные рекомендации `Opus`:

- в repair заменить `удали или перефразируй` на жёсткое `удали полностью`;
- в generation добавить прямое правило против расширения факта:
  - если факт говорит `не нужно учить слова`, нельзя писать `не потребуется подготовка`;
- для `theater_history` закрыть loophole вида `постановщики расскажут...`;
- для `party_theme_program` добавить negative examples, а не только абстрактный запрет на “ощущения участников”;
- в audit явно проверять `fact expansion`, а не только голые unsupported claims.

## 3. Что с Gemini

`Gemini` review в этой итерации не получен, несмотря на два отдельных запуска canonical command с моделью `gemini-3.1-pro-preview`.

Зафиксированный статус:

- provider error: `429 RESOURCE_EXHAUSTED`
- reason: `MODEL_CAPACITY_EXHAUSTED`
- это внешний capacity blocker, а не ошибка локального harness

Поэтому для этой итерации second opinion состоит только из `Opus` + локального synthesis.

## 4. Итоговое решение

Текущая ветка `ice-cream iter2`:

- сохраняется только как `dry-run/duel`;
- baseline `Smart Update` остаётся production эталоном;
- следующий раунд нужно делать не через rewrite, а через micro-tuning prompt-contracts.

Приоритет следующего раунда:

1. Ужесточить `repair_narrative_core` до delete-only поведения для unsupported content.
2. Добавить explicit anti-expansion rule в `generate_narrative_core`.
3. Закрыть loophole attribution/interpretation для `theater_history`.
4. Добавить few-shot style negative examples для `party_theme_program`.
5. Пересобрать `screening_card` вокруг positive allowlist, а не только запретов.
6. Исправить `primary_failure_stage`, чтобы ambiguity upstream считалась failure только если реально протекла в final miss.
