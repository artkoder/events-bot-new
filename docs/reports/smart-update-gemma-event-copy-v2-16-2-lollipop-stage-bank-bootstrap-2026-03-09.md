# Smart Update Gemma Event Copy V2.16.2 Lollipop Stage-Bank Bootstrap

Дата: 2026-03-09

Основание:

- [lollipop funnel design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-funnel-design-brief-2026-03-09.md)
- [lollipop salvage matrix](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-salvage-matrix-2026-03-09.md)
- [lollipop seed-bank consultation synthesis](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-seed-bank-consultation-synthesis-2026-03-09.md)
- [machine-readable stage bank bootstrap](/workspaces/events-bot-new/artifacts/codex/stage_bank/smart_update_lollipop_stage_bank_v2_16_2_2026-03-09.json)

## 1. Что сделано в этом раунде

Собран стартовый `stage bank` для `lollipop`.

В этом документе фиксируется не весь будущий pipeline, а стартовая исследовательская рамка:

- baseline `Smart Update` остаётся нетронутым;
- `lollipop` живёт только как `dry-run`;
- stage-bank собирается не как “по одной версии на шаг”, а как bank candidate-версий;
- работа идёт family-by-family;
- первая активная family: `facts.extract`.

## 2. Что здесь считается `stage bank`

Для `lollipop`:

- `stage registry` = весь каталог известных stage/version;
- `stage bank` = стартовый исследовательский набор families и candidate-версий, с которыми реально начинается funnel.

На старте банк должен хранить:

- provenance stage;
- evidence по кейсам;
- hypothesized fit по типам source/event;
- статус stage:
  - `active_candidate`
  - `deferred_outside_v1`
  - позже добавятся `retired`, `priority`, `rescue`.

## 3. Базовый принцип исполнения

`lollipop v1` строится как:

- broad candidate generation;
- потом `select / merge / priority`;
- затем следующий шаг.

Не как:

- early routing;
- одна guessed-ветка;
- один output.

Поэтому stage-bank с самого начала проектируется как multi-version bank.

## 4. Family research cycle

Каждая family проходит один и тот же цикл:

1. `broad_run`
2. `raw_family_review`
3. `narrow_consultation`
4. `prompt_expansion`
5. `rerun`
6. `shortlist`
7. `select_or_merge_design`
8. `integration`

Важно:

- `merge/select` не проектируется вслепую до того, как family созреет;
- сначала сохраняются и рассматриваются все raw outputs до мержа;
- именно на этом этапе появляется понимание, какие версии выигрывают на каких типах source/event.

## 5. Порядок families

Стартовый порядок сейчас такой:

1. `facts.extract`
2. `facts.type`
3. `facts.merge.tier1`
4. `facts.merge.tier2`
5. `facts.priority`
6. `hook.seed`
7. `hook.select`
8. `pattern.signal`
9. `layout.plan`
10. `pack.compose`
11. `pack.select`
12. `writer.final_4o.spec`

Это означает:

- сначала добиваем factual side;
- потом строим editorial side;
- `4o` подключается только после готового upstream pack.

## 6. Стартовый кейсбук

На старте зафиксирован кейсбук `12` событий:

Core `6`:

- `2673`
- `2687`
- `2734`
- `2659`
- `2731`
- `2498`

Extension `6`:

- `2747`
- `2701`
- `2732`
- `2759`
- `2657`
- `2447`

Идея такая:

- сначала family-lab идёт уже не на одном маленьком наборе, а на `6 core + 6 extension`;
- этого достаточно, чтобы увидеть первые реальные win patterns по типам source/event;
- и при этом объём ещё не уходит в слишком тяжёлый массовый прогон.

## 7. Что считается результатом этого bootstrap-раунда

Этот раунд не является первым broad-run.

Он даёт:

- machine-readable bootstrap stage bank;
- согласованный family order;
- зафиксированный research cycle;
- первую активную family-report ветку для `facts.extract`.

## 8. Следующий практический шаг

Следующий рабочий раунд уже не организационный:

- берётся `facts.extract family`;
- по ней собирается raw prompt inventory;
- запускается broad-run по casebook;
- сохраняются все outputs до any merge/select;
- после этого делается отдельный family review report.
