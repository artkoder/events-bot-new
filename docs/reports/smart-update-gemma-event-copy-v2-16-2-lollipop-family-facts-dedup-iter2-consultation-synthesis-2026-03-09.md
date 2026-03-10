# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Dedup Iter2 Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

Этот synthesis закрывает `facts.dedup iter2` после:

- retune `baseline_diff.v2`;
- полного `12`-event run на текущем extract casebook;
- post-run consultation через `Opus` и `Gemini`.

Материалы:

- iter2 harness: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_dedup_family_v2_16_2_iter2_2026_03_09.py`
- iter2 raw JSON: `/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_dedup_family_v2_16_2_iter2_2026-03-09.json`
- iter2 lab report: `/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-dedup-lab-iter2-2026-03-09.md`
- consultation brief: `/workspaces/events-bot-new/artifacts/codex/tasks/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter2-brief-2026-03-09.md`
- `Opus`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter2-opus-2026-03-09.md`
- `Gemini`: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter2-gemini-3.1-pro-preview-2026-03-09.md`
- `Gemini` stderr: `/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-dedup-family-postrun-consultation-iter2-gemini-3.1-pro-preview-2026-03-09.stderr.txt`

## 2. What Iter2 Actually Improved

`iter2` действительно исправил главные prompt-level симптомы `iter1`:

- `REFRAME` перестал быть мёртвой категорией: `events_with_reframe = 10/12`;
- enrichment count перестал выглядеть квотированным: `enrichment_stddev = 2.976`;
- full fact accounting сохранён на всём кейсбуке;
- `cross_enrich` не дал дубликатов, но pipeline от этого не ломается.

Главный числовой итог:

```json
{
  "events": 12,
  "avg_enrichment_count": 3.75,
  "avg_reframe_count": 1.333,
  "events_with_reframe": 10,
  "enrichment_stddev": 2.976,
  "avg_auto_reclassified_missing_match_count": 1.0,
  "events_with_flags": 7
}
```

То есть prompt retune сработал, но раскрыл новый bottleneck.

## 3. Shared External Verdict

`Opus` и `Gemini` сошлись в главном:

- `facts.dedup` как architecture остаётся правильной;
- `facts.merge` пока рано;
- текущий блокер сидит не в общей схеме, а в cleaning/validation around `baseline_match`;
- нынешний `auto_reclassified_missing_match` создаёт false enrichments и загрязняет downstream.

Обе модели дали статус:

- `facts.dedup iter2 architecture`: `GO`
- `facts.merge now`: `NO-GO`

## 4. Where Consultants Disagree

Они расходятся не в диагнозе, а в способе фикса.

### 4.1. Opus

`Opus` считает, что надо менять контракт матча:

- отказаться от exact-text `baseline_match`;
- перейти на `baseline_match_index`;
- baseline facts нумеровать в prompt;
- cleaner валидирует индекс, а не строковое совпадение.

Плюс `Opus` отдельно подчёркивает:

- текущие false enrichments в `2734` почти целиком состоят из logistics/service facts;
- это не prompt misclassification, а brittle validation.

### 4.2. Gemini

`Gemini` соглашается с диагнозом, но спорит с переходом на индекс:

- индексный контракт он считает более хрупким для LLM из-за off-by-one и неверных ссылок;
- рекомендует сохранить текстовый `baseline_match`;
- но ослабить cleaner до normalized/fuzzy match вместо строгого `set.__contains__`.

Его версия следующего шага:

- lowercasing;
- punctuation stripping;
- substring / similarity check;
- только затем downgrade to `ENRICHMENT`.

## 5. My Synthesis

С кодом и артефактами лучше согласуется такой вывод:

- проблема действительно mechanical, не architectural;
- сейчас мы уже не в ситуации, где надо опять тюнить большой prompt pack;
- следующий раунд должен бить по validation layer, а не по extraction и не по `cross_enrich`.

Я не считаю разумным сразу тащить новую index-schema как основной путь. Это уже не micro-fix, а новый contract fork. Для ближайшего шага безопаснее и дешевле:

1. оставить `baseline_diff.v2` prompt как есть;
2. сделать `cleaner_match_relax`:
   - normalize punctuation/case/whitespace;
   - exact-normalized match first;
   - then narrow fuzzy acceptance only for `COVERED/REFRAME`;
3. rerun на тех же `12` событиях;
4. смотреть, падает ли `auto_reclassified_missing_match_count` к почти нулю без роста false covered.

Если это не сработает, тогда уже поднимать challenger-ветку с `baseline_match_index`.

То есть мой next-step verdict:

- primary path: `baseline_diff.v2.1 + cleaner_match_relax`
- challenger path, not mainline yet: `baseline_diff.v3_index_match`

## 6. Event-Level Interpretation

Ключевые кейсы iter2:

- `2734` concert: hotspot сидит в `facts.extract_support.v1`, где service/logistics facts почти все уходят в auto-reclassified bucket;
- `2732` party: agenda-enrichment в целом хороший, но часть baseline-adjacent facts всё ещё не удерживает match;
- `2673` presentation: category split уже живой, но на card/date facts cleaner всё ещё слишком хрупкий;
- `2747` screening: `reframe_zero` здесь не выглядит поломкой сам по себе, это скорее shape with near-total coverage.

Значит следующий раунд должен фокусироваться не на всех event types сразу, а на match validation for:

- date/time facts;
- venue facts;
- age limits;
- ticket/logistics lines.

## 7. Decision

Текущий статус:

- `facts.dedup iter2`: `GO as architecture, NO-GO as merge-ready output`
- `facts.merge`: `NO-GO`

Следующий обязательный шаг:

- не новый extract run;
- не merge;
- не broad prompt rewrite;
- а узкий `baseline_diff cleaner` retune с повторным `12`-event rerun.

## 8. Next Step

Канонический следующий раунд:

1. `facts.dedup.baseline_diff.v2.1`
2. `cleaner_match_relax`
3. same `12`-event casebook
4. post-run check:
   - `auto_reclassified_missing_match_count` near zero
   - `reframe_count` stays alive
   - no drop in fact accounting
   - no suspicious surge in `COVERED`

Только после этого можно снова поднимать вопрос `GO / NO-GO` для `facts.merge`.
