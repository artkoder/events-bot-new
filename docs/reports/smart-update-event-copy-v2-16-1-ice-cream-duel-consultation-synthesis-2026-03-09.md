# Smart Update Event Copy V2.16.1 Ice-Cream Duel Consultation Synthesis

Дата: 2026-03-09

Основание:

- [v2.16.1 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-design-brief-2026-03-09.md)
- [v2.16.1 ice-cream duel dry-run](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-1-ice-cream-duel-2026-03-09.md)
- `Opus` review: `artifacts/codex/reports/event-copy-v2-16-1-ice-cream-duel-opus-2026-03-09.md`
- `Gemini` review: `artifacts/codex/reports/event-copy-v2-16-1-ice-cream-duel-gemini-3.1-pro-preview-2026-03-09.md`

## 1. Что именно было проверено

Первый `Smart Update duel` для `v2.16.1` был проведён не как offline rewrite, а как shadow-сравнение двух движков на одном и том же event-level fact floor:

- `Smart Update` — текущий baseline runtime;
- `Smart Update ice-cream` — dry-run candidate, собранный из:
  - generic generation-contract из `2.15.3`;
  - точечных atomic-путей из `2.15.8`.

Покрыты 6 реальных событий с накопленными фактами, включая:

- `presentation`
- `lecture`
- `concert`
- `screening`
- `party` с secondary-source enrichment
- `theater` с `4` источниками

## 2. Сухой результат duel

- Событий: `6`
- Baseline total deterministic missing: `29`
- Ice-cream total deterministic missing: `32`

По кейсам:

| Event | Type | Baseline missing | Ice-cream missing | Предварительный вердикт |
|---|---|---:|---:|---|
| `2673` | presentation | 5 | 6 | baseline лучше |
| `2687` | lecture | 4 | 3 | mixed, локальный win ice-cream |
| `2734` | concert | 5 | 3 | mixed по счётчику, но baseline практичнее |
| `2659` | screening | 4 | 8 | baseline лучше с большим запасом |
| `2731` | party / multi-source | 7 | 7 | baseline лучше по полезности |
| `2498` | theater / 4-source | 4 | 5 | mixed, локальный stylistic win ice-cream |

Главный вывод уже на сухих числах:

- `ice-cream` пока не превосходит baseline даже по суммарному `missing`;
- локальные win есть, но они не перекрывают dangerous regressions.

## 3. Консенсус Opus и Gemini

Обе внешние модели практически сошлись:

- `ice-cream` в текущем виде не готов к runtime integration;
- baseline сейчас лучше держит fact-first инварианты;
- главные проблемы `ice-cream` не косметические, а архитектурные.

Совпадающие verdicts:

- `2659 screening` — критический провал `ice-cream` из-за world-knowledge bleed;
- `2673 presentation` — `ice-cream` не просто теряет факты, а выдумывает ответы на forward-looking facts;
- `2731 party` — tie по `missing` скрывает фактический проигрыш `ice-cream`, потому что теряются цена и playlist-link;
- `2734 concert` — нельзя считать кейс выигранным, если из текста выпал setlist;
- baseline нельзя заменять новой веткой без жёсткого post-generation grounding gate.

## 4. Что признано самым опасным

### 4.1. World-knowledge bleed

Самый опасный провал — `2659`.

`ice-cream` сгенерировал полноценный plot-summary романа/фильма:

- имена персонажей;
- интерпретацию сюжета;
- экзистенциальные выводы;
- длинную narrative arc, которой в фактах события не было.

Это disqualifying failure.

### 4.2. Fabrication of forward-looking facts

На `2673` факты говорят:

- на презентации расскажут, зачем проект появился;
- на презентации расскажут, какую проблему он решает.

`ice-cream` не сохранил это как promise-of-coverage, а придумал конкретные ответы.

Это отдельный класс hallucination, особенно опасный для fact-first pipeline.

### 4.3. Потеря secondary-source enrichment

На `2731` и частично на `2673` потерялись факты, ради которых вообще нужен multi-source accumulation:

- price;
- playlist URL;
- pre-registration;
- tea/cookies;
- другие support/logistics facts.

Если generation слой их выбрасывает, event-level accumulation теряет смысл.

### 4.4. Потеря shape-critical facts

На `2734` `ice-cream` дал более компактный prose, но выпал setlist.

Для концерта это неприемлемо: list of songs — high-value content, а не optional detail.

## 5. Что в ice-cream действительно стоит сохранить

Новый dry-run всё же не пустой.

Подтверждённые локальные плюсы:

- на `2687` lead и flow живее baseline;
- на `2498` heading-style у `ice-cream` интереснее и ближе к human-written copy;
- generic path действительно умеет давать менее повторяющийся prose;
- у ветки есть реальный потенциал в pattern-driven heading generation и более живом lead-writing.

Но эти плюсы можно переносить только вместе с safety rails.

## 6. Решение по ветке после первого duel

На текущем этапе:

- `Smart Update` остаётся единственным baseline runtime;
- `Smart Update ice-cream` остаётся только в `dry-run / duel`;
- `screening_card` ветку нельзя продвигать в runtime в текущем виде;
- текущий generic `2.15.3` path нельзя считать достаточно универсальным без более жёстких micro-contracts.

То есть `v2.16.1` продолжает жить как safe comparison branch, а не как replace-candidate.

## 7. Что тюнить дальше

Консенсусная следующая цель не в том, чтобы "написать ещё живее".

Следующий шаг должен быть таким:

1. Сохранить baseline Smart Update без изменений.
2. В `ice-cream` усилить generation-contract, а не переписывать заново весь runtime.
3. Вернуть жёсткое разделение:
   - `narrative core`
   - `infoblock / logistics / list facts`
4. Добавить anti-hallucination fence во все generation prompts.
5. Добавить post-generation fact audit / orphan-fact gate.
6. Перейти от одного generic path к shape-aware micro-contracts хотя бы для:
   - `presentation/project`
   - `lecture/person`
   - `concert/program-rich`
   - `screening`
   - `multi-source list-heavy / enrichment-heavy`

## 8. Практический next round

Следующий тюнинг-раунд должен идти не с rewrite, а с reuse already-proven pieces:

- из baseline:
  - fact-first event rebuild
  - headings/lists/blockquote preservation
  - support/logistics retention
- из `2.15.8 - 2.15.10`:
  - grounding audit idea
  - deterministic routing lessons
- из `2.15.3`:
  - modular prompt assembly, но уже без permissive generic prose mode

Первый новый экспериментальный пакет должен быть таким:

- `shape_aware_layout_plan`
- `generate_narrative_core`
- `assemble_logistics_block`
- `post_generation_fact_audit`

И только после этого — новый duel против baseline на том же harness.

## 9. Итоговый status

`v2.16.1` после первого duel не проиграл идеологически, но проиграл practically.

Полезный outcome этого раунда:

- baseline сохранён;
- duel harness собран;
- `Opus` и `Gemini` дали согласованный red-flag verdict;
- теперь уже ясно, что следующий шаг — не "общий новый красивый текст", а жёсткое встроенное ограничение генерации внутри fact-first Smart Update.

