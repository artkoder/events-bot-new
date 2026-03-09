# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.8 Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-8-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_8_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_8_2026_03_07.py`
- `docs/reports/smart-update-gemini-event-copy-v2-8-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-hypotheses-consultation-response-review.md`

## 1. Краткий verdict

`v2.8` не стал quality win.

Это не катастрофа уровня `v2.7`, но practical result всё равно отрицательный:

- runtime candidate: нет;
- как corrective round против `v2.7`: partial;
- как реальное улучшение относительно `v2.6`: нет.

Главный итог:

- `2734` заметно оздоровился;
- все остальные кейсы либо ухудшились, либо не решили core target;
- ветка системно потеряла sparse routing и снова раздула extraction.

## 2. Что сломалось системно

### 2.1. `v2.8` снова выбил все 5 кейсов в `fact_first`

Это самый жёсткий structural signal.

Итоговые counts:

- `2660`: `facts=8`
- `2745`: `facts=7`
- `2734`: `facts=4`
- `2687`: `facts=10`
- `2673`: `facts=12`

Даже `2660` и `2745`, которые реально выигрывали в более компактных ветках, здесь ушли в full fact-first path.

То есть rollback к dense extraction не восстановил прежнюю компактность, а наоборот усилил branch inflation.

### 2.2. Simplified hints оказались слишком слабыми

`v2.8` убрал опасные sentence templates, но вместе с этим потерял и часть управляющей силы.

Практически это видно так:

- в `2660` survive-нул `label-style` факт `Тема: теме противоречий мира.`;
- в `2687` survive-нуло `Лекция расскажет о ...`;
- в `2673` survive-нули сразу несколько `На презентации расскажут ...`.

То есть simplified hints не удержали extraction в clean factual zone.

### 2.3. Generation не может компенсировать плохой facts layer

`template-overuse control` сам по себе не спасает.

Если в `facts_text_clean` уже попали:

- label-style facts;
- explainy packaging;
- service/event framing;
- repeated intent facts;

то generation/revise не успевают надёжно всё это переварить без новых потерь.

## 3. Case-by-case

### 3.1. `2660`: regression after `v2.6`

Это явный regression case.

- `v2.6 missing=0`
- `v2.8 missing=4`

Проблемы:

- кейс снова ушёл в full fact-first вместо более удачной compact-ish dynamics;
- в facts попал broken label-style факт `Тема: теме противоречий мира.`;
- description повторяет один и тот же смысл разными формулировками;
- prose стала более explanatory и менее точной.

### 3.2. `2745`: regression remains severe

- `v2.6 missing=3`
- `v2.8 missing=6`

Проблемы:

- sparse case снова inflated до `facts=7`;
- description звучит clean, но бедно и обобщённо;
- core themes partly preserved, но density и точность ниже удачных раундов.

### 3.3. `2734`: единственный явный gain

Это лучший outcome раунда.

- `v2.7 missing=5`
- `v2.8 missing=2`
- forbidden markers ушли.

Но и здесь не всё чисто:

- extraction сжался до `facts=4`;
- остался label-style факт `Тема: великой любви ...`;
- часть program richness всё равно выглядит схлопнутой.

То есть это real improvement, но не идеальный template для переноса.

### 3.4. `2687`: главный target всё ещё не закрыт

- `v2.6 missing=5`
- `v2.7 missing=4`
- `v2.8 missing=4`
- forbidden `посвящ*` всё ещё жив

Проблемы:

- facts inflated до `10`;
- body лучше покрывает имена и связи, но core hygiene failure не устранён;
- лекционный канцелярит partly survives уже на facts layer.

### 3.5. `2673`: всё ещё слабый dense-social case

- `v2.6 missing=4`
- `v2.7 missing=5`
- `v2.8 missing=6`

Проблемы:

- facts inflated до `12`;
- intent frames survive в raw form;
- final text снова уходит в service tone вроде `Презентация расскажет...`;
- practical quality ниже даже относительно `v2.6`.

## 4. Что `v2.8` всё же доказал полезного

### 4.1. Чистый rollback лучше `v2.7`

`v2.8` подтвердил, что `v2.7` действительно был placement error.

Rollback убрал worst-case `safe-positive packaging` вроде:

- `Выставка носит название ...`
- избыточные `В центре встречи — ...`

То есть сам диагноз про ошибочный перенос narrative shaping в extraction был верным.

### 4.2. Но simple rollback недостаточен

Это главный practical вывод.

Откат к `dense data points`:

- убрал часть artificial packaging;
- но не вернул нужную discipline extraction;
- и не вылечил `посвящ*` / intent-heavy кейсы.

## 5. Root cause after `v2.8`

### 5.1. Extraction сейчас oscillates между двумя плохими режимами

- слишком shaping-heavy (`v2.7`)
- слишком loosely factual without enough control (`v2.8`)

Нужен третий режим:

- плотные self-contained facts;
- без narrative wrappers;
- но и без выживания label-style / intent-style / topic-tag facts.

### 5.2. `compact` / `fact_first` routing всё ещё слишком чувствителен к extraction noise

Пока 1-2 лишних факта легко выталкивают кейс из compact branch, любые extraction regressions мгновенно бьют по конечному качеству.

### 5.3. `посвящ*` нельзя лечить только revise

`2687` показывает, что even stronger human-readable revise issue всё ещё не гарантирует clean final output.

Значит, проблема живёт раньше: в extraction/facts layer и, возможно, в missing-repair/revise interplay.

## 6. Bottom line

`v2.8` — useful evidence round, но не candidate round.

Практический вывод:

- **accept as evidence**: да;
- **accept as candidate**: нет;
- **нужен новый Gemini round**: да, уже по реальному `v2.8` failure/gain mix.

Самый важный вопрос для следующей консультации:

- как получить `dense but controlled extraction`, который не раздувает facts,
- не пропускает label/intent garbage,
- и при этом не возвращает narrative shaping обратно в Extraction.
