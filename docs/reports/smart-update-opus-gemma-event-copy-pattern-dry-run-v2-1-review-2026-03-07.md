# Smart Update Opus Gemma Event Copy Pattern Dry Run V2.1 Review — 2026-03-07

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-1-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_1_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_1_2026_03_07.py`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_5events_2026-03-07.json`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-v2-dryrun-quality-consultation-response-review.md`

## 1. Краткий verdict

`v2.1` не стал quality win.

Это всё ещё **не candidate для переноса в код бота**.

Более того, по сумме результатов `v2.1` сейчас выглядит:

- хуже `v2` на `2660`, `2745`, `2734`, `2687`;
- лучше `v2` только на `2673`, но всё равно хуже `v1` по coverage;
- заметно дороже по runtime: `538.8s` против `371.3s` у `v2`.

То есть узкий patch pack из `v2.1` не решил главные проблемы и открыл новые.

## 2. Event-by-event

### 2.1. `2660`

`v2.1` ухудшил кейс.

- sparse event ушёл в `fact_first_v2_1` из-за лишнего publishable fact;
- в description появился отдельный service-like блок `### Продолжительность` с датой;
- `forbidden = date_ru_words`;
- prose стал более “литературным”, но и более editorialized:
  - `многогранность форм и текстур`
  - `красоту в несовершенстве`
  - `гармонию в конфликте`
- missing не улучшился: `3`.

Итог: `baseline > v2 ≈ v2.1`, при этом `v2.1` ещё и грязнее по forbidden markers.

### 2.2. `2745`

Это самый явный regression.

- extraction repair затащил в `facts_text_clean` сервисную detail:
  - `Дополнительно добавлено 7 мест ... запись через Веронику Вишнёвую`
- duplicate core fact survived:
  - `Спектакль о взаимоотношениях...`
  - `Спектакль рассказывает о взаимоотношениях...`
- branch ушёл в `fact_first_v2_1`, хотя сам кейс по сути остался sparse;
- итоговый текст включает логистику, которую вообще не нужно было тащить в narrative description;
- diagnostics: `missing=6`, `forbidden=['tickets', 'date_ru_words']`.

Итог: `v2.1` здесь хуже baseline, хуже `v1` и хуже `v2`.

### 2.3. `2734`

`v2.1` не решил ни одну из двух главных проблем `v2`.

- `посвящ*` остался и в facts, и в final description;
- tracklist не восстановлен, хотя repair hints это явно требовали;
- `copy_assets.program_highlights` уже содержит названия песен, но `facts_text_clean` их не содержит;
- значит extraction / copy_assets расходятся, а generation всё равно строится по слабому facts set;
- diagnostics: `missing=4`, что хуже `v2 missing=3`.

Итог: `v1 > v2 > v2.1`.

### 2.4. `2687`

`v2.1` снова не решил Opus P0/P1 цели.

- `посвящ*` остался;
- whole-body metatext тоже остался:
  - `Мероприятие представляет собой лекцию...`
  - `Лекция фокусируется...`
- repair pass добавил service-like fact:
  - `Мероприятие проходит в формате лекции`
- hallucination / unsupported embellishment никуда не делись:
  - `Наталья Гончарова и Ольга Розанова внесли вклад в развитие авангарда`
  - `Любовь Попова искала новые формы и смыслы`
- diagnostics: `missing=4`, хуже `v2 missing=3`, сильно хуже `v1 missing=1`.

Итог: `v1` всё ещё сильнее, несмотря на свои старые недостатки.

### 2.5. `2673`

Это единственный частичный плюс `v2.1`.

- `missing` улучшился относительно `v2`: `6 -> 4`;
- `forbidden=[]`;
- service/CTA hygiene чище, чем раньше.

Но проблем всё равно слишком много:

- duplicate facts survived прямо во входном наборе;
- description всё ещё generic и секционно шумный;
- heading `О платформе «Собакусъел»` остаётся weak/generic;
- lead и первая секция всё ещё дублируют смысл друг друга;
- секция `Для представителей креативной среды` остаётся filler;
- coverage всё равно хуже `v1 missing=1`.

Итог: это partial recovery against `v2`, но не победа.

## 3. Что сломалось системно

### 3.1. Extraction repair pass оказался ненадёжным

Это главный вывод `v2.1`.

Новый repair-pass должен был:

- убрать `посвящ*`;
- сжать inflated facts;
- вернуть program items;
- убрать service contamination.

На практике он:

- оставил `посвящ*` в `2734` и `2687`;
- не вернул tracklist в `2734`;
- протащил service detail в `2745`;
- оставил дубли в `2673`;
- добавил service/meta fact `Мероприятие проходит в формате лекции` в `2687`.

То есть сам новый `repair stage` пока нельзя считать trustworthy.

### 3.2. Preservation floor + repair дают branch inflation

На sparse cases это особенно заметно.

`2660` и `2745` ушли в standard branch не потому, что реально стали richer events, а потому что:

- facts set раздулся;
- в него попали лишние или service-adjacent строки;
- routing начал опираться на уже загрязнённый output.

Это ухудшает композицию и повышает риск forbidden leakage.

### 3.3. `copy_assets` и `facts_text_clean` живут разной жизнью

Самый наглядный кейс — `2734`.

Там:

- `copy_assets.program_highlights` уже знает треклист;
- `facts_text_clean` треклист не знает;
- generation строится по слабому facts set;
- итоговый текст получается слабее, чем мог бы.

Это значит, что contract между extraction output и generation source of truth всё ещё плохо согласован.

### 3.4. Generation prompt всё ещё позволяет editorial drift

На `2660` и `2687` видно, что даже после prompt-tightening модель всё ещё добавляет:

- оценочные обобщения;
- красивости без жёсткого grounding;
- semi-journalistic flourishes, которых нет в фактах.

То есть anti-embellishment rule в текущем виде всё ещё недостаточно “sticky” для Gemma.

## 4. Что в `v2.1` всё же полезно

Полностью провальным этот раунд тоже нельзя назвать.

Полезные сигналы есть:

1. `2673` показал, что stronger preservation может реально уменьшать catastrophic coverage loss.
2. Сам ход `extract -> repair -> generate` теоретически полезен, но текущая реализация repair-pass слабая.
3. Новый прогон очень явно локализовал главный bottleneck:
   - не generation-only;
   - а именно extraction contract + post-extraction trust model.

## 5. Нужен ли новый этап консультации с Opus

**Да.**

Но это уже должен быть не общий brainstorm и не повтор старого `v2 patch pack`.

Нужна **узкая критическая консультация по реальному failure of v2.1**.

Главный вопрос к Opus теперь такой:

- почему repair-pass не выполняет свои own issue hints;
- что из `v2.1` надо откатить;
- какой минимальный `v2.2` patch worth trying;
- и нужно ли вообще сохранять extraction-repair как отдельную стадию.

## 6. Следующий разумный шаг

1. Отправить Opus полный comparative пакет:
   - baseline
   - `v1`
   - `v2`
   - `v2.1`
2. Попросить не защищать прошлые рекомендации, если `v2.1` их опровергает.
3. Получить от Opus не просто critique, а:
   - corrected root-cause map;
   - `keep / modify / rollback / remove` по `v2.1` changes;
   - узкий `v2.2` patch plan.

## 7. Bottom line

`v2.1` не приблизил нас к implementation.

Он полезен как diagnostic round:

- мы увидели, что именно не работает;
- подтвердили, что extraction-side fixes критичны;
- и получили новый grounded пакет для консультации.

Но сам `v2.1` в текущем виде — **ещё не решение**.
