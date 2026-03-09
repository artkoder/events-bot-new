# Smart Update Event Copy V2.13 Post-Run Consultation Synthesis

Дата: 2026-03-08

Связанные материалы:

- `artifacts/codex/tasks/event-copy-v2-13-postrun-complex-consultation-brief.md`
- `artifacts/codex/reports/event-copy-v2-13-postrun-complex-consultation-claude-opus.md`
- `artifacts/codex/reports/event-copy-v2-13-postrun-complex-consultation-gemini-3.1-pro.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-review-2026-03-08.md`
- `docs/reports/smart-update-gemma-event-copy-v2-13-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-5-events-2026-03-08.md`

## 1. Bottom Line

Мой итог после `v2.13` dry-run, `Opus`, `Gemini` и локальной проверки такой:

- `v2.13` лучше baseline overall.
- `v2.13` не является чистым overall-win против `v2.12`.
- `v2.13` лучше `v2.12` как hygiene-safe candidate.
- `v2.13` хуже `v2.12` на части rich cases по factual sharpness.
- `v2.13` всё ещё не готов к production rollout.

Чётко:

- baseline total missing = `22`
- `v2.12` total missing = `14`
- `v2.13` total missing = `14`
- `v2.13 forbidden = 0`

То есть:

- относительно baseline это реальный прогресс;
- относительно `v2.12` это mixed round: тот же total missing, но лучше hygiene и неравномерное качество на rich cases.

## 2. Что подтвердилось локально

### 2.1 Coverage metric действительно слишком буквальный

Тезис `Opus` и `Gemini` о broken coverage metric я проверил по коду.

В [smart_event_update.py](/workspaces/events-bot-new/smart_event_update.py#L1457) `_find_missing_facts_in_description(...)` делает проверку через:

- `_norm_text_for_fact_presence(...)`
- затем обычный `needle not in desc_n`

В [smart_event_update.py](/workspaces/events-bot-new/smart_event_update.py#L1390) нормализация:

- casefold
- `ё -> е`
- упрощение кавычек/тире/пробелов

Но там нет:

- лемматизации;
- морфологической нормализации;
- синонимического матчинга.

Быстрый локальный reproduction на `2745` это подтверждает:

- факт `камерный спектакль, атмосфера интимности`
- текст `Камерная постановка создает атмосферу интимности`

текущий checker всё равно помечает как missing.

Это означает:

- метрика реально завышает missing;
- и часть прошлых dry-run verdict была искажена;
- но это не отменяет реальные prose/factual problems на `2734` и `2673`.

### 2.2 Реальные текстовые проблемы всё равно остались

Даже если discount false negatives:

- `2734` действительно слишком sentimental / editorialized;
- `2734` реально теряет usefulness, потому что song list не доезжает в финальный текст;
- `2673` действительно explanation-heavy и generic;
- `2687` действительно weaker than `v2.12` по структуре и specificity.

То есть `Opus` прав про broken compass, но неправ, если из этого делать вывод, что основной remaining problem только в метрике.

## 3. Где Opus и Gemini совпали

Есть сильная зона согласия, и я её принимаю.

### 3.1 `full-floor normalization` откатывать не надо

Обе внешние модели не советуют возвращаться к старому `subset extraction -> merge-back` пути.

Это согласуется и с моими локальными выводами.

### 3.2 `Targeted repair` в текущем виде слаб

Обе модели считают текущий repair-pass малополезным:

- он завязан на шумную метрику;
- он лечит симптом, а не причину;
- он может ухудшать already-good prose.

Я принимаю этот вывод.

### 3.3 Normalizer сейчас слишком агрессивно режет content on rich cases

Особенно важно:

- program lists;
- named items;
- structured agenda / capability blocks.

Это явно видно на `2734`, и я принимаю это как реальный blocker.

### 3.4 `v2.13` лучше baseline

И `Opus`, и `Gemini`, и мой собственный review сходятся:

- baseline слишком шаблонен;
- `v2.13` уже заметно лучше по prose hygiene;
- `2660` и `2745` это подтверждают.

## 4. Где я не принимаю ответы на веру

### 4.1 Я не принимаю тезис Opus “почини метрику, потом думай дальше” как достаточный

Да, метрический слой broken.

Но:

- `2734` остаётся плохим текстом даже при human reading;
- `2673` остаётся generic even without literal checker;
- `2687` реально стал flatter than `v2.12`.

Поэтому `metric fix` нужен, но сам по себе не решает core quality problem.

### 4.2 Я не принимаю в лоб тезис Gemini, что shape taxonomy почти не нужна

`Gemini` слишком резко редуцирует проблему к `fact density`.

Я согласен, что:

- density — более сильный фактор, чем я считал раньше.

Но не согласен, что shape не важен вообще:

- `presentation_project` и `lecture_person` всё же ломаются не совсем одинаково;
- у них разные preservation risks;
- значит shape-aware contracts пока полностью выбрасывать не надо.

### 4.3 Я не принимаю literal “regex-first” или pure deterministic semantics

Это против product reality:

- бот работает на тысячах heterogeneous posts;
- semantics надо держать через LLM;
- deterministic слой должен быть support-only.

## 5. Реальный verdict по `v2.13`

### Against baseline

`v2.13` лучше baseline overall.

Почему:

- лучше total missing;
- zero forbidden;
- более естественный и профессиональный prose на лучших кейсах;
- меньше шаблонных heading patterns;
- сильнее sparse/event-copy quality.

### Against `v2.12`

`v2.13` не является clean overall win against `v2.12`.

Мой вердикт такой:

- как `quality-safe branch`, `v2.13` немного лучше;
- как `rich-case factual branch`, `v2.13` пока хуже.

Расшифровка:

- `2660`: `v2.13` лучше или не хуже `v2.12`
- `2745`: `v2.13` лучше `v2.12`
- `2734`: `v2.13` не лучше `v2.12`
- `2687`: `v2.13` хуже `v2.12`
- `2673`: `v2.13` хуже `v2.12`

То есть current branch still not rollout-ready.

## 6. Что я беру в `v2.14`

Следующий шаг я теперь вижу так.

### 6.1 Сразу поправить evaluation harness

Не production text generation, а именно evaluation layer.

Нужно:

- убрать exact-substring dogma;
- добавить более tolerant fact-presence check для русского;
- минимум: morphology-aware support;
- допустимо: separate LLM judge only for evaluation, если deterministic вариант окажется слишком слабым.

Это не нарушает `LLM-first` policy, потому что речь о measurement/support layer.

### 6.2 Упростить, но не откатывать normalization

Normalizer в `v2.14` должен:

- чистить framing;
- сохранять lists / names / titles / concrete program details;
- не убивать structured content вроде `в программе: ...`.

Иными словами:

- `clean & preserve`, а не `clean & erase`.

### 6.3 Убрать `targeted repair` из default path

В следующем experimental round я бы не делал `targeted repair` обязательным.

Вместо этого:

- лучше один сильный constrained generation;
- и отдельный validation layer после него;
- repair только при реально подтверждённом issue, а не по шумному missing-list.

### 6.4 Усилить generation против editorial drift

Здесь я принимаю core advice от `Gemini`:

- generation нужен более жёсткий anti-marketing / anti-sentimental contract;
- особенно для rich cases.

Но я не хочу превращать prompt в wall of bans again.

Что беру:

- более чёткая editor persona;
- прямой ban на promotional filler;
- прямой ban на profession hallucination;
- explicit preservation of concrete lists/items.

### 6.5 Допускаю split-call для rich events

Здесь user guidance была верная: не быть слишком консервативным.

Если для real quality gain нужен split, это допустимо.

Мой current bias:

- sparse cases оставлять короткими и однопроходными;
- для richer cases протестировать `normalize/group -> constrained assembly`;
- но делать это как narrow experiment, а не сразу как новый production truth.

## 7. Рабочий образ `v2.14`

На сегодня я вижу `v2.14` как такой experimental plan:

1. `raw_facts`
2. shape/density detection
3. `clean-and-preserve normalization`
4. deterministic dedup / hygiene
5. generation:
   - compact single-pass for sparse
   - possibly bucketed/constrained assembly for rich
6. post-generation validation
7. repair only on truly validated issues

Ключевая цель:

- не просто снова держать `missing ~= 14`;
- а реально улучшить `2734 / 2687 / 2673` без отката prose wins на `2660 / 2745`.

## 8. Final Decision

Консультационный цикл по `v2.13` можно считать закрытым.

Я не вижу смысла делать ещё один внешний раунд до новой локальной итерации.

Следующий разумный шаг:

- собрать `v2.14` с учётом этого synthesis;
- прогнать те же 5 кейсов;
- и уже потом снова оценить результат против baseline, `v2.12` и `v2.13`.
