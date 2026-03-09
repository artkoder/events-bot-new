# Smart Update Opus Gemma Event Copy Final Implementation Calibration Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-final-impl-calibration-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-quality-first-calibration-response-review.md`
- `smart_event_update.py`

## 1. Краткий verdict

Этот ответ Opus сильный и полезный.
Но сам по себе он **ещё не закрывает pre-implementation phase**.

Почему:

- он хорошо докручивает новую quality-first схему;
- но почти не делает отдельного migration audit того, что уже полезно работает в текущем коде;
- а именно это сейчас критично, чтобы redesign не потерял сильные стороны существующего flow.

Поэтому мой вывод такой:

- ответ принимается как сильный implementation calibration;
- но **ещё один очень узкий консультационный этап перед кодом всё-таки нужен**;
- этот этап должен быть не про новые идеи, а про `preservation / migration matrix`.

И да, здесь я сознательно ставлю приоритет на качество текста.
Именно поэтому не хочу потерять удачные элементы текущего подхода под видом “чистого redesign”.

## 2. Что в ответе Opus действительно сильное

### 2.1. Dense fact handling наконец стало конкретным

Это реальное улучшение.

Раньше риск “все факты включены => wall of text” был назван, но не решён.
Здесь уже есть:

- density tiers;
- правила, когда использовать список;
- grouping strategy;
- отдельное правило для visitor conditions;
- здравая оговорка про instruction noise у Gemma.

Это можно прямо использовать как основу implementation.

### 2.2. Missing-facts repair placement описан инженерно

Opus не просто говорит “оставим repair layer”.
Он:

- указывает текущее место этого шага в merge flow;
- предлагает runtime placement для fact-first path;
- задаёт threshold;
- связывает это с TPM posture;
- вводит trigger-rate как health metric.

Это сильный practical improvement.

### 2.3. TPM-aware posture стал заметно лучше

Сильная коррекция по сравнению с прошлым ответом:

- исчезает фокус на micro-latency;
- появляется batch / throughput / throttle / repair-rate framing.

Это намного ближе к реальной operational задаче.

### 2.4. `evidence_span` для traceability — сильная идея

Из всего ответа это одна из самых полезных technical refinements.

По сравнению с простым word overlap это намного лучше:

- для русского;
- для OCR noise;
- для fallback logic;
- для объяснимости routing decisions.

Если это реализуемо достаточно стабильно, это сильно улучшит и `scene_cues`, и `contrast_or_tension`.

## 3. Что ответ всё ещё не закрывает

### 3.1. Он недостаточно защищает сильные стороны текущего runtime

Это главный remaining gap.

В текущем коде уже есть полезные вещи, которые нельзя случайно потерять:

1. `epigraph_fact` / `_pick_epigraph_fact`
- текущий flow уже умеет открывать текст цитатой или сильным фактом;
- это важный источник живости и композиционной вариативности.

2. `_sanitize_fact_text_clean_for_prompt`
- полезная pre-sanitize прослойка перед generation;
- особенно для stubborn forbidden roots вроде `посвящ...`.

3. `_fact_first_remove_posv_prompt`
- отдельный targeted repair step;
- ugly, но practically useful.

4. `_cleanup_description`
- normalization pipeline после generation;
- bullet normalization, blockquote normalization, dedupe, heading cleanup.

5. `_collect_policy_issues`
- текущий runtime уже умеет ловить:
  - heading count
  - lead paragraph shape
  - duplicate headings
  - micro-sections
  - epigraph presence
  - forbidden markers

6. `SMART_UPDATE_VISITOR_CONDITIONS_RULE`
- это уже рабочее знание о том, что условия участия нельзя терять.

7. `SMART_UPDATE_FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE`
- очень важный safeguard для program-heavy events.

8. `_facts_text_clean_from_facts`
- отсекает anchors / infoblock noise;
- сохраняет полезный смысл participant-chat facts без URL.

9. текущий `Style C` в `_fact_first_description_prompt`
- это уже действующая попытка писать `Сцена → смысл → детали`;
- redesign должен быть эволюцией этого сильного места, а не его обнулением.

В новом ответе Opus часть этого контекста implicitly совместима с его предложениями, но явно preservation plan не дан.

### 3.2. `evidence_span` сильный, но требует точной операционализации

Идея хорошая, но есть важные practical details, которые ещё не разрулены:

- из какого именно source поля span должен быть verbatim:
  - clipped `source_text`?
  - `raw_excerpt`?
  - `poster_texts`?
  - объединённого payload?
- что делать, если extraction prompt работает на clipped payload, а runtime check сравнивает с чуть другим normalization layer;
- как не потерять хорошие cases на OCR / punctuation / whitespace normalization.

То есть сама идея сильная, но contract ещё надо уточнить.

### 3.3. Dense tiers по количеству фактов — полезны, но пока грубоваты

`13-20 facts = dense` это хороший start.
Но count alone не всегда отражает реальную плотность.

Пример:

- 14 коротких пунктов lineup и 14 длинных содержательных фактов — это два разных случая;
- 10 фактов, из которых 5 — visitor conditions, и 10 фактов чисто тематических — тоже два разных текста.

Значит для implementation либо:

- нужны чуть более гибкие tier signals;
- либо coverage/revise должны уметь это компенсировать.

### 3.4. Missing-facts repair нельзя превращать в привычную костыль-систему

Opus это уже частично понимает, и это плюс.

Но practical risk остаётся:

- если post-generation repair будет срабатывать слишком часто,
- система формально сохранит полноту,
- но quality-first generation окажется недостаточно сильной сама по себе.

То есть repair call должен быть safety net, а не hidden fourth mandatory pass.

## 4. Что из текущего подхода точно нельзя потерять

Если формулировать совсем жёстко, при implementation нельзя потерять следующие вещи:

### 4.1. Эпиграф и работа с сильной цитатой

Нужно сохранить и осмысленно встроить:

- `_pick_epigraph_fact`
- blockquote rules
- запрет пересказывать эпиграф в теле

Это уже сейчас даёт хороший compositional lift.

### 4.2. Предсанитизация проблемных фактов перед generation

Нужно сохранить идею:

- не только ловить forbidden markers post hoc;
- но и заранее снижать вероятность копирования токсичных лексем из facts.

Это одно из сильных practical решений текущего runtime.

### 4.3. Cleanup / normalization layer после generation

Нельзя делать вид, что новые prompts сами всё вылечат.

Текущий cleanup pipeline выполняет полезную работу:

- нормализует списки;
- чинит blockquotes;
- убирает структурный мусор;
- подчищает heading artifacts.

Это должно остаться частью системы.

### 4.4. Policy checks как runtime guardrails

Текущий `_collect_policy_issues` уже содержит много полезных guardrails.
Новый quality-first design должен их расширять, а не заменять абстрактными quality_flags без явного migration.

### 4.5. Visitor conditions / compact program lists

Это уже наработанное доменное знание.
Нельзя потерять его в пользу более “красивых”, но более общих pattern prompts.

## 5. Нужен ли ещё один этап консультации

Да.

Но только **один** и только **очень узкий**.

Не нужен ещё один redesign round.
Не нужен новый brainstorming.

Нужен **pre-implementation preservation/migration round**:

- что из текущего runtime сохраняем как есть;
- что адаптируем;
- что заменяем;
- что удаляем позже;
- и где именно в новой architecture живёт каждая уже полезная эвристика.

Это выглядит оправданным даже с учётом того, что:

- после написания кода всё равно будет ещё один Opus round на fine-tuning prompts.

Почему всё же нужен ещё один этап уже сейчас:

- fine-tuning после кода будет оптимизировать уже реализованную структуру;
- а сейчас надо не потерять working heuristics до того, как структура застынет в коде.

## 6. Bottom line

Если коротко:

- этот ответ Opus сильный и implementation-relevant;
- он существенно продвинул dense facts, repair placement, TPM framing и traceability;
- но он всё ещё недостаточно отвечает на вопрос “что ценного из текущего кода нельзя потерять”;
- поэтому я считаю, что **ещё один узкий pre-code consultation stage нужен**;
- и его тема должна быть не новая концепция, а `preservation / migration matrix`.
