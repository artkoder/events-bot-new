# Smart Update Opus Gemma Event Copy Pattern Redesign Follow-up Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-followup-response.md`
- `smart_event_update.py`

## 1. Краткий verdict

Этот follow-up ответ Opus важен не только сам по себе.
Он ещё и показывает, что **предыдущий наш narrowing был местами слишком жёстким**.

Главный пересмотр позиции:

- да, safety/debuggability важны;
- но в предыдущем review я местами переоценил их относительно главной цели;
- главная цель здесь всё-таки: **более естественный и качественный итоговый текст**, который реально работает в ограничениях Gemma и TPM.

Поэтому обновлённый вывод такой:

- low-risk пакет по-прежнему принимается;
- medium-risk ветку нужно делать аккуратно;
- но не надо чрезмерно обеднять её до такой степени, что она перестанет давать заметный quality lift.

Иными словами:

- **часть моего прежнего сопротивления была избыточной**;
- **часть возражений Opus здесь конструктивна и её стоит принять**.

## 2. Где предыдущий review был слишком консервативен

### 2.1. Полный вынос `scene_led` из v1 был, вероятно, ошибкой

Это самая важная самокоррекция.

Почему:

- без `scene_led` активные patterns в v1 почти все остаются объясняющими:
  - `topic_led`
  - `program_led`
  - `value_led`
  - `person_led`
- то есть текст структурно разнообразнее, но всё ещё стартует в одном регистре:
  - объяснить;
  - определить;
  - развернуть смысл.

Для задачи “убрать ощущение LLM-шаблона” этого недостаточно.

Ключевой runtime-факт:

- текущий `_fact_first_description_prompt` уже сам тянет модель в режим `Сцена → смысл → детали`.

То есть если мы полностью вырезаем `scene_led` из первой enriched ветки, мы рискуем не просто упростить систему, а **отступить от уже существующей stylistic ambition** в [smart_event_update.py](/workspaces/events-bot-new/smart_event_update.py#L1724).

Обновлённая позиция:

- `scene_led` стоит оставить в quality-oriented v1;
- но включать только с очень консервативным gate;
- при провале gate делать fallback в `topic_led`.

### 2.2. Полностью runtime-only routing был переоценён

Мой предыдущий review слишком уверенно предполагал, что почти всё можно надёжно вывести из primitive fields в Python.

На практике это не везде так.

Особенно это касается semantic distinctions типа:

- человек действительно является narrative center;
- список в source — это реальная программа, а не просто перечень тем.

Если всё это пытаться вывести через regex и простые эвристики по LLM-generated strings, получается не настоящий deterministic routing, а его имитация.

Здесь Opus справедливо указывает на слабое место:

- иногда 1–2 semantic helper signals от LLM могут быть практичнее, чем агрессивный runtime-only dogma.

Обновлённая позиция:

- полный пакет `routing_features` из 6 boolean действительно избыточен;
- но **1-2 targeted helper fields** как часть v1 вполне заслуживают рассмотрения;
- особенно если они проходят runtime sanity-check.

### 2.3. Полный defer `contrast_or_tension` тоже мог быть лишним

Мой прежний аргумент про editorializing risk был корректен, но слишком общий.

Если в source уже есть явное grounded противопоставление, поле типа `contrast_or_tension` может быть:

- дешёвым;
- редким;
- но очень полезным способом сделать lead живее.

Это особенно важно для:

- кинопоказов;
- выставок;
- событий с ясно сформулированным “не про X, а про Y”.

Обновлённая позиция:

- `contrast_or_tension` не должно быть routing dependency;
- но как optional source-grounded lead aid его не стоит убивать автоматически;
- при условии строгой traceability check.

## 3. Где Opus в этом follow-up действительно убедителен

### 3.1. Он правильно атакует главную продуктовую проблему

Самая сильная часть ответа:

- Opus прямо говорит, что без `scene_led` мы получим более аккуратную, но всё ещё объясняющую систему;
- а значит выигрыш по “human / natural text” может оказаться слабее, чем хотелось бы.

С этим трудно спорить.

### 3.2. Он не просто защищает richer design, а предлагает guards

Это важно.

В ответе есть не просто “оставьте больше полей”, а попытка добавить:

- runtime grounding gate для `scene_cues`;
- runtime traceability gate для `contrast_or_tension`;
- sanity checks для helper booleans.

То есть Opus не предлагает безусловную свободу модели.
Он предлагает quality-oriented design с дополнительными предохранителями.

### 3.3. Он честно режет часть собственных ранних идей

Полезно, что в follow-up он сам:

- убирает `tone_hint` из v1;
- не возвращает large boolean pack целиком;
- держит `quote_led` вне v1.

Это делает его позицию заметно более инженерной, чем в первом broad redesign ответе.

## 4. Что всё ещё требует осторожности

### 4.1. `scene_cues` и `contrast_or_tension` нельзя принимать “на доверии”

Здесь мой прежний скепсис остаётся частично верным.

Да, эти поля могут сильно улучшить lead.
Но они же и самые опасные с точки зрения subtle hallucination.

Проблема:

- предложенные Opus runtime checks на word overlap полезны;
- но в текущем виде они всё ещё грубые и могут либо пропускать слабые hallucinations, либо ложно блокировать хорошие сигналы.

Практический вывод:

- эти поля стоит оставить в v1;
- но их gating logic ещё нужно доработать перед implementation.

### 4.2. Hybrid routing хорош как идея, но helper booleans должны быть строго оправданными

Здесь я бы не переходил от одной крайности к другой.

Не стоит:

- ни жёстко требовать pure runtime routing любой ценой;
- ни снова раздувать extraction schema множеством routing-like outputs.

Обновлённая practical posture:

- helper booleans должны быть максимум точечными;
- их value надо обосновывать не теоретически, а по реальным misroute cases;
- `is_speaker_led` выглядит разумнее, чем `has_true_program_list`, но оба ещё заслуживают дополнительной калибровки.

### 4.3. Порядок приоритетов patterns всё ещё не до конца стабилен

Даже в новом follow-up остаётся спорный момент:

- где именно должен стоять `scene_led` относительно `value_led`;
- когда живой grounded scene важнее, чем explicit why-go reason;
- когда наоборот explanation of value важнее, чем живая деталь.

Это уже не только инженерный, но и редакторский вопрос.

Именно здесь я бы не принимал окончательное решение “с листа”.

### 4.4. Нужны acceptance criteria не только по architecture, но и по качеству текста

Сейчас у нас уже достаточно идей.
Не хватает более жёсткой рамки валидации:

- где именно quality lift считается достаточным;
- как измерять naturalness vs template feel;
- какой дополнительный token overhead считаем допустимым;
- сколько false positive / false negative по `scene_led` и `value_led` терпимо.

Без этого можно бесконечно спорить о схеме, но не о результате.

## 5. Обновлённая практическая позиция

### 5.1. Что я бы принял после этого follow-up

Сейчас я бы принял уже не stripped-down v1, а **quality-first v1**:

- `core_angle`
- `format_signal`
- `program_highlights`
- `experience_signals`
- `why_go_candidates`
- `credibility_signals`

Optional, но уже потенциально активные в v1:

- `voice_fragments`
- `subformat`
- `scene_cues`
- `contrast_or_tension`

Точечные helper fields — под дополнительную калибровку:

- `is_speaker_led`
- `has_true_program_list`

Deferred:

- `tone_hint`
- `quote_led` routing
- large `routing_features` pack

### 5.2. Какие patterns выглядят разумными для quality-first v1

Я бы сейчас рассматривал v1 уже не как 5-pattern subset, а как:

- `topic_led`
- `program_led`
- `compact_fact_led`
- `person_led`
- `value_led`
- `scene_led`

При этом:

- `quote_led` остаётся outside v1;
- `scene_led` включается только при safe gate;
- `scene_led` и `value_led` требуют ещё одной калибровки по precedence.

### 5.3. Как я бы сформулировал главный урок для себя

Предыдущий review был полезен как защита от runaway complexity.
Но он местами слишком сильно оптимизировал:

- debuggability;
- агрессивную минимальность schema;
- operational neatness;

за счёт:

- живости текста;
- композиционной вариативности;
- реального шанса убрать ощущение “это снова объясняющий LLM copy”.

Для этой задачи это было смещение приоритетов.

## 6. Нужен ли ещё один раунд с Opus

Теперь да, но не широкий и не повторный brainstorming.

Нужен **узкий calibration round**:

- не о том, стоит ли делать pattern-driven redesign вообще;
- а о том, как собрать `quality-first v1`, который:
  - не душит лучшие идеи;
  - не режет quality ради искусственной экономии;
  - не раздувает schema без реального payoff;
  - реально помещается в Gemma/TPM constraints.

## 7. Bottom line

Если коротко:

- follow-up ответ Opus убедительно показал, что часть моего предыдущего narrowing была слишком жёсткой;
- особенно это касается `scene_led`, optional richer signals и слишком догматичного runtime-only routing;
- теперь разумнее двигаться не к “самой безопасной” v1, а к **достаточно сильной v1, которая заметно улучшает качество текста и остаётся жизнеспособной в рамках TPM/Gemma**;
- следующий шаг — один calibration round и затем локальная implementation ветка.
