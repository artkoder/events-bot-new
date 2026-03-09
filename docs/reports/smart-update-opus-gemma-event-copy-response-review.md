# Smart Update Opus Gemma Event Copy Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-response.md`
- `artifacts/codex/opus_gemma_event_copy_casebook_latest.md`
- `artifacts/codex/opus_gemma_event_copy_lexicon_latest.md`
- `artifacts/codex/fact_first_gemma_5events_2026-03-01_v28.md`

## 1. Краткий verdict

Ответ Opus сильный и в целом попал в правильный architectural direction.

Что выглядит особенно удачным:

- проблема действительно разделена правильно: часть сидит в extraction, часть в generation;
- deterministic mode routing выглядит сильнее, чем ещё один planner-call;
- `why go` трактуется как evidence-based value explanation, а не как отдельный рекламный ритуал;
- отдельный critic-pass не предлагается, вместо этого усиливается уже существующий coverage/revise path;
- heading palette, anti-template checks и A/B ladder сформулированы прагматично.

Но перед внедрением есть 4 ключевых возражения / follow-up темы:

1. `copy_assets` extraction поверх одного только `facts_text_clean` слишком lossy.
2. `why go` gate через жёсткое `>=2` причин выглядит грубо.
3. `quote_led` gate по одному правилу `>=8 слов` слишком слаб.
4. routing на основе raw counts (`program_highlights >= 4`) может misroute several formats.

Итоговая позиция:

- `direction accepted`
- `response not final`
- нужен узкий follow-up раунд с фокусом на call budget, gating quality и source-grounding.

## 2. Что в ответе Opus подтверждается

### 2.1. Диагноз текущей шаблонности

Opus корректно попал в основные слабости current output:

- единый ритм;
- механический lead;
- “бухгалтерские” headings;
- list-as-text anti-pattern;
- однородный tone для разных жанров;
- повторяемость facts across lead/body.

Это хорошо бьётся с нашими локальными наблюдениями по `fact_first_gemma_5events_2026-03-01_v28.md`.

### 2.2. Mode routing как центральная идея

Предложение разделить тексты на:

- `compact_notice`
- `reported_preview`
- `program_led`
- `quote_led`

выглядит sound.

Особенно ценно, что routing у Opus:

- deterministic;
- дешёвый;
- объяснимый;
- не тянет новый LLM-слой ради самого факта routing.

### 2.3. Inline why-go вместо отдельной секции

Это важное попадание. Мы как раз хотели:

- не делать `### Зачем идти` по умолчанию;
- не превращать ценность события в sales paragraph;
- встраивать why-go как 1-2 grounded предложения внутри narrative.

### 2.4. Extended coverage вместо нового critic call

Это тоже сильное решение.

Идея расширить текущий coverage дополнительными checks:

- `template_feel`
- `weak_lead`
- `weak_heading`
- `redundancy`

выглядит operationally аккуратно и не раздувает runtime цепочку.

## 3. Что требует коррекции или follow-up

### 3.1. Главный риск: `copy_assets` поверх `facts_text_clean` может быть already too late

Самое важное расхождение с ответом Opus — место, где он хочет делать `copy_assets extraction`.

Его вариант:

- сначала обычный facts extraction;
- потом отдельный лёгкий call уже по `facts_text_clean`.

Проблема:

- если первый pass уже **не сохранил** голос, цитату, experience detail или тональную подсказку,
  то второй pass по `facts_text_clean` этого уже не восстановит;
- значит часть нужных сигналов будет теряться именно там, где мы и хотим их спасти.

Поэтому более надёжные варианты выглядят так:

1. **Расширить первый source-backed extraction call**
- один JSON-ответ возвращает и atomic facts, и `copy_assets`.

2. **Если делать второй call, то не по одному `facts_text_clean`**
- а по `source_text/raw_excerpt/poster_texts` плюс уже выделенным фактам как safety context.

Это сейчас главный пункт follow-up.

### 3.2. `why go` gate через `>=2` причин слишком жёсткий

Count-based rule удобен, но editorially груб.

Есть события, где существует **одна** очень сильная grounded причина:

- редкий headliner;
- сильная award signal;
- one-off format;
- единственная крупная ретроспектива;
- необычный post-screening discussion.

Поэтому лучше просить у Opus strength-based модель:

- `why_go_score` или
- `why_go_candidates[]` с весами `strong / regular`.

Более реалистичное правило:

- `1 strong` reason **или** `2 regular` reasons.

### 3.3. `quote_led` gate по одному только количеству слов слабоват

Правило:

- `voice_fragments[0].word_count >= 8`

слишком легко пропустит:

- слоган;
- poster line;
- псевдо-цитату без говорящего;
- длинную, но пустую декларацию;
- служебную строку, если extraction ошибся.

Нужен более жёсткий gate:

- желательно speaker attribution;
- смысловая плотность;
- стоп-лист для slogan / CTA / афишных лозунгов;
- запрет на auto-quote-mode для lines, которые не открывают тему события.

### 3.4. Routing по количеству `program_highlights` может misroute

Условие:

- `program_highlights.length >= 4`

слишком coarse.

Оно может неправильно увести в `program_led`:

- концерт с треклистом;
- лекцию с большим набором тем;
- спектакль с длинным перечнем наград;
- кинопоказ с большим списком discussion topics.

Нужны более содержательные features, например:

- `participatory_signal`
- `has_stepwise_action`
- `hands_on_materials`
- `speaker_led_signal`
- `performance_signal`
- `has_true_program_list`

То есть не только число пунктов, а тип пунктов.

### 3.5. Cost posture выглядит чуть оптимистично

Opus пишет, что +1 extra call и ~+500-800 tokens per event “пренебрежимо”.

В directionally sense это может оказаться правдой.
Но operationally это нельзя принимать на веру.

Нужно считать на реальной нагрузке:

- median / p95 latency;
- токены на событие;
- влияние на bursts Telegram/VK импорта;
- долю событий, где revised pass и так уже срабатывает.

То есть тезис не отвергается, но требует real canary verification, а не rhetorical acceptance.

## 4. Что уже выглядит implementable почти без спора

Это можно считать strong candidate set:

1. heading palette + heading stop-list;
2. inline why-go instead of dedicated section;
3. `quote_led` как отдельный narrative mode;
4. deterministic mode routing как концепция;
5. extended coverage flags:
   - `template_feel`
   - `weak_lead`
   - `weak_heading`
   - `redundancy`
6. A/B rubric и staged rollout ladder.

## 5. Что теперь рационально спросить у Opus

Следующий раунд должен быть уже не общим, а narrow:

1. Готов ли Opus пересобрать architecture так, чтобы:
   - либо `copy_assets` шли из первого source-backed extraction call;
   - либо второй call видел не только `facts_text_clean`, но и исходный source context?

2. Как он предлагает заменить `>=2 why_go_candidates` на более редакторски точный gate?

3. Как именно он предлагает фильтровать `quote_led`, чтобы туда не просачивались slogans и служебные фразы?

4. Какие deterministic features лучше raw-count thresholds для `program_led` routing?

5. Какой из его prompt drafts он считает:
   - immediate low-risk candidate;
   - medium-risk experiment;
   - пока только research hypothesis?

## 6. Практический следующий шаг

Рационально подготовить короткий follow-up prompt к Opus:

- подтвердить, что direction mostly accepted;
- зафиксировать 4 узких возражения;
- попросить revised minimal-call architecture;
- попросить stronger gates для `why go`, `quote_led` и `program_led`.

Это лучше, чем спорить со всем отчётом целиком:

- synthesis у Opus уже сильный;
- теперь нужна инженерная шлифовка под текущий Smart Update runtime.
