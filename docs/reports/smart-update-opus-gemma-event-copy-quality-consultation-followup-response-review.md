# Smart Update Opus Gemma Event Copy Quality Consultation Follow-up Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-followup-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md`

## 1. Краткий verdict

Этот follow-up ответ Opus заметно лучше откалиброван, чем предыдущий.

Главное улучшение:

- он реально принял критику по `baseline optics vs actual quality priorities`;
- точнее разделил `pattern failure` и `harness failure`;
- дал намного более полезную рамку для `copy_assets boundary`;
- и выдал уже почти implementation-minded `corrected v2 patch set`.

Мой текущий вывод:

- этот ответ **в целом сильный и полезный**;
- большинство его v2 направлений я принимаю;
- но перед локальной доработкой prototype нужен ещё **один последний узкий раунд**:
  - не про концепцию;
  - а про `prioritization / prompt compression / Gemma-realistic patch pack`.

## 2. Что в ответе Opus особенно сильное

### 2.1. Наконец нормально откалиброван `pattern vs harness`

Это, возможно, самая важная коррекция.

Особенно сильный момент:

- `2687` сломался не потому, что `program_led` обязательно плохой;
- а потому, что revise-loop и dedup discipline у prototype v1 были сломаны.

Это правильный way of reading the dry-run.

### 2.2. `copy_assets boundary` стала заметно зрелее

Его 4-level contract — один из самых полезных outputs всей консультационной линии:

- Level 1: fact-backed
- Level 2: source-backed / traceable
- Level 3: inferred soft aid
- Level 4: invented

Это намного лучше, чем и мой более ранний жёсткий скепсис, и его предыдущий blanket подход.

### 2.3. `content-preservation floor` — сильная формулировка

Это намного лучше, чем старое “не уменьшать raw fact count”.

Правильная мысль:

- не count parity как самоцель;
- а сохранение publishable content coverage;
- с явными допустимыми причинами, когда факт можно drop'нуть.

Это очень полезно для следующей extraction iteration.

### 2.4. Его corrected `v2 patch set` уже почти годится как implementation queue

Особенно strong set:

1. anti-duplication rule
2. anti-dup runtime guard
3. sparse routing correction
4. epigraph/blockquote recovery
5. weak-heading ban
6. CTA detection
7. revised extraction preservation rule

Это выглядит как хороший `v2 core subset`.

## 3. Что в ответе всё ещё требует осторожности

### 3.1. Он всё ещё местами слишком благосклонен к идее, что patterns “mostly sound”

Это вероятно, но ещё не доказано.

У меня бы здесь была более осторожная формулировка:

- current evidence says many failures are implementation-level;
- but this **не доказывает**, что pattern library уже оптимальна.

Особенно спорно:

- `program_led` для `2687` всё ещё не выглядит окончательно закрытым вопросом;
- там вполне возможно, что `topic_led + structured list` дал бы ещё лучшее качество.

То есть я бы не превращал текущую pattern taxonomy в “почти утверждённую”.

### 3.2. `experience_signals` как Level 3 soft aid — полезная идея, но опасная

Да, blanket drop не нужен.
Но practical risk всё ещё высокий:

- эти поля легко провоцируют tone drift;
- легко становятся источником красивого, но unsupported prose;
- Gemma особенно любит использовать такие мягкие сигналы слишком буквально.

Поэтому на практике:

- держать можно;
- но использовать в v2 очень осторожно;
- возможно вообще не давать generation прямого права verbatim использовать их.

### 3.3. `compact_fact_led` без headings — допустимо, но это надо аккуратно зафиксировать

Opus правильно смягчил прежний bias по headings.
Но здесь всё равно нужен точный контракт.

Иначе можно получить два плохих режима:

- over-structured sparse text;
- или наоборот рыхлый unformatted абзац, который плохо выглядит на Telegraph.

То есть idea принята, но требует более точного format contract.

### 3.4. Ответ всё ещё слишком широк для немедленного переноса в prompt changes

И это главный practical point.

Даже хороший corrected v2 patch set пока ещё:

- слишком длинный;
- слишком богатый на micro-rules;
- не упакован в `Gemma-realistic prompt budget`.

То есть в нынешнем виде это сильный design memo, но ещё не компактный implementation pack.

## 4. Что стоит принять уже сейчас

### 4.1. Принять почти без изменений

1. anti-duplication в generation prompt
2. anti-duplication check после revise/repair
3. sparse routing correction
4. epigraph / blockquote recovery как cross-pattern enhancement
5. weak heading ban
6. CTA detection
7. content-preservation floor

### 4.2. Принять с сужением / уточнением

1. `format_signal`:
   - `event_type` as hard default
   - LLM as fallback/refinement only

2. `credibility_signals`:
   - source-traceable only
   - не обязаны быть в `facts_text_clean`
   - generation uses carefully

3. `experience_signals`:
   - keep only as soft optional aid
   - not a strong factual surface
   - likely deprioritized in first v2 patch pack

4. `compact_fact_led`:
   - accept
   - but exact formatting contract must be narrower

### 4.3. Пока отложить

1. `merge value_led + topic_led`
2. `reduce pattern set from 6 to 4`
3. any large-scale pattern taxonomy simplification

Для этого пока просто не хватает dry-run width.

## 5. Нужен ли ещё один этап консультаций

**Да, но это уже должен быть финальный узкий этап перед локальной v2 доработкой.**

Не нужен новый review of outputs.
Не нужен новый architecture round.

Нужен `prioritization/compression round`:

- как превратить сильные идеи из follow-up response + quality improvements catalog
  в компактный, Gemma-friendly `v2 patch pack`;
- что exactly включать в generation prompt;
- что exactly включать в revise prompt;
- что exactly держать в runtime checks;
- что пока не тащить в v2, чтобы не утонуть в instruction noise.

## 6. Bottom line

Этот follow-up ответ Opus я считаю **почти достаточным по содержанию**.

Но следующий шаг должен быть уже не “ещё одна критика”, а:

- финальная приоритизация;
- компрессия;
- сборка компактного `v2 quality patch pack`.

То есть мы уже почти вышли из режима исследования и входим в режим controlled local iteration.
