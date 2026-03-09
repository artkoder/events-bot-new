# Smart Update Opus Gemma Event Copy Follow-up Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-followup-response.md`
- `artifacts/codex/opus_gemma_event_copy_followup_prompt_latest.md`

## 1. Краткий verdict

Этот follow-up ответ Opus уже выглядит не как brainstorming, а как usable implementation ladder.

Главный вывод:

- **архитектурный direction принимается**
- **новый Opus-раунд перед low-risk implementation не нужен**
- **можно переходить к локальной реализации safe subset и A/B подготовке**

По сравнению с предыдущим ответом Opus действительно снял все 4 главных возражения:

1. `copy_assets` больше не вытягиваются из одного только `facts_text_clean`;
2. `why go` больше не привязан к тупому `>=2`;
3. `quote_led` больше не включается по одному word-count;
4. `program_led` больше не зависит от raw count `program_highlights`.

## 2. Что в ответе Opus теперь выглядит сильным

### 2.1. Merged extraction вместо lossy second pass

Это ключевое улучшение.

Новый вариант:

- один source-backed extraction call;
- в нём же возвращаются и `facts`, и `copy_assets`;
- bucketization остаётся runtime-only.

Это правильно по двум причинам:

- не теряем `voice / experience / tone / credibility` между pass-ами;
- не плодим лишний LLM-call ради signals, которые уже доступны в source payload.

### 2.2. Убран `structure_hint` из extraction

Это тоже удачное решение.

В предыдущем ответе `structure_hint` сидел внутри extraction schema. Сейчас Opus убрал его и отдал mode choice runtime-логике.

Это лучше, потому что:

- routing становится объяснимым;
- снижается зависимость от LLM mood;
- проще делать A/B и later debug.

### 2.3. Why-go gate теперь редакторски реалистичнее

Правило:

- `1 strong OR 2 regular`

заметно лучше прежнего fixed-threshold.

Это уже ближе к реальному editorial reasoning:

- иногда одной сильной причины достаточно;
- иногда нужны две supporting reasons;
- бедный source не будет artificially “натягиваться” на value paragraph.

### 2.4. Quote-led переведён из “сразу внедрять” в research-only

Это важная зрелая коррекция со стороны Opus.

Теперь он сам признаёт, что:

- `voice_fragments` extraction хрупок;
- quote-led mode нельзя безопасно включать до реальной статистики по extraction quality.

Это хороший сигнал: response не пытается любой ценой продавить красивую идею в runtime раньше времени.

### 2.5. Implementation posture стал реально прагматичным

Opus разделил изменения на 3 класса:

1. `low-risk immediate candidates`
2. `medium-risk experiments`
3. `research-only`

Это уже usable engineering posture, а не просто хороший текст о промптах.

## 3. Что можно принять почти без спора

Ниже — набор, который выглядит разумным кандидатом на локальную реализацию уже сейчас.

### 3.1. Low-risk immediate set

Из follow-up ответа прямо просится в работу:

1. heading palette + heading stop-list;
2. anti-redundancy rule в generation;
3. extended coverage checks:
   - `template_feel`
   - `weak_lead`
   - `weak_heading`
   - `redundancy`
4. lead variety instructions;
5. compact sizing rule для бедных sources.

Сильная сторона этого набора:

- он не требует schema migration;
- не трогает extraction JSON;
- не добавляет новый routing layer;
- и при этом бьёт ровно в самые заметные проблемы current output.

### 3.2. Medium-risk set

Это уже не “сразу в runtime”, а честный A/B territory:

1. merged extraction `facts + copy_assets`;
2. deterministic mode routing on top of `copy_assets`;
3. strength-based why-go gate;
4. inline why-go generation.

Это выглядит перспективно, но уже требует:

- нового JSON schema;
- устойчивого parsing;
- проверки, что enriched extraction не деградирует core facts.

### 3.3. Research-only set

Логично пока не продвигать в production-candidate:

1. `quote_led`
2. tone-adaptive generation

С этим я согласен. Оба пункта слишком чувствительны к extraction quality и stylistic drift.

## 4. Что всё ещё требует осторожности

Несмотря на сильный follow-up, несколько real risks остаются.

### 4.1. Merged extraction всё ещё может ухудшить качество базовых facts

Даже если call count не растёт, schema становится ощутимо богаче.

Риск:

- модель начнёт тратить больше внимания на `copy_assets`;
- core `facts[]` станут менее чистыми или менее стабильными.

Поэтому acceptance criterion должен включать не только качество `copy_assets`, но и сохранность:

- числа валидных facts;
- точности facts bucketization downstream;
- стабильности списков и важных narrative facts.

### 4.2. `strong / regular` — полезно, но potentially noisy label

Это уже лучше, чем raw count.
Но сама классификация `strength` тоже может плавать.

Практический вывод:

- не стоит слишком рано завязывать сильные stylistic decisions на `strength`, пока не увидим distribution на реальных events;
- до A/B можно держать why-go conservative even inside medium-risk branch.

### 4.3. `voice_fragments` всё ещё останутся самым хрупким полем

Opus это признал, и с этим стоит согласиться.

Даже если `quote_led` пока отложен, поле `voice_fragments` всё равно может пригодиться позже:

- для inline quote;
- для better epigraph selection;
- для richer but bounded narrative.

Но до реальной валидации это поле не должно быть hard dependency для routing.

### 4.4. Budget posture по-прежнему нужно мерить, а не верить на слово

Тот факт, что extra calls не выросли, очень хороший.
Но `+600-800 tokens/event` всё равно надо проверять на:

- median;
- p95;
- burst imports;
- реальную долю revise-trigger случаев.

То есть риск уже ниже, но operational validation всё ещё обязательна.

## 5. Практическая позиция после follow-up

Теперь позиция выглядит так:

### Принять

- revised architecture as target direction;
- low-risk immediate set as near-term implementation candidate;
- medium-risk set as A/B branch;
- quote-led and tone-adaptive as research-only.

### Не делать пока

- не включать `quote_led` в runtime routing;
- не начинать с merged extraction до baseline/golden dataset;
- не считать, что why-go уже “решён” без реальных examples review.

## 6. Что теперь рационально делать дальше

После этого follow-up рациональная последовательность уже локальная, не внешняя:

1. Внедрить low-risk prompt/coverage улучшения.
2. Подготовить golden set из 20 событий.
3. Только после этого собирать medium-risk branch:
   - merged extraction
   - routing
   - why-go gate
4. Проверять medium-risk branch через A/B, а не сразу через production switch.

## 7. Bottom line

Если коротко:

- первый ответ Opus был хороший concept proposal;
- follow-up ответ уже похож на production-minded design;
- дальше главная работа — не ещё один раунд переписки, а аккуратная локальная реализация и проверка.
