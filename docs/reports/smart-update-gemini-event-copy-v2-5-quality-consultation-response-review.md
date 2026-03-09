# Smart Update Gemini Event Copy V2.5 Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-5-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-5-quality-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_5_2026_03_07.py`

## 1. Краткий verdict

Это сильный и достаточно grounded Gemini response.

Главное:

- он правильно понял, что `v2.5` лучше `v2.4`, но не лучше `v2.3`;
- не стал снова тянуть нас в большой redesign;
- точно локализовал слабость `soft hints`;
- дал несколько действительно patchable Gemma-friendly moves.

Мой итог:

- **новый Gemini-раунд сейчас не нужен**;
- response можно использовать как основу для `v2.6`;
- но не всё из него надо брать буквально.

## 2. Что в ответе Gemini принимаю

### 2.1. Диагноз `soft hints -> over-abstraction`

Это принимаю.

Он точно попал в корень `v2.5`:

- grouped program hints были слишком мягкими;
- Gemma интерпретировала их как право обобщить названия;
- human-readable anti-`посвящ*` issue оказался слишком advisory.

Это хорошо совпадает с реальными кейсами:

- `2734`
- `2687`
- `2673`

### 2.2. Compact branch нужен не 1-to-1 facts dump, а смысловое слияние

Это сильная рекомендация.

`2660` действительно показал, что:

- coverage может улучшаться;
- но compact prose деградирует, если модель вынуждена механически вставлять каждый факт отдельным блоком.

Разрешение сливать смыслово близкие факты в 1-2 предложения выглядит правильным.

### 2.3. Anti-bureaucracy rule для `На презентации расскажут о...`

Это тоже сильный point.

`2673` действительно страдает не только от heading quality, а от самих канцелярских facts, которые generation слишком буквально переносит в prose.

Prompt-level guidance писать сразу о предмете, а не о том, что “о нём расскажут”, worth testing.

### 2.4. Явный ban на label-style facts вроде `Тема: ...`

Это тоже принимаю.

`2734` показал, что label-style extraction — реальный regression marker, а не косметика.

## 3. Что беру только с поправками

### 3.1. Hard ban на `посвящ*`

Принимаю только с modification.

Gemini прав, что:

- текущий issue недостаточно силён;
- Gemma слишком легко игнорирует мягкую просьбу.

Но буквальный rollback к pure lexical hard-ban already known risky:

- мы уже видели, как такие запреты ведут к неестественным rewrites;
- в ответе Gemini встречается пример `Событие рассказывает о...`, а это снова metatext и editorially слабый ход.

То есть правильно не просто “запретить корень капсом”, а:

- пометить это как critical issue;
- сразу давать 2-4 допустимых replacement patterns;
- не разрешать metatext replacement.

### 3.2. Template `В программе: "А", "Б", "В"`

Идея сильная, но тоже не как универсальная догма.

Это особенно полезно для:

- концертов;
- репертуарных программ;
- явных performance lists.

Но нельзя превращать любой list-like fact в обязательную форму `В программе: ...`, иначе:

- лекции;
- выставки;
- person-led cases

начнут звучать искусственно.

Итог:

- accept with modification;
- использовать как preferred pattern только для real program-item cases.

### 3.3. Keep `<= 6` routing

Здесь тоже только partial accept.

Gemini прав, что threshold пока полезен practically.
Но я не принимаю это как settled rule.

Причина:

- `2734` как раз показывает, что один лишь count threshold всё ещё brittle.

То есть:

- для ближайшего `v2.6` его можно оставить;
- но параллельно нужен additional compact gate по content shape.

## 4. Что в ответе Gemini считаю слишком жёстким

### 4.1. ALL CAPS severity как universal solution

Идея повысить severity правильная.
Но буквальный стиль:

- `СТРОГО ЗАПРЕЩЕНО`
- `ТЫ ОБЯЗАН`
- `КРИТИЧЕСКАЯ ОШИБКА`

может сработать, а может сделать prompt tone unnecessarily brittle.

Я бы не копировал этот регистр в лоб во все prompts.
Лучше:

- явно пометить issue как blocking;
- сформулировать действие конкретно;
- не превращать весь prompt в shouting mode.

### 4.2. Почти полный фокус на extraction

Gemini правильно усиливает extraction.
Но слегка недооценивает, что часть проблем всё ещё живёт в generation contract:

- `2660` — это уже не только extraction problem;
- `2673` — это не только facts phrasing, но и generation literalism.

Поэтому `v2.6` не должен быть extraction-only.

## 5. Что точно пойдёт в следующий локальный round

Из ответа Gemini я беру следующее:

1. Убрать мягкость из critical issue wording.
2. Запретить label-style facts (`Тема: ...`, `Идея: ...`).
3. Для real program-item cases дать явный grouped template вместо vague grouping hint.
4. Разрешить compact branch сливать overlapping facts в одно предложение.
5. Добавить anti-bureaucracy rule против literal transfer вроде `На презентации расскажут о ...`.

## 6. Что не беру как готовую истину

- Не принимаю ALL CAPS prompt style как готовый стандарт.
- Не принимаю `<= 6` как окончательный routing law.
- Не принимаю replacement `Событие рассказывает о...`; это editorially weak metatext.

## 7. Bottom line

Этот Gemini response полезен и actionable.

Если коротко:

- **accept**: diagnosis про `soft hints`, compact dedup permission, anti-bureaucracy, label-style fact ban;
- **accept with modification**: hard anti-`посвящ*`, grouped program template, routing `<= 6`;
- **reject as literal copy**: shouting prompt tone и metatext replacements.

Практический вывод:

- новый Gemini round сейчас не нужен;
- следующий шаг уже локальный `v2.6` patch pack и новый dry-run.
