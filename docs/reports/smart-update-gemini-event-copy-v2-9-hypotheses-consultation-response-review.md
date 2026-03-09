# Smart Update Gemini Event Copy V2.9 Hypotheses Consultation Response Review

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-9-hypotheses-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-hypotheses-consultation-brief.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-sanitizer-followup-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-prompt-context.md`

## 1. Краткий verdict

Это сильный pre-run response.

Главное:

- Gemini правильно признал, что `Тема: ...` rewrite в sanitizer — critical blocker;
- хорошо откалибровал `content-aware anti-splitting`;
- не потащил нас в новый complexity layer через pre-generation fact gate;
- сохранил полезные generation-side improvements `v2.8`.

Мой итог:

- **ещё один pre-run consultation round не нужен**;
- можно идти в локальный `v2.9`;
- но часть wording всё равно нужно брать не literally, а с санитарной правкой.

## 2. Что в ответе Gemini принимаю

### 2.1. `v2.9` строим от `v2.8`

Это принимаю полностью.

### 2.2. `Тема: ...` branch надо убрать

Это core change `v2.9`.

### 2.3. `Content-aware anti-splitting`

Тоже принимаю.

Главное — не перепутать anti-splitting со схлопыванием program richness.

### 2.4. `Fact quality gate = no`

Согласен.

На этом этапе дополнительный LLM gate действительно скорее маскировал бы проблему upstream, чем лечил её.

## 3. Что беру только с поправками

### 3.1. Extraction examples должны быть noun-phrase shaped, но не слишком сухими

Gemini прав по направлению, но есть риск over-drying.

То есть примеры вроде:

- `Жизнь и творчество русских художниц`

хороши;

а вот слишком короткие:

- `Искусство`

неприемлемы.

### 3.2. `ОШИБКА:` да, но без общего ужесточения всего prompt

Принимаю только как локальное усиление hints, а не как смену общего тона системы.

## 4. Что не беру буквально

### 4.1. `Generation сейчас работает нормально`

Это верно только при clean facts.

Поэтому я не трактую это как повод вообще не смотреть на generation wording.

Правильнее:

- generation-side `v2.8` improvements пока сохраняем;
- но если `v2.9` снова даст stale prose, этот слой придётся revisit.

## 5. Что реально пойдёт в `v2.9`

Из ответа Gemini беру:

1. убрать `Тема:` rewrite в sanitizer;
2. добавить content-aware anti-splitting;
3. усилить label/intent-style bans в extraction;
4. перевести hints в `ОШИБКА:` tone;
5. сохранить generation-side hygiene rules `v2.8`.

## 6. Bottom line

Этот response достаточно сильный, чтобы закрыть pre-run consultation stage по `v2.9`.

Практический вывод:

- **идти в локальный `v2.9` dry-run**: да;
- **новый pre-run раунд**: нет.
