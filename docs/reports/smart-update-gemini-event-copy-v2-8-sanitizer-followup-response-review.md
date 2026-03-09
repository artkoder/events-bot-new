# Smart Update Gemini Event Copy V2.8 Sanitizer Follow-Up Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-8-sanitizer-followup-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-sanitizer-followup-brief.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-dryrun-quality-consultation-response-review.md`
- `docs/reports/smart-update-gemini-event-copy-v2-8-prompt-context.md`

## 1. Краткий verdict

Это сильный и materially important follow-up response.

Gemini правильно перестроил диагноз:

- label-style artifacts в `v2.8` идут не только из extraction prompt;
- существующий prompt-facing sanitizer действительно является частью root cause map;
- без калибровки этого слоя `v2.9` был бы собран на неполной модели причин.

Мой итог:

- **дополнительный консультационный раунд больше не нужен**;
- консультационный цикл по `v2.8` можно считать закрытым;
- следующий шаг уже локальный `v2.9`.

## 2. Что в ответе Gemini принимаю

### 2.1. `Тема: ...` нужно убрать из prompt-facing sanitizer path

Это принимаю полностью.

Самый важный practical вывод:

- именно этот deterministic rewrite сейчас сам производит тот мусор, против которого мы боремся downstream.

Следствие:

- в `v2.9` нельзя оставлять ветку `<event> посвящена ... -> Тема: ...`.

### 2.2. `Content-aware anti-splitting`

Это тоже принимаю.

Правильная формулировка:

- не дробить один и тот же смысл на 3-4 факта;
- но не схлопывать distinct program items, имена и названия.

Это как раз тот balance, которого не хватало между `v2.7` и `v2.8`.

### 2.3. Жёстче constraints на label/intent-style data shapes

Принимаю.

После `v2.8` уже достаточно evidence, что мягкие hints для Gemma слишком легко игнорируются.

## 3. Что беру только с поправками

### 3.1. `disable in experimental v2.9`

Смысл верный, но implementation надо делать аккуратно.

Я не трактую это как:

- полностью вырубить весь `_sanitize_fact_text_clean_for_prompt`.

Правильнее:

- отключить именно branch, генерирующий `Тема: ...`;
- сохранить остальные безопасные sanitize behaviors, если они не меняют смысл.

### 3.2. Tone hints: `ОШИБКА:` да, но без shouting

С этим согласен.

То есть:

- усилить seriousness полезно;
- но не превращать prompt в alarmist wall of caps.

## 4. Что теперь реально пойдёт в `v2.9`

Из этого follow-up беру:

1. убрать `Тема:` rewrite из prompt-facing sanitizer branch;
2. добавить content-aware anti-splitting в extraction;
3. сделать harder ban на label-style и intent-style data shapes;
4. сохранить `template-overuse control`;
5. не возвращать `v2.7`-style safe-positive wrappers.

## 5. Bottom line

Этот follow-up закрыл важный blind spot.

Практический вывод:

- **consultation cycle closed**: да;
- **следующий шаг**: локальный `v2.9` dry-run;
- **дальше снова консультация**: уже только по реальному `v2.9` результату, а не до него.
