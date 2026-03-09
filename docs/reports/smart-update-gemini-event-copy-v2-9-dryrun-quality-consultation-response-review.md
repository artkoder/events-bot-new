# Smart Update Gemini Event Copy V2.9 Dry-Run Quality Consultation Response Review

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-dryrun-quality-consultation-brief.md`
- `docs/reports/smart-update-gemini-event-copy-v2-9-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-9-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-9-review-2026-03-08.md`

## 1. Краткий verdict

Это сильный и practical response.

Ещё один консультационный подэтап перед `v2.10` действительно не нужен.

Gemini хорошо прочитал main shape `v2.9`:

- sanitizer bypass был реальным win;
- extraction остаётся главным bottleneck;
- новый intermediate quality gate не нужен;
- `2687` и `2673` упираются в плохую упаковку dense fact sets.

Но переносить его советы literally нельзя.

Самая важная поправка:

- его идея `List Consolidation` сильная;
- его literal examples `Художницы: ...` и `В программе: ...` слишком рискованные, потому что легко возвращают colon-style template / label-style artifacts.

## 2. Что в ответе Gemini принимаю

### 2.1. `sanitizer bypass` надо сохранять

Принимаю полностью.

`v2.9` подтвердил, что это был точный corrective fix.

### 2.2. Core bottleneck всё ещё в Extraction

Тоже принимаю полностью.

Response правильно не пытается снова лечить это новым downstream stage.

### 2.3. `List Consolidation` как missing concept

Это самая ценная новая формулировка в ответе.

Практически Gemini прав:

- нам нужен не просто anti-splitting;
- нам нужен явный контракт, как именно упаковывать lists of names / topics / program items в 1-2 плотных facts.

### 2.4. `Action-oriented hints` worth testing

Это тоже сильная идея.

Текущие `ОШИБКА: ...` действительно слишком диагностические и недостаточно операционные.

## 3. Что беру только с поправками

### 3.1. `[ПЛОХО] / [ОТЛИЧНО]` примеры

Идея полезная, но только в очень compact форме.

Риск:

- если дать слишком много explicit pairs,
- Gemma начнёт literal-copy паттерн,
- а не поймёт shape transformation.

То есть это `accept with modification`.

### 3.2. `List Consolidation` examples

Здесь нужна важная коррекция.

Примеры Gemini:

- `Художницы: ...`
- `В программе: ...`

слишком близки к label-style packaging.

Правильнее тестировать no-label варианты вроде:

- `Лекция охватывает творчество ...`
- `Программа включает ...`
- `Среди героинь лекции ...`

### 3.3. `Revise scaffold` для `посвящ*`

Берём, но без ужесточения в shouting-style.

Текущий human-readable blocking scaffold в целом уже неплох;
проблема пока не в нём одном.

## 4. Что не принимаю буквально

### 4.1. `КРИТИЧЕСКАЯ ОШИБКА` как основной стиль post-extract hints

Не принимаю буквально.

Мы уже видели, что слишком жесткий caps / alarm framing не даёт гарантированного gain и может толкать Gemma в awkward obedience.

### 4.2. Generation-side list formatting как существенный next lever

Не переоцениваю это.

Generation guidance про аккуратные списки можно слегка уточнить, но main leverage всё равно выше в extraction.

Иначе мы снова начнём лечить symptoms downstream.

## 5. Что пойдёт дальше в `v2.10`

Из ответа Gemini реально беру такой narrow patch pack:

1. сохранить sanitizer bypass из `v2.9`;
2. заменить vague anti-splitting на explicit `list consolidation` rule;
3. добавить 1-2 compact `[плохо] -> [хорошо]` examples для intent-style transformation;
4. перевести `_pre_extract_issue_hints` в action-oriented форму `ТВОЯ ЗАДАЧА: ...`;
5. при необходимости слегка уточнить post-extract hint для `посвящ*`, но без нового stage.

## 6. Bottom line

Response quality высокая.

Практический итог:

- **consultation loop before `v2.10`**: stop;
- **blind accept**: нет;
- **go to local `v2.10` dry-run**: да.

Главный take-away:

- `v2.9` закрыл тему synthetic labels;
- следующий рычаг уже не в sanitizer,
- а в том, чтобы заставить Gemma упаковывать dense lists и intent-heavy source text в более publishable fact units.
