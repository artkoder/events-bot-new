# Smart Update Gemini Event Copy V2.6 Dry-Run Quality Consultation Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-6-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-6-dryrun-quality-consultation-brief.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_6_2026_03_07.py`

## 1. Краткий verdict

Это сильный и достаточно grounded Gemini response.

Главное:

- Gemini правильно прочитал `v2.6` как **net-positive, но не production-ready**;
- хорошо отделил реальные wins (`2660`, `2734`) от structural failures (`2687`, `2673`);
- очень точно локализовал слабость многослойных negative constraints в Extraction;
- дал практичный `v2.7` vector без нового architectural redesign.

Но принимать его literal patch pack нельзя.

Мой итог:

- **новый консультационный раунд сейчас не нужен**;
- response достаточно сильный, чтобы идти в локальный `v2.7`;
- часть prompt examples надо брать только с серьёзной поправкой на factual grounding.

## 2. Что в ответе Gemini принимаю

### 2.1. Positive transformation direction

Это принимаю как главный next-step.

Gemini правильно увидел root cause:

- Gemma плохо исполняет стек из `не делай X / не пиши Y / запрети Z`;
- когда ей не дают образца переписывания, она выбирает плохой обходной путь:
  - `Тема: ...`
  - `Мероприятие анонсирует...`
  - `Презентация посвящена...`

То есть сам direction:

- меньше слепых запретов;
- больше безопасных transformation patterns;

выглядит верным.

### 2.2. `2660` и `2734` действительно подтверждают working moves

Gemini здесь прав:

- compact merge permission — реальный win;
- program-aware protection — тоже реальный win;
- `2734` оздоровился не случайно, а потому что performance/program case получил более пригодный contract.

### 2.3. `2687` и `2673` — это extraction/fact-shaping problem прежде всего

С этим тоже согласен.

Да, generation и revise не безупречны.
Но основной провал всё же upstream:

- label-style facts;
- agenda/metatext facts;
- insufficiently transformed `посвящ*`.

## 3. Что беру только с поправками

### 3.1. Позитивные примеры нельзя превращать в фабрикацию

Это мой главный контраргумент к Gemini.

Примеры вроде:

- `На презентации расскажут об устройстве платформы` -> `Платформа устроена так: ...`
- `расскажут, какую проблему решает проект` -> `Проект решает проблему: ...`

небезопасны.

Причина:

- в raw fact есть topic/agenda;
- но нет factual content о том, **как именно** устроена платформа и **какую именно** проблему она решает.

Такие replacements могут толкать Gemma к выдумыванию.

Правильная безопасная версия должна быть другой:

- `В центре презентации — устройство платформы и её возможности.`
- `На встрече разберут причины появления проекта, его задачи и возможности для участников.`
- `Лекция о творчестве художниц...`

То есть positive transformation нужна, но только в forms that preserve uncertainty/agenda honestly.

### 3.2. Anti-metatext нельзя трактовать как абсолютный ban на любой event framing

Gemini чуть перегибает, когда предлагает почти полный запрет на конструкции типа:

- `Презентация посвящена...`
- `Мероприятие анонсирует...`

Суть проблемы не в любом event framing, а в:

- generic service metatext;
- пустой канцелярщине;
- bureaucratic self-reference.

Но такие конструкции, как:

- `В центре лекции — ...`
- `Программа вечера построена вокруг ...`
- `На встрече разберут ...`

могут быть вполне нормальными и grounded.

### 3.3. `<= 6` не надо объявлять решённым законом

На этот раунд routing действительно не выглядит главным blocker.

Но формулировка Gemini:

- `работает абсолютно корректно`

слишком сильная.

Правильнее:

- для `v2.7` routing можно не трогать;
- но считать его окончательно закрытым вопросом пока нельзя.

## 4. Что не беру буквально

### 4.1. Literal `ВМЕСТО -> ПИШИ` примеры, которые требуют новых фактов

Это reject as literal copy.

Если proposed pattern требует конкретики, которой нет в raw fact, его нельзя вставлять в Extraction prompt напрямую.

### 4.2. Полный запрет на форматные указатели

Нельзя требовать, чтобы Gemma всегда говорила только “о предмете”, как будто это не lecture/presentation/case-announcement.

Для части событий честный human-sounding framing всё же нужен.

### 4.3. `Routing healthy` как завершённый вывод

Это пока не принимаю как окончательную истину.

## 5. Что реально пойдёт в `v2.7`

Из ответа Gemini я беру следующее:

1. Перевести extraction contract в режим safe positive transformation examples.
2. Переписать `_pre_extract_issue_hints`, чтобы они показывали допустимые grounded rewrites.
3. Усилить standard generation anti-bureaucracy, но без запрета на любой event framing.
4. Усилить revise/policy wording для `посвящ*`, чтобы оно требовало полной rewrite, а не косметической правки.
5. Не трогать routing в `v2.7`, кроме already existing blocker logic.

## 6. Bottom line

Этот Gemini response достаточно сильный, чтобы закрыть текущий consultation round.

Если коротко:

- **accept**: diagnosis про negative constraints, positive transformation direction, no-more-theory verdict;
- **accept with modification**: anti-metatext rules, `<= 6`, rewrite examples;
- **reject as literal**: любые transformation patterns, которые заставляют Gemma додумывать отсутствующие детали.

Практический вывод:

- **ещё один Gemini раунд сейчас не нужен**;
- следующий шаг уже локальный: собрать `v2.7` на safe positive transformations и прогнать его на тех же 5 событиях.
