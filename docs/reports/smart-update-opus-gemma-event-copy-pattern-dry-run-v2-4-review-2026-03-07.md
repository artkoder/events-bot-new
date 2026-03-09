# Smart Update Opus Gemma Event Copy Pattern Dry-Run V2.4 Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-4-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_v2_4_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v2_4_2026_03_07.py`
- `docs/reports/smart-update-gemini-event-copy-v2-3-dryrun-quality-consultation-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-3-review-2026-03-07.md`

## 1. Краткий verdict

`v2.4` не стал quality win и не готов ни к runtime, ни к новому внешнему раунду как “почти готовая” ветка.

Это был полезный corrective experiment, но по сумме результатов он хуже `v2.3`.

Что реально подтвердилось:

- anti-conversational heading rule полезен;
- stronger program preservation действительно частично помогает `2734`;
- сам Gemini round не был пустым.

Что показал dry-run:

- patch pack оказался слишком широким для sparse/richer balance;
- `2660` и `2745` потеряли выигрыш `compact_fact_led`;
- `2687` заметно регресснул;
- `2673` избавился от question-headings, но стал тяжелее и слабее по coverage;
- `2734` улучшился только частично и всё ещё тащит `посвящ*`.

Мой итог:

- **ветку `v2.4` отклоняю как candidate**;
- следующий шаг должен быть локальным corrective round, а не новый consultation by default.

## 2. Что в `v2.4` реально улучшилось

### 2.1. `2673`: question-headings ушли

Это главный подтверждённый плюс patch pack.

В `v2.3` были:

- `Что в программе?`
- `Что ещё нужно знать?`
- `Зачем создан проект ...?`

В `v2.4` это заменилось на более редакторские:

- `Устройство и возможности платформы`
- `Программа мероприятия`
- `Детали посещения`

То есть рекомендация Gemini про non-question headings была верной.

### 2.2. `2734`: quoted-program preservation частично сработал

Тут виден реальный частичный плюс:

- `missing=7 -> 5`;
- extraction уже попытался вернуть `«Лучший город земли»`;
- структура текста стала чуть собраннее по сравнению с `v2.3`.

Но improvement ограниченный:

- один title всё равно потерялся downstream;
- `посвящ*` остался;
- кейс всё ещё не production-grade.

## 3. Где `v2.4` провалился

### 3.1. `2660` и `2745`: поломка sparse routing

Это самый неприятный системный эффект.

Оба кейса ушли из `compact_fact_led` в `fact_first_v2_4`:

- `2660`: `missing 2 -> 3`
- `2745`: `missing 3 -> 4`

И главное тут не только counts.
Тексты реально стали тяжелее, секционнее и менее живыми.

Это прямой удар по самому сильному достижению `v2.3`.

### 3.2. `2687`: регресс на хорошем grounded case

`2687` был одним из лучших кейсов `v2.3`, а в `v2.4`:

- `missing 1 -> 3`;
- текст стал суше и бюрократичнее;
- в facts снова попали label-style ходы вроде `Тема: ...` и metatext `Лекция расскажет...`.

Это важный сигнал, что extraction patch pack оказался слишком агрессивным и неустойчивым.

### 3.3. `2673`: structural hygiene улучшилась, но итоговый текст стал хуже

Это важная оговорка.

Да, question-headings исчезли.
Но вместе с этим:

- `missing 5 -> 7`;
- текст стал более служебным и менее естественным;
- prose ушёл в explanatory/bureaucratic mode;
- появились почти канцелярские куски вроде `Будет представлен обзор функционала...`.

То есть local fix по headings не превратился в overall quality gain.

### 3.4. `2734`: anti-`посвящ*` фактически не сработал

Это ключевой провал patch pack.

Несмотря на:

- stronger prompt rules;
- возврат `посвящ*` как policy issue;

финальный текст всё равно содержит:

- `посвящён великой любви ...`

Значит текущая реализация guardrail слишком слабая.

Отдельно важно:

- issue `forbidden_marker(посвящ*)` слишком абстрактен;
- Gemma могла просто проигнорировать opaque marker без human-readable rewriting instruction.

## 4. Что `v2.4` доказал про рекомендации Gemini

### 4.1. `accept`: anti-conversational headings

Это подтверждено практикой.

Но это надо сохранять отдельно, без сопутствующего routing drift.

### 4.2. `accept with modification`: program preservation

Идея сильная, реализация пока нет.

Dry-run показывает:

- quoted titles действительно стоит сохранять;
- но reinforcement не должен ломать branch routing;
- и restored titles нельзя терять после pre-consolidation / floor.

### 4.3. `reject as implemented`: current anti-`посвящ*`

Сама цель верная.
Но текущая реализация не сработала.

Нужен следующий уровень точности:

- не `forbidden_marker(посвящ*)`,
- а явная human-readable policy issue:
  - `В тексте нельзя использовать корень «посвящ». Перепиши через тему/историю/фокус события.`

## 5. Почему `v2.4` оказался хуже `v2.3`

Я вижу 4 причины.

### 5.1. Patch pack оказался слишком широким для sparse cases

Усиление fact preservation увеличило fact count и перевело `2660` / `2745` в другой branch.

То есть проблема не только в prose, а в том, что:

- маленькая правка в extraction/facts layer поменяла routing behavior.

### 5.2. Dedup/preservation баланс сломался

`2745` особенно хорошо это показывает:

- в `facts_text_clean` оказались почти дубли:
  - `В постановке исследуются ...`
  - `Постановка исследует ...`

Значит:

- content-preservation floor сработал;
- но последующий dedup уже не выровнял слой достаточно чисто.

### 5.3. Program title reinforcement проходит extraction, но не survives fully downstream

`2734` это демонстрирует буквально:

- `«Лучший город земли»` вернулся в `extracted_facts_initial`;
- но исчез из `facts_text_clean` и из финального текста.

Значит next fix нужен не только на extraction prompt, а и на downstream preservation.

### 5.4. Policy issue wording слишком машинное

Gemma намного лучше реагирует на:

- конкретную редакторскую инструкцию,

чем на:

- opaque diagnostic token.

Это особенно заметно по `посвящ*`.

## 6. Нужен ли новый внешний consultation round сейчас

**Нет.**

Пока не нужен.

Причина:

- failure map уже достаточно локальная и понятная;
- новый внешний раунд сейчас даст меньше пользы, чем ещё одна узкая локальная коррекция;
- мы уже знаем, какие именно 3-4 вещи надо чинить next.

## 7. Что я предлагаю как следующий шаг

Следующий шаг: не `v2.5 redesign`, а узкий corrective round поверх `v2.3` или `v2.4`.

Я бы тестировал такое:

1. Вернуть sparse routing для `2660` / `2745`:
   - либо threshold `<= 6`;
   - либо density-aware routing вместо голого fact count.

2. Оставить anti-conversational heading rule:
   - это verified improvement.

3. Исправить anti-`посвящ*`:
   - human-readable policy issue вместо `forbidden_marker(посвящ*)`;
   - без deterministic regex rewrite.

4. Сохранить named program items после reinforcement:
   - не терять их после pre-consolidation / floor.

5. Почистить post-floor dedup:
   - чтобы не тащить почти дубли как в `2745`.

## 8. Bottom line

`v2.4` полезен как evidence round, но не как candidate branch.

Самые важные выводы:

- Gemini был прав про question-headings;
- частично прав про named program preservation;
- этого оказалось недостаточно для overall quality win;
- текущий patch pack сломал sparse wins `v2.3`.

Значит practical answer сейчас такой:

- не интегрировать;
- не отправлять это сразу наружу как “почти готовый” результат;
- сделать ещё один локальный corrective round с очень узким scope.
