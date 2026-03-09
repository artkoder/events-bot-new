# Smart Update Gemini Event Copy V2.10 Dry-Run Quality Consultation Response Review

Дата: 2026-03-08

Связанные материалы:

- `docs/reports/smart-update-gemini-event-copy-v2-10-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-10-dryrun-quality-consultation-brief.md`
- `docs/reports/smart-update-gemini-event-copy-v2-10-prompt-context.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-5-events-2026-03-08.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-10-review-2026-03-08.md`

## 1. Краткий verdict

Это сильный post-run response.

Ещё один консультационный подэтап перед `v2.11` не нужен.

Gemini правильно прочитал core shape `v2.10`:

- `list consolidation` itself is real;
- main failure moved to project/presentation nominalization;
- `2660` looks like anti-quote / over-literal packaging failure;
- новый stage по-прежнему не нужен.

Но переносить его советы literally нельзя.

Главная поправка:

- structural alternatives вроде `В центре внимания — ...` полезны;
- но если сделать их too hard and universal, Gemma снова начнёт штамповать один opening pattern.

## 2. Что в ответе Gemini принимаю

### 2.1. `Nominalization` — главный новый рычаг

Это самое сильное место ответа.

Gemini правильно увидел, что `2673` упёрся не просто в intent-style ban, а в неумение превратить:

- `зачем появился`
- `какую проблему решает`
- `как устроена`

в компактные noun-phrase facts.

Это выглядит более точной framing, чем просто очередной `anti-intent`.

### 2.2. `Anti-quote rule` worth testing

Тоже принимаю.

`2660` действительно показал, что текущий contract местами толкает Gemma в safe-but-awful quoting behavior.

### 2.3. `No extra stage`

Согласен.

Response не уводит нас в новый repair gate и это правильно.

## 3. Что беру только с поправками

### 3.1. Hard structural alternative for `посвящ*`

Идея полезна, но не в виде single mandatory scaffold.

Если жёстко заставить Gemma всегда писать:

- `В центре внимания — ...`

мы снова рискуем получить template overuse.

Правильнее:

- дать 2-3 safe structural alternatives,
- но не один compulsory opening.

### 3.2. `Targeted hint routing by event_type`

Это `accept with modification`.

Да, presentation/project cases реально требуют другой hint emphasis.
Но не хочется вводить полноценный branching explosion.

Лучше:

- мягкий targeted hint injection для `presentation / business / project`,
- а не новый parallel extraction contract для half the system.

## 4. Что не принимаю буквально

### 4.1. Полная прозрачность причины `2660`

Gemini почти наверняка прав насчёт anti-quote,
но я бы не утверждал, что это уже доказанный single root cause.

Там мог сыграть и общий rhetorical over-structuring.

То есть anti-quote worth testing,
но как hypothesis, а не как settled fact.

### 4.2. `List consolidation` only for entity cases

Не сужаю правило так резко.

`List consolidation` itself still valuable broadly,
проблема скорее в том, что для project/presentation sources нужен second transformation rule:

- сначала nominalize concept blocks,
- потом already consolidate them.

## 5. Что реально пойдёт дальше в `v2.11`

Из ответа Gemini беру такой narrow patch pack:

1. добавить extraction-level anti-quote rule;
2. добавить explicit nominalization examples for clause-style intents;
3. сделать `посвящ*` repair more structural, but with 2-3 allowed scaffolds, not one;
4. добавить targeted project/presentation hint emphasis;
5. сохранить sanitizer bypass, generation hygiene and list consolidation core.

## 6. Bottom line

Response quality высокая.

Практический итог:

- **consultation loop before `v2.11`**: stop;
- **blind accept**: нет;
- **go to local `v2.11` dry-run**: да.

Главный take-away:

- `v2.10` доказал ценность `list consolidation`;
- следующий рычаг уже не в общей плотности,
- а в более точной transform-логике для project/presentation clauses and anti-quote control.
