# Smart Update Gemini Event Copy V2.15.3 Dry-Run Text Consultation Review

Дата: 2026-03-08

Основание:

- Raw Gemini report: [event-copy-v2-15-3-dryrun-text-consultation-gemini-3.1-pro-2026-03-08.md](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-15-3-dryrun-text-consultation-gemini-3.1-pro-2026-03-08.md)
- Session log: [session-2026-03-08T19-25-e483ce8e.json](/home/vscode/.gemini/tmp/events-bot-new/chats/session-2026-03-08T19-25-e483ce8e.json)
- [v2.15.3 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-3-5-events-2026-03-08.md)
- [v2.15.3 texts vs baseline](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-3-texts-vs-baseline-2026-03-08.md)
- [v2.15.3 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-prompt-context-2026-03-08.md)
- [v2.15.3 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-3-review-2026-03-08.md)

## 1. Короткий verdict

Ответ Gemini полезный и в целом хорошо откалиброван именно по text-quality проблемам `2.15.3`.

Самое ценное:

- очень точно разобран `2673`;
- хорошо пойман новый failure mode `2687` с ложной логической связкой вокруг "британского следа";
- полезно зафиксированы weak openings, stop-phrases и syntax-level positive transformations для Gemma.

Но ответ нельзя принимать буквально как общий oracle verdict.

Итог:

- **worth taking**
- **accept with modification**
- как guide для следующего prompt round — да
- как final verdict по сравнению с baseline — нет

## 2. Что принимаю

### 2.1. `2673` действительно остаётся главным editorial blocker

Gemini здесь попал точно.

Что реально плохо:

- текст начинает не с сути проекта, а с второстепенной программы;
- фраза `Платформа предоставляет задачи и возможности` действительно звучит искусственно;
- в конце остаётся agenda-recap вместо живого project presentation prose.

Это совпадает и с нашим собственным чтением кейса.

### 2.2. `2687` сломан не только стилистически, но и логически

Gemini правильно заметил, что проблема `2687` уже не в false epigraph, как раньше, а в другой ошибке:

- heading и следующий абзац заставили Gemma стянуть разные блоки фактов в один ложный смысловой узел;
- "русские художницы с британскими корнями" оказалось слишком широким umbrella для ближайших facts.

Это важное уточнение:

- текущая проблема здесь уже не extraction-only;
- она сидит в interaction между plan, heading style и generation.

### 2.3. Project lead rule для `presentation_project` надо ужесточать

Это сильный и practical carry.

Принимаю принцип:

- для `presentation_project` первое предложение обязано называть сам проект / продукт / платформу;
- развлечения, программа и второстепенные детали не должны открывать текст.

Это один из самых полезных next-step signals.

### 2.4. Регалии и статусы надо интегрировать в первое упоминание, а не бросать в конец

Это тоже полезно.

`2734` действительно проигрывает, когда регалия:

- не интегрирована в lead;
- а просто приклеена в последний момент.

Такой prompt-level rule для Gemma выглядит реалистичным и недорогим.

### 2.5. Нужен stronger anti-abstract prose contract

Gemini справедливо указывает на фразы вроде:

- `сложные философские вопросы`
- `хрупкая грань`
- `имена часто остаются в тени`

Это действительно типичный LLM drift.

Сдвиг от abstract praise к concrete detail-driven prose я принимаю как важную следующую цель.

## 3. Что беру только с поправками

### 3.1. Общий verdict `v2.15.3` хуже baseline

Gemini формулирует это слишком категорично.

Я бы уточнил так:

- по pure editorial quality baseline местами действительно сильнее;
- но как system candidate `v2.15.3` не хуже baseline во всём подряд;
- `2745` и partly `2687` уже лучше baseline по сумме readability/hygiene.

То есть тезис полезен как warning, но не как абсолютный итог.

### 3.2. Hard-ban список stop-phrases

Список слабых фраз полезен.
Но переносить это в giant ban wall нельзя.

Правильнее:

- короткий stop-phrase bank;
- несколько positive transformations;
- downstream validation на persistent failures.

Не:

- ещё одна огромная отрицательная prompt wall.

### 3.3. Blanket rule `use_headings=false` для `lecture_person` при facts `< 8`

Сигнал полезен, но literal threshold выглядит слишком грубо.

Проблема `2687` не только в количестве facts, а в типе semantic load:

- grouped names
- subgroup relation
- высокая вероятность ложной heading metaphor

Я принимаю не threshold как догму, а более мягкую мысль:

- lecture/person-rich cases с перечислениями нуждаются в более жёстком heading gate;
- иногда вообще безопаснее идти без headings.

### 3.4. Сохранение quoted program items как hard invariant

Это полезно для `2734`, но с поправкой.

Нельзя превращать любой program-rich case в raw list dump.

Принимаю только:

- сильнее защищать конкретные именованные пункты;
- не обобщать их до пустой формулы вроде `песни, созданные для ...`.

## 4. Что не принимаю буквально

### 4.1. Перевод hard bans в основной semantic core

Gemini местами толкает в сторону более механического governance.

Я этого не беру:

- нельзя снова превращать систему в regex-first semantic editor;
- repair и validation могут стать чуть жёстче;
- но смысл и prose должны по-прежнему рождаться в LLM path.

### 4.2. Некоторые literal positive transformations

Например:

- `Начинай текст со строгой формулы: "[Название] — это [суть проекта]"`

Как bias это полезно.
Как обязательный literal template — слишком рискованно и снова может сделать прозу одинаковой.

Нужно брать как structural hint, а не как жёсткий унифицированный lead template.

## 5. Что это значит для следующего шага

Если брать только подтверждённые практикой вещи, то следующий round должен нести:

1. Stronger `presentation_project` lead contract:
   - первый абзац всегда начинается с сути проекта, а не с развлекательной программы.
2. Stronger anti-abstract prose block:
   - меньше псевдофилософии;
   - больше concrete detail-driven phrasing.
3. Better heading policy for lecture/person cases:
   - headings либо сильнее ограничены, либо вообще выключаются для fragile grouped-person cases.
4. Status/regalia integration rule:
   - регалии интегрируются в первое упоминание имени.
5. Narrow protection for named program items:
   - program facts нельзя обобщать до пустой общей формулы.

## 6. Финальный вывод

Gemini на этом шаге был особенно полезен не по architecture, а по тексту:

- naturalness;
- AI-cliche detection;
- lead quality;
- heading quality;
- prompt-level guidance for Gemma.

То есть консультация дала не новую архитектуру, а хороший следующий corrective pack именно по prose behavior.
