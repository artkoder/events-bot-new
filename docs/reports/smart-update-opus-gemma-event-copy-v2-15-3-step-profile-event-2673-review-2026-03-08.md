# Smart Update Gemma Event Copy V2.15.3 Step Profile Review — Event 2673

Дата: 2026-03-08

Основание:

- [step profile report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-v2-15-3-step-profile-event-2673-2026-03-08.md)
- Full trace JSON: [trace.json](/workspaces/events-bot-new/artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/trace.json)
- [v2.15.3 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-3-5-events-2026-03-08.md)
- [v2.15.3 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-3-prompt-context-2026-03-08.md)

## 1. Зачем этот profile важен

`2673` сейчас один из самых проблемных кейсов.

Step-profile нужен, чтобы не гадать:

- проблема в prompt'ах;
- проблема в planner semantics;
- проблема в validation/repair;
- или проблема вообще раньше, в shape/fact preparation.

Этот trace показывает: основной провал сидит не в одном месте, а в связке `normalize -> deterministic plan -> generate`.

## 2. Короткий verdict

Главный вывод:

- `2.15.3` на `2673` ломается **не потому, что repair слабый**;
- и **не потому, что regex/sanitizer всё испортил**;
- а потому, что система уже до generation подаёт Gemma неверный rhetorical priority.

Дополнительный важный сигнал:

- single-event rerun на `2673` дал лучший deterministic result, чем batch dry-run `2.15.3`;
- это значит, что у prompt-pack сейчас есть ещё и проблема stability / variance, а не только неправильной логики.

То есть:

- system может иногда сгенерировать заметно более приличный вариант даже без смены architecture;
- но результат пока недостаточно воспроизводим, чтобы считать это solved case.

То есть:

1. `normalize_floor` даёт факты в формально чистом виде, но не расставляет важность;
2. deterministic planner безошибочно, но неумно кладёт в lead факт про сценическую программу;
3. generation obediently открывает текст собаками и стихами;
4. targeted repair улучшает literal coverage, но закрепляет agenda-recap prose.

Это хороший сигнал:

- core проблема здесь решаема prompt engineering'ом и planner contract'ом;
- не нужно уходить в regex-first.

## 3. Сильные стороны по шагам

### 3.1. Step B/C `normalize_floor`

Сильное:

- сервисный мусор и registration correctly dropped;
- `расскажут о ...` трансформировано в более предметные chunks;
- quote hallucination вообще не участвует;
- facts не потерялись wholesale.

То есть normalize step не катастрофический.

### 3.2. Step G/H `generate_description`

Сильное:

- текст self-contained;
- нет headings leak;
- нет epigraph leak;
- нет forbidden markers;
- Gemma реально слушает shape-specific prose constraints лучше, чем в старых rounds.

### 3.3. Validation / repair pipeline

Сильное:

- pre-repair missing correctly показывает, что generation слишком много обобщил;
- targeted repair реально снижает missing с `7` до `2`;
- repair при этом не ломает формат.

То есть support layer технически работает.

## 4. Слабые стороны по шагам

### 4.1. Главная слабость `normalize_floor`: facts стали чище, но semantic hierarchy не появилась

Normalize step делает:

- `задачи платформы`
- `возможности платформы`
- `причины появления проекта`
- `проблема, которую решает проект`

Формально это clean facts.
Практически это очень weak facts для generation:

- они agenda-like;
- они плохо читаются как publishable semantic units;
- они почти неизбежно провоцируют робота на recap sentence.

Отдельная проблема:

- `проект «Собакусъел» — социальная сеть...` не поднимается до clearly dominant fact;
- entertainment fact и project fact остаются слишком "равноправными".

### 4.2. Главная слабость planner: deterministic lead selection выбрал неправильный opening anchor

Формально planner не сломан:

- shape=`presentation_project`
- pattern=`scene_led`
- no headings
- no epigraph

Но structural plan всё равно плохой:

- `lead: facts 1,2`
- а `fact 1` = `выступления артистов, шоу дрессированных собак и чтение стихов`

Именно здесь система закладывает будущий провал лида.

Для `presentation_project` такой plan не должен быть допустим.

### 4.3. Главная слабость generation: obedient but wrong

Generation prompt не ломается технически.
Он делает то, что ему дали.

Проблема в том, **как** он делает это:

- усиливает второстепенную программу (`артистичные трюки`, `смешные номера`);
- invents embellishment not present in facts;
- смещает проект во второе предложение;
- превращает agenda-like facts в bureaucratic prose:
  - `платформа предоставляет возможности`
  - `платформа устроена так, чтобы...`

То есть prompt пока не удерживает:

- project-first rhetoric;
- anti-embellishment discipline;
- anti-agenda recap discipline.

### 4.4. Repair чинит coverage, но не чинит rhetoric

Repair полезен как factual patch.
Но именно на `2673` он показывает свой предел:

- literal missing снижается;
- prose quality не растёт;
- last sentence становится ещё более agenda-like:
  - `Также будут затронуты задачи платформы...`

Это означает:

- repair step не должен быть главным средством улучшения copy;
- ему не хватает issue type `bad lead / agenda recap / secondary-first opening`.

## 5. Где именно живёт root cause

Если разложить по важности:

### Root cause 1

`presentation_project` сейчас не имеет достаточно жёсткого `lead priority contract`.

Пока этого нет, даже хороший normalize будет иногда вести к плохому opening.

### Root cause 2

`normalize_floor` для project cases создаёт слишком много noun-phrase agenda facts и слишком мало publishable semantic facts.

То есть:

- `задачи платформы`
- `возможности платформы`
- `устройство платформы`

слишком слабы как facts for prose.

### Root cause 3

Generation prompt всё ещё допускает decorative inflation:

- `артистичные трюки`
- `смешные номера`
- `пространство для ...`
- `предоставляя возможности ...`

Это уже не coverage issue, а style control issue.

## 6. Что это значит practically

### 6.1. Что не надо делать

Не надо лечить `2673` через:

- новые regex semantics;
- ещё один giant ban-list;
- ещё один full rewrite pass;
- дополнительный heavy planner step.

### 6.2. Что стоит менять в prompt engineering

Для `presentation_project` нужен следующий prompt pack correction:

1. `normalize_floor`
   - меньше abstract agenda nouns;
   - больше project-definition and problem-solution phrasing;
   - stronger instruction that one fact must explicitly answer `что это за проект`.

2. `shape_and_format_plan`
   - hard rule: для `presentation_project` lead не может начинаться с program/entertainment fact, если есть project-definition fact.

3. `generate_description`
   - first sentence must name the project/product/platform;
   - entertainment/program details only after project identity is established;
   - stronger anti-embellishment examples;
   - stronger anti-agenda-recap rule.

4. `targeted_repair`
   - должен уметь чинить не только missing facts, но и `secondary-first opening`;
   - должен получать issue type вроде `bad_project_lead`.

## 7. Самые полезные следующие гипотезы

Для узкого prompt-tuning round по `2673` я бы тестировал в таком порядке:

1. В `normalize_floor` заменить weak facts:
   - `задачи платформы`
   - `возможности платформы`
   - `устройство платформы`
   на более publishable fact forms, но без выдумывания.

2. В planner добавить жёсткий guard:
   - если shape=`presentation_project` и есть project-definition fact, он обязан быть в lead.

3. В generation prompt добавить explicit anti-secondary opening:
   - нельзя начинать с программы, если событие — презентация проекта.

4. В generation prompt усилить anti-embellishment:
   - нельзя добавлять `артистичные`, `смешные`, `проникновенные`, если этого нет в facts.

5. В repair prompt добавить отдельную issue category:
   - `opening starts with secondary detail instead of project identity`.

## 8. Финальный вывод

Этот profile уже полезен сам по себе.

Он показывает:

- текущая проблема `2673` хорошо локализуется;
- prompt tuning здесь действительно имеет смысл;
- но tuning должен идти не "вообще по тексту", а по конкретным step contracts.

Это именно тот материал, который уже имеет смысл нести в следующий узкий `Opus`/`Gemini` round:

- с реальным trace;
- с prompt files;
- с exact per-step artifacts;
- и с чётким вопросом не про архитектуру вообще, а про качество prompt behavior на каждом шаге.
