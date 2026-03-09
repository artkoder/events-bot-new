# Smart Update Gemma Deep Research Impact On V2.15.2

Дата: 2026-03-08

Исходные документы:

- [using-gemma-deep-research-report.md](/workspaces/events-bot-new/docs/researches/using-gemma-deep-research-report.md)
- [using-gemma-deep-research-report-gemini.md](/workspaces/events-bot-new/docs/researches/using-gemma-deep-research-report-gemini.md)
- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)

## 1. Что реально было дано на вход

Теперь содержательный материал есть в обоих документах:

- [using-gemma-deep-research-report.md](/workspaces/events-bot-new/docs/researches/using-gemma-deep-research-report.md)
- [using-gemma-deep-research-report-gemini.md](/workspaces/events-bot-new/docs/researches/using-gemma-deep-research-report-gemini.md)

Но их полезность несимметрична:

- основной широкий practical signal по-прежнему приходит из первого, большого research report;
- второй, Gemini-based report короче, более догматичен по ряду советов, но даёт несколько полезных Gemma-specific уточнений по formatting, schema discipline и управлению креативностью.

## 1.1. Что конкретно взято из `using-gemma-deep-research-report.md`

Из основного research report приняты:

- self-contained prompts для Gemma;
- multi-step short-chain как default;
- per-step decoding profiles;
- structure-first JSON/planner outputs;
- короткие positive examples вместо длинных few-shot walls;
- fine-tuning как legit next-step path при plateau.

Не взято как immediate architectural action:

- `RAG` как core fix для event-copy;
- function-calling как relevant change для текущего prose flow;
- safety classifiers как решение текущих text-quality проблем.

## 1.2. Что конкретно взято из `using-gemma-deep-research-report-gemini.md`

Из Gemini-based research приняты:

- prompts лучше работают в секционированной форме (`ROLE / RULES / OUTPUT / FACTS`);
- schema semantics важны не меньше, чем schema shape;
- repetition-control стоит рассматривать как optional prose experiment;
- скрытая step decomposition полезна, но reasoning не должен утекать в final output.

Не принято как core approach:

- heavy visible SSR внутри generation;
- massive negative prompting как основной способ style-control;
- превращение generation prompt в длинную аналитическую инструкцию.

## 2. Что из deep research полезно именно для нашей задачи

### 2.1. Self-contained prompts for Gemma

Самый practically useful сигнал:

- Gemma IT не должна предполагать скрытую `system`-роль;
- правила каждого независимого вызова должны жить внутри самого запроса;
- нельзя рассчитывать на то, что предыдущий шаг или "общий системный контекст" магически перенесётся в следующий call.

Фактическое влияние:

- в `2.15.2` и далее каждый step должен быть self-contained;
- critical rules нельзя держать только в общих абстрактных инструкциях "где-то сверху".

### 2.2. Multi-step chains are the right default

Отчёт прямо подтверждает то, к чему мы уже пришли dry-run-практикой:

- для Gemma короткие цепочки вызовов с ясными ролями устойчивее, чем один giant prompt;
- validate/repair паттерн — нормальный production path, а не "костыль".

Фактическое влияние:

- `normalize -> plan -> generate -> validate -> narrow repair` остаётся правильной базой;
- не нужно откатываться к одному всеядному prompt.

### 2.3. Decoding profiles must be step-specific

Очень полезный сигнал:

- JSON extraction / planning / repair должны быть максимально детерминированными;
- main prose generation может быть умеренно креативнее;
- одинаковый decoding profile для всех шагов — плохая идея.

Фактическое влияние:

- low temperature / schema-first для `normalize_floor`, planner и repair;
- умеренно более свободный режим только для main prose step.

### 2.4. Structure-first outputs beat freeform explanations

Отчёт полезно подчёркивает:

- для extraction/classification не нужен chain-of-thought в output;
- лучше структурированный результат, evidence-like grounding и валидация;
- длинные reasoning outputs только раздувают prompt budget и ломают format compliance.

Фактическое влияние:

- planner остаётся structural-only;
- normalize остаётся JSON-only;
- repair не должен превращаться в "объясни, как ты думаешь".

### 2.5. Short high-quality examples are useful, giant few-shot walls are not

Это хорошо совпадает с нашими dry-run выводами:

- Gemma лучше держит короткие positive transformations и anti-pattern examples;
- длинные few-shot блоки легко начинают конкурировать с основными правилами.

Фактическое влияние:

- в prompts стоит использовать короткие `плохо -> хорошо` примеры;
- не стоит тащить большие few-shot packs внутрь happy-path generation.

### 2.6. Fine-tuning is a real next step if prompting plateaus

Это важный стратегический вывод, а не immediate action item.

Если после `2.15.x`:

- prose quality остаётся нестабильной;
- repair loop rate не падает;
- JSON/schema compliance приходится постоянно чинить prompt-ами;

то fine-tuning Gemma на curated event-copy dataset становится не "роскошью", а рациональным следующим шагом.

Фактическое влияние:

- не сейчас, но это нужно держать как legit path после стабилизации architecture и prompt contracts.

### 2.7. Markdown-style sectioning inside prompts helps Gemma separate instruction layers

Новый Gemini report полезно акцентирует:

- Gemma лучше держит запрос, когда instruction blocks визуально и семантически разведены;
- крупные section labels вроде `РОЛЬ`, `ПРАВИЛА`, `ФОРМАТ ОТВЕТА`, `ФАКТЫ` помогают не смешивать задачу, ограничения и входные данные;
- это особенно полезно там, где нет нативной надёжной `system`-роли.

Фактическое влияние:

- в self-contained prompts для `normalize_floor`, planner и generation нужно сохранять явную секционную структуру;
- это не значит делать prompts длиннее, а значит делать их более организованными.

### 2.8. Schema semantics matter, not only schema shape

Gemini report отдельно усиливает мысль, которая в первом отчёте была слабее выражена:

- для Gemma важно не только наличие JSON schema;
- полезны semantic hints на уровне enum choices, field purpose и output expectations.

Фактическое влияние:

- в planner и normalize contracts стоит усиливать field descriptions/examples там, где модель чаще ошибается;
- особенно это относится к:
  - `pattern`
  - `use_epigraph`
  - `use_headings`
  - `blocks.kind`

### 2.9. Repetition control should be treated as a prose-quality lever, but only experimentally

Новый Gemini report много внимания уделяет repetition penalty и контролю повторов.

Полезное зерно здесь есть:

- repeated wording — одна из реальных болей event-copy;
- для Gemma repetition control может быть отдельным inference lever, а не только prompt problem.

Но это надо брать осторожно:

- не как новый mandatory decoding rule;
- а как optional experiment для main prose step, если API/runtime поддерживает этот параметр без побочных эффектов.

Фактическое влияние:

- добавить repetition-control в backlog of experiments можно;
- в текущую архитектуру `2.15.x` это не должно встраиваться как core assumption.

### 2.10. Visible step-by-step reasoning is still not a fit for final event-copy outputs

Gemini report местами советует SSR / явное пошаговое выполнение.

Это полезно только частично.

Что принимается:

- скрытая декомпозиция задачи на шаги полезна;
- micro-planning и structural decomposition уже доказали ценность.

Что не принимается:

- выводить reasoning chain;
- превращать generation prompt в numbered analytical procedure;
- заставлять финальный event text нести следы внутреннего "шаг 1 / шаг 2 / шаг 3" мышления.

Фактическое влияние:

- decomposition остаётся архитектурной;
- финальный prose step должен оставаться чистым editorial output, а не trace of reasoning.

## 3. Что из deep research не меняет текущую архитектуру

### 3.1. RAG / EmbeddingGemma

Для текущей задачи это не главный рычаг.

Почему:

- наши факты уже приходят из source posts;
- проблема сейчас не в недостающем внешнем знании;
- проблема в quality of normalization, planning and generation.

Вывод:

- `RAG` полезен как общая Gemma best practice, но не как next-step core fix для event-copy `2.15.x`.

### 3.2. Function calling

Это полезно для других частей системы, но почти не влияет на текущий event-copy prose flow.

### 3.3. Safety classifiers like ShieldGemma

Это может быть полезно позже для более широкого moderation layer, но не решает наши текущие проблемы:

- false quote
- generic headings
- agenda-like prose
- epigraph misuse

### 3.4. Aggressive negative prompting and heavy SSR as default style-control

Во втором research report есть полезные наблюдения, но и потенциально опасный уклон:

- слишком сильная опора на massive negative constraints;
- тяжёлые role blocks;
- явные step-by-step instructions прямо внутри generation.

Для нашей задачи это скорее риск, чем core solution.

Почему:

- мы уже видели, что Gemma на длинных wall-of-rules начинает терять часть инструкций;
- текущая задача — не analytical memo, а живой event-copy text;
- visible SSR легко делает prose более искусственным и тяжёлым.

## 4. Фактическое влияние на `2.15.2` brief

Из deep research в brief стоит добавить и закрепить:

1. Каждый Gemma step — self-contained prompt; не полагаться на скрытую `system`-роль.
2. У каждого step свой decoding profile:
   - JSON/planner/repair — максимально детерминированно
   - prose generation — умеренно креативно
3. Planner и normalization остаются structure-first, без CoT и без freeform reasoning.
4. В prompts использовать короткие high-quality positive transformations, а не длинные few-shot walls.
5. Явно секционировать prompts через короткие смысловые блоки (`ROLE`, `RULES`, `OUTPUT`, `FACTS`), а не сваливать всё в один абзац.
6. Усилить semantic guidance внутри JSON contracts и planner outputs, а не надеяться только на type-shape.
7. В success criteria добавить:
   - JSON validity rate
   - repair loop rate
   - plan compliance rate
8. Repetition-control оставить как optional prose experiment, не как core architectural dependency.
9. Fine-tuning зафиксировать как legit next-step path, если prompt-only refinement перестанет давать ощутимый прогресс.

## 5. Bottom line

Deep research не перевернул архитектуру, но дал важную калибровку:

- текущий `LLM-first`, short-step, validate/repair direction — правильный;
- giant prompts и скрытые "system" assumptions для Gemma — плохой путь;
- следующий уровень зрелости — не больше regex, а лучше step discipline, better prompt sectioning, per-step decoding, stronger schema semantics и, при необходимости, fine-tuning.
