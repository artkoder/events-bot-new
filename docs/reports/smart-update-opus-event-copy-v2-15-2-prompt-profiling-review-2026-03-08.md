# Smart Update Opus Event Copy V2.15.2 Prompt Profiling Review

Дата: 2026-03-08

Основание:

- [Opus raw prompt profiling report](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-15-2-opus-prompt-profiling-2026-03-08.md)
- [Opus raw JSON](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-15-2-opus-prompt-profiling-2026-03-08.json)
- [v2.15.2 design brief](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-design-brief-2026-03-08.md)

## 1. Verification

Консультация проведена одним strict one-shot запуском `Claude CLI`.

Проверка raw JSON:

- `modelUsage` содержит только `claude-opus-4-6`
- second run / fallback не запускались

Это валидная `Opus-only` консультация.

## 2. High-level verdict

Ответ `Opus` полезный и в целом точно попадает в prompt-level bottleneck `2.15.2`:

- главный риск сейчас действительно не в общей идее atomic flow, а в перегрузе основного prose prompt;
- inline ban-lists и дублирование enforcement между prompt и validator для Gemma вредны;
- structural planning действительно нужно держать максимально сухим;
- repair без facts — слабая конструкция.

Но часть его architecture moves я не принимаю буквально:

- не считаю правильным автоматически убивать Step B как отдельную сущность;
- не считаю, что positive examples нужно полностью убрать;
- не считаю, что pattern library нужно свести к минимализму любой ценой.

Итог:

- prompt-profiling `Opus` принимается как сильная основа для `2.15.2`;
- architecture changes берутся selectively, с учётом наших целей по text quality и variability.

## 3. Что принимаю

### 3.1. Dynamic prompt assembly

Это, вероятно, самый полезный конкретный совет в ответе.

Принятие:

- основной generation prompt должен собираться в Python;
- blocks для headings / epigraph / list должны попадать в prompt только when active;
- stop-phrase enforcement не должен занимать место внутри каждого generation call.

Почему это правильно:

- это снижает prompt load без semantic drift;
- это масштабируется;
- это не превращает систему в regex-heavy core.

### 3.2. Register over persona

Принятие:

- перейти от абстрактной persona к более короткому register anchor;
- например, `пиши как для раздела культуры городской газеты`.

Почему:

- Gemini тоже указывал, что Gemma лучше слушается concrete style framing;
- это короче и чище, чем "ты редактор культурной афиши".

### 3.3. Quote metadata inside normalization

Принятие:

- `normalize_floor` должен уметь возвращать:
  - `has_direct_quote`
  - `quote_text`
  - `quote_speaker`

Почему:

- epigraph/blockquote не должны решаться на "интуиции" generation step;
- это делает quote gating grounded и explainable.

### 3.4. Repair must receive facts

Принятие:

- `targeted_repair` должен получать issues + original facts + description;
- repair не должен работать "в вакууме".

Почему:

- без facts Gemma часто лечит banned phrase на другой канцелярит;
- repair становится менее grounded.

### 3.5. Anti-duplication stays P0

Это не новость, но `Opus` правильно подтверждает приоритет.

Принятие:

- anti-dup remains central in generation + validation;
- duplication нельзя считать вторичным cosmetic defect.

## 4. Что принимаю с поправками

### 4.1. Step B should shrink, but not vanish blindly

`Opus` предлагает фактически слить planner в generation или перевести его почти полностью в Python.

Моя позиция:

- да, planner должен стать маленьким;
- да, format gates лучше считать cheap/deterministic;
- но полностью отказываться от отдельного planning layer рано.

Почему:

- user-facing цель включает вариативность текста и реальные macro-patterns;
- tiny/hybrid planner полезен для ambiguous rich cases;
- отдельный planning layer снижает риск того, что generation снова станет единственным overloaded super-prompt.

Итог для `2.15.2`:

- Step B сохраняется;
- но становится `deterministic / tiny-hybrid`, а не полноценным free-choice prose planner.

### 4.2. Pattern reduction to four core execution patterns

`Opus` предлагает сократить library с 6 до 4 execution patterns:

- `scene_led`
- `quote_led`
- `person_led`
- `program_led`

Я это принимаю как internal execution layer, но с оговоркой:

- внешне мы не обязаны считать, что "паттернов стало меньше" в editorial sense;
- `theme` и `project` можно сохранить как higher-level editorial cards или aliases, если они потом мапятся на 4 execution shapes.

Это компромисс:

- Gemma получает более чёткие structural contracts;
- мы не теряем мысль, что живые редакторы используют больше одного крупного шаблона.

### 4.3. Anti-pattern lead examples

`Opus` предпочитает 2-3 short bad examples вместо набора good templates.

Это полезно, но не как абсолют:

- Gemini ранее был прав про positive transformations;
- поэтому для `2.15.2` лучше не чистая negative framing и не набор positive templates;
- а short structural rule + a few anti-pattern examples + minimal positive transformation wording.

### 4.4. Deterministic format gates

Принятие с поправкой:

- `epigraph`, `headings`, `list_block` действительно должны по возможности gate-иться детерминистически;
- но optional tiny LLM override для genuinely ambiguous richer cases можно сохранить как support function.

Главное:

- optional enhancer не должен иметь power испортить otherwise-clean output.

## 5. Что не принимаю как есть

### 5.1. Полный отказ от Step B

Не принимаю.

Причина:

- это слишком быстро снова вернёт всё решение в overloaded generation prompt;
- а мы как раз от этого уходим.

### 5.2. Полный отказ от positive examples

Не принимаю.

Причина:

- earlier Gemini guidance и наш own cycle показали, что для Gemma полезны позитивные transformation hints;
- проблема не в positive guidance как таковой, а в template-heavy examples.

### 5.3. Тотальная ставка на Python-side pattern selection

Не принимаю как догму.

Причина:

- shape heuristics нужны;
- но слишком жёсткий deterministic selector легко станет новой fragile routing table.

## 6. What this changes in 2.15.2

После этого review архитектурная форма `2.15.2` выглядит так:

### Step A. `normalize_floor`

LLM.

Responsibilities:

- clean factual normalization of the whole floor;
- quote metadata extraction;
- preservation of grouped program/person blocks.

### Step B. `shape_and_format_plan`

Deterministic by default, tiny-hybrid when needed.

Responsibilities:

- narrow available patterns;
- gate `epigraph` / `headings` / `list_block`;
- optionally return `fact_ids` blocks for richer cases;
- no prose.

### Step C. `generate_description`

Main prose step.

Responsibilities:

- dynamic prompt built from:
  - compact core rules
  - pattern-specific structural hint
  - only active formatting blocks
- no duplicated inline stop-phrase wall
- register-based style anchor

### Step D. `validate_description`

Deterministic.

Responsibilities:

- lexical bans
- generic headings
- duplication
- service leakage
- epigraph/body duplication

### Step E. `targeted_repair`

Conditional LLM.

Responsibilities:

- local fix only
- issues + facts + current description

## 7. Top accepted carries for implementation

Если сжать до самого полезного:

1. Build generation prompt dynamically.
2. Remove duplicated inline stop-phrase wall from generation.
3. Use register, not persona.
4. Add quote metadata to normalization output.
5. Pass facts into repair.
6. Shrink pattern library at execution level to sharper structural shapes.
7. Keep planner structural-only and small.

## 8. Bottom line

`Opus` не дал новый architectural miracle, но дал очень полезный prompt-level correction:

- проблема не в том, что `2.15.2` слишком LLM-first;
- проблема в том, что Gemma всё ещё грозило выдать слишком много обязанностей в одном prose prompt;
- правильный ход теперь не в новых ban-lists и не в regex escalation;
- а в коротких role-separated prompts, dynamic assembly и deterministic enforcement там, где semantic reasoning не нужен.
