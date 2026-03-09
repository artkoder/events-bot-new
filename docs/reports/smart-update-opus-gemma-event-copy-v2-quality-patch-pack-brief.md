# Smart Update Opus Gemma Event Copy V2 Quality Patch Pack Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-followup-response.md`
- `docs/reports/smart-update-opus-gemma-event-copy-quality-consultation-followup-response-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-text-quality-improvements.md`
- `docs/reports/smart-update-opus-gemma-event-copy-text-quality-improvements-review.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md`

## 1. Контекст

Мы уже прошли:

- теоретическую проработку pattern-driven redesign;
- preservation/migration audit текущего runtime;
- baseline dry-run на 5 реальных событиях;
- experimental pattern-driven dry-run на тех же 5 событиях;
- quality consultation с Opus по реальным output-кейсам;
- follow-up recalibration response от Opus;
- отдельный каталог text-quality improvements.

На этой стадии проблема уже не в нехватке идей.
Проблема в том, что сильные идеи нужно перевести в компактный, рабочий и Gemma-реалистичный `v2 patch pack`.

## 2. Что уже понятно

### 2.1. Реальные quality blockers prototype v1

По реальному dry-run видно, что у pattern prototype v1 главные проблемы такие:

- duplication после revise/repair;
- неверный routing sparse cases в `value_led`;
- слишком слабый anti-CTA / anti-filler hygiene;
- слишком generic headings;
- extraction местами режет publishable facts слишком агрессивно;
- prompt surface пока не даёт достаточно сильной anti-metatext / anti-embellishment дисциплины.

### 2.2. Что в предложениях Opus уже выглядит сильным

С высокой уверенностью полезны:

- anti-duplication rule в generation prompt;
- anti-duplication runtime guard;
- content-preservation floor вместо raw fact-count floor;
- sparse routing в compact branch;
- CTA detection;
- ban / downrank generic headings;
- anti-metatext rule;
- anti-embellishment rule;
- filler phrase denylist;
- восстановление epigraph / blockquote как conditional enhancement, а не обязательного baseline mimicry.

### 2.3. Что пока нельзя тащить в v2 wholesale

Пока слишком рано или слишком шумно:

- большой набор stylistic micro-rules;
- question-led openings;
- heavy paragraph-level и sentence-level prose gates;
- богатый few-shot блок без жёсткой компрессии;
- радикальное сокращение pattern library без более широкого dry-run.

## 3. Главная задача следующего раунда

Нужно попросить Opus не придумывать ещё больше идей,
а **приоритизировать и сжать** уже собранные предложения в implementation-ready subset.

Ключевой вопрос:

> Какой минимально-шумный, но всё ещё действительно качественный `v2 quality patch pack`
> даст заметное улучшение текста на Gemma без перегруза prompt и runtime?

## 4. Что нужно получить от Opus

### 4.1. Приоритизация

Каждое существенное предложение должно получить verdict:

- `must_include_v2`
- `include_if_compact`
- `defer`
- `reject_for_now`

### 4.2. Компактные prompt blocks

Нужны короткие, implementation-ready блоки для:

- generation prompt;
- lead guidance;
- revise prompt.

Они должны быть короткими, прямыми и пригодными для Gemma.

### 4.3. Minimal runtime gates

Нужен shortlist runtime checks, которые реально стоят своей сложности:

- duplicate guard;
- CTA detection;
- weak heading detection;
- metatext lead detection;
- возможно ещё 1-2 high-signal guard, но не больше.

### 4.4. Compact sparse contract

Нужен точный contract для sparse events:

- нужны ли headings;
- нужен ли blockquote;
- какая длина целевого текста;
- как избежать и over-structuring, и bland wall-of-text.

## 5. Критерии качества для этого раунда

Ответ Opus должен помогать двигаться к тексту, который:

- сохраняет publishable facts;
- не уходит в unsupported embellishment;
- не разваливается на дубли и filler;
- остаётся естественным, профессиональным и Telegraph-readable;
- реально работает на Gemma в рамках разумного TPM / runtime budget.

## 6. Итоговая позиция

Новый consultation round нужен.

Но это уже не open-ended analysis и не новый architecture debate.
Это **финальный узкий раунд по сборке `v2 quality patch pack`** перед следующей локальной итерацией prototype.
