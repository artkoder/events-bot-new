# Smart Update Gemini Event Copy V2.6 Hypotheses Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-5-review-2026-03-07.md`
- `docs/reports/smart-update-gemini-event-copy-v2-5-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-5-quality-consultation-response-review.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_5_2026_03_07.py`

## 1. Зачем нужна эта консультация

Нам нужен не ещё один общий review, а точечная консультация по нашим текущим рабочим гипотезам для `v2.6`.

Смысл раунда:

- не просто снова спросить “что улучшить”;
- а проверить, насколько наши собственные предположения реально защищены evidence;
- и не упускаем ли мы более сильный bottleneck.

## 2. Что уже известно

По текущему циклу у нас есть довольно стабильная картина:

- `v2.5` лучше `v2.4`;
- `v2.5` всё ещё не лучше `v2.3` по суммарному качеству текста;
- главный root cause остаётся в раннем слое:
  - extraction;
  - fact shaping;
  - routing;
  - revise severity.

Также уже подтверждено:

- мягкие hints Gemma интерпретирует слишком свободно;
- grouped program hints без явного шаблона ведут к over-abstraction;
- `compact` ветка на overlapping facts начинает дублировать смысл;
- human-readable anti-`посвящ*` лучше opaque marker, но пока не forceful enough;
- один только `<= 6` routing threshold всё ещё brittle.

## 3. Наш текущий hypothesis pack для `v2.6`

Сейчас мы предполагаем следующее.

### H1. Главная проблема `v2.5` — не architecture, а слабая строгость prompt contract

То есть:

- новый большой redesign не нужен;
- нужен более жёсткий и более структурный prompt contract для Gemma.

### H2. Нужно убрать мягкость из extraction-hints и policy issues

Вместо vague phrasing вроде:

- “по возможности переформулируй”
- “предпочтительно собери”

нужны более конкретные блоки с allowed patterns и explicit blocking wording.

### H3. Для program-item cases нужен явный grouped pattern

Не общая идея “сгруппируй”, а pattern вроде:

- `В программе: "A", "B", "C"`

Но только для real performance/program lists, а не как универсальный шаблон для любых list-like facts.

### H4. Нужно жёстко запретить label-style facts

В первую очередь:

- `Тема: ...`
- `Идея: ...`
- `Цель: ...`

Потому что это не narrative-ready facts и они ухудшают и extraction, и generation.

### H5. Compact branch нужен explicit permission to merge overlapping facts

То есть:

- не mechanical one-fact-one-sentence;
- а смысловое объединение пересекающихся facts в 1-2 живых предложения.

Это прежде всего нужно для `2660`.

### H6. Standard generation нужен anti-bureaucracy rule

Если facts содержат конструкции типа:

- `На презентации расскажут о задачах...`

то generation должен сразу писать о задачах/устройстве/смысле, а не повторять бюрократическую рамку.

Это прежде всего кейс `2673`.

### H7. Routing `<= 6` пока можно оставить, но только как временный локальный gate

Мы НЕ считаем это final truth.
Но предполагаем, что для ближайшего `v2.6` это пока допустимо, если:

- extraction станет менее шумным;
- а для richer cases позже добавим content-aware gate.

### H8. Следующий шаг должен быть локальным `v2.6` dry-run, а не ещё одним долгим consultation loop

То есть мы предполагаем, что после этой hypothesis-консультации надо уже идти в код experimental harness, а не проводить ещё 2-3 раунда теории.

## 4. Что мы хотим проверить у Gemini

Нас интересует:

1. Какие из этих гипотез сильные и подтверждаются evidence.
2. Какие из них частично верны, но сформулированы неточно.
3. Какие из них вторичны по сравнению с более важным bottleneck.
4. Достаточен ли этот hypothesis pack для `v2.6`.
5. Нужен ли ещё один consultation round до кода, или уже пора в локальную реализацию.

## 5. Самое важное требование

Нужен не общий opinion, а hypothesis-by-hypothesis critique с конкретными рекомендациями для Gemma prompts.

То есть мы просим Gemini:

- спорить с нашими предположениями, если они слабые;
- не соглашаться автоматически;
- указывать, где мы переоцениваем один слой и недооцениваем другой;
- давать patchable prompt guidance, а не абстрактную теорию.
