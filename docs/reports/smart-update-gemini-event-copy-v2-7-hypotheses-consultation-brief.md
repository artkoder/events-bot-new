# Smart Update Gemini Event Copy V2.7 Hypotheses Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md`
- `docs/reports/smart-update-gemini-event-copy-v2-6-dryrun-quality-consultation-response.md`
- `docs/reports/smart-update-gemini-event-copy-v2-6-dryrun-quality-consultation-response-review.md`
- `artifacts/codex/experimental_pattern_dryrun_v2_6_2026_03_07.py`

## 1. Зачем нужна эта консультация

Нам нужен не ещё один общий review, а узкая pre-run консультация перед `v2.7`.

Смысл раунда:

- не спорить снова обо всей архитектуре;
- а проверить, насколько наш новый `v2.7` hypothesis pack действительно безопасен и quality-positive;
- особенно не тащит ли он нас в неgrounded rewrites ради борьбы с канцеляритом.

## 2. Что уже известно после `v2.6`

Картина стала достаточно чёткой:

- `v2.6` — net-positive round, но не runtime candidate;
- `2660` и `2734` подтвердили полезность stricter contract;
- `2687` и `2673` показали, что main blocker теперь в lecture/presentation extraction;
- Gemini был прав про проблему многослойных negative constraints;
- но часть предложенных им positive transformation examples оказалась слишком сильной и рискованной для factual grounding.

## 3. Что мы не хотим сломать

В `v2.7` нельзя потерять:

- compact gains на `2660`;
- performance/program recovery на `2734`;
- текущий routing;
- content-preservation discipline;
- anti-CTA / anti-service / anti-metatext hygiene.

## 4. Наш текущий hypothesis pack для `v2.7`

### H1. Главная задача `v2.7` — не новый redesign, а safe positive transformation в Extraction

То есть:

- меньше blind bans;
- больше grounded rewrite models;
- без добавления новых смыслов, которых нет в raw facts.

### H2. Positive transformation examples должны быть agenda-safe

Мы предполагаем, что нельзя писать примеры вроде:

- `На презентации расскажут об устройстве платформы` -> `Платформа устроена так: ...`
- `расскажут, какую проблему решает проект` -> `Проект решает проблему: ...`

если этих деталей нет в raw fact.

Вместо этого нужны safe forms вроде:

- `В центре презентации — устройство платформы и её возможности.`
- `На встрече разберут причины появления проекта, его задачи и возможности для участников.`
- `Лекция о творчестве художниц...`

### H3. Label-style / `посвящ*` / intent transfer надо лечить прежде всего в Extraction и pre-hints

Мы предполагаем, что главный patch pack должен быть сосредоточен на:

- extraction prompt;
- `_pre_extract_issue_hints`;
- возможно `_extract_issue_hints`.

А не на routing.

### H4. Standard generation нужен более строгий anti-bureaucracy, но не total ban on event framing

Мы предполагаем, что надо запретить:

- `мероприятие анонсирует...`
- `проект призван стать...`
- `будет представлен обзор...`

Но не ломать human-sounding формулы вроде:

- `В центре лекции — ...`
- `На встрече разберут ...`
- `Программа вечера построена вокруг ...`

### H5. Revise для `посвящ*` и похожих формулировок должен стать сильнее

Не advisory, а blocking rewrite instruction.

Но rewrite должен:

- сохранять factual grounding;
- не заменять одну форму на другую equally weak.

### H6. Routing в `v2.7` лучше не трогать

Наше текущее предположение:

- `<= 6` + `program_rich_blocker` пока оставить;
- источник проблем `v2.6` сейчас не routing, а extraction/fact shaping.

### H7. Следующий шаг после этой консультации — локальный `v2.7` dry-run

То есть мы предполагаем, что после этого раунда надо уже идти в код, а не запускать ещё одну теоретическую дискуссию.

## 5. Что мы хотим проверить у Gemini

Нас интересует:

1. Достаточно ли strong этот `v2.7` hypothesis pack.
2. Где мы всё ещё рискуем лечить канцелярит ценой hallucination-like overreach.
3. Действительно ли routing можно пока оставить без изменений.
4. Какие safe prompt-level rewrites для Gemma реально worth testing next.

## 6. Самое важное требование

Нужен не общий opinion, а critique по гипотезам с **конкретными Gemma-friendly формулировками**.

Особенно важно:

- не предлагать rewrites, которые требуют додумывания отсутствующих деталей;
- спорить с нами, если safe-positive pack всё ещё неточен;
- давать patchable wording, а не abstraction.
