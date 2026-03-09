# Smart Update Opus Gemma Event Copy Quality Consultation Brief

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md`
- `artifacts/codex/gemma_event_copy_dry_run_5events_2026-03-07.json`
- `artifacts/codex/gemma_event_copy_pattern_dry_run_5events_2026-03-07.json`
- `artifacts/codex/experimental_pattern_dryrun_v1_2026_03_07.py`
- `smart_event_update.py`

## 1. Контекст

Мы долго вместе с Opus проектировали `quality-first / pattern-driven` направление для улучшения текстов описаний событий в fact-first подсистеме Smart Update.

Ключевая разница с прошлой фазой:

- теперь есть не только теоретическая схема;
- есть **реальный baseline dry-run** текущего flow;
- и есть **реальный experimental dry-run** нового pattern-driven prototype на тех же 5 событиях.

То есть дальнейшая консультация уже должна опираться не на “архитектурные идеи вообще”, а на конкретные outputs:

- вот факты;
- вот baseline текст;
- вот experimental pattern text;
- вот где реально текст лучше;
- вот где реально хуже или грязнее.

## 2. Главная цель консультации

Нам нужна **максимально критическая и свободная quality consultation** по реальным результатам.

Приоритеты:

1. естественный, человеческий, профессиональный текст;
2. полнота и точность фактов;
3. логичность, связность, композиция;
4. чистота формулировок и отсутствие сервисного мусора;
5. адекватность выбранного narrative pattern;
6. реализуемость на Gemma в рамках TPM / practical runtime.

Важно:

- мы **не защищаем текущий prototype**;
- мы **не просим подтверждать уже придуманные идеи**;
- мы хотим, чтобы Opus свободно предложил любые улучшения, если они объективно поднимут качество.

## 3. Что уже видно по текущим dry-run

### 3.1. Baseline current flow

Baseline прогон показал:

- текущий flow ещё часто даёт сухой или слишком “объясняющий” текст;
- в ряде кейсов есть coverage drift;
- на sparse cases текст быстро становится корректным, но слишком общим;
- на некоторых events baseline всё ещё тянет в service leakage и шаблонные секции.

### 3.2. Experimental pattern-driven prototype

Prototype уже дал первые реальные сигналы:

- `2687` и `2673` выглядят promising;
- `2734` mixed: лучше hygiene и coverage, но extraction слишком агрессивно режет facts;
- `2660` и `2745` показывают, что новый branch пока не production-ready.

То есть pattern-driven direction уже нельзя считать purely speculative.
Но и переносить его в код бота “как есть” пока нельзя.

## 4. На что особенно просим смотреть

Ниже — список quality-dimensions, которые Opus должен оценивать **не формально, а редакторски и инженерно одновременно**.

### 4.1. Качество самих текстов

- естественность и человечность;
- профессиональность и журналистская чистота;
- отсутствие шаблонности;
- отсутствие press-release / promo tone;
- отсутствие unsupported embellishment;
- отсутствие awkward или тяжёлых фраз;
- адекватность заголовков, блоков, ритма абзацев;
- правильная работа с blockquote / epigraph;
- отсутствие микросекций и пустых headings.

### 4.2. Полнота и точность

- не потеряны ли важные факты;
- не заменены ли факты слишком общим пересказом;
- не добавлены ли неподтверждённые объясняющие конструкции;
- не возникает ли “красивый, но менее точный” текст;
- не слишком ли груб текущий lexical/deterministic missing metric.

### 4.3. Pattern quality

- верно ли вообще выбраны patterns в каждом кейсе;
- где routing логичен, а где нет;
- где pattern реально улучшает композицию;
- где pattern только добавляет structure noise;
- не слишком ли часто `value_led` побеждает;
- нужен ли текущий набор patterns целиком или его стоит упростить/пересобрать.

### 4.4. Pipeline quality

Нужно отдельно оценить качество по стадиям:

1. `facts extraction / filtering`
2. `copy_assets extraction`
3. `routing`
4. `generation prompt`
5. `revise / missing-facts repair`
6. `cleanup / hygiene`
7. `evaluation methodology`

То есть нам нужен не только “приговор текстам”, но и diagnosis:

- где именно pipeline производит quality loss;
- что является root cause;
- какие изменения дадут реальный quality lift.

## 5. Что Opus может свободно предлагать

Opus НЕ ограничен текущей формулировкой prototype.

Разрешается свободно предложить:

- переписать extraction contract;
- изменить/сузить/объединить patterns;
- поменять precedence / routing rules;
- усилить или упростить copy_assets;
- вернуть часть baseline-эвристик;
- изменить generation/revise prompts;
- усилить cleanup/runtime checks;
- предложить другой poor-source policy;
- изменить критерии качества и evaluation rubric;
- предложить новый comparison protocol;
- рекомендовать не внедрять часть текущего prototype вообще.

Главный критерий:

- improvement должен быть **объективно полезен для качества текста**, а не просто концептуально красив.

## 6. Что особенно важно не потерять

Даже в режиме свободной критики нельзя забывать про уже принятые guardrails:

- `полнота фактов = P0`;
- moderate token/runtime growth допустим, если даёт quality lift;
- результат должен оставаться рабочим для Gemma;
- нельзя терять сильные стороны текущего runtime:
  - `_facts_text_clean_from_facts`
  - `_sanitize_fact_text_clean_for_prompt`
  - `_pick_epigraph_fact`
  - `_find_missing_facts_in_description`
  - `_llm_integrate_missing_facts_into_description`
  - `_cleanup_description`
  - `_collect_policy_issues`
  - visitor conditions / compact list safeguards

## 7. Что мы хотим получить от Opus

Нужен ответ не в духе “направление хорошее / плохое”, а в инженерно-редакторском формате.

Желаемый output:

### 1. `Event-by-event review`

Для всех 5 кейсов:

- baseline vs pattern verdict;
- где текст лучше;
- где текст хуже;
- что потеряно;
- что выдумано или editorialized;
- подходит ли выбранный pattern;
- какой pattern был бы лучше, если текущий неверен.

### 2. `Cross-case quality diagnosis`

По всей выборке:

- главные повторяющиеся проблемы;
- сильные стороны prototype;
- слабые стороны prototype;
- где baseline по-прежнему выигрывает.

### 3. `Pipeline failure map`

По стадиям:

- extraction
- filtering
- copy_assets
- routing
- generation
- revise/repair
- cleanup
- evaluation

Для каждой стадии:

- что работает;
- что ломает quality;
- что надо keep/tune/rewrite/remove.

### 4. `Prioritized improvements`

Нужен список улучшений по приоритету:

- `P0` — самые сильные и необходимые;
- `P1` — важные, но вторые по очереди;
- `P2` — позже / экспериментально.

Желательно для каждого improvement:

- expected quality impact;
- risk;
- prompt/runtime/extraction ownership.

### 5. `Concrete next iteration`

Нужен pragmatic recommendation:

- что именно делать в следующем local prototype v2;
- что не трогать;
- что отложить;
- стоит ли после этого идти ещё в один dry-run before implementation;
- или уже можно переносить subset в код.

## 8. Bottom line

Это уже консультация не “про концепцию”, а **про реальное качество текста**.

Opus должен максимально критично посмотреть:

- на тексты;
- на facts retention;
- на связность и композицию;
- на уместность patterns;
- на pipeline root causes;
- и предложить любые изменения, которые реально поднимут качество.
