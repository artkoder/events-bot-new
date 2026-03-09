# Smart Update Opus Gemma Event Copy Quality-First Calibration Response Review

Дата: 2026-03-07

Связанные материалы:

- `docs/reports/smart-update-opus-gemma-event-copy-quality-first-calibration-response.md`
- `artifacts/codex/opus_gemma_event_copy_quality_first_calibration_prompt_latest.md`
- `docs/reports/smart-update-opus-gemma-event-copy-pattern-redesign-followup-response-review.md`
- `smart_event_update.py`

## 1. Краткий verdict

Это пока **самый сильный и самый practically useful ответ Opus** по этой линии.

Главное улучшение:

- полнота фактов наконец поставлена в P0 не декларативно, а как сквозной invariant;
- patterns прямо определены как `structure-only`, а не как предлог сократить содержание;
- quality-first рамка теперь совместима с fact-first discipline, а не конфликтует с ней.

Общий вывод:

- direction принимается;
- low-risk пакет можно считать устойчивым;
- medium-risk branch теперь имеет гораздо более зрелый contract;
- но перед local implementation остаются 4 узких вопроса, которые лучше ещё докрутить.

## 2. Что в этом ответе выглядит особенно сильным

### 2.1. `Все факты должны войти` теперь реально зацементировано

Это ключевой сдвиг.

Opus не ограничился фразой “конечно, coverage важен”.
Он протянул этот принцип через:

- generation rules;
- budget scaling;
- coverage check;
- revise prompt;
- decision о сохранении missing-facts safety net;
- acceptance criteria;
- rollback triggers.

Это именно то, чего не хватало раньше: не красивый narrative redesign сам по себе, а redesign, который не ломает основную fact-first гарантию.

### 2.2. `scene_led` получил уже не красивую идею, а рабочий contract

В предыдущих ответах `scene_led` был скорее promising concept.
Здесь он стал намного практичнее:

- extraction contract для `scene_cues`;
- runtime gate;
- blocklist generic atmosphere phrases;
- fallback logic;
- отдельный запрет подменять факты сценой.

Это заметно повышает шанс, что `scene_led` даст quality lift, а не случайные hallucinated openings.

### 2.3. Helper set ужат до одного осмысленного boolean

Это сильный компромисс.

Вместо большого набора routing booleans Opus оставляет только:

- `is_speaker_led`

Это выглядит гораздо реалистичнее:

- не раздувает schema чрезмерно;
- оставляет LLM только там, где semantic judgment действительно трудно вывести тупой эвристикой;
- сохраняет идею hybrid routing без лишнего шума.

### 2.4. Prompt family стала заметно зрелее

Особенно сильный ход:

- вынести `P0: ВСЕ факты включены` в shared preamble для всех patterns.

Это важнее, чем кажется.
Так patterns перестают быть “стилистическими режимами” и становятся именно организационными режимами текста.

### 2.5. Acceptance criteria наконец пригодны для engineering use

В ответе уже есть:

- hard gate по missing facts;
- правило “v1 не может терять факты чаще current flow”;
- quality uplift criteria;
- rollback triggers.

Это уже можно положить в реальный implementation plan и A/B валидацию.

## 3. Что всё ещё требует осторожности

### 3.1. `All facts included` работает только если `facts_text_clean` остаётся чистым

Это самый важный structural risk, который ответ не снимает до конца.

Если upstream extraction начнёт возвращать:

- лишние почти-дубликаты;
- слишком мелкие service-like details;
- плохо сгруппированные visitor conditions;

то требование “все факты должны войти” начнёт толкать generation к:

- list soup;
- тяжёлому телеграф-тексту;
- потере естественности уже не из-за patterns, а из-за качества fact inventory.

То есть после этого ответа ещё важнее становится upstream discipline:

- cleanliness;
- dedupe;
- фактическая полезность каждого item в `facts_text_clean`.

### 3.2. Missing-facts safety net сохранён правильно, но место в runtime пока не до конца определено

С этим ответом Opus я скорее согласен:

- `_llm_integrate_missing_facts_into_description` не надо слишком рано объявлять `remove-later`.

Но есть практический нюанс:

- в текущем fact-first ядре [smart_event_update.py](/workspaces/events-bot-new/smart_event_update.py#L2064) этот шаг не является частью стандартного generation loop;
- сейчас он реально используется в merge-oriented flow, а не как default post-revise step.

Поэтому остаётся важный implementation question:

- на каком именно residual missing threshold этот extra repair call включается;
- и как это отражается в call-budget / TPM posture.

### 3.3. Traceability gates для `scene_cues` и `contrast_or_tension` ещё слишком rough

В текущем ответе gates уже полезные, но пока довольно грубые:

- word overlap;
- content-word counts;
- blocklist generic phrases.

Для русского текста этого может не хватить:

- морфология;
- парафраз;
- OCR noise;
- различия между exact phrase и “похожим пересказом”.

То есть general direction правильный, но practical contract для traceability ещё можно усилить.

### 3.4. Latency thresholds в acceptance criteria не очень соответствуют реальной цели

Это важное расхождение с текущей продуктовой рамкой.

В ответе есть ограничения вроде:

- latency `<= 300ms`
- unacceptable `> 500ms`

Но для нашей задачи сейчас приоритет другой:

- не micro-latency per event;
- а разумная работа в рамках TPM / throughput / import stability.

Если quality lift заметный, то сама по себе более долгая обработка не является проблемой.
Проблема начинается только тогда, когда дизайн operationally ломается.

Значит acceptance criteria по performance нужно формулировать не в таких жёстких миллисекундных числах, а в более релевантных operational metrics.

### 3.5. `scene_led` vs `value_led` всё ещё лучше считать A/B-sensitive зоной

Opus дал уже намного более разумное правило:

- strong why-go → `value_led`
- regular-only value → `scene_led`

Это хороший шаг.
Но редакторски это всё ещё спорная граница.

Поэтому я бы не делал этот precedence “законом природы” до просмотра реальных кейсов на golden set.

## 4. Практическая позиция после этого ответа

### 4.1. Что можно принять уже сейчас

Без большого спора я бы принял:

1. `полнота фактов = P0` как жёсткий invariant;
2. сохранение fact-proportional budget;
3. shared completeness preamble во все pattern prompts;
4. `scene_led` с conservative gate;
5. `contrast_or_tension` как optional non-routing lead aid;
6. helper set из одного boolean `is_speaker_led`;
7. расширенный coverage check с quality flags;
8. rollback criteria по fact loss.

### 4.2. Что ещё надо докалибровать до implementation

Остаются 4 практических вопроса:

1. как не превращать dense fact sets в тяжёлый текст при требовании “все факты должны войти”;
2. где именно и при каком residual threshold должен включаться missing-facts repair call;
3. какими должны быть TPM-aware / throughput-aware acceptance thresholds;
4. как усилить traceability contract для `scene_cues` и `contrast_or_tension`.

### 4.3. Нужен ли ещё один Opus-раунд

Да, но уже совсем узкий.

Не нужен новый redesign.
Нужен **final implementation calibration round** по этим 4 вопросам.

После этого ответа Opus уже не нужно просить снова придумывать pattern system.
Нужно попросить:

- закрыть оставшиеся implementation holes;
- и выдать более operationally precise guidance.

## 5. Bottom line

Если коротко:

- этот ответ реально приблизил нас к рабочей quality-first архитектуре;
- Opus наконец встроил полноту фактов в саму ткань design, а не как постфактум оговорку;
- strongest part ответа — сочетание richer text design с coverage discipline;
- remaining work уже не концептуальная, а калибровочная: dense facts, repair-call placement, TPM-aware thresholds, stronger traceability.
