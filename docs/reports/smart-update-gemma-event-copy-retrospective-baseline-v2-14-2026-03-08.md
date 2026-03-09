# Smart Update Gemma Event Copy Retrospective — Baseline → V2.14

Дата: 2026-03-08

Связанные канонические материалы:

- [baseline dry-run](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-dry-run-5-events-2026-03-07.md)
- [v1 pattern dry-run](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-5-events-2026-03-07.md)
- [v2.6 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-6-review-2026-03-07.md)
- [v2.12 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-12-review-2026-03-08.md)
- [v2.13 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-13-review-2026-03-08.md)
- [v2.14 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-14-review-2026-03-08.md)
- [text-quality improvements catalog](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-text-quality-improvements.md)
- [first Gemini review](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-first-consultation-response-review.md)
- [v2.12 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-12-prompt-context.md)
- [v2.13 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-13-prompt-context.md)
- [v2.14 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-14-prompt-context.md)

## 1. Зачем этот документ

Этот документ нужен по двум причинам.

Первая: за цикл `baseline -> v2.14` накопилось много локальных review, prompt-context и dry-run отчётов. Без общей ретроспективы легко потерять уже найденные сильные решения и снова наступать на те же проблемы.

Вторая: требования действительно начали "плавать". Поэтому ниже сначала фиксируются устойчивые требования, а уже потом разбираются версии.

Важно:

- это retrospective-only документ;
- здесь не выбирается новая архитектура "по вере";
- здесь не используется мнение внешних моделей как истина;
- выводы опираются на реальные dry-run артефакты и review по `baseline`, `v1`, `v2...v2.14`.

## 2. Зафиксированные требования, которые дальше не должны плавать

Это не новые идеи, а то, что уже многократно подтвердилось по истории прогонов.

### 2.1. Главная цель

Главная цель не "выиграть по одной метрике", а получить **предсказуемо хороший итоговый текст описания события**:

- естественный;
- профессиональный;
- не шаблонный;
- не рекламный;
- без нейросетевых клише;
- при этом grounded и без потери фактов.

### 2.2. Семантический core должен оставаться LLM-first

Для тысяч очень разных постов нельзя строить semantic core на регулярках.

Допустимы только support-layer вещи:

- санация явно запрещённых маркеров;
- exact/near dedup;
- safety checks;
- deterministic validation;
- caps/filters на сервисный мусор.

Но регулярки и deterministic слой не должны становиться главным местом, где "создаётся смысл".

### 2.3. Полнота фактов остаётся P0, но не выше качества текста вообще всегда

История показала две крайности:

- baseline часто держит coverage лучше и стабильнее;
- более поздние ветки иногда звучат заметно лучше baseline, но теряют часть важной конкретики.

Правильный компромисс:

- coverage нельзя отпускать;
- но оптимизация только под literal `missing` ломает prose;
- итоговый verdict всегда должен учитывать и coverage, и publishability текста.

### 2.4. Нельзя терять уже найденные prose-wins

Нельзя смотреть только на архитектуру.

Даже если новая схема умнее технически, её нельзя принимать, если она:

- возвращает канцелярит;
- генерирует шаблонные headings;
- тянет модель к `лекция расскажет...`, `мероприятие будет интересно...`, `это не ..., а ...`;
- делает текст хуже редакторски, чем сильные раунды вроде `v2.6` или `v2.13`.

### 2.5. Требование масштабируемости

Нужна не система, которая случайно хорошо пишет именно 5 кейсов.

Нужна система, которая:

- масштабируется на тысячи разнородных постов;
- не зависит от длинного хвоста ручных regex-патчей;
- переносит variability source shapes;
- не разваливается при добавлении новой выборки.

## 3. Где и как уже прорабатывали борьбу с нейросетевыми клише

Ниже не просто список проблем, а история того, что уже было осознано и частично решено.

### 3.1. До dry-run цикла

Ещё до `v1-v2.14` уже были собраны:

- внутренние паттерны сильных Telegram-анонсов;
- внешний casebook по event pages и культурной журналистике;
- журналистский лексикон и safe phrase bank;
- отдельный каталог quality-improvements для event-copy.

То есть борьба с шаблонностью не появилась "вдруг" на поздних раундах. Она была частью задачи почти с самого начала.

### 3.2. Baseline-phase: первые явные дефекты

В baseline были явно замечены:

- слабые leads типа `Спектакль рассказывает...`, `Лекция расскажет...`, `Это ...`;
- generic headings типа `О событии`, `Основная идея`, `Ключевые мотивы`;
- monotone sentence structure;
- unsupported embellishment;
- filler-обороты;
- press-release tone.

Это зафиксировано, в частности, в [text-quality improvements catalog](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-text-quality-improvements.md).

### 3.3. Ранние pattern rounds: борьба с шаблонностью через routing и pattern variety

В `v1-v2.3` основной ход был таким:

- уйти от одного шаблонного Style C;
- добавить routing;
- разрешить более разные narrative patterns;
- ослабить жёсткие lexical bans;
- убрать обязательные headings для sparse cases;
- добавить positive examples вместо части запретов.

Это дало реальные gains по naturalness, особенно в `v2.3`.

### 3.4. Средние раунды: клише начали ловить уже как отдельный класс дефектов

В `v2.4-v2.11` явно и многократно фиксировались:

- `question-style headings`;
- `label-style facts` (`Тема: ...`, `Цель: ...`);
- `посвящ*`;
- `метатекст`;
- `канцелярит`;
- `inline quote contamination`;
- `не ..., а ...` конструкции;
- service-style headings вроде `Формат мероприятия`.

То есть к этому моменту проблема уже была не "текст просто плоховат", а "есть конкретный список AI-prose failure patterns".

### 3.5. Поздние раунды: anti-cliche стало частью architecture, но не всегда удерживалось

В `v2.12-v2.14` уже явно удерживались цели:

- no generic headings;
- no `посвящ*`;
- no anti-metatext drift;
- no full editorial over-rewrite;
- no noisy dirty merge.

Но история показала и другую проблему:

- часть уже решённых клише потом возвращалась через новые слои;
- пример: `v2.14` вернул бюрократию уже не через sanitize/facts, а через LLM outline.

Главный вывод:

- борьба с AI clichés уже велась много раз;
- но она не была сведена в единый устойчивый набор non-negotiables;
- из-за этого решения периодически "терялись" при архитектурных поворотах.

## 4. Сильные решения, которые уже доказали пользу и не должны потеряться

### 4.1. Из baseline

- LLM-first semantic core.
- Fact-first discipline.
- Самодостаточный body.
- Existing cleanup/policy layer.
- Простая и более production-like архитектура.

### 4.2. Из `v2.3`

- Sparse no-heading contract.
- Dynamic formatting.
- Positive examples вместо части brittle lexical bans.
- Body self-sufficiency.
- Лёгкая pre-consolidation как направление.

### 4.3. Из `v2.6`

- Реальные prose wins против baseline на части кейсов.
- Более живые headings и менее шаблонная подача.
- Strong compact handling на `2660`.
- Strongest recovery на `2734` внутри старой pattern-family.

### 4.4. Из `v2.11`

- Anti-quote rule действительно полезен.

### 4.5. Из `v2.12`

- Full-floor normalization как architectural shift.
- Уход от destructive dirty merge.
- Возврат к LLM-first semantic reshaping.

### 4.6. Из `v2.13`

- `forbidden = 0` на 5 кейсах.
- Более короткий exemplar-driven generation.
- Отказ от full editorial rewrite pass.
- Тексты на части кейсов выглядят заметно менее шаблонными, чем baseline.

### 4.7. Из `v2.14`

- Split-call сам по себе не ошибка.
- Для rich cases дополнительная структурная помощь может быть полезной.
- Но outline должен быть structural, а не prose-like.

## 5. Что по ключевым этапам говорили другие модели

Этот раздел добавлен специально, чтобы retrospective была не только моей внутренней реконструкцией цикла.

Важно:

- ниже перечислены оценочные сигналы от `Opus` и `Gemini` из уже существующих consultation/review документов;
- это не "истина по умолчанию";
- рядом сразу фиксируется, где эти оценки были приняты, а где нет.

### 5.1. Ранний цикл: `v2.2 -> v2.3`

`Gemini` в [first consultation review](/workspaces/events-bot-new/docs/reports/smart-update-gemini-event-copy-first-consultation-response-review.md) и в review по `v2.3` считал, что:

- bottleneck был не в "ещё одном pattern", а в `fact fragmentation + rigid prompting`;
- sparse cases не должны автоматически получать headings;
- жёсткие lexical bans вроде anti-`посвящ*` ломают русский и лучше заменяются positive examples;
- `v2.3` стал реальным шагом вперёд.

Это было принято в существенной части, и retrospective подтверждает:

- `v2.3` действительно стал первой полезной mixed-positive версией после ранних regression rounds;
- sparse no-heading contract и body self-sufficiency были реальным gain.

Где я с `Gemini` не соглашался полностью:

- pre-consolidation не была "серебряной пулей";
- не все negative constraints вредны, вредны именно brittle lexical bans.

### 5.2. Средний цикл: `v2.4 -> v2.6`

`Gemini` в response/review по `v2.4` и `v2.6` настаивал, что:

- `v2.4` по сумме хуже `v2.3`;
- надо возвращаться к `v2.3` base, а не развивать `v2.4`;
- Gemma плохо исполняет большие negative ban-lists;
- для `2687` и `2673` проблема была в том, что модель не умеет красиво трансформировать `посвящ...`, `Тема: ...`, `На презентации расскажут...`, если ей не дать positive transformation patterns;
- anti-bureaucracy нужно формулировать явно.

Это retrospective подтверждает частично:

- rollback к `v2.3` base действительно был правильным;
- `v2.6` стал одним из strongest prose rounds;
- борьба с label-style, `посвящ*` и bureaucratic self-reference действительно была отдельным техническим фронтом.

Но `Gemini` тогда всё же недооценивал:

- насколько fragile остаётся `lecture/person-rich` family;
- насколько легко хорошие positive examples начинают сами закреплять новый шаблон.

### 5.3. Architectural turn: `v2.11 -> v2.12`

Здесь был самый важный внешний disagreement.

В [v2.12 consultation synthesis](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-12-consultation-synthesis-2026-03-08.md):

- `Opus` тянул в сторону более deterministic cleaning и жёстче трактовал итог как `baseline > v2.12`;
- `Gemini` предлагал сохранить `LLM-first`, но заменить старый `subset extraction -> dirty merge` на `full-floor normalization`.

В retrospective я фиксирую это явно:

- по architecture stronger была линия ближе к `Gemini`;
- по caution against uncontrolled prose drift `Opus` тоже давал полезный сигнал;
- но тезис `baseline overall still better` нельзя принимать в лоб уже на `v2.12`, потому что `v2.12` впервые beat baseline по coverage и местами уже был редакторски живее baseline.

То есть честная формула такая:

- `Opus` был полезен как источник архитектурного скепсиса;
- `Gemini` была ближе к правильному стратегическому направлению;
- финальная позиция собрана не из "доверия одной модели", а из synthesis.

### 5.4. `v2.13`: strongest external consensus

В [v2.13 post-run synthesis](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-13-postrun-consultation-synthesis-2026-03-08.md) есть наиболее сильная зона согласия:

- и `Opus`, и `Gemini` согласились, что `full-floor normalization` откатывать не надо;
- обе модели признали, что `v2.13` лучше baseline overall;
- обе модели увидели, что `targeted repair` в текущем виде слаб;
- обе модели согласились, что broken coverage metric шумит и literal substring-check переоценивает `missing`.

Где disagreement оставался:

- `Opus` слишком охотно хотел сводить remaining problem к broken metric;
- `Gemini` слишком резко редуцировала проблему к fact density, недооценивая shape differences.

Retrospective это учитывает:

- `v2.13` действительно один из лучших candidate rounds цикла;
- но проблемы `2673` и частично `2687` были реальными даже без literal-check noise.

### 5.5. `v2.14`: согласие про outline, но не про rollback

В [v2.14 post-run synthesis](/workspaces/events-bot-new/docs/reports/smart-update-event-copy-v2-14-postrun-consultation-synthesis-2026-03-08.md) `Opus` и `Gemini` дали очень полезный общий сигнал:

- проблема не в самом split-call;
- проблема в том, что outline стал писать prose и headings;
- outline должен стать structural-only, лучше через `fact_ids`;
- generation должен сам рождать headings и body;
- semantic core нельзя снова сдвигать в regex-first semantics.

Это retrospective принимает почти полностью.

Где я сознательно не беру внешние ответы на веру:

- нельзя автоматически считать, что достаточно "сделать outline fact-only" и проблема решена;
- нельзя опять забывать про цель по итоговому тексту и ориентироваться только на coverage/structure.

### 5.6. Итог по внешним оценкам

Если смотреть трезво, внешний слой консультаций дал 4 особенно полезные вещи:

- помог раньше заметить, где regressions действительно structural, а не случайные;
- несколько раз правильно заставлял вернуться к более узкой и менее хрупкой версии;
- отдельно и многократно подсветил AI clichés как инженерный класс дефектов, а не как "эстетический вкус";
- помог выйти к `full-floor normalization`, который сам по себе стал самым важным архитектурным шагом второй половины цикла.

Но не менее важно и другое:

- ни `Opus`, ни `Gemini` нельзя считать oracle;
- в истории были случаи, где `Opus` был слишком консервативен;
- и случаи, где `Gemini` была слишком оптимистична насчёт positive examples или слишком упрощала diagnosis.

Именно поэтому в этом retrospective их оценки присутствуют явно, но рядом всегда сохранён мой собственный synthesis.

## 6. Архитектурные фазы цикла

### 6.1. Baseline family

`baseline` — это production-like fact-first runtime:

- LLM extraction;
- `facts_text_clean`;
- один основной LLM generation prompt в духе Style C;
- coverage-check / revise;
- deterministic policy checks, cleanup и targeted anti-`посвящ*` repair.

По смыслу это LLM-first схема с небольшим deterministic support-layer.

### 6.2. Pattern-prototype family

Это линия `v1 -> v2.11`.

Общий замысел:

- enriched extraction;
- дополнительные `copy_assets`;
- routing по patterns;
- разные generation branches;
- потом revise / repair / hygiene / merge / shaping.

Сильная сторона этой семьи:

- она реально подняла ceiling по prose quality;
- она показала, что baseline style не потолок.

Слабая сторона:

- архитектура стала всё более case-sensitive;
- deterministic shaping и merge постепенно стали слишком важными;
- часть локальных fix-ов перестала быть масштабируемой.

### 6.3. Full-floor normalization family

Это линия `v2.12 -> v2.14`.

Главный сдвиг:

- перестать выбирать "лучшее подмножество facts";
- нормализовать всю релевантную fact base через LLM;
- потом уже чистить и собирать текст.

Это был правильный возврат к LLM-first подходу после перегруза pattern family support-логикой.

## 7. Версия за версией: архитектура, плюсы, минусы, влияние на текст, масштабируемость

Ниже — максимально простая версия истории. Архитектура указана коротко, с акцентом на то, где LLM, а где deterministic support.

### Baseline

- Архитектура:
  `LLM extraction -> facts_text_clean -> Style C fact-first generation -> coverage/revise -> deterministic cleanup`
- Где LLM:
  extraction, generation, coverage/revise.
- Где deterministic:
  sanitize, forbidden checks, cleanup, targeted anti-`посвящ*` repair.
- Что было сильным:
  стабильность, coverage discipline, простота, production-scale пригодность.
- Что было слабым:
  шаблонный текст, generic headings, flat leads, filler, unsupported embellishment, service leakage на части кейсов.
- Влияние на итоговый текст:
  тексты рабочие и publishable чаще, но часто плоские и слишком "конструкторные".
- Масштабируемость:
  высокая. Это лучшая по operational simplicity архитектура из всех уже испытанных.

### V1

- Архитектура:
  `enriched extraction -> copy_assets -> deterministic routing -> pattern-aware generation -> revise/repair`
- Где LLM:
  extraction, `copy_assets`, generation, revise.
- Где deterministic:
  routing и часть hygiene.
- Что было сильным:
  большой coverage jump против baseline, более разнообразный текст, сильные кейсы `2687` и `2673`.
- Что было слабым:
  noisy `copy_assets`, слишком частый `value_led`, CTA leak, poor-source cases слабые, branch fragility.
- Влияние на итоговый текст:
  впервые стало видно, что текст может быть заметно лучше baseline, но качество сильно гуляло.
- Масштабируемость:
  средняя или ниже. Слишком зависимо от routing и качества `copy_assets`.

### V2

- Архитектура:
  та же pattern-family, но с quality-block patch pack поверх `v1`.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  больше hygiene / routing / post-fix logic.
- Что было сильным:
  попытка убрать дубли и поправить prose-hygiene.
- Что было слабым:
  не стало quality win, sparse и compact кейсы не стабилизировались, `посвящ*` и другие машинные формулы жили дальше.
- Влияние на итоговый текст:
  местами чище, но ещё не лучше `v1` как направление.
- Масштабируемость:
  низкая-средняя. Польза была слишком локальной.

### V2.1

- Архитектура:
  `v2` плюс более тяжёлый repair-oriented path.
- Где LLM:
  extraction, generation, extra repair/revise.
- Где deterministic:
  hygiene и guards.
- Что было сильным:
  локально лечил часть кейсов.
- Что было слабым:
  слишком много repair, latency выросла, открылись новые regressions.
- Влияние на итоговый текст:
  система стала хрупче; prose не стала надёжно лучше.
- Масштабируемость:
  низкая. Для тысяч постов такой repair-heavy path плох.

### V2.2

- Архитектура:
  subtractive correction `v2.1`, часть repair-pass убрана.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  cleanup, validation, targeted anti-`посвящ*`.
- Что было сильным:
  важный шаг назад от overly heavy repair architecture.
- Что было слабым:
  `посвящ*`, unsupported embellishment и brittle prompting остались.
- Влияние на итоговый текст:
  кое-где cleaner, но всё ещё не победа над baseline.
- Масштабируемость:
  средняя, но quality ceiling всё ещё низкий.

### V2.3

- Архитектура:
  `v2.2` плюс sparse no-heading, positive examples, body self-sufficiency, light pre-consolidation.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  лёгкая dedup/pre-consolidation, policy checks.
- Что было сильным:
  первый по-настоящему mixed-positive round после `v1`; `2660`, `2745`, `2687` заметно оздоровились; `2673` стал лучше читаемо.
- Что было слабым:
  `2734` сломался; pre-consolidation оказалась полезной, но слишком чувствительной.
- Влияние на итоговый текст:
  важный шаг к менее шаблонному тексту; sparse formatting стал заметно профессиональнее.
- Масштабируемость:
  средняя. Направление полезное, но ещё слишком case-sensitive.

### V2.4

- Архитектура:
  `v2.3` плюс anti-question headings, stronger preservation and routing tweaks.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  routing, more hygiene checks.
- Что было сильным:
  убрал question-headings как отдельный AI-prose паттерн.
- Что было слабым:
  сломал sparse routing, вернул label-style / bureaucratic framing, стал хуже `v2.3`.
- Влияние на итоговый текст:
  формально более "организован", но реально менее естественен.
- Масштабируемость:
  низкая. Слишком широкий patch pack для маленькой пользы.

### V2.5

- Архитектура:
  rollback к `v2.3` базе с более узкими правками по `посвящ*` и grouped program preservation.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  hygiene, grouping helpers.
- Что было сильным:
  лучше `v2.4`, аккуратнее по тону.
- Что было слабым:
  всё ещё не лучше `v2.3`; `посвящ*` не был стабильно добит.
- Влияние на итоговый текст:
  больше control, но без реального breakthrough.
- Масштабируемость:
  средняя, но без ясного upside.

### V2.6

- Архитектура:
  та же pattern-family, но с более строгим extraction/generation contract, label-style ban и anti-bureaucracy intent.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  routing, hygiene, policy checks.
- Что было сильным:
  один из лучших prose rounds; `2660`, `2745`, `2734` выглядят сильнее baseline; headings и текст в части кейсов заметно лучше baseline.
- Что было слабым:
  `2687` жёстко регресснул, `2673` остался бюрократичным, `посвящ*` вернулся.
- Влияние на итоговый текст:
  это была одна из самых убедительных демонстраций, что текст можно сделать лучше baseline, но не ценой стабильности.
- Масштабируемость:
  средняя или ниже. Quality ceiling высок, но architecture слишком неравномерна между source shapes.

### V2.7

- Архитектура:
  narrative shaping перенесён глубже в extraction через safe-positive transformations.
- Где LLM:
  extraction стал ещё более "редакторским", generation продолжал pattern path.
- Где deterministic:
  support/hygiene.
- Что было сильным:
  локально пытался решить negative-constraints проблему.
- Что было слабым:
  fact inflation, generic packaging, regression on previously good cases.
- Влияние на итоговый текст:
  тексты стали менее плотными и более искусственно "упакованными".
- Масштабируемость:
  низкая. Плохой пример того, как LLM extraction можно перегрузить стилистической задачей.

### V2.8

- Архитектура:
  corrective round после `v2.7`, но ещё внутри старой pattern-family; проблема support-layer sanitizer стала явно видна.
- Где LLM:
  extraction, generation, revise.
- Где deterministic:
  sanitizer, hygiene, validation.
- Что было сильным:
  `2734` частично восстановился; стало видно, что часть проблемы не в prompt, а в sanitizer.
- Что было слабым:
  `Тема:` contamination, потеря sparse branch, generation не могла компенсировать грязный facts layer.
- Влияние на итоговый текст:
  показал limits generation-only fixes.
- Масштабируемость:
  средняя, но architecture явно упиралась в support-layer drift.

### V2.9

- Архитектура:
  `v2.8` плюс sanitizer bypass и stronger extraction hints, без нового stage.
- Где LLM:
  extraction, generation.
- Где deterministic:
  sanitizer bypass, cleanup, validation.
- Что было сильным:
  доказал, что synthetic `Тема:` contamination была реальной support-layer проблемой.
- Что было слабым:
  dense lecture/presentation cases всё равно не вылечились; `ОШИБКА:`-style hints оказались слабыми.
- Влияние на итоговый текст:
  меньше артефактов, но качества текста как класса ещё не прибавилось.
- Масштабируемость:
  средняя. Это важный corrective step, но не стратегия.

### V2.10

- Архитектура:
  `v2.9` плюс `list consolidation`, action-oriented hints и `[плохо] -> [хорошо]` examples.
- Где LLM:
  extraction стал более shape-aware.
- Где deterministic:
  всё ещё поддержка/hygiene.
- Что было сильным:
  `list consolidation` реально помог на `2734` и `2687`.
- Что было слабым:
  `2660` и `2673` развалились; examples местами закрепляли сам канцелярский шаблон.
- Влияние на итоговый текст:
  доказал, что shape-specific extraction может дать wins, но универсальный prompt всё ещё unsafe.
- Масштабируемость:
  средняя-низкая. Слишком сильная зависимость от source shape.

### V2.11

- Архитектура:
  `v2.10` плюс anti-quote, post-merge semantic dedup, scoped list consolidation, clause-style nominalization.
- Где LLM:
  extraction, generation.
- Где deterministic:
  semantic dedup и merge cleanup стали важнее.
- Что было сильным:
  anti-quote реально полезен; `2734` сильно поправился.
- Что было слабым:
  `2687` и `2673` разлетелись, merge/floor interaction оказалась слишком хрупкой.
- Влияние на итоговый текст:
  quality win не случился; зато стало ясно, что старая pattern-family почти исчерпана.
- Масштабируемость:
  низкая. Здесь deterministic shaping уже слишком глубоко вмешивался в смысл.

### V2.12

- Архитектура:
  новая family:
  `raw_facts -> shape detection -> full-floor LLM normalization -> deterministic cleanup/dedup/cap -> generation -> optional editorial review`
- Где LLM:
  normalization всей fact base, generation.
- Где deterministic:
  cleanup/dedup/cap, validation.
- Что было сильным:
  первый раунд новой architecture; лучше baseline по coverage (`14 < 22`); вернул LLM-first semantic core.
- Что было слабым:
  `посвящ*` ещё жил, `2673` был explanation-heavy, editorial pass добавлял хрупкость.
- Влияние на итоговый текст:
  одна из самых важных архитектурных побед всего цикла; headings и текст на части кейсов уже менее шаблонны, чем baseline.
- Масштабируемость:
  хорошая потенциально. Это первый реально масштабируемый post-baseline direction.

### V2.13

- Архитектура:
  `v2.12` без full editorial pass:
  `full-floor normalization -> cleanup/dedup -> shorter exemplar-driven generation -> deterministic validation -> targeted repair`
- Где LLM:
  normalization, generation, узкий repair.
- Где deterministic:
  validation, hygiene, narrow fallback cleanup.
- Что было сильным:
  `forbidden = 0`; лучше baseline по coverage (`14 < 22`) и по hygiene; `2660` и `2745` сильные; текст на части кейсов явно лучше baseline по naturalness.
- Что было слабым:
  plateau по total missing относительно `v2.12`; `2673` всё ещё explanation-heavy; `2687` чуть потерял coverage.
- Влияние на итоговый текст:
  одна из лучших точек цикла, если смотреть именно на publishability текста, а не только на raw coverage.
- Масштабируемость:
  хорошая. Архитектура компактнее и ближе к тому, что реально можно масштабировать.

### V2.14

- Архитектура:
  `v2.13` плюс split-call для rich cases:
  `full-floor normalization -> cleanup/dedup -> LLM story outline -> exemplar-driven generation -> deterministic validation`
- Где LLM:
  normalization, outline, generation.
- Где deterministic:
  validation/hygiene.
- Что было сильным:
  `2734` улучшился; `2673` стал лучше называть проект и презентацию; система сознательно ушла от regex-heavy drift.
- Что было слабым:
  новый failure mode `outline-generated bureaucracy`; `2687` вернул `посвящ*`; `2673` получил бюрократические headings.
- Влияние на итоговый текст:
  подтвердил, что split-call может помочь, но prose-like outline быстро возвращает machine-like language.
- Масштабируемость:
  средняя-хорошая по идее, но только если outline станет structural и перестанет писать за автора.

## 8. Что версии сделали с итоговым текстом

### 8.1. Что baseline делал лучше многих экспериментальных версий

- Держал общий fact-first discipline.
- Реже разваливался архитектурно.
- Был проще и operationally понятнее.
- Чаще был "нормально publishable" без катастрофических сбоев.

### 8.2. Что baseline делал хуже сильных экспериментальных версий

- Тексты чаще были шаблонными.
- Headings часто были generic.
- Лиды были плоскими.
- Было больше ощущение "LLM summary", а не культурного анонса.

### 8.3. Где качество текста реально обгоняло baseline

Наиболее убедительно это происходило в:

- `v1` локально;
- `v2.3` на sparse и lecture cases;
- `v2.6` на нескольких кейсах сразу;
- `v2.13` по сочетанию hygiene + naturalness.

Именно поэтому нельзя сводить вывод цикла к "baseline всё равно лучше". Это неверно.

### 8.4. Где происходил разрыв между coverage и quality

Самый частый паттерн цикла:

- версия улучшает prose;
- но теряет часть sharp factual content;
- либо наоборот, держит coverage, но текст становится канцелярским или machine-like.

Это и есть главный engineering challenge задачи.

## 9. Что было масштабируемым, а что нет

### 9.1. Наиболее масштабируемые идеи

- baseline LLM-first fact-first flow;
- `full-floor normalization`;
- shorter exemplar-driven generation;
- deterministic validation вместо semantic rewriting;
- shape detection как routing hint, но не как жёсткая rule engine.

### 9.2. Наименее масштабируемые идеи

- heavy repair pipeline;
- semantically active regexes;
- dirty merge/floor clean+dirty facts;
- слишком сложный pattern routing с noisy `copy_assets`;
- prose-like outline, который сам пишет headings и смысловые рамки;
- попытка лечить всё через всё более длинный ban-list prompt.

## 10. Общие выводы

### 10.1. Мы не ходили по кругу полностью, но частично действительно теряли уже найденное

Это трезвый вывод.

Что реально происходило:

- часть циклов давала новые знания;
- часть решений реально поднимала качество;
- но отсутствие одного канонического retrospective/state-of-play документа приводило к тому, что найденные wins не всегда превращались в stable requirements.

### 10.2. Лучшее направление на сегодня — не baseline и не старая pattern-family, а новая LLM-first architecture семьи `v2.12-v2.14`

Причина:

- baseline хорош как stable floor;
- pattern-family показала ceiling по prose;
- но именно `v2.12-v2.14` впервые совместили:
  - LLM-first semantic core;
  - победу над baseline по coverage;
  - заметно менее шаблонный текст.

### 10.3. Лучший единичный candidate по качеству текста сейчас ближе к `v2.13`, чем к baseline или `v2.14`

Не потому, что `v2.13` идеален.

А потому что он:

- лучше baseline по coverage;
- лучше baseline по hygiene;
- заметно менее шаблонен;
- при этом не вернул `v2.14`-бюрократию и не утонул в `v2.11`-style fragility.

### 10.4. Что обязательно надо сохранить перед `v2.15`

- baseline fact discipline и cleanup layer;
- отказ от regex-first semantics;
- `v2.6`-уровень амбиции по качеству текста;
- `v2.11` anti-quote;
- `v2.12` full-floor normalization;
- `v2.13` shorter exemplar-driven generation и zero-forbidden hygiene;
- осознание, что headings и lead quality важны не меньше coverage.

### 10.5. Что нельзя повторять

- смыслообразующие regex-патчи;
- dirty merge clean и dirty facts в одном floor;
- длинные стены запретов вместо понятных exemplars;
- full editorial rewrite pass по умолчанию;
- prose-like outline, который сам заносит бюрократию;
- оптимизацию только под один literal missing-check без редакторской оценки текста.

## 11. Bottom line

Если смотреть издалека, то история `baseline -> v2.14` выглядит так:

- baseline дал надёжный, но шаблонный production floor;
- старая pattern-family доказала, что текст можно сделать заметно лучше baseline, но слишком часто делала это нестабильно;
- новая family `v2.12-v2.14` вернула цикл к правильной LLM-first логике и впервые дала architecture, которая одновременно:
  - сильнее baseline по coverage;
  - потенциально сильнее baseline по prose;
  - и лучше масштабируется, чем старая pattern-family.

То есть цикл был не бессмысленным.

Но твоя претензия по двум пунктам справедлива:

- требования нужно было раньше собрать в один устойчивый документ;
- и anti-cliche / anti-template wins действительно нельзя больше терять при каждом новом архитектурном повороте.
