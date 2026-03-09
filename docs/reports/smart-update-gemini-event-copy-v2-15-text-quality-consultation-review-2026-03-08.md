# Smart Update Gemini Event Copy V2.15 Text Quality Consultation Review

Дата: 2026-03-08

Связанные материалы:

- raw Gemini report: `artifacts/codex/reports/event-copy-v2-15-text-quality-consultation-gemini-3.1-pro-2026-03-08.md`
- brief sent to Gemini: `artifacts/codex/tasks/gemini_event_copy_v2_15_text_quality_consultation_brief_2026-03-08.md`
- session log: `/home/vscode/.gemini/tmp/events-bot-new/chats/session-2026-03-08T16-08-39eb4b01.json`
- current design brief: [smart-update-gemma-event-copy-v2-15-design-brief-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-design-brief-2026-03-08.md)
- retrospective: [smart-update-gemma-event-copy-retrospective-baseline-v2-14-2026-03-08.md](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-retrospective-baseline-v2-14-2026-03-08.md)

## 1. Проверка модели

Session log подтверждает:

- `model = gemini-3.1-pro-preview`

То есть консультация проведена реальной `gemini-3.1-pro-preview`, а не auto-fallback моделью.

## 2. Краткий verdict

Это полезная и в целом сильная консультация именно по **качеству текста**.

Главное, что Gemini сделал правильно:

- сместил разговор с абстрактного "стиля журналиста" к конкретным текстовым правилам;
- хорошо попал в проблему новых шаблонов внутри pattern-library;
- дал много практических замечаний по leads, headings, lists, epigraphs и lexical hygiene;
- предложил полезный stop-phrase слой именно для русского AI-prose.

Но архитектурную часть Gemini по-прежнему нужно читать осторожно:

- он снова тянет часть решений в deterministic enforcement сильнее, чем стоит;
- часть hard bans слишком жёсткие и могут снова сделать текст сухим или искусственным;
- некоторые structural rules полезны как heuristics, но опасны как абсолютные законы.

Мой итог:

- **новый Gemini-раунд сейчас не нужен**;
- text-quality слой ответа стоит использовать;
- но брать его надо через `accept / accept with modification / reject`, а не как oracle.

## 3. Что в ответе Gemini особенно сильное

### 3.1. Критика "стиля культурного журналиста" как пустой абстракции

Это сильный и своевременный удар по одному из слабых мест текущего brief.

Gemini прав:

- для Gemma формулы вроде `пиши как культурный журналист` слишком расплывчаты;
- без синтаксических и риторических ограничений модель быстро уезжает в:
  - пресс-релиз;
  - пафос;
  - канцелярит;
  - pseudo-literary filler.

Это не значит, что persona не нужна вообще.
Но persona должна быть поддержана более concrete writing rules.

### 3.2. Сильный фокус на syntax-level prose quality

Особенно полезны его предложения про:

- активный залог;
- прямой вход в предмет;
- запрет на оценочные эпитеты;
- приоритет concrete nouns и strong verbs;
- запрет на философские или meta-intro openings.

Это полезнее, чем просто очередной список "не используй клише".

### 3.3. Правильная тревога про "новую шаблонность"

Gemini справедливо заметил важный риск:

- если pattern-library останется слишком абстрактной, Gemma быстро превратит каждый pattern в новую повторяемую формулу.

Это очень важный warning.

Отсюда сильный вывод:

- patterns нужны;
- но они должны быть не "темами", а более concrete structural cards.

### 3.4. Сильные замечания про headings, epigraphs и lists

Полезные точки:

- headings должны быть semantic, а не interface-like;
- epigraph нужен только когда он реально усиливает текст;
- list formatting должно быть строго связано с homogeneous payload, а не с "красивым видом".

Это хорошо совпадает с нашими ретроспективными выводами по baseline, `v2.6` и `v2.13`.

### 3.5. Стоп-фразы и русскоязычный anti-bureaucracy слой

Это один из самых practically useful блоков ответа.

Полезные candidate bans / discouraged phrases:

- `мероприятие`
- `данное событие`
- `погрузиться в атмосферу`
- `уникальная возможность`
- `никого не оставит равнодушным`
- `будет интересно как ..., так и ...`
- `не просто X, а настоящее Y`

Это не вся будущая stoplist, но как стартовый lexical layer — сильный материал.

## 4. Что принимаю почти без поправок

### 4.1. Добавить в `v2.15.2` syntax-level quality rules

Принимаю.

Нужно усилить brief и generation prompts правилами такого класса:

- без meta-openers;
- с прямым входом в предмет;
- без рекламных эпитетов;
- с приоритетом strong verbs / concrete nouns;
- без цепочек родительных падежей и канцелярита.

### 4.2. Усилить anti-evaluation / anti-filler contract

Принимаю.

Нужен более явный ban на:

- оценочные похвалы;
- "обещает стать";
- "позволит погрузиться";
- "уникальная возможность";
- "настоящий праздник".

### 4.3. Явно уточнить правила для leads

Принимаю.

Особенно полезные идеи:

- первый абзац должен входить в сам предмет события;
- не начинать с философских разгонов;
- не начинать с `Это событие`, `Лекция расскажет`, `Спектакль рассказывает`.

### 4.4. Сделать headings semantic-only

Принимаю.

Нужны:

- blacklist generic headings;
- heuristic "если heading подходит к любому событию, он плохой";
- более concrete heading palette.

### 4.5. Уточнить list logic

Принимаю.

Списки должны использоваться только для:

- реальной программы;
- homogeneous payload;
- named items.

Не для:

- абстрактных преимуществ;
- промо-обещаний;
- pseudo-features.

## 5. Что принимаю, но только с модификацией

### 5.1. Pattern-library critique

Gemini прав по сути, но его ответ нельзя понимать как "patterns не нужны".

Принимаю такой вывод:

- pattern-library надо делать более concrete;
- patterns должны быть ближе к structural delivery modes, а не к слишком общим narrative archetypes.

Не принимаю вывод:

- что из-за риска новой шаблонности patterns нужно фактически сворачивать.

Правильная коррекция для `v2.15.2`:

- сохранить pattern-layer;
- переписать patterns как more explicit structural cards;
- дополнить их better exemplars.

### 5.2. Deterministic pattern selection

Gemini предлагает почти полностью rule-based pattern choice.

Принимаю только частично:

- deterministic / cheap metadata should dominate;
- LLM не должен свободно фантазировать pattern;
- но optional tiny fallback для ambiguous cases всё ещё полезен.

То есть:

- pattern choice mostly deterministic;
- not 100% python-only dogma.

### 5.3. Stopword DB / zero-tolerance filter

Как validation/repair layer — принимаю.

Как semantic core — не принимаю.

Правильная форма:

- stopword DB после generation;
- narrow repair if violated;
- без попытки через regex выправлять сам смысл текста.

### 5.4. Более жёсткий `why it matters`

Gemini предлагает почти фактически ban-нуть любую попытку value proposition.

Полностью не принимаю.

Но принимаю уточнение:

- `why it matters` должен быть очень узким;
- только when clearly evidenced;
- никогда не должен превращаться в promo;
- по умолчанию лучше не писать, чем написать generic value sentence.

### 5.5. Epigraph rules

Gemini слишком жёстко связывает эпиграф только с explicit quote field.

Это слишком узко.

Принимаю ослабленную версию:

- epigraph допустим только когда есть реально сильный source-backed quote-like fact;
- он не должен быть synthetic;
- он не должен дублировать первый абзац;
- body обязан быть самодостаточным.

### 5.6. Two-stage generation

Это интересная гипотеза, но пока только experiment-only.

В production-core пока не беру.

Причина:

- это увеличивает complexity и call count;
- пока не доказано, что выигрыш на Gemma будет устойчивым;
- сначала стоит попробовать улучшить one-pass generation contract.

## 6. Что не принимаю

### 6.1. Полный отказ от "why it matters"

Слишком жёстко.

Это важный editorial move, когда он grounded.
Проблема не в самом ходе, а в его generic/promo реализации.

### 6.2. Абсолютный тезис, что macro-patterns почти неизбежно превратятся в зло

Не принимаю.

Это скорее warning, чем verdict.
У нас уже было достаточно evidence, что pattern variation сама по себе полезна для prose quality.

### 6.3. Жёсткое правило "никаких headings" в pattern snippets

Это слишком flattening move.

Baseline и часть сильных rounds показали:

- headings полезны на rich cases;
- проблема не в headings вообще, а в generic headings.

### 6.4. Сильный уклон в rule-based Python governance

Не принимаю как основное направление.

Это может быть useful support layer.
Но если довести этот совет до конца, мы снова уйдём в drift от LLM-first policy.

## 7. Что реально меняет ответ Gemini для `v2.15.2`

### 7.1. Нужно переписать pattern layer

Не выкинуть, а сделать жёстче и конкретнее.

Вместо слишком общих названий:

- больше structural cards;
- больше pattern-specific exemplars;
- меньше абстрактных meta-labels.

### 7.2. Нужно усилить generation contract именно на уровне фраз и синтаксиса

Это, вероятно, самый high-impact practical takeaway.

Нужны:

- banned opener families;
- discouraged bureaucratic verbs;
- preferred subject-verb structures;
- list/heading heuristics;
- anti-evaluation lexical guidance.

### 7.3. Нужно завести explicit stop-phrase layer

Не как core semantics, а как:

- validation set;
- repair trigger set;
- prompt blacklist for generation/refine.

### 7.4. Нужно сильнее защищать epigraph/list formatting от decorative use

Это хорошо совпадает с нашими собственными выводами.

Не любой epigraph полезен.
Не любой list делает страницу лучше.

## 8. Рабочий patch pack для `v2.15.2`

Если превращать этот ответ в реальные изменения, я бы взял именно такой узкий набор:

1. Переписать quality section generation prompt из абстрактного style-language в syntax-level rules.
2. Ввести явный blacklist для openers.
3. Ввести explicit stop-phrase bank для bureaucratic/promo Russian AI-prose.
4. Переписать pattern library из abstract molds в structural cards с short exemplars.
5. Оставить headings только для rich cases и усилить semantic heading rules.
6. Жёстче описать list usage: only homogeneous named payload.
7. Уточнить epigraph contract и anti-dup rule между epigraph and body.
8. Оставить pattern selection mostly deterministic, но не превращать его в pure regex governance.
9. Ужесточить `why it matters` gate, но не ban-нуть его полностью.
10. Оставить two-stage generation только как later experiment, не как immediate core move.

## 9. Bottom line

Gemini был особенно полезен там, где речь шла не об архитектуре, а о **качествах самого текста**:

- leads;
- headings;
- lexical hygiene;
- anti-bureaucracy;
- epigraph/list discipline;
- practical prompt phrasing for Gemma.

Это стоит использовать.

Но architecture-level советы Gemini всё ещё надо фильтровать:

- часть из них слишком rule-heavy;
- часть слишком категорична;
- часть может снова сделать систему безопасной, но слишком сухой.

Поэтому мой итог такой:

- **response useful**;
- **new Gemini round not needed now**;
- **take text-quality layer seriously**;
- **carry a filtered patch pack into `v2.15.2`, not the whole response verbatim**.
