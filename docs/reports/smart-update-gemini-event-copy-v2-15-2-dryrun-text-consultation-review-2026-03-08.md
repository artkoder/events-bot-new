# Smart Update Gemini Event Copy V2.15.2 Dry-Run Text Consultation Review

Дата: 2026-03-08

Основание:

- Raw Gemini report: [event-copy-v2-15-2-dryrun-text-consultation-gemini-3.1-pro-2026-03-08.md](/workspaces/events-bot-new/artifacts/codex/reports/event-copy-v2-15-2-dryrun-text-consultation-gemini-3.1-pro-2026-03-08.md)
- Session log: [session-2026-03-08T17-31-012517ce.json](/home/vscode/.gemini/tmp/events-bot-new/chats/session-2026-03-08T17-31-012517ce.json)
- [v2.15.2 dry-run report](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-5-events-2026-03-08.md)
- [v2.15.2 prompt context](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-15-2-prompt-context-2026-03-08.md)
- [v2.15.2 review](/workspaces/events-bot-new/docs/reports/smart-update-opus-gemma-event-copy-pattern-dry-run-v2-15-2-review-2026-03-08.md)

## 1. Короткий verdict

Ответ Gemini полезный и в целом хорошо откалиброван именно по text-quality проблемам `v2.15.2`.

Самое ценное:

- он правильно поднял `quote/epigraph` как top blocker;
- отдельно заметил `plan -> output` mismatch на `2660`;
- не увёл разговор обратно в giant-architecture debate и дал конкретные Gemma prompt-level рекомендации.

Но на слово я его не принимаю.

Итог:

- ответ **worth taking**;
- как guide для следующего `2.15.3` — да;
- как final oracle по общему verdict — нет.

## 2. Что принимаю

### 2.1. False quote / false epigraph сейчас действительно главный blocker

Gemini прав по сути:

- `2687` сломан ложной цитатой из соседнего digest item;
- `2673` сломан уже generation-stage quote hallucination;
- `2734` показывает, что даже source-backed fragment ещё не делает хороший epigraph.

Это сейчас важнее, чем ещё один раунд борьбы с общим канцеляритом.

### 2.2. Нужен жёсткий format gate по реальному output, а не только по prompt intention

Gemini прав:

- если `use_headings=false`, то одних инструкций в generation prompt недостаточно;
- mismatch надо ловить детерминированно на финальном тексте.

Это относится не к semantic core, а к safe output enforcement, поэтому такой deterministic слой допустим.

### 2.3. `2673` всё ещё слишком agenda-like

Это тоже точный диагноз.

Даже при улучшении factual framing `2673` местами остаётся текстом вида:

- `устройство и цели платформы`
- `формат встречи`

То есть case ещё не дошёл до по-настоящему живой project presentation prose.

### 2.4. Pure prose quality `v2.15.2` пока не clean win над `v2.13`

Это неприятно, но близко к правде.

На coverage/hygiene `v2.15.2` сильнее.
На чистой редакторской цельности `v2.13` всё ещё местами безопаснее.

## 3. Что беру только с поправкой

### 3.1. Quote extraction для digest cases

Gemini предлагает почти полностью блокировать `quote_led` для дайджестов.

Это полезный сигнал, но я беру его только с поправкой:

- не blanket-ban на любой digest;
- а более строгий `event-local quote gate`.

То есть:

- quote должен быть привязан к самому event fragment;
- quote не должен приезжать из соседнего анонса;
- короткие обрывки и titles не должны активировать epigraph.

### 3.2. Жёсткая project normalization

Gemini верно требует сильнее чистить project/presentation metatext.

Но его пример:

- `как устроена платформа` -> `платформа позволяет находить партнеров`

слишком unsafe, если конкретная возможность явно не подтверждена фактами.

Принимаю только принцип:

- agenda-style framing надо убирать;
- но semantic rewrite не должен подменять факт inferred benefit statement.

### 3.3. Stop-headings и stop-openers

Gemini тут полезен.

Но использовать это нужно как компактный bank:

- несколько weak openers;
- несколько weak headings;
- несколько positive replacements.

Не как новую giant ban wall.

## 4. Что не принимаю

### 4.1. Тезис, что `v2.15.2` хуже `v2.13` и `v2.14` вообще

С этим я не согласен.

По system-level сумме:

- coverage лучше;
- forbidden = `0`;
- `2673` factual framing лучше;
- `2745` лучший кейс текущего round.

Правильнее так:

- `v2.15.2` сильнее как architecture candidate;
- слабее `v2.13` на части pure prose cases;
- не production-ready из-за quote/format failures.

### 4.2. Слишком жёсткое сужение `quote_led`

Gemini почти сводит `quote_led` к лекциям, стендапам и авторским встречам.

Это слишком узко.

`quote_led` может быть уместен и вне этих форматов, но только при реальной сильной цитате.
Проблема сейчас не в самом pattern, а в слабом quote gate.

## 5. Что это значит для следующего шага

Если из ответа Gemini брать только dry-run-confirmed вещи, то следующий `2.15.3` patch pack должен быть узким:

1. Жёсткий anti-hallucination contract для epigraph:
   - эпиграф можно использовать только дословно из `quote_text`.
2. Event-local quote gate:
   - titles, fragments и digest-leak quotes не должны активировать `quote_led`.
3. Deterministic format enforcement:
   - `use_headings=false` -> никакие `###` не проходят.
4. Epigraph position lock:
   - blockquote допустим только первым блоком.
5. Project-case prose cleanup:
   - меньше agenda headings;
   - меньше `устройство/цели/формат встречи`-тона;
   - больше нормальной presentation lead logic.

## 6. Финальный вывод

Gemini был полезен именно там, где сейчас главный риск:

- ложные цитаты;
- format slippage;
- robotized project prose.

Его ответ не отменяет наш собственный review, а хорошо его усиливает.

Следствие:

- дополнительный Gemini round сейчас не нужен;
- перед следующим раундом с `Opus` имеет смысл сначала локально собрать `2.15.3` именно вокруг этих text-quality блокеров.
