# Smart Update Identity Longrun

Статус: завершённый offline dry-run на свежем prod-snapshot `2026-03-06`.

Основные артефакты:
- итоговый long-run proposed модели: `artifacts/codex/smart_update_identity_longrun_20260306_v7.json`
- markdown-сводка long-run: `artifacts/codex/smart_update_identity_longrun_20260306_v7.md`
- baseline current-prompt на расширенном gold-наборе: `artifacts/codex/smart_update_identity_longrun_20260306_v6.json`
- targeted checks после финальной настройки: `artifacts/codex/smart_update_identity_targeted_checks_20260306.json`
- кейсбук расследованных дублей: `docs/reports/smart-update-duplicate-casebook.md`

## 1. Короткий вывод

Тестируемая quality-first подсистема identity-resolution внутри Smart Update на расширенном gold-наборе из `32` кейсов дала `32/32` acceptable decisions:
- `20` safe-merge кейсов ушли в `merge`;
- `12` must-not-merge кейсов не ушли в merge (`11 gray`, `1 different`);
- `safe_merge_misses = 0`;
- `must_not_merge_failures = 0`.

Это уже лучше текущего прод-подхода не на уровне “ощущений”, а на одном и том же наборе реальных прод-кейсов:
- текущий current-prompt baseline acceptable только `20/32`;
- по сути он пытается `merge` у всех `32` кейсов;
- все `12` must-not кейсов current prompt хотел автосклеить.

## 2. Что именно сравнивалось

### Текущий прод-пайплайн

```text
Extractor / importer
  -> Smart Update candidate
  -> source_url / source anchor checks
  -> shortlist by date(+city)
  -> early location/time filtering
  -> deterministic exact/related-title checks
  -> fat-shortlist LLM match_or_create
  -> binary decision:
       merge/update
       or create
  -> facts merge / rewrite / Telegraph / queues
```

Свойства текущего пайплайна:
- бинарное решение `merge` или `create`;
- промежуточного безопасного состояния `gray` нет;
- shortlist может потерять правильный existing event ещё до LLM;
- current prompt излишне склонен к merge на schedule/repertoire/series кейсах.

### Тестируемый quality-first вариант внутри Smart Update

```text
Extractor / importer
  -> Smart Update candidate + source metadata
  -> hard guards
       expected_event_id
       single-event source ownership
       safe source-anchor reuse
  -> shortlist by broad anchors
  -> Identity Resolver (внутри Smart Update)
       profile build
       deterministic evidence scoring
       merge / gray / different routing
       mandatory pairwise LLM triage
  -> final decision:
       merge
       gray_create_softlink
       create
  -> facts merge / rewrite / Telegraph / queues
```

Свойства тестируемого варианта:
- `merge` разрешается только при сильном identity-proof;
- `gray` используется как нормальное состояние, а не как скрытый `create`;
- LLM обязателен, но не на fat-shortlist, а как compact pairwise judge;
- ошибочная склейка считается более тяжёлой ошибкой, чем дубль.

## 3. Что вошло в модель

### Hard rules

- `expected_event_id` для linked-source enrichment: linked-source не должен создавать новый `event`.
- single-event source ownership guard: один single-event `source_url` не должен владеть несколькими active event без явной multi-event природы.
- venue/title confusable normalization: `№11` vs `№ 11`, `Авe` vs `Аве`, `Bar` vs `Бар`.

### Deterministic signals

- title exact / related / alias / quoted-title bridge;
- explicit time vs `door_time` / `start_time`;
- venue core / city / venue noise rescue;
- specific `ticket_link`;
- `poster_hash` overlap;
- follow-up post signals;
- canonical time correction;
- multi-session risk:
  - один и тот же post/page породил несколько same-day sessions;
  - оба времени явно перечислены в одной афише;
  - записи созданы почти одновременно;
- schedule-parent risk:
  - umbrella-title вроде `... и другие события`;
  - режим работы/праздничная программа;
  - child event из schedule нельзя автоматически склеивать с umbrella parent.

### Mandatory LLM triage

Gemma используется только как pairwise judge по компактному structured payload:
- два event profile;
- evidence flags и blockers;
- решение: `merge | gray | different`;
- merge mode: `regular | time_correction | followup_update | same_source_alias | brand_item | none`.

## 4. Итоги прогона

### Gold benchmark

Источник: `artifacts/codex/smart_update_identity_longrun_20260306_v7.json`

- `gold_total = 32`
- `safe_merge = 20`
- `must_not_merge = 12`
- `gold_acceptable = 32`
- `gold_acceptable_rate = 1.0`
- `gold_predictions = {'merge': 20, 'gray': 11, 'different': 1}`

Что это значит:
- все подтверждённые дубль-кейсы из кейсбука модель склеила;
- все расширенные control-case на ложные merge она удержала от автосклейки.

### Current baseline

Источник:
- `artifacts/codex/smart_update_identity_longrun_20260306_v6.json`
- `artifacts/codex/smart_update_identity_targeted_checks_20260306.json`

Итог по тому же расширенному gold-набору:
- acceptable только `20/32`
- current prompt тянет в `merge` все `32/32` кейса
- это значит:
  - `0` safe-merge misses
  - `12` must-not-merge failures

Проще говоря: текущий prompt хорошо ловит дубли, но слишком плохо различает
`same event` и `same family / same program / same source`.

### Additional mined pairs

Источник: `artifacts/codex/smart_update_identity_longrun_20260306_v7.json`

На `48` дополнительно mined прод-парах:
- `merge = 6`
- `gray = 14`
- `different = 28`

Это хороший знак именно для quality-first режима:
- модель не пытается массово схлопывать всё похожее;
- спорные пары в большинстве случаев остаются в `gray` или `different`.

## 5. Что long-run реально нашёл

### Удачные дополнительные merge-кластеры

1. `2761/2815/2816/2817` — `Маленькие женщины`
- это подтверждённый дубль-кластер одного показа;
- модель устойчиво собирает `brand_item + exact title` комбинацию.

2. `282/360` — `Зойкина квартира`
- это корректный `time_correction` кейс;
- у одной записи время пустое, у другой `19:00`.

### Новые must-not кейсы, добавленные из майнинга в gold

Именно mined long-run выявил класс, которого не было в исходном кейсбуке:
- `1293/1294` `Щелкунчик` — два показа в один день;
- `2005/2007` `Северное сияние` — дневные сеансы с одной play page;
- `1715/1716` `Лукоморье` — матине и вечерний показ;
- `1310/1311` `Текстуры. Глина` — digest-derived same-day sessions;
- `1419/1420` `Столярный авангард` — same-day workshop sessions;
- `1453/1454` `Петя Перепёлкин` — афиша явно перечисляет `12:00` и `15:00`;
- `2777/2828` — umbrella schedule post Третьяковки против конкретного child event.

Это важный вывод: без long-run по mined прод-парам модель выглядела бы сильно лучше, чем есть на самом деле.

## 6. Residual risk

После финальной итерации long-run всё ещё оставил один заметный false-positive class:

### `2743/2744/2745` — праздничная программа Музея изобразительных искусств

Кластер:
- `2743` `8 Марта в Музее изобразительных искусств`
- `2744` `Акция «Вам, любимые!» в Музее изобразительных искусств`
- `2745` `Бесплатная экскурсия в Музей изобразительных искусств`

Проблема:
- все три derived из одного и того же VK post;
- `source_text` совпадает байт-в-байт;
- `ticket_link` общий;
- current deterministic scorer всё ещё считает это сильным identity-proof.

Почему это важно:
- это не “ещё один дубль”, а оставшийся риск ошибочной склейки umbrella holiday program и её конкретных дочерних пунктов.

Вывод:
- proposed модель уже достаточно хороша для shadow-rollout и для внедрения hard guards;
- но перед агрессивным auto-merge для same-source multi-child schedule кейсов нужен ещё один guardrail:
  - `same_source_multi_child_risk` для schedule/holiday posts, где один source породил несколько child event с одинаковым payload.

## 7. TPM и latency

Наблюдения по длинным прогонам:
- pairwise triage prompt в среднем около `5.3k` chars;
- input tokens по живым вызовам обычно были порядка `1.4k .. 2.4k`;
- с pacing `10s` long-run не упирался в TPM;
- были отдельные transient reserve RPC warnings / local-fallback reserve, но прогон завершался без срыва.

Практический вывод:
- такой quality-first LLM-pass можно держать без fat-shortlist fan-out;
- TPM-профиль получается лучше контролируемым, чем у current full-shortlist prompt;
- latency спорных кейсов будет выше, но это соответствует целевому приоритету качества.

## 8. Что уже можно встраивать в Smart Update

### Можно внедрять уже сейчас

1. `expected_event_id` hard guard для linked-source enrichment.
2. single-event source ownership guard.
3. выравнивание runtime location/title normalization с dedup-логикой.
4. identity resolver внутри Smart Update с результатом:
   - `merge`
   - `gray_create_softlink`
   - `create`
5. pairwise LLM triage вместо fat-shortlist decision prompt.
6. shadow logging:
   - `identity_trace`
   - `deterministic_reasons`
   - `blockers`
   - `triage_decision`
   - `final_decision`

### Что ещё не стоит считать завершённым

Не закрыт до конца класс:
- same-source multi-child holiday/schedule programs, где raw payload одинаковый, а child titles разные.

Именно его нужно добить до production auto-merge rollout, если приоритет остаётся на минимизации ложных склеек.

## 9. Рекомендованный rollout

1. Встроить identity resolver в Smart Update как подсистему матчинга, не меняя остальной merge/rewrite pipeline.
2. Включить shadow mode:
   - current runtime decision остаётся прежним;
   - новая модель только логирует своё решение и trace.
3. Сначала включить только hard guards:
   - `expected_event_id`
   - linked-source no-create
   - single-event source ownership
4. Потом включить `gray_create_softlink` для части источников.
5. Только после отдельного добивания `same_source_multi_child_risk` включать полноценный auto-merge для same-source holiday/schedule derived cases.
