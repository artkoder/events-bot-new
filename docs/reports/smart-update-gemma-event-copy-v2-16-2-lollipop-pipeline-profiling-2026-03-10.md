# Smart Update Gemma Event Copy V2.16.2 Lollipop Pipeline Profiling

Дата: 2026-03-10

## 1. Scope

Профиль собран по текущему активному пути `lollipop`:

```text
facts.extract
-> facts.dedup iter3
-> facts.merge iter5
-> facts.prioritize iter3
-> editorial.layout iter2
-> writer_pack.compose iter2
-> writer.final_4o iter2
```

Важно:

- это не один сплошной rerun в один момент времени, а сборка по уже существующим каноническим артефактам;
- `facts.prioritize iter3` канонически работает поверх `facts.merge iter5`;
- `source.scope` и mixed-phase interceptor сюда не включены;
- где family хранит `duration_s`, использованы эти значения;
- где family не хранит явный runtime, использован rough wall-clock по trace-файлам (`input/result` `mtime`).

## 2. Family Timing Table

| Family | Active run | Avg s/event | Sum s/12 | Main cost driver |
|---|---|---:|---:|---|
| `facts.extract` | `2026-03-09` | `227.037` | `2724.450` | широкая multi-stage extraction bank |
| `facts.dedup` | `iter3` | `77.788` | `933.460` | `baseline_diff` |
| `facts.merge` | `iter5` | `89.059` | `1068.710` | `resolve` |
| `facts.prioritize` | `iter3` | `18.824` | `225.889` | `weight` |
| `editorial.layout` | `iter2` | `17.896` | `214.751` | `plan` |
| `writer_pack.compose` | `iter2` | `0.094` | `1.129` | deterministic compose/select |
| `writer.final_4o` | `iter2` | `4.018` | `48.218` | final `4o` call |

## 3. Step Detail

### `facts.extract`

- average total: `227.037s/event`
- total: `2724.450s`
- median: `190.810s/event`
- max: `339.350s/event`

Самые дорогие extract stages по среднему времени:

1. `facts.extract_profiles.v1` — `25.329s/event`
2. `facts.extract_program_shape.v1` — `23.059s/event`
3. `facts.extract_program.v1` — `22.250s/event`
4. `facts.extract_stage.v1` — `15.191s/event`
5. `facts.extract_setlist.v1` — `14.862s/event`

Вывод:

- основной runtime branch сидит именно здесь;
- даже до дедупликации уже тратится больше половины всего текущего `extract -> final` времени.

### `facts.dedup iter3`

- average total: `77.788s/event`
- total: `933.460s`
- median: `73.470s/event`
- max: `128.080s/event`

Разбивка:

- `baseline_diff_results`: `901.620s` total across `84` launches, `10.734s` average launch
- `cross_enrich_results`: `31.840s` total across `4` launches, `7.960s` average launch

Самые медленные события:

1. `2673` — `128.080s`
2. `2447` — `100.880s`
3. `2701` — `96.960s`

Вывод:

- `cross_enrich` почти ничего не стоит;
- runtime практически полностью определяется `baseline_diff`.

### `facts.merge iter5`

- average total: `89.059s/event`
- total: `1068.710s`
- median: `78.310s/event`
- max: `172.440s/event`

Разбивка по recorded stages:

- `bucket_result`: `26.206s/event`
- `resolve_result`: `45.170s/event`
- `emit_result`: `17.683s/event`

Самые медленные события:

1. `2447` — `172.440s`
2. `2673` — `126.770s`
3. `2732` — `114.560s`

Вывод:

- `resolve` остаётся главным bottleneck всей merge-family;
- merge-family дороже любой отдельной downstream family.

### `facts.prioritize iter3`

- family wall-clock average: `18.824s/event`
- family wall-clock total: `225.889s`

Recorded stage durations:

- `weight`: `13.631s/event`
- `lead`: `4.682s/event`
- combined recorded stage time: `18.312s/event`

Вывод:

- почти весь downstream semantic cost живёт в `weight`;
- `lead` ощутим, но это не главный latency risk.

### `editorial.layout iter2`

- family wall-clock average: `17.896s/event`
- family wall-clock total: `214.751s`
- `plan` recorded average: `17.118s/event`

Вывод:

- layout почти сравнялся по цене с `facts.prioritize`;
- вместе эти две families составляют основную цену downstream tail.

### `writer_pack.compose iter2`

- family wall-clock average: `0.094s/event`
- family wall-clock total: `1.129s`
- max: `0.424s/event`

Вывод:

- stage практически бесплатный;
- любые дополнительные deterministic checks здесь почти не повлияют на latency.

### `writer.final_4o iter2`

- family wall-clock average: `4.018s/event`
- family wall-clock total: `48.218s`
- median: `4.298s/event`
- max: `6.212s/event`
- average attempts/event: `1.083`
- max attempts: `2`
- average prompt tokens/event: `1805.5`
- average completion tokens/event: `168.2`

Самые медленные события:

1. `2732` — `6.212s`
2. `2447` — `5.545s`
3. `2673` — `4.989s`
4. `2659` — `4.547s`
5. `2731` — `4.506s`

Вывод:

- final writer заметно дешевле, чем кажется интуитивно;
- по latency это не bottleneck downstream tail.

## 4. Downstream Tail

Для текущей активной downstream части:

- `facts.prioritize -> editorial.layout -> writer_pack.compose -> writer.final_4o`
- average total: `40.832s/event`
- total for `12` events: `489.987s`

Самые медленные downstream события:

1. `2673` — `95.784s`
2. `2731` — `74.462s`
3. `2732` — `52.428s`
4. `2734` — `40.256s`
5. `2447` — `32.067s`

Для `2673` почти весь хвост сидит не в final writer, а в:

- `facts.prioritize`: `46.940s`
- `editorial.layout`: `43.801s`

## 5. End-to-End Approximation

Если сложить текущий активный путь `facts.extract -> writer.final_4o` по лучшим доступным метрикам, получаем rough operational estimate:

- `434.716s/event`
- `7.245 min/event`
- `5216.592s` на `12` событий
- `86.943 min` на весь текущий `12`-event casebook

Это именно operational approximation, а не benchmark:

- upstream использует recorded `duration_s`;
- downstream частично использует wall-clock по trace `mtime`;
- run собирается из нескольких канонических итераций.

## 6. Main Takeaways

1. Основной runtime живёт upstream: `facts.extract + facts.dedup + facts.merge` дают примерно `393.884s/event`, то есть около `90.6%` всего текущего пути.
2. Downstream tail сравнительно дешёвый: `40.832s/event`, то есть около `9.4%` total path.
3. Внутри downstream tail почти вся цена сидит в `facts.prioritize + editorial.layout` (`36.720s/event`, около `89.9%` downstream tail).
4. `writer_pack.compose` практически бесплатен, а `writer.final_4o` не является главным latency bottleneck.
5. Если следующий retune касается только clarity/structure в writer-tail, latency budget лучше тратить не на новые LLM families, а на точечные contract changes в уже существующих `facts.prioritize` / `editorial.layout` / deterministic pack logic.
