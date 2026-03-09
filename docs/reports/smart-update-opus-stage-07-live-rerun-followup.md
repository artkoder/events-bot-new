# Smart Update Opus Stage 07 Live Rerun Follow-up

Дата: 2026-03-07

Stage 07 = точечный follow-up после следующего локального live rerun на подготовленной casebook БД.

Этот этап уже не про `matryoshka` и не про общий deterministic subset.
Новый главный сигнал здесь другой:

- runtime false positive на giveaway / promo post;
- тяжёлый same-source zoo schedule post, который не завис, но дал mixed result;
- и отдельный operational хвост по defer/lock path.

Связанные материалы:
- `docs/reports/smart-update-opus-stage-index.md`
- `docs/reports/smart-update-opus-stage-06-live-validation.md`
- `docs/reports/smart-update-stage-06-live-validation-response.md`
- `docs/reports/smart-update-duplicate-casebook.md`
- `artifacts/codex/smart_update_casebook_vk_reimport_prep_v3_latest.md`
- `artifacts/codex/smart_update_stage_07_live_rerun_followup_latest.md`
- `artifacts/codex/smart_update_stage_07_live_rerun_followup_latest.json`

## 1. Setup

Для rerun использовалась подготовленная копия snapshot:

- `artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.casebook_vk_reimport_v3.sqlite`

Смысл `v3` prep:

- удалены `86` casebook/stage06 event rows;
- восстановлены `32` связанных `vk_inbox` rows;
- вставлены `2` отсутствовавших `vk_inbox` rows;
- `remaining_case_events = 0`;
- `pending_rows_total = 38`.

Важно:

- `v3` prep строился от bundle + stage06 extras;
- не все исторические zoo child-events попали в этот delete-set;
- поэтому zoo rerun ниже нужно читать осторожно: часть сигнала confounded уже существовавшими rows.

## 2. Aggregate Rerun Result

По операторскому Telegram log за 2026-03-07:

| Run | limit | processed | imported | rejected | failed | deferred | created | updated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/vk_auto_import 1` | `1` | `1` | `1` | `0` | `0` | `0` | `1` | `0` |
| `/vk_auto_import 3` | `3` | `3` | `2` | `0` | `0` | `1` | `7` | `1` |
| **Итого** | `4` | `4` | `3` | `0` | `0` | `1` | `8` | `1` |

Позитивный baseline signal:

- `https://vk.com/wall-194968_17306` импортировался чисто:
  - created `event 2843`
  - `Сёстры`
  - `2026-03-07 19:00`

То есть rerun сам по себе не был “сломан”.

## 3. Main Findings

### 3.1. `giveaway_false_positive_baltika_cska`

Статус: новый содержательный blocker.

Observed runtime:

- `https://vk.com/wall-86702629_7354`
- created `event 2850`
- title:
  - `Розыгрыш билетов на матч «Балтика» — «ЦСКА»`
- date:
  - `2026-03-14`
- venue:
  - `Ростех Арена`

DB verification:

- у `event 2850` source text — это чистый giveaway / contest post;
- attendable event invite в source text нет;
- в тексте есть:
  - giveaway framing `РОЗЫГРЫШ`;
  - prize framing `главный приз — два билета на матч`;
  - mechanics:
    - подписка;
    - комментарии;
    - лайки;
    - итоги `10 марта`.

Проблема:

- система интерпретировала матч как event subject для импорта;
- но по смыслу это не анонс посещаемого события, а promo post, где матч фигурирует только как prize.

Почему это важно:

- это не duplicate miss и не gray;
- это прямой false positive / pseudo-event creation;
- для product quality это заметная ошибка публичного вывода.

Рабочая гипотеза:

- текущий `giveaway_no_event` guard слишком мягкий;
- mention of a real match/date is трактуется как “underlying event facts”;
- upstream parse / draft extraction тоже, вероятно, не различает:
  - `event as prize/reference`
  - vs `event as attendable subject`.

### 3.2. `zoo_heavy_schedule_post`

Статус: содержательно mixed, но не “зависание”.

Observed runtime:

- `https://vk.com/wall-48383763_39377`
- processed for `1460.6s`
- result:
  - created `2844`, `2845`, `2846`, `2847`, `2848`, `2849`
  - updated `2699`

Это подтверждает:

- пост действительно очень тяжёлый по числу child-events;
- wall-clock большой, но pipeline завершился корректно;
- сама жалоба “процесс завис” здесь не подтверждается.

### 3.3. Zoo Duplicate Residue Is Confounded

После rerun в DB одновременно живут:

- `2695` and `2846`
  - `Ветеринарный экспресс: о чём молчат животные`
  - `2026-03-14 11:00`
- `2696` and `2847`
  - `Экскурсия «Следы времени» по Калининградскому зоопарку`
  - `2026-03-15 11:00`
- `2698` and `2849`
  - `Экскурсия «Следы времени» в Калининградском зоопарке`
  - `2026-03-28 11:00`

При этом:

- `2699` for `2026-03-29` обновился, а не задублировался.

Ключевой нюанс:

- pre-existing rows `2695/2696/2698/2699` не входили в bundle delete-set `v3` prep;
- то есть rerun сравнивал новые zoo drafts не с полностью очищенной family, а с partly surviving family.

Практический вывод:

- эти zoo duplicates нельзя честно трактовать как чистый regression against Stage 04 alone;
- но это всё равно важный runtime signal:
  - same-source multi-event reimport конвергирует не полностью;
  - часть child-events update'ится, часть создаётся заново;
  - значит есть либо pre-match title/extraction gap, либо shortlist/action asymmetry внутри одного same-source schedule post.

### 3.4. `deferred_locked_row_after_rate_limit`

Статус: operational issue, не merge-quality.

Observed runtime:

- `https://vk.com/wall-212760444_4543`
- deferred on TPM
- row `vk_inbox.id = 4750` остался:
  - `status = locked`
  - `locked_by = 8336351413`
  - `review_batch = auto:1772892308`

Важно:

- это не обязательно bug;
- текущий defer path намеренно держит row locked до stale-timeout, чтобы тот же unbounded run не подбирал её снова;
- stale unlock идёт позже.

Но как test ergonomics signal это всё равно неудобно:

- оператор после run видит не `pending`, а `locked`;
- следующий rerun может выглядеть “грязным”, хотя это просто backoff behaviour.

## 4. Что этот этап реально меняет относительно Stage 06

Stage 06 был про:

- `matryoshka` false merge;
- `oncologists`;
- `makovetsky`;
- `little_women`;
- `vistynets`;
- zoo false-friend control.

Stage 07 добавляет новый класс проблемы:

- false positive на giveaway/promo post,
  а не только ошибки merge/dedup.

Это важно для следующего Opus-раунда, потому что теперь вопрос уже шире:

- не только “как лучше merge existing events”;
- но и “как не превращать promo/prize posts в события вообще”.

## 5. Что мы хотим спросить у Opus

### 5.1. Giveaway False Positive

Нужен engineering verdict:

1. primary cause:
   - deterministic guard too weak,
   - upstream extraction miss,
   - prompt miss,
   - или комбинация;
2. какой safe next step лучше:
   - tighter deterministic `giveaway_no_event`,
   - upstream parse prompt change,
   - schema-level distinction (`subject event` vs `prize/reference event`),
   - или комбинация.

Отдельно нужен prompt-level advice:

- как переформулировать extraction prompt, чтобы пост вида:
  - `розыгрыш`
  - `главный приз — билеты на матч`
  - `итоги позже`
  не порождал event candidate про матч.

### 5.2. Zoo Same-Source Multi-Event Rerun

Нужен аккуратный разбор:

1. насколько этот signal можно считать содержательным, а не prep-confounded;
2. если смотреть по сути:
   - почему same-source family частично converged (`2699`),
   - а частично нет (`2695/2696/2698`);
3. какой safe mitigation рациональнее:
   - deterministic same-source convergence upgrade,
   - upstream draft-title improvement,
   - prompt rule для multi-event extraction,
   - или keep-gray / no-action до clean rerun.

### 5.3. Prompt-Level Patch Pack

Нужен не абстрактный advice, а маленький practical patch pack:

- какие именно инструкции стоит добавить в upstream parse prompt;
- какие — в Smart Update matching / create prompts;
- какие сигналы нельзя переоценивать;
- какие explicit anti-patterns надо проговорить модели.

Особенно интересуют два класса:

1. `giveaway / prize / contest` posts
2. `same-source schedule / recurring multi-event` posts

## 6. Предварительный внутренний вывод

На текущий момент у нас два разных follow-up трека:

1. code/system track:
   - false-positive giveaway fix;
   - возможно, cleanup DB prep для zoo family;
2. consultation track:
   - попросить Opus дать узкие инженерные предложения и prompt changes,
     а не уходить в новый общий redesign.

Именно это и есть цель Stage 07 handoff.
