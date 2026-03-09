# Smart Update Opus Stage 06 Live Validation

Дата: 2026-03-07

Stage 06 = первый live validation-pass после локальной реализации narrowed deterministic subset.

Это уже не dry-run и не abstract alignment.
Это проверка на реальном локальном прогоне VK auto-import по очищенному casebook snapshot.

Связанные материалы:
- `docs/reports/smart-update-opus-stage-index.md`
- `docs/reports/smart-update-opus-stage-05-brief.md`
- `docs/reports/smart-update-stage-05-final-response.md`
- `docs/reports/smart-update-duplicate-casebook.md`
- `artifacts/codex/smart_update_casebook_vk_reimport_prep_latest.md`
- `artifacts/codex/smart_update_stage_06_live_validation_latest.md`
- `artifacts/codex/smart_update_stage_06_live_validation_latest.json`

## 1. Setup

Для live validation использовалась подготовленная копия snapshot:

- `artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.casebook_vk_reimport_v1.sqlite`

Перед прогоном в этой БД было сделано:

- удалены `83` casebook event ids;
- восстановлены в `pending` связанные `vk_inbox` rows;
- вставлены `2` отсутствовавших VK rows;
- всего к reimport было возвращено `33` VK queue rows;
- покрытие через VK queue у casebook ограничено `22 / 35` кейсами;
- ещё `13 / 35` кейсов не VK-linked и этим прогоном не валидируются.

## 2. Aggregate Run Result

По операторскому Telegram log за 2026-03-07:

| Run | limit | processed | imported | rejected | failed | deferred | created | updated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/vk_auto_import 1` | `1` | `1` | `0` | `0` | `0` | `1` | `0` | `0` |
| `/vk_auto_import 1` | `1` | `1` | `0` | `0` | `1` | `0` | `0` | `0` |
| `/vk_auto_import 3` | `3` | `3` | `2` | `0` | `0` | `1` | `3` | `0` |
| `/vk_auto_import 20` | `20` | `20` | `10` | `1` | `0` | `9` | `11` | `4` |
| **Итого** | `25` | `25` | `12` | `1` | `1` | `11` | `14` | `4` |

Главная operational картина:

- `TPM` pressure остаётся очень тяжёлым: `11` deferred rows на `25` processed;
- Gemma дважды падала в `match_create_bundle` с переключением на `4o`;
- один VK row дал `bad gemma parse response`;
- wall-clock для `limit=20` получился `6267s`, что слишком дорого даже для quality-first режима.

## 3. Что этот live run реально показал

### 3.1. `matryoshka_exhibition_duplicate`

Статус: самый опасный live finding.

Observed runtime:

- `https://vk.com/wall-9118984_23492` создал `event 2849`:
  - `Путешествие матрешки`
  - `2026-03-07 .. 2026-04-05`
  - venue: `Музей Изобразительных искусств`
- `https://vk.com/wall-9118984_23524` **не** смёржился в `2849`;
- вместо этого он обновил `event 2746`:
  - `Акция «Музей. Музы и творцы» в Музее изобразительных искусств`
  - source list у `2746` теперь содержит и `23596`, и `23524`.

Почему это выглядит как real false merge, а не просто duplicate miss:

- source `23524` по casebook относится к `matryoshka_exhibition_duplicate`;
- text `23524` прямо про выставку `Путешествие Матрешки`;
- `2746` — это другой museum holiday/program child event;
- после live run `23524` оказался source-ом у `2746`, а не у `2849`.

Вывод:

- это уже не “LLM оставил duplicate”;
- это live false-merge class;
- по severity это хуже `makovetsky` duplicate и должно стать главным blocker для следующего раунда.

### 3.2. `makovetsky_chekhov_duplicate`

Статус: must-merge miss сохранился.

Observed runtime:

- `https://vk.com/wall-100137391_163885` создал `event 2846`;
- `https://vk.com/wall-100137391_163915` создал `event 2847`;
- оба события:
  - `2026-07-11`
  - `00:00`
  - `Янтарь холл`
  - один и тот же poster/text lineage;
- runtime не смёржил их.

Это соответствует old casebook intuition:

- duplicate safer than false merge;
- но residual gray LLM layer пока не дотягивает этот класс до merge.

### 3.3. `oncologists_svetlogorsk_duplicate`

Статус: unresolved и, похоже, шире исходной пары.

Observed runtime:

- `https://vk.com/wall-30777579_14694` создал:
  - `2850` for `2026-03-23`
  - `2851` for `2026-03-24`
  - `2852` for `2026-03-26`
- все три с title `Бесплатные приемы детских онкологов`;
- при этом venue у них стал `Научная библиотека`, хотя source text описывает multi-city campaign post.

Потом:

- `https://vk.com/wall-211997788_2805` создал:
  - `2853` for `2026-03-23`, venue `Светлогорск`
  - `2854` for `2026-03-26`, venue `Зеленоградск`
- и обновил pre-existing `2711` for `2026-03-24`.

Потом:

- `https://vk.com/wall-151577515_24685` создал `2855`:
  - `Бесплатный приём детского онколога`
  - `2026-03-23`
  - `Детская поликлиника`, `Светлогорск`

Что это значит:

- must-merge pair `campaign post -> city-specific post` по Светлогорску не закрыт;
- на `2026-03-23` сейчас есть как минимум `2850`, `2853`, `2855`;
- здесь одновременно видны:
  - residual merge miss;
  - venue/source-name leakage (`Научная библиотека`);
  - runtime confusion между city-level и venue-level representation.

### 3.4. `hudozhnitsy_5way_cluster`

Статус: partial coverage only.

Observed runtime:

- `https://vk.com/wall-212760444_4506` создал `event 2856`;
- cluster posts `4543` и `4545` в текущей БД после прогона остались `rejected`;
- therefore cluster merge quality по `hudozhnitsy` этим live run не доказана.

Это важно:

- отсутствие false merge тут пока не означает, что класс решён;
- мы просто не дошли до полноценного runtime comparison между cluster members.

### 3.5. Дополнительный prod-signal вне этого VK live run: `little_women_cluster`

Статус: unresolved must-merge cluster, уже видимый на публичной витрине.

Это не результат именно данного локального VK reimport прогона.
Но это свежий production signal, который нельзя игнорировать перед следующим раундом с Opus.

Observed on prod `/daily`:

- `Сто семнадцатый показ киноклуба westside movieclub`
- `Маленькие женщины`
- `Маленькие женщины`

Все три карточки описывают один и тот же screening:

- `2026-03-07 19:30`
- `Сигнал`, `Космонавта Леонова 22`
- ticket link из источника: `https://vk.cc/cV8eI8`
- фильм `Little Women` / `Маленькие женщины`
- обсуждение после показа

Почему это важно:

- case уже был в casebook как `little_women_cluster`;
- теперь есть дополнительное подтверждение, что это не только historical polluted cluster;
- это user-visible duplicate family, который доходит до prod `/daily`.

Практически этот сигнал говорит о двух незакрытых проблемах:

- alias `бренд клубного показа` vs `название фильма` всё ещё плохо схлопывается;
- polluted cluster/source-ownership эффект всё ещё способен доживать до публичного выхода.

### 3.6. Дополнительный snapshot signal: `vistynets_fair_duplicate`

Статус: snapshot-confirmed duplicate, но не live blocker уровня `matryoshka`.

Observed in the current prepared DB:

- `event 2674`
  - `Ярмарка «Вкусов Виштынецкой возвышенности»`
  - `2026-03-07 12:00`
  - sources:
    - `https://vk.com/wall-211015009_857`
    - `https://t.me/tastes_of_vistynets/1261`
  - venue: `Дизайн-резиденция Gumbinnen, Ленина 29, Гусев`
  - city: `Гусев`
- `event 2784`
  - `Ярмарка «Вкусов Виштынецкой возвышенности»`
  - `2026-03-07 12:00`
  - source:
    - `https://t.me/gumbinnen/1181`
  - venue: `Дизайн-резиденция Gumbinnen`
  - city: `Калининград`

Почему это полезно:

- exact same title;
- exact same slot;
- очевидно одна и та же ярмарка в `Gumbinnen / Гусев`;
- но есть city/location noise и нет ticket-proof.

Это значит:

- кейс хорош как secondary control для `exact-title cross-source duplicate with city noise`;
- он не доказывает новый live regression сам по себе;
- но его стоит показать Opus как дополнительный unresolved family, который не должен требовать broad venue override.

### 3.7. Дополнительный prod control: `zoo_reptile_vs_generic_excursion_false_friend`

Статус: свежий prod `/daily` control на `must_not_merge`.

Observed pair:

- snapshot event `2694`
  - `Экскурсия «Тайны панциря и чешуи, или О тех, кого не любят»`
  - `2026-03-07 11:00`
  - `Калининградский зоопарк`
  - ticket `https://vk.cc/cUYxJb`
- свежая prod `/daily` карточка вне snapshot
  - `Экскурсии в зоопарке Калининграда`
  - `2026-03-07 11:00`
  - `Зоопарк`
  - ticket `https://vk.cc/cVbnez`

Почему это важно:

- по slot/venue пара выглядит dangerously similar;
- но смысл разный:
  - первая карточка про конкретную guided reptile excursion из цикла `Другой зоопарк`;
  - вторая про generic/self-guided zoo visit с картой и отдельной бесплатной регистрацией;
- registration links разные.

Это значит:

- кейс полезен не как merge target, а как свежий false-friend control;
- `same slot + same venue family + excursion wording` нельзя переоценивать;
- особенно на площадках, где одновременно живут child-excursions и generic visitor activities.

## 4. Operational Findings, которые нельзя смешивать с merge-quality

### 4.1. TPM / fallback pressure

Live run подтвердил исходное ограничение:

- compact pairwise baseline обязателен не “теоретически”, а practically;
- даже на narrowed test queue Gemma регулярно упирается в `TPM`;
- repeated deferrals заметно замедляют real validation.

Это не отменяет merge-quality findings, но влияет на выводы:

- часть casebook VK rows просто не была полноценно проверена;
- `not solved` и `not reached due to TPM` надо разделять.

### 4.2. `bad gemma parse response`

`https://vk.com/wall-212233232_1680` после defer дал:

- `Результат: ошибка извлечения событий (drafts)`
- `Причина: bad gemma parse response`

Это скорее parser/JSON robustness issue, а не direct duplicate-merge question.

### 4.3. Possible queue/idempotency anomaly

В одном batch `https://vk.com/wall-211997788_2805` был обработан дважды как шаги `14/20` и `15/20`.

При этом в текущей БД для `(group_id=211997788, post_id=2805)` есть один `vk_inbox` row.

Это выглядит как отдельная internal queue/idempotency проблема.
Её не стоит выносить на Opus как primary consultation topic, но её нельзя путать с semantic matching quality.

## 5. Что уже ясно после live run

1. Narrowed deterministic subset сам по себе не закрывает residual gray runtime classes.
2. `makovetsky` duplicate остаётся acceptable-safe miss, но всё ещё miss.
3. `oncologists` остаётся unresolved и осложняется venue leakage.
4. `matryoshka` live false merge сейчас самый важный blocker.
5. `little_women_cluster` остаётся актуальным must-merge gap и уже виден в prod `/daily`.
6. `vistynets_fair_duplicate` подтверждает ещё один cross-source must-merge class с шумом в `city/location`, но без признаков false merge.
7. `zoo_reptile_vs_generic_excursion_false_friend` добавляет свежий prod control против false merge при одинаковом zoo slot.
8. Coverage по live VK run неполная:
   - часть casebook VK rows была deferred;
   - часть была rejected;
   - часть non-VK кейсов этим методом вообще не валидируется.

## 6. Что нужно от следующего Opus-раунда

Не нужен ещё один общий redesign.

Нужен короткий консультационный pass по трём вещам:

1. Как квалифицировать и блокировать `matryoshka`-class false merge без возврата к broad heuristics.
2. Как обрабатывать `oncologists` campaign/city-specific family:
   - что должно merge;
   - что должно оставаться separate;
   - какие сигналы слишком шумные для deterministic.
3. Что делать с residual gray LLM policy после live validation:
   - `makovetsky`
   - `matryoshka`
   - `oncologists`
   - `little_women`
   - `vistynets_fair_duplicate`
   - `zoo_reptile_vs_generic_excursion_false_friend`
   - partial `hudozhnitsy`

Отдельно можно спросить короткий operational appendix:

- как уменьшить `TPM` pain без fat-shortlist и без расширения merge bias.
