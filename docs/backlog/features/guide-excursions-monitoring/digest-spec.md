# Guide Excursions Digest Spec

> **Status:** Implementation-ready digest and delivery spec  
> **Scope:** UX, versioned component naming, temporary media scheme, Telegram output constraints.

Примечание: спека синхронизирована с live `Opus` audit из `2026-03-14`; итоговая рекомендация для MVP — не fully deterministic digest, а deterministic shell + batched LLM copy поверх fact pack.

Связанные документы:

- feature overview: `docs/backlog/features/guide-excursions-monitoring/README.md`
- architecture: `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- LLM-first pack: `docs/backlog/features/guide-excursions-monitoring/llm-first.md`
- eval pack: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- casebook: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- existing digest baseline: `docs/features/digests/README.md`
- live `Opus` audit artifact: `artifacts/codex/reports/guide-excursions-monitoring-opus-audit-2026-03-14.md`

## 1. Что было не доработано в MVP до этой спеки

До этой спеки в MVP уже были проработаны:

- source monitoring;
- facts-first entity model;
- source taxonomy;
- digest families;
- scheduler/admin surfaces;
- general stats integration.

Но не были до конца формализованы:

- versioned naming для всех ключевых компонентов;
- Telegram UX самого digest bundle;
- временная схема работы с картинками и видео без постоянного storage;
- чёткий contract, как `on_request` предложения отсекаются от public digest;
- E2E manual flow с понятными действиями оператора.

Эта спека закрывает именно эти пробелы.

## 2. Версионируемые компоненты MVP

Для будущего versioning у компонентов должны быть короткие и устойчивые имена.

### `Trail Scout v1`

Назначение:

- fetch Telegram posts;
- grouped albums;
- OCR;
- prefilter;
- post taxonomy.

Contract:

- source-in → candidate JSON out.

Operational note:

- `Trail Scout v1` — это отдельный Kaggle notebook, максимально переиспользующий текущий `TelegramMonitor` runtime skeleton; bot runtime не сканирует исходные Telegram-каналы сам.

### `Route Weaver v1`

Назначение:

- guide/template/occurrence match;
- fact merge;
- status reconcile;
- digest eligibility decision.

Contract:

- candidate-in → canonical rows + fact claims + digest deltas out.

### `Lollipop Trails v1`

Назначение:

- surface-specific copy from fact pack batch-режимом.

MVP scope:

- `title`
- `digest_blurb`
- deterministic shell keeps `guide/date/audience/meeting-point/price/seats/booking` outside the writer call

### `Trail Digest v1`

Назначение:

- ranking;
- grouping;
- digest text rendering;
- preview/publish pipeline.

Contract:

- digest candidates in → preview/publish bundle out.

### `Media Bridge v1`

Временный media-delivery слой до появления постоянной infra на Yandex Cloud.

Назначение:

- взять media refs из исходных Telegram-постов;
- выбрать и дедуплицировать нужные фото/видео;
- превратить их в bot-reusable media через `forward -> file_id`;
- отправить медиагруппу в целевой digest без отдельного permanent hosting.

### `Guide Atlas v1`

Назначение:

- operator read-model;
- `/guide_excursions`
- `/guide_recent`
- `/guide_sources`
- `/guide_digest`
- `/general_stats`.

### `Guide E2E Pack v1`

Назначение:

- канонический manual E2E flow;
- Gherkin сценарии;
- operator-friendly test path без сложных команд и параметров.

## 3. Вывод по полноте MVP

После добавления `Media Bridge v1` и `Guide E2E Pack v1` MVP уже покрывает все необходимые контуры для реализации первой рабочей версии:

- intake;
- facts-first merge;
- short-form text generation;
- digest building;
- temporary media reuse;
- admin UX;
- scheduler/reporting;
- manual E2E path.

Что сознательно остаётся вне MVP:

- постоянное object storage для digest media;
- public static pages;
- generated preview image по модели `/3di`;
- rich long-form template/guide pages.

### 3.1. Text policy for MVP

MVP не должен превращаться ни в fully deterministic шаблонизатор, ни в дорогой `LLM per card`.

Каноника:

- `Trail Digest v1` собирает layout, ordering, numbering и omission policy deterministic-слоем;
- `Lollipop Trails v1` пишет только `title + digest_blurb` batch-режимом из fact pack;
- длина `digest_blurb` выбирается по плотности фактов, а не по объёму raw source text;
- если LLM path временно недоступен, digest может уйти с deterministic fallback, но это аварийный режим, а не целевой дизайн.

## 4. Telegram UX ограничения, которые определяют дизайн

Нельзя проектировать digest абстрактно. Есть жёсткие Telegram-ограничения:

- media group: до `10` media items;
- caption у альбома фактически одна и ограничена `1024` символами;
- длинный digest-текст в caption быстро становится нечитаемым;
- смешанные photo/video albums возможны, но их нужно ограничивать по размеру и длине.

Из этого следует базовое решение:

- digest остаётся **text-first**;
- media group является **визуальным companion bundle**, а не носителем всего текста;
- при длинном списке digest отправляется как **2-message bundle**:
  - `message A`: media group with short header;
  - `message B`: full structured text digest.

### 4.1. Hard split rules for Telegram

Чтобы не было белого пятна в реализации:

- максимум `8` карточек в одном text message;
- максимум `5` occurrence media items в одном media companion bundle;
- если digest text превышает `4096` символов или `8` карточек, отправляется continuation message;
- continuation header обязателен:
  - `Продолжение дайджеста экскурсий (2/2)`
  - или `Продолжение last call (2/2)`;
- media group прикрепляется только к первой части выпуска;
- numbering карточек сквозное: `1..N`, без перезапуска в continuation.

## 5. Рекомендуемый шаблон digest bundle

### 5.1. `new_occurrences`

#### Mode A: compact album

Используется, если:

- `items <= 4`
- полный текст помещается в caption
- media set компактный и понятный.

Формат:

- media group;
- caption на первом media item;
- отдельное короткое action message не требуется.

#### Mode B: split bundle

Используется по умолчанию для MVP, если:

- `items > 4`
- caption длиннее `1024`
- full text длиннее `4096`
- есть mix photo/video
- есть хотя бы один item без media.

Формат:

1. media group с короткой caption:
   - заголовок family;
   - 1 короткая вводная строка;
   - legend вида `В альбоме карточки 1-4`.
2. следом отдельное text message:
   - полный digest.

### 5.2. Recommended text layout for `new_occurrences`

Заголовок:

```text
Новые экскурсии гидов: 4 находки на ближайшие дни
```

Вводная:

```text
Собрали новые публичные экскурсии, которые появились в мониторинге сегодня. В альбоме показаны карточки 1-4.
```

Карточка:

```text
1. ❤️ Город К. Женщины, которые вдохновляют
Татьяна Удовенко + Юлия Гришанова
🗓 Сб, 16 марта, 11:00
👥 Для кого: взрослым, местным и туристам, кто любит городской сторителлинг
🧭 О чём: прогулка по женским сюжетам города и его скрытым биографиям
📍 Встреча: у Матери России
💸 2000 ₽
🎟 Места: осталось 3
✍️ Запись: @username
📣 Канал: @guide_channel
```

Правила:

- `Для кого` выводим только если он короткий и уверенный;
- `О чём` всегда одно предложение;
- каждая карточка отделяется одной пустой строкой;
- numbers `1..N` каноничны и используются в legend для media group.

### 5.2.a. Digest design system

Если кратко: digest должен читаться как **редакторская карта решений**, а не как бесконечный анонсный текст.

Иерархия внутри карточки всегда такая:

1. что это за маршрут;
2. кто ведёт;
3. когда;
4. для кого;
5. чем интересен;
6. где встреча;
7. сколько стоит;
8. как записаться.

Практические правила дизайна:

- первая строка должна отвечать на вопрос `что выбрать`;
- вторая строка — `кто проводит`;
- logistics lines должны быть короткими и однотипными, чтобы глаз сканировал выпуск быстро;
- `Для кого` ставим выше `О чём`, потому что для этой фичи fit важнее редакторской красоты;
- если поля нет, строку не показываем вообще;
- не использовать более одного visual marker на карточку:
  - `❤️` только если likes выше медианы своего окна;
  - `⚠️` только для last-call/status family.

Рекомендуемый набор маркеров:

- `🗓` дата/время
- `👥` для кого
- `🧭` о чём
- `📍` встреча
- `💸` цена
- `🎟` места/статус
- `✍️` запись
- `📣` канал

### 5.2.b. Omission policy

Digest не должен выдумывать недостающие поля.

Если факта нет:

- не писать строку вовсе;
- `Цена уточняется` допустимо только если в facts есть явный signal `уточняется`;
- `Места ограничены` допустимо только если это прямо grounded в source.

Дополнительное publication rule:

- occurrences с датой дальше чем `today + 30 days` не публикуются в public digest family;
- они остаются видимыми в admin surfaces и могут попасть в будущие long-horizon digest families.

### 5.2.c. Reference split-bundle example from real casebook posts

Media caption:

```text
Новые экскурсии гидов
3 свежие находки из мониторинга. В альбоме карточки 1-3, ниже полный разбор.
```

Full text:

```text
1. ❤️ Город К. Женщины, которые вдохновляют
Татьяна Удовенко + Юлия Гришанова
🗓 7 марта, 11:00
👥 Для кого: взрослым, местным и туристам, кому интересны городские истории
🧭 О чём: прогулка по женским сюжетам проспекта Мира и скрытым биографиям города
📍 Встреча: у Матери России
💸 2000 ₽
🎟 Места: количество мест ограничено
✍️ Запись: @Yulia_Grishanova
📣 Канал: @tanja_from_koenigsberg

2. У Тани на районе: Закхайм и окрестности
Татьяна Удовенко
🗓 14 марта, 11:00
🧭 О чём: маршрут по Закхайму и соседним городским слоям без туристического глянца
💸 2000 ₽
✍️ Запись: @reiseleiterin_tanja
📣 Канал: @tanja_from_koenigsberg

3. Путешествие на ферму осетра и улиток
Профи-тур
🗓 15 марта, 11:00
👥 Для кого: взрослым и компаниям, которым интересны гастрономические выезды
🧭 О чём: поездка на ферму с рассказом о разведении осетров и дегустацией
📍 Выезд: от Дома Советов
💸 2500 ₽
🎟 Места: количество мест ограничено
📣 Канал: @excursions_profitour
```

Это не “эталон формулировок пословно”, а эталон иерархии, плотности и читаемости.

### 5.3. Recommended text layout for `last_call`

Заголовок:

```text
Экскурсии гидов: мало мест и важные апдейты
```

Карточка:

```text
1. ⚠️ Расширенная экскурсия по Зеленоградску
Наталья Котова
🗓 12 марта, 12:00
🎟 Статус: появилось одно свободное место
📍 Встреча: у супермаркета «Спар», ул. Тургенева, 1Б
⏳ Формат: прогулка 4+ часа с перерывом на обед
✍️ Запись: @gid_zelenogradsk_kotova_natalia
📣 Канал: @gid_zelenogradsk
```

Правила:

- компактнее, чем `new_occurrences`;
- главный акцент на `status delta`;
- `Для кого` особенно полезен для operator/group programs.

### 5.3.a. Reference `last_call` digest example

```text
Экскурсии гидов: мало мест и важные апдейты

1. ⚠️ Расширенная экскурсия по Зеленоградску
Наталья Котова
🗓 12 марта, 12:00
🎟 Статус: появилось одно свободное место
📍 Встреча: у супермаркета «Спар», ул. Тургенева, 1Б
✍️ Запись: @gid_zelenogradsk_kotova_natalia
📣 Канал: @gid_zelenogradsk
```

Важно:

- `last_call` не должен использовать template-only или group-only operator offers как public status cards;
- для family `last_call` status delta важнее красивого summary.

## 6. Media Bridge v1: временная схема работы с картинками и видео

### 6.1. Принцип

До появления постоянной infra картинки и видео **не нужно грузить в отдельный storage** только ради digest.

Временная схема:

1. `Trail Scout v1` в Kaggle сохраняет **media references** и выбранный candidate media per occurrence;
2. серверный бот по этим refs делает temporary relay:
   - `forwardMessage` выбранного source message в helper/admin chat;
   - из returned `Message` извлекает `photo.file_id` / `video.file_id`;
   - staging relay-message сразу можно удалить;
3. в БД/сессии хранится уже `bot_file_id`, а не доступ к user-session;
4. `Trail Digest v1` собирает новый custom media group через Bot API;
5. после публикации Telegram сам хранит media в целевом канале.

Ключевой принцип: **Telethon не нужен на стороне бота**.  
Единственная user-session живёт в Kaggle / `Trail Scout v1`.

### 6.1.a. Почему именно `forward -> file_id`

Для сборки **нового** digest album из media разных источников нужен bot-reusable media identifier.

Практический MVP-путь:

- `forwardMessage` возвращает sent `Message`;
- из него можно извлечь `file_id`;
- затем `sendMediaGroup` может собрать новый album уже из этих `file_id`.

Это лучше, чем чистый `copyMessages`-relay, потому что `copyMessages` удобен для копирования существующих сообщений, но не создаёт из произвольных source items новый curated album.

### 6.1.b. Ограничения и fallback

Если `forwardMessage` не сработал:

- protected content;
- источник недоступен для bot-forward;
- media type не подходит;

тогда fallback такой:

1. Kaggle подготавливает bot-visible staging media;
2. сохраняется `bot_file_id` или staging ref;
3. если и это не удалось, digest уходит как text-only.

### 6.2. Что нужно сохранять в данных

Минимальный media ref:

- `source_username`
- `message_id`
- `grouped_id`
- `media_kind`
  - `photo`
  - `video`
- `media_index`
- `sha256` или другой stable content hash
- `image_phash` для фото
- `mime_type`
- `duration_sec` для видео
- `width/height`
- `ocr_text` при наличии
- `bridge_status`
  - `pending`
  - `forward_file_id`
  - `staged_file_id`
  - `failed`
- `bot_file_id` при наличии
- `staging_chat_id` / `staging_message_id` при наличии
- `bridge_error` при наличии

Этого достаточно, чтобы не держать Telethon в боте и всё равно переиспользовать source media без постоянной external hosting схемы.

### 6.3. Как выбирать media для digest

Правило MVP:

- максимум `1` media asset на occurrence;
- prefer order:
  1. релевантное фото;
  2. короткое видео;
  3. без media.

Дополнительные правила:

- дедуп по `sha256` и `image_phash`;
- не брать одно и то же фото дважды в один выпуск;
- длинные ролики лучше не использовать;
- `last_call` по умолчанию предпочитает фото, а не видео.

### 6.4. Recommended bridge mode for MVP

Рекомендуемый режим:

- helper/admin chat, доступный боту;
- bot делает `forwardMessage` выбранного source media;
- из forwarded `Message` достаёт `file_id`;
- forwarded relay-message удаляется;
- в БД/сессии сохраняется `bot_file_id`;
- все bridge operations логируются;
- daily cleanup sweep дочищает relay chat от застрявших сообщений;
- publish path использует только Bot API и не зависит от Telethon.

Это лучший временный вариант для curated digest, потому что позволяет собрать новый album из media разных источников.

### 6.4.a. Почему это согласуется с уже существующими паттернами проекта

В проекте уже есть реальный bot-pattern:

- форвард сообщения в admin chat;
- чтение returned message;
- удаление forwarded relay-message.

То есть такой operational стиль уже не чужд текущему коду.

### 6.5. Как публиковать media group

Пайплайн публикации:

1. `Trail Digest v1` строит `media_manifest`;
2. `Media Bridge v1` читает `bot_file_id`;
3. bot sends `sendMediaGroup`:
   - `InputMediaPhoto` для фото;
   - `InputMediaVideo` для видео;
4. caption у первого элемента содержит только short header;
5. full digest text отправляется отдельным сообщением, если нужен split mode.

Fallback:

- если `bot_file_id` нет, но есть staged fallback, используем staged ref;
- если и это не сработало, digest уходит как text-only.

### 6.6. Temporary cache

Рекомендуемая схема:

- helper/admin chat для temporary relay;
- `bot_file_id` как основной bridge artifact;
- optional staging helper path только как fallback;
- best-effort cleanup relay-messages сразу после extraction.

Это временный operational cache, а не боевое хранилище.

### 6.7. Failure policy

Если media не удалось скачать или отправить:

- digest всё равно должен уходить как text-only;
- инцидент попадает в operator report;
- occurrence не теряется из-за media failure.

## 7. Видео в MVP

Видео поддерживаем, но консервативно.

Правила:

- максимум `2` видео на один digest bundle;
- video only if:
  - ролик короткий;
  - размер безопасен для sendMediaGroup;
  - ролик реально помогает понять экскурсию;
- если есть сомнения, берём фото.

Видео не должны ломать скорость и надёжность публикации.

## 8. Future path after MVP

### После появления Yandex Cloud storage

`Media Bridge v1` можно заменить на persistent media pipeline:

- upload canonical digest assets;
- stable URLs;
- reuse across pages, stories and digests.

### После появления preview generation

Следующий шаг после media group:

- отказаться от raw media group;
- перейти к generated preview по принципу `/3di`;
- raw source media оставить как source material, а не финальную выдачу.

Эту миграцию стоит планировать как:

- `Media Bridge v1`
- `Digest Preview Generator v2`

## 9. Operator UX для MVP

Главное меню `/guide_excursions` должно использовать простые и запоминаемые действия:

- `Run light scan`
- `Run full scan`
- `Recent findings`
- `Preview new digest`
- `Preview last call`
- `Publish to @keniggpt`
- `Send test report`
- `Sources`
- `Stats`

`Send test report` нужен отдельно:

- для ручного smoke пути;
- для E2E;
- для быстрой проверки, что feature live even when digest is empty.

## 10. Итоговая рекомендация по реализации digest

Для MVP самый прагматичный вариант такой:

- делать text-first digest;
- публиковать media group как visual companion;
- media готовить через `forward -> file_id`, а Kaggle staging держать как fallback;
- не городить отдельное storage до Yandex Cloud этапа;
- `on_request` программы не публиковать автоматически;
- `Для кого` сделать коротким, но first-class полем в тексте и фактах.
