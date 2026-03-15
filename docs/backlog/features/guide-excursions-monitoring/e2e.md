# Guide Excursions E2E Plan

> **Status:** Planned / backlog  
> **Intent:** канонический live E2E сценарный пакет для будущей реализации guide excursions monitoring.

Связанные документы:

- feature overview: `docs/backlog/features/guide-excursions-monitoring/README.md`
- MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- digest spec: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- eval pack: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- E2E guide: `docs/operations/e2e-testing.md`
- E2E index: `docs/operations/e2e-scenarios.md`

## 1. Принципы operator UX для E2E

Пользовательский сценарий должен быть таким, чтобы оператор не вспоминал длинные команды и параметры.

Поэтому для MVP канонический flow должен строиться вокруг одной команды и понятных кнопок:

- `/guide_excursions`
- `Run light scan`
- `Run full scan`
- `Recent findings`
- `Preview new digest`
- `Preview last call`
- `Publish to @keniggpt`
- `Send test report`
- `Sources`
- `Stats`

## 2. Что именно должен покрывать live E2E

Обязательные проверки:

- фича находится через простую команду;
- scan запускается из UI без параметров;
- бот показывает понятный progress/report;
- preview digest строится после scan;
- publish работает в `@keniggpt`;
- `on_request` / `private-group-only` предложения не попадают в public digest;
- media group формируется из source media;
- ручной test report можно отправить независимо от наличия digest items.
- результаты frozen cases можно сравнить с `eval-pack.md`.

## 3. Предусловия

- live bot запущен локально в polling режиме;
- test user Telethon авторизован;
- test user добавлен админом в `@keniggpt`;
- guide monitoring feature включена;
- seed sources уже заведены в БД;
- для сценариев с media в мониторинге есть хотя бы один source post с фото или видео.

## 4. Канонический manual flow

1. Оператор открывает чат с ботом.
2. Отправляет `/guide_excursions`.
3. Нажимает `Run full scan`.
4. Дожидается финального operator report.
5. Нажимает `Preview new digest`.
6. Проверяет preview bundle.
7. Нажимает `Publish to @keniggpt`.
8. Проверяет, что digest появился в целевом канале.

Отдельный smoke flow:

1. Оператор открывает `/guide_excursions`.
2. Нажимает `Send test report`.
3. Бот отправляет тестовый report с текущим статусом guide monitoring.

## 5. Gherkin сценарии

Ниже не “идея”, а канонический draft будущего `.feature` файла.

```gherkin
# language: ru
Функция: Мониторинг экскурсий гидов

  Предыстория:
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом

  Сценарий: Главное меню мониторинга экскурсий открывается без сложных параметров
    Когда я отправляю команду "/guide_excursions"
    Тогда я жду сообщения с текстом "Мониторинг экскурсий"
    И под сообщением должны быть кнопки: "Run light scan, Run full scan, Recent findings, Preview new digest, Preview last call, Publish to @keniggpt, Send test report, Sources, Stats"

  Сценарий: Оператор вручную запускает полный мониторинг экскурсий
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    Тогда я жду сообщения с текстом "Guide monitoring started"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я жду сообщения с текстом "sources_scanned"
    И я жду сообщения с текстом "occurrences_created"

  Сценарий: Оператор получает ручной тестовый отчёт без запуска дайджеста
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Send test report"
    Тогда я жду сообщения с текстом "Guide monitoring test report"
    И я жду сообщения с текстом "sources_total"
    И я жду сообщения с текстом "last_full_scan"

  Сценарий: После полного скана можно собрать preview нового дайджеста
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Preview new digest"
    Тогда я жду сообщения с текстом "Guide digest preview"
    И я жду сообщения с текстом "Новые экскурсии гидов"

  Сценарий: Дайджест публикуется в тестовый канал keniggpt
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Preview new digest"
    И я жду сообщения с текстом "Guide digest preview"
    И я нажимаю инлайн-кнопку "Publish to @keniggpt"
    Тогда я жду сообщения с текстом "Guide digest published"
    И в канале "@keniggpt" должен появиться новый digest "Новые экскурсии гидов"

  Сценарий: Публичный дайджест не включает on-demand предложения
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Preview new digest"
    Тогда сообщение preview не должно содержать "по запросу"
    И сообщение preview не должно содержать "только для организованных групп"

  Сценарий: Recent findings показывает on-demand результаты как internal findings
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Recent findings"
    Тогда я жду сообщения с текстом "on_demand"
    И я жду сообщения с текстом "template"

  Сценарий: Preview нового дайджеста отправляется как media group + text bundle
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Preview new digest"
    Тогда бот должен отправить альбом медиа для guide digest
    И после альбома бот должен отправить текстовый digest

  Сценарий: В preview bundle могут использоваться исходные фото и видео из source posts
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run full scan"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Preview new digest"
    Тогда в альбоме guide digest должно быть хотя бы одно фото или видео
    И preview report должен содержать "media_bridge=forward_file_id"

  Сценарий: Light scan даёт оперативный digest last call
    Когда я отправляю команду "/guide_excursions"
    И я нажимаю инлайн-кнопку "Run light scan"
    Тогда я жду сообщения с текстом "Guide monitoring started"
    И я жду долгой операции с текстом "Guide monitoring finished"
    И я нажимаю инлайн-кнопку "Preview last call"
    Тогда я жду сообщения с текстом "Guide digest preview"
```

## 6. Какие step definitions понадобятся дополнительно

С высокой вероятностью потребуются новые generic steps:

- `в канале "<username>" должен появиться новый digest "<title>"`
- `бот должен отправить альбом медиа для guide digest`
- `после альбома бот должен отправить текстовый digest`
- `preview report должен содержать "<text>"`
- `сообщение preview не должно содержать "<text>"`

Все эти шаги достаточно общие, чтобы потом переиспользовать их и в других media-digest фичах.

## 7. Что важно проверить в Telegram UI, а не только в behave

При live E2E нужно смотреть не только на pass/fail шагов, но и на operator messages:

- есть ли понятный start/finish report;
- есть ли breakdown `sources_scanned`, `posts_prefiltered`, `llm_checked`;
- видит ли оператор, что digest пустой из-за `digest_eligible=no`, а не из-за ошибки;
- видит ли оператор, что media fallback сработал или не сработал;
- приходит ли `Send test report` даже при пустом результате.

## 8. Что считать успешным MVP E2E

MVP можно считать готовым к ручному использованию, если live E2E подтверждает:

- оператор запускает scan одной кнопкой;
- digest preview и publish доступны без параметров;
- `@keniggpt` получает digest bundle;
- `on_request` офферы не утекли в public digest;
- media group и text digest приходят как связанная пара;
- есть отдельный ручной test report путь.
