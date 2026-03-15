# Source parsing (извлечение событий из источников)

Фича отвечает за извлечение и обновление событий из внешних источников: театры, отдельные сайты (например pyramida.info), спецпроекты Дом искусств, а также другие источники, которые будут добавляться.

## Точки входа

- Команда супер‑админа: `/parse` (и диагностический режим `/parse check`).
  - Точечный запуск: `/parse <source>` (например `/parse dramteatr`, `/parse tretyakov`) чтобы не гонять все источники.
- Автозапуск по расписанию: `ENABLE_SOURCE_PARSING=1` (см. `source_parsing/commands.py`).
- Из VK review UI: кнопки “Извлечь …” для ссылок на поддерживаемые источники.

## Как события попадают в БД (важно)

Все результаты `/parse` сохраняются **через Smart Update** (`smart_event_update.smart_event_update`):

- `Event.search_digest` (в UI иногда называют “короткое описание”) используется как краткий дайджест.
- `Event.description` — **полное** описание события, которое публикуется на странице события в Telegraph.
- Ежедневные анонсы (daily posts) должны показывать именно `Event.search_digest` как one‑liner, а не полный `Event.description`.
- В ежедневных анонсах заголовок события должен вести на Telegraph страницу события (а не на Telegram/VK пост-источник).
- Для каждого источника создаются записи `event_source` и “факты” в `event_source_fact`, чтобы было видно вклад каждого источника в мердж.
- Telegram Monitoring канонизирует `location_name/location_address/city` через `docs/reference/locations.md` + `docs/reference/location-aliases.md` ещё до создания `EventCandidate`, чтобы `/daily` и merge-path не расходились по написанию площадок.

### LLM-first guardrails для VK/TG parse

- Для VK/TG draft extraction сохраняется LLM-first подход: массовые смысловые решения принимаются в prompt/parser, а не детерминированным “переписыванием” текста после разбора.
- Отдельный targeted guard теперь добавляет в parse prompt узкий hint для giveaway/contest постов: если матч/концерт/другое событие упомянуто только как приз розыгрыша, parser должен вернуть `[]`, а не создавать pseudo-event.
- Downstream Smart Update дублирует это как safety-net (`skipped_giveaway`), чтобы prize-only promo пост не проходил даже при неудачном upstream parse.
- Для image-heavy intro posts (`листайте афиши`, `смотрите карточки`, weekly schedule wrapper без конкретных событий в тексте) parse prompt теперь явно разрешает вернуть `[]` как штатный результат, а не пытаться “додумать” события из обёртки.
- Gemma parse path теперь жёстче требует чистый JSON (`[]` или объект с `events`) и, если Gemma после repair всё равно отдаёт битый JSON, переключается на fallback `4o` вместо немедленного падения.
- Для VK multi-poster / schedule posts intake дополнительно схлопывает exact duplicate child drafts внутри одного parsed batch только при совпадении `date + explicit time + venue + normalized title`; это узкий safety-net против двойного извлечения одной и той же карточки из карусели/афиш.

### Каноничность сайта (/parse) при конфликтах

Источник сайта/парсера считается **каноническим** (trust high):

- если Telegram был импортирован первым, последующий `/parse` должен смержиться в тот же `event_id`;
- при противоречиях в “якорных” полях (дата/время/место) побеждает факт из `/parse` (сайт), а не Telegram.

### Source-aware дедупликация в `/parse` (важно)

Для найденного в БД события `/parse` теперь проверяет, есть ли у него источник именно этого сайта (`parser:<source>`):

- если такого parser-источника **нет**, запускается Smart Update и источник сайта добавляется в `event_source` (это не “Пропущено”);
- если parser-источник этого сайта **уже есть**, выполняется лёгкий путь (ticket/link update) без лишнего LLM-мерджа;
- `⏭️ Пропущено` в отчёте используется только для реальных skip-статусов Smart Update (например `skipped_nochange`), а не для успешного merge.

Это снижает лишнюю нагрузку на LLM и делает отчёт `/parse` честным для E2E-проверок.

### Очередь обновления month/weekend страниц

- Для созданных/обновлённых событий `/parse` использует общий `schedule_event_update_tasks` (как и VK/TG), где `month_pages`/`weekend_pages` ставятся как debounce-задачи с `next_run_at = now + 15 минут`.
- В финальном safeguard `_process_parsing_files` гарантирует постановку задач по затронутым месяцам и выходным, и тоже ставит их отложенно (`+15 минут`), чтобы не было немедленной пересборки Telegraph-страниц после массового прогона.

### Расписание автозапуска

- `ENABLE_SOURCE_PARSING=1` — включить ежедневный запуск.
- `SOURCE_PARSING_TIME_LOCAL=04:30` — локальное время запуска (HH:MM).
- `SOURCE_PARSING_TZ=Europe/Kaliningrad` — таймзона для локального времени.
- `ENABLE_SOURCE_PARSING_DAY=1` — включить дневной запуск.
- `SOURCE_PARSING_DAY_TIME_LOCAL=14:15` — локальное время дневного запуска (HH:MM).
- `SOURCE_PARSING_DAY_TZ=Europe/Kaliningrad` — таймзона дневного запуска.

Если значения не заданы, используется 04:30 по Europe/Kaliningrad. Дневной запуск пропускает Kaggle, если страницы источников не изменились с последнего успешного прогона.

## Документация по источникам

- Театры (/parse): `docs/features/source-parsing/sources/theatres/README.md`
- Дом искусств: `docs/features/source-parsing/sources/dom-iskusstv/README.md`
- Pyramida: `docs/features/source-parsing/sources/pyramida/README.md`
- Третьяковка: `docs/features/source-parsing/sources/tretyakov/README.md`
- Филармония: `docs/features/source-parsing/sources/philharmonia/README.md`
- Qtickets: `docs/features/source-parsing/sources/qtickets/README.md`
- Universal Festival Parser: `docs/features/source-parsing/sources/festival-parser/README.md`
- Каноника по фестивальным сериям/выпускам и очереди: `docs/features/festivals/README.md`

## Артефакты

Все выгрузки/логи/результаты прогонов хранить в `artifacts/` (см. `artifacts/README.md`).

## Задачи

Связанные backlog items/планы/отчёты — в `docs/features/source-parsing/tasks/README.md` (без копирования контента).
