# Qtickets (Kaliningrad): парсер событий

> **Linear:** EVE-56  
> **Статус:** implemented  
> **Цель:** добавить Qtickets как регулярный источник в `source_parsing` (как theatres/philharmonia), с нормальной дедупликацией и обновлениями.

## Обзор

Qtickets для Калининграда: `https://kaliningrad.qtickets.events/`.

## Regression Note: March 15, 2026

- Live reproduction on **March 15, 2026** confirmed that the current Playwright parser still finds **40** event cards on `kaliningrad.qtickets.events`, and all parsed items keep a valid `parsed_date`.
- The fragile point was downstream draft extraction: some Qtickets cards have sparse descriptions, so `/parse` now passes explicit structured facts (`date`, `time`, `venue`, `ticket status`, `price`, `url`) to the LLM together with the raw description.
- Qtickets currently runs via its own Kaggle kernel `ParseQtickets` / `zigomaro/parse-qtickets` and is not bundled into `ParseTheatres`.

Парсер должен извлекать события (и, если есть, расписание по датам/времени) и прогонять их через стандартный пайплайн `source_parsing`:

- нормализация/обогащение через LLM (единый формат, `search_digest`, `event_type`, `topics`);
- сохранение в БД через `upsert_event`;
- повторные прогоны не создают дубликатов, а корректно обновляют статус билетов/недостающие поля.

## Как это должно встраиваться (как у других парсеров)

### Запуск

Рекомендуемый режим для Qtickets — **регулярный запуск** вместе с `/parse` и автозапуском `ENABLE_SOURCE_PARSING=1`.

Дополнительно (опционально): поддержать точечный запуск по ссылке из VK review, если Qtickets URL встречается в постах.

### Компоненты

- Kaggle kernel: `kaggle/ParseQtickets/` (Playwright/requests; зависит от того, SPA ли сайт и есть ли API).
- Python модуль: `source_parsing/qtickets.py` (запуск Kaggle + разбор JSON).
- Интеграция в общий раннер: `source_parsing/handlers.py` + `source_parsing/commands.py` (добавить новый `source` + (опционально) `/parse check`).
- Тесты: `tests/test_qtickets.py` (юнит на разбор и нормализацию, плюс пара регрессий на дедуп).

## Контракт данных: что отдаёт парсер

`source_parsing` сейчас работает с моделью `TheatreEvent` (`source_parsing/parser.py`). Для Qtickets придерживаемся её (или совместимого JSON).

Текущий рабочий JSON-контракт `qtickets_events.json`:

- `date_raw`
- `parsed_date`
- `parsed_time`
- `location`
- `photos[]`
- `ticket_price_min`
- `ticket_price_max`
- `ticket_status`

Backend parser держит backward compatibility со старым форматом (`date/time/image_url/price_min/price_max`), но новым notebook-выгрузкам нужно придерживаться именно этого контракта.

## Особенности источника (Qtickets)

1) Сайт использует **динамическую подгрузку списка** событий при прокрутке (infinite scroll).
2) Поэтому базовый “технический план” парсинга:
   - Playwright (headless) открывает главную страницу;
   - скроллит вниз и ждёт подгрузки, пока новые карточки перестанут появляться;
   - фиксирует “полный список” карточек и дальше ходит в детали.

Важно: это не “опция”, а обязательное требование для корректного покрытия “все найденные события”.

### Обязательные поля (`TheatreEvent`)

| Поле | Тип | Зачем нужно |
|---|---|---|
| `title` | `str` | LLM-нормализация, дедуп по названию |
| `url` | `str` | `ticket_link`/`source_post_url`, якорь источника (см. ниже) |
| `ticket_status` | `available|sold_out|unknown` | обновления статуса без пересоздания |
| `location` | `str` | маппинг в `location_name`, дедуп по площадке |
| `date_raw` | `str` | базовый парсинг даты/времени (`parse_date_raw`) |

### Рекомендуемые поля (для качества и стабильности)

| Поле | Тип | Зачем нужно |
|---|---|---|
| `parsed_date` / `parsed_time` | `YYYY-MM-DD` / `HH:MM` | лучше, чем парсить `date_raw`; снижает ошибки |
| `description` | `str` | исходник для LLM, влияет на `search_digest`/описание |
| `photos[]` | `list[str]` | афиши для `EventPoster`/OCR и дальнейшего Smart Update |
| `ticket_price_min/max` | `int` | диапазон цен |
| `age_restriction` | `str` | добавляется в описание/карточку |
| `scene` | `str` | зал/сцена (входит в `normalize_location_name`) |
| `source_type` | `str` | метка источника (предлагается: `qtickets`) |

### Правило “одно событие = одна дата+время”

Если у события на Qtickets есть несколько сеансов (разные даты/время), то **каждый сеанс — отдельный `TheatreEvent`**.
Это соответствует текущей логике `source_parsing` и упрощает корректные обновления `ticket_status`.

## Дедуп и обновления (как сейчас работает `source_parsing`)

### Поиск существующего события

Используется `find_existing_event(...)` (`source_parsing/parser.py`) — матчинг по:

1) `date + time` (в первую очередь),  
2) затем `location` + fuzzy-match `title`,  
3) отдельные правила для placeholder-времени (`00:00`) и “полного апдейта”.

### Что обновляем при совпадении

Ориентир: поведение `/parse` (см. `docs/features/source-parsing/sources/theatres/README.md`):

- минимально: `ticket_status` + `ticket_link`;
- при `needs_full_update`: добиваем недостающие поля (время, описание, фото, пушкинская карта, цены).

## Увязка с будущим Smart Update (EVE-60)

Smart Update должен стать единой точкой мерджа для всех источников. Чтобы Qtickets “встал” без боли:

1) **Не терять якорь источника**: `url` (страница Qtickets) всегда сохранять в `source_post_url` (а позже — в `event_source`).
2) **Постеры как first-class данные**: `photos[]` максимально полные; важно для `poster_hash` и дедупа по изображениям.
3) **Не менять якорные поля автоматически** (контракт Smart Update): `date/time/location_name` — только если поле было пустым или это “placeholder”.
4) Готовить данные так, чтобы их можно было преобразовать в `EventCandidate` без потерь:
   - `source_type='site'`, `source_url=url`,
   - `ticket_link=url` (текущий контракт `source_parsing`: одна ссылка, обычно = `url`),
   - `posters[]` (скачанные/приведённые к Catbox при импорте),
   - `raw_text`/`html` (если есть) — для LLM merge.

## План разработки и проверки

1) Анализ сайта (Playwright): понять структуру списка/деталей и где лежит “истина” по расписанию (сеансы/даты/время).
2) Локальный прототип: собрать “все события” с `https://kaliningrad.qtickets.events/` с учётом infinite scroll.
3) Kaggle kernel: формировать JSON в формате `TheatreEvent[]`.
4) Интеграция в `source_parsing`: прогнать через LLM (`build_event_drafts_from_vk`) и сохранить.
5) Регрессия на дедуп: повторный прогон не должен создавать новые записи.
6) Тестовый E2E: минимум 2 события, проверка в БД:
   - `search_digest` заполнен (при создании через LLM),
   - `ticket_link/source_post_url` указывает на Qtickets,
   - `ticket_status` обновляется при изменениях.

## Принятые решения (по требованиям)

- **Что парсим:** все события, которые удаётся обнаружить на `https://kaliningrad.qtickets.events/` (после полной подгрузки списка).
- **Horizon:** “всё найденное на сайте” (без искусственного ограничения по датам на уровне парсера).
- **`ticket_link` / `url`:** ссылка на карточку события вида `https://kaliningrad.qtickets.events/210564-1402-valentin-2000`.

## Open Questions (нужно уточнить перед реализацией)

1) Где “истина” по расписанию: на странице события есть все даты/сеансы или нужен отдельный виджет/эндпоинт (уточняется на этапе Playwright-анализа)?
