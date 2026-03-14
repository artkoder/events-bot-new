# Event Static Pages (собственные статические страницы событий)

> **Status:** Not implemented / Research backlog  
> **Goal:** уйти от Telegraph как основной public-surface событий к собственным статическим HTML-страницам на домене проекта с мобильным дизайном, SEO/GEO и нормальной индексацией для поисковиков и AI search.

## Зачем нужен переход

Текущие Telegraph-страницы решают задачу “быстро показать карточку события в Telegram”, но слабо подходят для следующего этапа продукта:

- дизайн почти не контролируется и плохо масштабируется на desktop + mobile;
- нет полноценного контроля над `robots`, `canonical`, `sitemap.xml`, `lastmod`, внутренней перелинковкой и GEO-структурой;
- нет нормальной surface-модели для других мессенджеров, включая MAX;
- сложно делать страницу “первоисточником” с явными структурированными фактами;
- нельзя полноценно строить SEO/AI-discovery вокруг собственного домена;
- operational-контур Telegraph остаётся хрупким из-за preview/cache особенностей Telegram.

Целевое направление: **публичный статический HTML на собственном домене** с данными из основной БД, где событие имеет постоянный URL и страницу можно одинаково удобно открыть из Telegram, MAX, поисковика и обычного браузера.

## Product target

### Базовая формула

`static HTML + permanent event URL + JSON-LD Event + open indexing + up-to-date facts`

### Обязательные требования

- у каждой сущности события должна быть отдельная постоянная URL-страница: `/event/<slug>`;
- для повторяющихся событий предпочтительна модель “отдельный `Event` на каждую дату” + опциональный `EventSeries`;
- страницы должны быть mobile-first, но без деградации desktop-версии;
- страница должна быть полезной сама по себе, а не тонким дублем Telegram/VK/билетного сайта;
- основной контент должен отдаваться в HTML, а не быть спрятанным в JS;
- у страницы должны быть корректные `title`, `description`, `canonical`, `lastmod`, Open Graph и внутренняя перелинковка;
- сайт должен отдавать `sitemap.xml` и позволять нормальную индексацию через `robots.txt`;
- для события должна строиться JSON-LD разметка `schema.org/Event`, согласованная с видимым контентом;
- на странице должны быть явно извлекаемые факты: дата, timezone, адрес, город, площадка, организатор, цена/бесплатно, язык, возрастной рейтинг, online/offline, ссылка на оригинал, дата последнего обновления;
- AI-краулеры не должны блокироваться по умолчанию, если цель включает видимость в AI-search;
- апдейты события должны быстро переезжать на страницу: отмена, перенос, sold out, смена площадки, новая ссылка на билет.

### Минимальный JSON-LD contract

На уровне backlog фиксируем минимум для `Event` page:

- `name`
- `startDate`
- `location` и/или `eventAttendanceMode`
- `description`
- `image`
- `organizer`
- `offers` или явная ticket/signup ссылка
- `eventStatus`

## Что должно появиться в surface model

Первая обязательная сущность публичного сайта:

- `EventPage`
  - одна страница на конкретное событие / конкретную дату;
  - канонический URL: `/event/<slug>`;
  - при повторяемости событие не “перезаписывается в одну карточку”, а живёт отдельными occurrence pages.

Поверх неё позже можно строить:

- страницы города;
- страницы даты/месяца;
- страницы категории;
- страницы площадки;
- страницы серии событий;
- спец-треки вроде экскурсий.

Важно: отдельные listing/index pages полезны для discoverability, но **не заменяют** страницу конкретного события.

## Архитектурная гипотеза

### Infra direction

Целевая гипотеза на текущий момент:

- статические HTML-артефакты собираются вне runtime Telegram-бота;
- публикация идёт в Yandex Cloud Object Storage как в S3-compatible storage;
- домен привязан к bucket/build target;
- поверх storage при необходимости добавляются CDN, HTTPS, кастомные error/redirect rules и preview environments;
- критичный контент рендерится как готовый HTML, без зависимости от client-side data fetch для первого экрана.

### Что это значит для доменов

Если обычные события и экскурсии пойдут на **разные домены**, это надо считать не “темой на потом”, а базовым архитектурным ограничением:

- отдельный домен практически означает отдельный bucket / отдельную публикацию;
- значит, renderer и пайплайн должны изначально поддерживать несколько site targets;
- доменная развязка не должна ломать единый data contract по событиям.

## Связь с текущим Telegraph-контуром

Переход затронет не только внешний URL:

- `telegraph_build` в будущем, вероятно, будет заменён или дополнен отдельным `static_page_build`;
- ссылки в админ-отчётах, дайджестах и будущих public-posts должны уметь показывать новый canonical URL;
- month/weekend/festival/event public surfaces нужно будет мигрировать поэтапно, а не одним big-bang;
- Telegraph может понадобиться как временный compatibility layer, но не как target architecture.

## TODO до детального проектирования

### 1. Аудит реальных данных по событиям

- [ ] Выгрузить репрезентативную выборку событий минимум за последние `6-12` месяцев по основным типам: концерты, выставки, лекции, фестивали, экскурсии, ярмарки.
- [ ] Посчитать completeness по полям, критичным для event page и JSON-LD: `title`, `date`, `time`, `timezone`, `city`, `location_name`, `location_address`, `description`, `image`, `organizer`, `ticket_link`, `price`, `age_restriction`, `language`, `source_url`, `updated_at`, `status`.
- [ ] Отдельно измерить долю событий, где факты есть только в OCR/медиа, а не в основном тексте.
- [ ] Отдельно измерить долю событий с несколькими датами, `end_date`, recurring-связями и linked-occurrences.
- [ ] Зафиксировать по каждому event-type, какие поля реально можно считать обязательными, а какие пока заполняются слишком нестабильно.

Deliverable:

- матрица полноты полей по `event_type` и по `source_type`;
- список data gaps, блокирующих rich event pages.

### 2. Исследование источников и source-of-truth

- [ ] Разобрать реальные цепочки “оригинальный источник -> агрегатор -> репост -> билетный сайт” на живых примерах.
- [ ] Зафиксировать, какой источник считается каноническим для каждого класса событий.
- [ ] Проверить, насколько часто текущие `source_url` ведут не на первоисточник, а на перепаковку.
- [ ] Составить правила provenance для страницы: когда показывать original source, когда ticket site, когда оба.
- [ ] Отдельно исследовать recurring/duplicate кейсы, где одно и то же событие гуляет по нескольким каналам или площадкам.

Deliverable:

- правила canonical-source и source provenance для public pages;
- shortlist случаев, где текущая модель `Event` требует усиления.

### 3. Исследование identity и URL-модели

- [ ] На реальных событиях проверить, где достаточно `Event`, а где нужен `EventSeries`.
- [ ] Определить стабильные правила slug generation и slug survival после правок заголовка.
- [ ] Определить, как жить с переносами, отменами, сменой площадки и split/merge случаев.
- [ ] Протестировать, какие типы событий должны иметь отдельный permanent URL на каждую дату без исключений.
- [ ] Сверить, как linked events, long-run exhibitions и фестивальные сущности должны влиять на cross-links, а не на canonical URL.

Deliverable:

- проект правил для `slug`, redirect и canonical identity;
- набор edge cases с примерами из БД.

### 4. Исследование freshness и жизненного цикла страницы

- [ ] Посчитать, насколько часто по событиям происходят апдейты: `sold_out`, `waitlist`, `cancelled`, `rescheduled`, `venue_changed`, `ticket_link_changed`.
- [ ] Определить, какие апдейты должны вызывать немедленный rebuild страницы и `lastmod`.
- [ ] Проверить, как часто события после публикации меняют факты в последние `24/72` часа перед началом.
- [ ] Зафиксировать SLA свежести для time-sensitive event pages.

Deliverable:

- матрица lifecycle-сигналов и rebuild-триггеров;
- требования к очереди публикации и инвалидации.

### 5. Исследование SEO, GEO и internal linking

- [ ] Проверить, какие GEO-факты уже есть в данных, а какие придётся достраивать: город, район, координаты, площадка, timezone.
- [ ] Определить минимальный набор index/listing pages для внутренней перелинковки: город, дата, категория, площадка.
- [ ] Оценить, какие поля обязательны для search snippets и rich results, а какие можно добавлять позже.
- [ ] Сформировать contract для `title`, `meta description`, `canonical`, `breadcrumbs`, `robots`, `sitemap.xml`, `lastmod`.
- [ ] Зафиксировать policy по AI crawlers, включая `OAI-SearchBot`, без конфликта с обычной индексацией.

Deliverable:

- SEO/GEO metadata contract;
- список недостающих полей и нормализаций.

### 6. Исследование UI, контента и дизайна на реальных карточках

- [ ] На живой выборке страниц определить обязательные content blocks для mobile-first layout.
- [ ] Проверить, что реально нужно пользователю на первом экране на телефоне и что переносится ниже на desktop.
- [ ] Определить единые паттерны для cover-image, gallery, CTA, блока “источник”, “как попасть”, “другие даты”.
- [ ] Отдельно проверить, как должны выглядеть страницы для exhibition, festival, excursion, screening, fair и lecture shapes.
- [ ] Проверить требования к share-preview для Telegram и MAX: title, image ratio, excerpt, fallback image.

Deliverable:

- набор page-blocks и design constraints для первого прототипа;
- список event-shape differences, которые нельзя игнорировать в renderer.

### 7. Исследование infra и публикации в Yandex Cloud

- [ ] Подтвердить целевую схему деплоя: bucket layout, custom domain, HTTPS/certificates, CDN, invalidation, error pages.
- [ ] Определить, как разделять production, preview и локальную публикацию.
- [ ] Зафиксировать требования к артефактам сборки: HTML, assets, JSON-LD, sitemap, robots.
- [ ] Решить, нужен ли один multisite renderer или отдельные build targets по доменам.
- [ ] Сразу проверить, как будет жить отдельный excursions domain/bucket, если его захотят включить раньше общей миграции.

Deliverable:

- infra decision memo;
- список env/secrets/config для rollout.

### 8. Исследование миграции с Telegraph

- [ ] Собрать перечень текущих пользовательских и админских поверхностей, которые завязаны на `telegraph_url`.
- [ ] Определить, где нужен dual-run период `Telegraph + static`, а где можно быстро переключиться на canonical site URL.
- [ ] Решить, что делать со старыми Telegraph-ссылками в уже опубликованных Telegram-постах.
- [ ] Определить порядок миграции: `event pages -> month/weekend -> festival pages -> auxiliary surfaces`.
- [ ] Сформировать список кода и джобов, которые будут затронуты миграцией.

Deliverable:

- phased migration plan без big-bang;
- dependency map по коду и runtime jobs.

## Research exit criteria

Переход к детальному проектированию имеет смысл только после того, как готовы:

- data completeness audit по реальным событиям;
- правила source-of-truth и identity;
- SEO/GEO metadata contract;
- подтверждённая infra-схема под Yandex Cloud;
- migration plan с понятным coexistence period для Telegraph.

## Связь с экскурсиями

Экскурсии не стоит проектировать как отдельный “Telegraph-хвост”. Для них надо сразу держать совместимую future-схему:

- та же логика permanent URL на конкретный `GuideExcursionOccurrence`;
- при необходимости отдельный excursions domain и отдельный bucket;
- дополнительные сущности `GuideProfile` и `GuideExcursionTemplate` идут поверх occurrence pages, а не вместо них;
- одинаковые требования к HTML, JSON-LD, indexability, freshness и source provenance;
- общий renderer/data contract, даже если бренд и домен у экскурсий будут отдельные.

Связанный backlog:

- `docs/backlog/features/guide-excursions-monitoring/README.md`
