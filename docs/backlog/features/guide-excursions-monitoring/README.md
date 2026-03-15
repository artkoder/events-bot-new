# Guide Excursions Monitoring (мониторинг экскурсий гидов)

> **Status:** Not implemented / In design  
> **Current scope:** Telegram only. VK sources are postponed to a later iteration.

Цель: находить анонсы и апдейты по экскурсиям гидов в выбранных Telegram-источниках трека (`guide`, `guide_project`, `excursion_operator`, `organization_with_tours`, `aggregator`), собирать по ним отдельный дайджест и публиковать его в Telegram без смешивания с обычными культурными событиями (`/daily`, month pages, weekend pages).

Исследовательская база для этой спеки:

- каноника текущего Telegram Monitoring: `docs/features/telegram-monitoring/README.md`
- пост-метрики и медианы: `docs/features/post-metrics/README.md`
- Smart Update / fact-first: `docs/features/smart-event-update/README.md`
- future public pages / own domain / SEO-GEO: `docs/backlog/features/static-event-pages/README.md`
- выгрузка примеров через Telethon: `artifacts/codex/excursion_posts_2026-03-14.json`
- расширенный casebook по каналам: `docs/backlog/features/guide-excursions-monitoring/casebook.md`
- facts-first architecture for `guide / template / occurrence`: `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- LLM-first prompt pack and stage map: `docs/backlog/features/guide-excursions-monitoring/llm-first.md`
- frozen expected outputs and metrics: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- external review brief for `Opus`: `docs/backlog/features/guide-excursions-monitoring/opus-audit-brief.md`
- digest/media delivery spec: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- planned live E2E scenarios: `docs/backlog/features/guide-excursions-monitoring/e2e.md`
- implementation-ready MVP: `docs/backlog/features/guide-excursions-monitoring/mvp.md`

## Зачем это отдельная фича

Экскурсии гидов отличаются от обычного потока событий:

- один и тот же маршрут живёт как серия повторяющихся выходов, а не как одноразовое событие;
- важные факты часто размазаны по нескольким постам: анонс, перенос, sold out, отчёт, анонс повтора;
- один и тот же выход может одновременно появляться у автора, у соорганизатора и у агрегатора;
- часть постов нужна не для публикации дайджеста, а для накопления профиля гида и “типовой экскурсии”;
- эти карточки не должны попадать в общие public-поверхности текущего бота.

## Что показал анализ примеров

По предоставленным URL через Telethon были проверены тексты, просмотры, реакции, медиа и альбомы.

### Наблюдения по формату постов

1. **Мульти-анонс в одном посте — это норма.**  
   Примеры: `tanja_from_koenigsberg/3895`, `tanja_from_koenigsberg/3873`, `gid_zelenogradsk/2684`, `amber_fringilla/5739`, `amber_fringilla/5806`, `alev701/631`.  
   Один пост может содержать 2-5 отдельных выходов, иногда ещё и лекции/ужины рядом.

2. **Есть отдельные “операционные” посты без полного анонса.**  
   Примеры: `gid_zelenogradsk/2705` (освободилось одно место), `amber_fringilla/5676` (перенос), `gid_zelenogradsk/2517` (напоминание накануне), `vkaliningrade/4585` (точка встречи на завтра).  
   Такие посты должны обновлять уже найденную экскурсию, а не создавать новую.

3. **Есть отчёты и рефлексивные посты, которые полезны для шаблона экскурсии, но не для мгновенного дайджеста.**  
   Примеры: `tanja_from_koenigsberg/3702`, `gid_zelenogradsk/2508`, `amber_fringilla/5782`, `amber_fringilla/5347`.  
   Они описывают атмосферу, маршрут, отзывы, развитие темы и могут давать ссылку на будущий повтор.

4. **Один и тот же маршрут встречается в нескольких каналах.**  
   Примеры:
   - `Город К. Женщины, которые вдохновляют`: `tanja_from_koenigsberg/3895`, `amber_fringilla/5739`, `amber_fringilla/5773`
   - `Innenstadt: жизнь в кольце`: `tanja_from_koenigsberg/3873`, `tanja_from_koenigsberg/3830`, `amber_fringilla/5560`, `amber_fringilla/5676`
   - `У Тани на районе: Закхайм и окрестности`: `tanja_from_koenigsberg/3895`, `amber_fringilla/5739`, `vkaliningrade/4576`
   Значит, нужен слой дедупликации “типовая экскурсия/конкретный выход”, а не только дедуп постов.

5. **Посты часто смешивают экскурсии с нецелевыми сущностями.**  
   Примеры: `gid_zelenogradsk/2684` (экскурсии + лекции), `amber_fringilla/5739` (экскурсии + гастроужины), `amber_fringilla/5742` (экоквест, не экскурсия).  
   Простого regex по слову `экскурсия` недостаточно.

6. **Caption альбома может лежать не в том сообщении, которое дали ссылкой.**  
   Пример: `katimartihobby/1842` входит в альбом, а текст фактически лежит в `1844`.  
   Значит, мониторинг обязан собирать grouped media целиком, а не анализировать только один `message_id`.

### Наблюдения по метрикам

На выборке из предоставленных постов медианы по каналам получились такими:

- `tanja_from_koenigsberg`: `views≈1910`, `reactions≈82`
- `amber_fringilla`: `views≈1002`, `reactions≈86`
- `gid_zelenogradsk`: `views≈607`, `reactions≈32`
- `katimartihobby`: `views≈438`, `reactions≈46`
- `alev701`: `views≈594`, `reactions≈20` (одна точка в выборке)

Вывод: ранжирование должно быть **только внутри своего канала/окна**, а не по абсолютным цифрам между гидами.

## Уточнённая taxonomy источников

На уровне фичи источники стоит разделять минимум на пять top-level archetypes:

- `guide_personal`
  - личный канал конкретного гида; source of truth по личности, narrative и прямому контакту
- `guide_project`
  - брендовый канал гида-проекта; может смешивать экскурсии с доп. услугами, travel и lifestyle контентом
- `excursion_operator`
  - экскурсионный оператор / агентство, у которого экскурсии и групповые поездки являются главным продуктом, но publisher не равен конкретному гиду
- `organization_with_tours`
  - организация/движение/институция, для которой экскурсии важны, но не являются единственным продуктом
- `aggregator`
  - агрегаторный канал, который перепаковывает и продаёт/собирает туры нескольких гидов

Практический приоритет источников для occurrence merge:

1. `guide_personal`
2. `guide_project`
3. `excursion_operator`
4. `organization_with_tours`
5. `aggregator`

Casebook с разбором каналов и примерами: `docs/backlog/features/guide-excursions-monitoring/casebook.md`

## Рекомендация по границе проекта

### Рекомендуемый путь сейчас

Делать **внутри текущего репозитория и текущего админ-бота**, но как **отдельный content track**.

Причины:

- уже есть Telethon/Kaggle-пайплайн, OCR, хранение метрик, LLM gateway с лимитами, scheduler, `/a`, digest-паттерны и админ-UI;
- ожидаемый масштаб (`~100` Telegram-каналов) не требует отдельного сервиса по нагрузке;
- большая часть сложности здесь не в инфраструктуре, а в доменной логике: дедуп, факты, типовые маршруты, апдейты мест/переносов.

### Когда имеет смысл выносить в отдельный проект

Отдельный проект имеет смысл только если одновременно появятся:

- свой публичный бренд и отдельная аудитория;
- отдельный редакторский цикл и отдельные команды/операторы;
- полноценный VK ingestion;
- отдельные guide pages / excursion pages / stories pipeline, слабо связанные с культурной афишей текущего бота.

Пока это выглядит как следующий этап, а не стартовая точка.

## Анализ: один бот или два бота

### Сценарий A: один бот, два контент-трека

Плюсы:

- максимум переиспользования уже работающего кода;
- один набор секретов, один deploy, один scheduler;
- проще добавить команды в `/a`;
- проще использовать общие post metrics, LLM limiter, Kaggle launcher и Telethon auth.

Минусы:

- нужна строгая изоляция выдачи, чтобы экскурсии гидов не утекали в `/daily`, month pages и weekend pages;
- текущие broad-query поверхности надо явно научить понимать новый трек.

### Сценарий B: отдельный бот/форк на базе текущего

Плюсы:

- чище продуктовая граница;
- проще отдельно развивать guide cards, template pages и stories;
- меньше риск случайно затронуть основной поток культурных событий.

Минусы:

- дублирование Telethon/Kaggle/LLM/scheduler/admin-routing;
- две кодовые базы и два operational контура;
- часть проблем всё равно останется той же самой: дедуп, OCR, кластеризация маршрутов, метрики.

### Итог

Для старта целесообразнее **один кодовый базис и один админ-бот**, но:

- с отдельной командой;
- с отдельным списком источников;
- с отдельным digest-потоком;
- с отдельным scheduler job;
- с отдельным content flag в данных.

## Нагрузка и почему нельзя просто “скопировать текущий TG Monitoring”

Текущий Telegram Monitoring имеет production baseline timeout:

- `TG_MONITORING_TIMEOUT_PER_SOURCE_MINUTES = 3.64`
- итоговый таймаут считается как `15 + sources * 3.64 * 1.3`, capped at `360 min`

Для `100` источников это даёт примерно `489` минут до cap, то есть текущий generic flow уже близок к потолку.

Вывод:

- прямой “полный аналог” текущего мониторинга для ещё сотни каналов делать не стоит;
- нужен **более дешёвый guide-specific notebook** с сильным deterministic prefilter до LLM;
- отдельный ключ `GOOGLE_API_KEY2` для этого потока уместен.

### Реалистичный стартовый budget

Для первой итерации разумно проектировать так:

- Telethon-скан по `10-20` свежим сообщениям на источник;
- regex/heuristic prefilter на стороне notebook;
- в LLM отдавать только кандидаты классов `announce`, `status_update`, `reportage_template_signal`;
- повторный rescanning держать отдельным коротким проходом по уже найденным активным экскурсиям.

Это радикально дешевле, чем гонять LLM по всем постам всех гидов.

## Kaggle notebook как каноническая граница

Это нужно зафиксировать отдельно: **сам мониторинг Telegram-каналов должен выполняться в Kaggle notebook**.

Не в runtime бота и не в отдельном новом ingestion service.

Рекомендуемая граница:

- Kaggle notebook делает Telethon fetch, grouped albums, OCR, prefilter и Gemma-pass;
- серверный бот получает только результат notebook run и делает import/merge/digest/admin reporting;
- Telethon user-session остаётся только на Kaggle стороне.

### Что именно переиспользуем из текущего Telegram Monitoring

Не нужно проектировать guide-monitoring “с нуля”.

Нужно максимально переиспользовать существующий стек:

- `kaggle/TelegramMonitor/telegram_monitor.ipynb` как runtime baseline;
- `source_parsing/telegram/service.py` как модель push/poll/download/recovery;
- `source_parsing/telegram/handlers.py` как модель import/reporting lifecycle;
- `source_parsing/telegram/split_secrets.py` как модель безопасной доставки секретов в Kaggle.

### Что форкается, а не копируется 1:1

- candidate schema;
- regex/heuristic prefilter;
- Gemma prompts / output contract;
- JSON result schema;
- source taxonomy и guide-specific post kinds.

### Что остаётся только на сервере

- facts-first merge для `guide / template / occurrence`;
- блоки `/guide_*` и `/general_stats`;
- digest ranking;
- публикация через bot-side media bridge.

## Что делать с хранением

### Не рекомендовано

Не стоит хранить первую версию как “обычные `Event` без дополнительных флагов”.

Причина: текущие `/daily`, month pages и weekend pages выбирают все `active + non-silent` события широкими запросами. Если просто добавить ещё событий типа `экскурсия`, они начнут попадать в общие поверхности.

### Рекомендуемый минимум

Если реализация остаётся внутри текущего бота, нужен отдельный content flag, например:

- `content_track = 'guide_excursions'`
- `exclude_from_daily = 1`
- `exclude_from_calendar_pages = 1`

При этом `event_type='экскурсия'` можно оставить как тип события, но этого **недостаточно** для изоляции поверхностей.

`silent` для этого использовать не рекомендуется: у него уже есть другой operational смысл.

### Нужна ли отдельная SQLite / local SQL DB

На старте — **нет**.

Рекомендуемая стратегия:

- канонические guide-occurrence / template / source tables хранить в основной базе;
- raw notebook outputs, deep scans, OCR debug и временные classification traces хранить в `artifacts/`;
- к отдельной local DB возвращаться только если guide-monitoring вырастет до отдельного интенсивного raw/cache слоя со своей частой записью.

Причина: на старте выгода одной общей operational базы выше, чем гипотетическая изоляция нагрузки.

Для implementation-first MVP рекомендуемый компромисс ещё жёстче: не использовать `event` table вовсе, а держать guide-monitoring в отдельных `guide_*` таблицах внутри той же SQLite. Детали: `docs/backlog/features/guide-excursions-monitoring/mvp.md`

## Нужен ли fact-based подход

### Короткий ответ

Да, но поэтапно.

### Рекомендация

1. **Итерация 1**  
   Собирать структурированные facts **по конкретному выходу экскурсии**:
   - guide names / organizer names
   - route title
   - date / time / duration
   - digest eligibility: public vs `on_request`
   - meeting point
   - price / free / donation / waitlist / sold out
   - seats information
   - `для кого`:
     возраст / класс / тип группы / locals vs tourists / tempo / интерактивность
   - signup link / contact
   - source channel / source post / metrics
   - OCR snippets

2. **Итерация 2**  
   Добавить уровень **типовой экскурсии**:
   - стабильное название маршрута
   - summary маршрута
   - `availability_mode`: `scheduled_public | on_request_private | mixed`
   - `audience_fit` как first-class fact family
   - характерные stops/themes
   - recurring evidence from past posts
   - отзывы / эмоции / манера подачи

3. **Итерация 3**  
   Добавить **карточку гида** и public static pages на собственном домене/бакете.

### Public pages сейчас или позже

На старте public pages для guide excursions **не нужны**.  
Сначала нужно стабилизировать:

- detection,
- occurrence dedup,
- template clustering,
- digest quality.

После этого public pages стоит включать уже **не через Telegraph**, а через тот же static-site контур, что и для обычных событий:

- отдельная страница на каждый `GuideExcursionOccurrence`;
- `Guide Profile`;
- `GuideExcursionTemplate`.

Если экскурсии получат отдельный бренд, это должно поддерживаться как **отдельный домен / отдельный bucket**, но без расхождения по data contract и renderer. Каноника по этому треку: `docs/backlog/features/static-event-pages/README.md`.

## Admin surfaces без публичных страниц

Так как публичные страницы на старте не планируются, admin-инструменты нужны сразу.

Минимальный набор:

- `/guide_excursions`
  - run now
  - preview digest
  - publish digest
  - send test report
  - active occurrences
  - source list
  - uncertain / aggregator-only findings
- `/guide_recent [hours]`
  - новые occurrences
  - status-updates
  - template-signals
  - aggregator-only findings
- `/guide_sources`
  - список источников, тип источника, health и медианы
- `/guide_digest`
  - выбор family дайджеста

Также отдельный блок должен появиться в `/general_stats`:

- run status / duration;
- sources scanned by type;
- posts scanned / prefiltered / llm_checked;
- occurrences new / updated;
- status updates;
- template signals;
- aggregator-only unresolved;
- digests built / published.

## Предлагаемая доменная модель

Архитектурная каноника с facts-first слоем и именованными компонентами: `docs/backlog/features/guide-excursions-monitoring/architecture.md`

### Итерация 1

- `GuideSource`
  - Telegram channel of guide / collaborator / aggregator
  - role: `guide|co_organizer|aggregator`
  - trust and source priority

- `GuideProfile`
  - минимальная карточка сущности `person|project|organization`
  - display name / marketing name
  - links to source channels
  - facts rollup по позиционированию и специализации

- `GuideExcursionTemplate`
  - минимальная типовая экскурсия / recurring route
  - canonical title
  - aliases / recurring anchors
  - short fact rollup по маршруту и особенностям

- `GuideExcursionOccurrence`
  - конкретный выход экскурсии
  - дата, время, цена, место встречи, статус мест, signup link
  - link to template, если он уже определён
  - participants / roles
  - best source post
  - engagement snapshot

- `GuideFactClaim`
  - append-only fact ledger для `guide|template|occurrence`
  - provenance, confidence, observed_at
  - факты накапливаются раньше, чем появляются полноценные public pages

- `GuideDigestIssue`
  - выпуск дайджеста новых экскурсий
  - список включённых occurrences

### Итерация 2+

- richer `GuideProfile`
  - positioning summary
  - strengths / bio
  - collaboration graph

- richer `GuideExcursionTemplate`
  - stronger clustering
  - references to past materials / reviews
  - long-form narrative blocks

### Логическая связь

Корректнее считать так:

- `GuideProfile M:N GuideExcursionTemplate`
- `GuideExcursionTemplate 1:N GuideExcursionOccurrence`
- `GuideProfile M:N GuideExcursionOccurrence`

Это лучше отражает реальность, чем плоская модель “один пост = одно событие” и жёсткое `1:N`, потому что в домене много коллабораций, проектов и организационных форматов.

## Предлагаемый detection pipeline

### 1. Источники

Отдельный список источников:

- личные каналы гидов;
- каналы соорганизаторов;
- агрегаторы (`vkaliningrade`) как fallback-источник.

VK-группы и VK-страницы в этой версии не входят.

### 2. Telethon fetch

Guide monitoring notebook должен:

- получать текст поста;
- собирать grouped albums целиком;
- сохранять `views`, `forwards`, `reactions`;
- вытягивать links / handles / телефоны;
- прогонять OCR по изображениям;
- сохранять media fingerprints для dedup картинок.

### 3. Deterministic prefilter до LLM

Нужны минимум три класса правил.

#### 3.1. High-recall candidate signals

- `экскур`, `прогулк`, `маршрут`, `по следам`, `авторская экскурсия`, `экопрогулка`
- явная дата
- явное время
- цена / `руб` / `₽`
- запись / бронирование / `@username` / телефон
- `мест нет`, `лист ожидания`, `осталось N мест`, `перенос`

#### 3.2. Candidate kind classification

После prefilter пост надо отнести к одному из видов:

- `announce_occurrence`
- `on_demand_offer`
- `status_update`
- `reportage`
- `template_signal`
- `ignore`

#### 3.3. Reject / downgrade signals

- лекции без экскурсии;
- гастроужины, квесты, спектакли, если экскурсия не является главным предметом поста;
- праздничные поздравления;
- общие размышления про профессию;
- афиши без фактов, если ни текст, ни OCR не дают за что зацепиться.

### 4. LLM confirmation

Gemma через существующий LLM Gateway должна подтверждать:

- это экскурсия или нет;
- это новый выход, `on-demand` offer, апдейт существующего выхода или материал для шаблона;
- кто гид/гиды;
- какие поля уверенные, какие uncertain;
- `для кого` подходит эта экскурсия;
- есть ли связь с уже известной типовой экскурсией.

### 5. Source priority

При конфликте источников:

1. исходный канал гида;
2. канал гида-проекта;
3. экскурсионный оператор;
4. канал соорганизатора / организации;
5. агрегатор.

Если экскурсия найдена только у агрегатора, это допустимый fallback, но при появлении исходного поста он должен становиться primary source.

## Повторное сканирование

Повторный scan нужен не только ради метрик.

Он нужен, чтобы ловить:

- sold out / waitlist / появилось место;
- перенос даты;
- уточнение места встречи;
- ссылку на повтор того же маршрута;
- новые отзывы / фото / эмоциональные куски для template page.

Рекомендация:

- ежедневный основной scan новых постов;
- короткий re-scan активных occurrences на окнах `0/1/3/7` суток;
- отдельный lookback по recent template-related posts без попадания в digest.

## Дайджест: что должно попасть в выпуск

### Принцип отбора

В дайджест попадают **только новые найденные выходы экскурсий**, а не все найденные посты.

`on_request` / `private-group-only` предложения без нормального публичного набора в digest автоматически не попадают. Они должны усиливать template/profile layer.

Один выпуск может содержать несколько гидов и несколько экскурсий.

### Приоритизация

Рекомендуемый ranking:

1. freshness: новый выход, ещё не публиковался в guide digest;
2. completeness: есть дата, время, маршрут, место встречи, цена, signup;
3. source priority: original guide > co-organizer > aggregator;
4. per-channel popularity: `views/likes` выше медианы своего канала;
5. scarcity signal: `осталось N мест`, `лист ожидания`, `мест мало`;
6. editorial value: новый маршрут / премьера / благотворительный / коллаборативный формат.

### Public marker

У названия экскурсии допустим лайк-маркер, если `likes > median_likes` внутри окна своего канала, по аналогии с текущими `⭐/👍` сигналами.

### Поля карточки в дайджесте

Для каждой карточки:

- гид / гиды;
- дата и время;
- название маршрута;
- одно короткое предложение “чем интересна эта экскурсия”;
- при наличии короткий marker `для кого`;
- место встречи;
- стоимость;
- статус мест: число / ограничено / не ограничено / waitlist / sold out;
- ссылка на запись;
- ссылка на канал гида.

Дополнительно:

- до одной картинки на карточку;
- картинки дедуплицируются по perceptual hash;
- если источник агрегаторный, в карточке всё равно нужен link на канал гида, если он распознан.

## Команды и discoverability

Рекомендуемая стартовая поверхность:

- `/guide_excursions` — меню фичи
- внутри: `run now`, `sources`, `preview digest`, `publish digest`, `stats`
- companion commands: `/guide_recent`, `/guide_sources`, `/guide_digest`

Через `/a` должны находиться фразы:

- “дайджест экскурсий гидов”
- “мониторинг гидов”
- “новые экскурсии гидов”
- “запусти мониторинг экскурсий”

## Семейства дайджестов

На старте целесообразны не один, а несколько families.

### Public families

1. `new_occurrences`
   - основной дайджест новых найденных выходов
2. `last_call`
   - осталось мало мест / waitlist / sold out / освободилось место
3. `weekend_soon`
   - подборка на ближайшие выходные или 3-7 дней
4. `premieres_and_new_routes`
   - первые выходы и новые маршруты
5. `popular_inside_channel`
   - relative-popularity подборка по медианам канала

### Admin families

- `aggregator_only`
- `status_changes`
- `uncertain_cluster`

Порядок внедрения:

1. `new_occurrences`
2. `last_call`
3. `weekend_soon`
4. `premieres_and_new_routes`
5. `popular_inside_channel`

## Предлагаемые итерации

### Iteration 1: мониторинг + дайджест

Делаем:

- Telegram-only scan;
- Telethon + OCR + regex prefilter + Gemma confirmation;
- facts-first storage, но с occurrence-first publishing;
- минимальные `GuideProfile` и `GuideExcursionTemplate` rows;
- append-only `GuideFactClaim` для `guide|template|occurrence`;
- digest preview/publish;
- отдельный scheduler;
- отдельный key `GOOGLE_API_KEY2`;
- admin surfaces и блок в `/general_stats`.

Не делаем:

- VK ingestion;
- public static pages;
- guide profile pages;
- template pages;
- Bento stories generation.

### Iteration 2: типовые экскурсии и карточки гидов

Делаем:

- stronger clustering repeated excursions into templates;
- richer guide profile summary;
- references to past materials;
- richer fact accumulation from reports/reviews;
- wider lollipop-based text generation beyond digest one-liners.

### Iteration 3: страницы и stories

Делаем:

- mobile-first static pages на собственном домене/бакете для `GuideExcursionOccurrence`;
- отдельный excursions domain/bucket при необходимости;
- cross-links между `GuideExcursionOccurrence`, `GuideProfile` и `GuideExcursionTemplate`;
- related-routes blocks между шаблонами и соседними маршрутами;
- Bento stories image generation;
- VK sources.

## Нерешённые вопросы

- Нужен ли публичный отдельный канал только под guide digests, или публикация пойдёт в существующий канал/сеть каналов.
- Где провести границу между `экскурсией`, `экопрогулкой`, `аудиоквестом`, `иммерсивной прогулкой`, `ужином с историей`.
- Нужен ли общий template, если маршрут сильно меняется от сезона к сезону.
- Какой минимум фактов считать обязательным для попадания в digest.

## Итоговая рекомендация

Начинать стоит **внутри текущего бота**, но не как “ещё один обычный event source”.

Правильный стартовый дизайн:

- отдельный guide-monitoring pipeline;
- отдельный source registry;
- отдельный digest surface;
- отдельный content flag в хранении;
- separate Kaggle notebook;
- separate Google key;
- fact accumulation с первого дня, но public static pages только после стабилизации template layer.
