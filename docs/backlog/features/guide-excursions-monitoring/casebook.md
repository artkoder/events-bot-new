# Guide Excursions Casebook

> **Status:** Research / design  
> **Research date:** 2026-03-14  
> **Method:** Telethon deep scan of selected public Telegram channels with grouped-album collapsing and post-shape heuristics.

Артефакты исследования:

- `artifacts/codex/guide_channels_deep_2026-03-14.json`
- `artifacts/codex/guide_channels_deep_summary_2026-03-14.json`
- `artifacts/codex/excursion_posts_2026-03-14.json`
- `artifacts/codex/guide_channel_excursions_profitour_2026-03-14.json`
- `artifacts/codex/guide_channel_excursions_profitour_summary_2026-03-14.json`

## 1. Цель кейсбука

Зафиксировать не только high-level дизайн фичи, но и реальные паттерны поведения источников:

- какие бывают типы каналов;
- как именно гиды публикуют экскурсии;
- какие посты создают больше всего шума для пайплайна;
- какие поверхности администрирования и дайджестов действительно нужны на старте.

## 2. Выборка каналов

В deep scan вошли:

- `tanja_from_koenigsberg`
- `gid_zelenogradsk`
- `katimartihobby`
- `amber_fringilla`
- `art_from_the_Baltic`
- `alev701`
- `vkaliningrade`
- `ruin_keepers`
- `twometerguide`
- `valeravezet`
- `excursions_profitour`

Это уже покрывает разные archetypes источников:

- личные каналы гидов;
- брендовые каналы гидов-проектов;
- экскурсионные операторы / агентства;
- организационные каналы, где экскурсии лишь часть работы;
- агрегаторный проект.

## 3. Уточнённая taxonomy источников

### 3.1. Top-level классы

#### `guide_personal`

Личный канал конкретного гида. Экскурсии идут от первого лица, запись ведётся напрямую, narrative привязан к личности автора.

Каналы:

- `tanja_from_koenigsberg`
- `gid_zelenogradsk`
- `katimartihobby`
- `amber_fringilla`
- `alev701`

Признаки:

- в `about` или в постах есть имя конкретного гида;
- запись идёт через личный контакт;
- отчёты, размышления и анонсы переплетены;
- часто есть коллаборации с другими гидами.

#### `guide_project`

Канал гида-проекта или брендового экскурсионного продукта. Персона автора важна, но поверхность уже проектная и может включать доп. услуги, другой город, travel/food контент.

Каналы:

- `twometerguide`
- `valeravezet`
- `art_from_the_Baltic`

Признаки:

- ярко выраженный бренд;
- смешение экскурсий с сопутствующим контентом;
- часто более “медийная” подача;
- выше риск false positives по travel/lifestyle текстам.

#### `guide_project_pending_review`

Отдельно фиксируем свежие добавления, которые уже должны участвовать в мониторинге, но ещё не прошли полный deep-scan разбор.

Каналы:

- `art_from_the_Baltic`

Текущее рабочее допущение для seed/runtime:

- включать в guide monitoring как `guide_project`;
- держать `trust_level=medium`;
- считать mixed-content source до первого полноценного case review;
- после накопления live материалов уточнить, это больше авторский экскурсионный проект, art-walk бренд или смешанный культурный канал.

#### `excursion_operator`

Экскурсионный оператор / агентство, для которого экскурсии и поездки являются основным продуктом, но publisher не равен конкретному гиду.

Каналы:

- `excursions_profitour`

Признаки:

- центр тяжести не на личности гида, а на packaged programs и бронировании;
- много `школьные группы`, `организованные группы`, `корпоративные` и других audience-led форматов;
- значимая часть предложений живёт как `по запросу`, `по вашему желанию`, `под группу`, `цена зависит от количества`;
- источник часто является source of truth по цене, бронированию и availability, но не по guide narrative.

#### `organization_with_tours`

Организация, движение, НКО, проект или институция, для которых экскурсии важны, но не являются единственным продуктом.

Каналы:

- `ruin_keepers`

Признаки:

- много контента про миссию, наследие, партнёров, публикации, пожертвования;
- экскурсии появляются как один из способов вовлечения;
- booking и фактология обычно оформлены аккуратно, но общий сигнал канала сильно шире экскурсионного.

#### `aggregator`

Канал, который собирает и репакует экскурсии/поездки от нескольких гидов или проектов.

Каналы:

- `vkaliningrade`

Признаки:

- централизованный телефон/почта/контакт;
- авторский гид указан внутри поста, но не всегда является publisher;
- часто бывают reminder-посты с местом встречи или повторная упаковка чужого анонса;
- такие каналы нельзя считать source of truth, но они важны как fallback и как источник оперативных апдейтов.

### 3.2. Practical priority by source type

Для occurrence-level merge приоритет рекомендуется такой:

1. `guide_personal`
2. `guide_project`
3. `excursion_operator`
4. `organization_with_tours`
5. `aggregator`

Но для некоторых кейсов апдейтов приоритет должен быть уже field-level:

- `meeting_point_update`: допускается из `aggregator`, если исходный guide-post молчит;
- `sold_out / few_seats`: допустимо из любого источника, но с пометкой source provenance;
- `route_summary`: предпочтительно из `guide_personal` или `guide_project`.
- `booking / price / group_size / on_request availability`: часто первичны именно у `excursion_operator`.

## 4. Профили каналов и что это значит для пайплайна

### `tanja_from_koenigsberg`

- Archetype: `guide_personal`
- Content shape: много месячных/сезонных анонсов, много коллабораций, много narrative вокруг маршрута.
- Deep-scan signal:
  - `sample_size=53`
  - `views_median≈1553`
  - `reactions_median≈114`
- Operational implication:
  - сильный источник для occurrence discovery;
  - route summaries и template signals часто лежат в prose, а не только в сухих расписаниях;
  - один маршрут может повторяться месяцами и в разных кооперациях.
  - коллаборации с co-guides часто прячутся внутри длинного narrative post body; occurrence-level `guide_names` могут знать двух людей, но public digest ещё должен резолвить упомянутые `@username` в точные публичные ФИО, чтобы не деградировать `Анастасия Туз` в неполное `Анна Туз`.

### `gid_zelenogradsk`

- Archetype: `guide_personal`
- Content shape: более структурные анонсы, но канал смешивает экскурсии, лекции, обзоры объектов и личные заметки.
- Deep-scan signal:
  - `sample_size=33`
  - `views_median≈655`
  - `reactions_median≈34`
- Operational implication:
  - нужна хорошая separation-логика `экскурсии vs лекции`;
  - status-updates важны: освобождение мест, точка встречи, reminder;
  - часть постов описывает чужие объекты/музеи и не должна попадать в digest.

### `katimartihobby`

- Archetype: `guide_personal`
- Content shape: narrative/авторская подача, у многих постов слабая “операционная упаковка”, часть анонсов спрятана в середине текста или в album-caption.
- Deep-scan signal:
  - `sample_size=60`
  - `views_median≈459`
  - `reactions_median≈36.5`
- Operational implication:
  - особенно важен grouped-album handling;
  - regex по “экскурсия” недостаточен, нужны route-title/intent signals;
  - хороший пример каналов, где “типовая экскурсия” важнее сухого анонса.

### `amber_fringilla`

- Archetype: `guide_personal` с сильной thematic identity
- Content shape: орнитология, природа, поездки, прогулки, коллаборации, экопросвет.
- Deep-scan signal:
  - `sample_size=46`
  - `views_median≈926`
  - `reactions_median≈80.5`
- Operational implication:
  - много mixed posts: экоквесты, nature-notes, сопутствующие активности;
  - route-building и long-form narrative особенно полезны для template layer;
  - повторяющиеся маршруты часто кросспостятся с другими гидами.
  - multi-announce posts вроде `@amber_fringilla/5806` требуют media diversification: если несколько published occurrences приходят из одного исходного поста, digest должен распределять разные `media_refs` по карточкам, а не повторять одну и ту же фотографию несколько раз подряд.

### `art_from_the_Baltic`

- Archetype: provisional `guide_project`
- Status: added to canonical monitoring seed on 2026-03-16 by operator request; detailed deep-scan notes still pending.
- Current operational assumption:
  - канал нужно мониторить как потенциальный источник авторских art-walk / экскурсионных анонсов;
  - до первого полноценного разбора не считать его высокодоверенным source of truth;
  - внимательно проверять mixed cultural noise и не путать экскурсии с обычными art/event-posts.

### `alev701`

- Archetype: `guide_personal`
- Content shape: исторические заметки, ландшафты, градостроительство, редкие короткие анонсы.
- Deep-scan signal:
  - `sample_size=75`
  - `views_median≈589`
  - `reactions_median≈42`
- Operational implication:
  - лента шумная для naive regex;
  - occurrence discovery здесь, скорее всего, будет через редкие schedule posts;
  - template enrichment может быть полезнее, чем массовый candidate flow.

### `vkaliningrade`

- Archetype: `aggregator`
- Content shape: плотная лента анонсов, оперативки и project posts.
- Deep-scan signal:
  - `sample_size=49`
  - `views_median≈303.5`
  - `reactions_median≈3`
- Operational implication:
  - полезен как coverage fallback;
  - пригоден для meeting-point reminders и rescue-cases;
  - опасен как source of truth для guide ownership и canonical signup-link.

### `ruin_keepers`

- Archetype: `organization_with_tours`
- Content shape: heritage-движение, поездки, прогулки, журналы, медиа, призывы поддержать организацию.
- Deep-scan signal:
  - `sample_size=25`
  - `views_median≈1479`
  - `reactions_median≈40`
- Operational implication:
  - сильный анонсный канал, но не guide-only;
  - экскурсии часто завязаны на тему наследия и на конкретного приглашённого гида;
  - organization-posts должны обогащать occurrence, но не переписывать guide profile.

### `excursions_profitour`

- Archetype: `excursion_operator`
- Content shape: экскурсионный оператор/агентство с сильным school-group и group-booking уклоном, смешением scheduled программ и `по запросу` предложений.
- Deep-scan signal:
  - `sample_size=120`
  - `views_median≈237`
  - `reactions_median≈0`
- Operational implication:
  - это не агрегатор: канал является source of truth по заявкам, цене, условиям группы и адаптации программы;
  - значимая часть предложений не должна превращаться в `public digest occurrence`, потому что это `on-demand` или `private-group-only` предложения;
  - канал очень богат на `audience fit` факты: возраст, школьные классы, профориентация, группы 20-30 человек, “подойдёт для школьников”, “адаптируется под возраст”;
  - для template layer это один из самых ценных источников, даже когда occurrence layer даёт мало публичных выпусков.

### `twometerguide`

- Archetype: `guide_project`
- Content shape: сильный медийный бренд, Калининград + Петербург, экскурсии + travel/service content.
- Deep-scan signal:
  - `sample_size=39`
  - `views_median≈2544.5`
  - `reactions_median≈119`
- Operational implication:
  - нельзя полагаться на default-region канала;
  - нужен жёсткий geographic filter на уровне occurrence;
  - высокий engagement не должен автоматически означать “сильный экскурсионный анонс” — часть постов вообще про еду, travel logistics или медийные новости.
  - отдельный negative class: generic travel calendars (`когда цветут тюльпаны / куда поехать / travel wishlist`) должны отсеиваться целиком и не попадать даже в sparse/template occurrence layer.

### `valeravezet`

- Archetype: `guide_project`
- Content shape: бренд-персона автобуса, туры, lifestyle, travel-blog, акции, партнёрские истории.
- Deep-scan signal:
  - `sample_size=77`
  - `views_median≈854`
  - `reactions_median≈16`
- Operational implication:
  - высокий риск false positives из-за “покатушек”, travel-tone и рекламных розыгрышей;
  - occurrence detection надо жёстко привязывать к фактам: дата, маршрут, цена, бронирование, география;
  - отличный пример why source taxonomy влияет на parser rules.

## 5. Taxonomy постов

Для guide monitoring стоит работать не только с бинарным `event / non-event`, а как минимум с такими post-kinds:

- `announce_multi`
  - несколько выходов в одном посте;
- `announce_single`
  - один явный выход;
- `on_demand_offer`
  - типовая экскурсия / программа `по запросу`, `под группу`, `по вашему желанию`, без нормального публичного occurrence;
- `status_update`
  - few seats, sold out, перенос, точка встречи, reminder;
- `reportage`
  - пост о прошедшей экскурсии, опыте, проверке маршрута;
- `template_signal`
  - пост о создании маршрута, философии экскурсии, сильных особенностях;
- `mixed_or_non_target`
  - лекции, гастроужины, квесты, поздравления, heritage/news noise;
- `other`
  - всё прочее.

### Почему это важно

Без этой taxonomy пайплайн будет делать две системные ошибки:

1. либо заваливать LLM лишними постами;
2. либо терять важные template/status signals и `on-demand` предложения, если смотреть только на “сухие анонсы”.

## 6. Ключевые кейсы для дедупа и связи источников

### Case A: один и тот же выход в канале гида и соорганизатора

Примеры:

- `Город К. Женщины, которые вдохновляют`
- `Innenstadt: жизнь в кольце`
- `У Тани на районе: Закхайм и окрестности`

Решение:

- occurrence dedup по `route title + date + time + organizers + city`;
- внутри occurrence хранить несколько `source posts`;
- выбирать `best_source_post` по source priority и completeness.

### Case B: агрегатор публикует экскурсию, которой нет у исходного гида

Пример:

- `vkaliningrade` как fallback-marketplace слой.

Решение:

- occurrence можно заводить из агрегатора, но со статусом `source_quality=fallback`;
- позже occurrence должен смержиться в original-guide-source при его появлении;
- admin surface должна отдельно показывать `aggregator-only occurrences`.

### Case C: пост-апдейт без полного анонса

Примеры:

- одно место освободилось;
- перенос даты;
- завтра встречаемся там-то.

Решение:

- status-update не должен создавать новый occurrence;
- он должен искать ближайший active occurrence той же экскурсии и патчить поля `seats_status`, `meeting_point`, `occurrence_status`, `notes`.

### Case D: отчёт о прошедшей экскурсии как сигнал типовой экскурсии

Примеры:

- пост с рефлексией;
- разведка маршрута;
- разбор, как создавался маршрут.

Решение:

- не публиковать в digest как новый выход;
- сохранять как `template_evidence`.

### Case E: `по запросу` / `только для организованных групп`

Примеры из `excursions_profitour`:

- школьные программы с `стоимость зависит от количества человек`;
- формулировки `выбирайте удобный день`, `по вашему желанию`, `программа может быть адаптирована под возраст и интересы участников`;
- `свободная дата` внутри operator-calendar без полноценного публичного набора.

Решение:

- не создавать публичный digest item только потому, что найден красивый offer post;
- сохранять это в `GuideExcursionTemplate` как `availability_mode=on_request_private` или `mixed`;
- при наличии конкретной публичной даты и открытого набора допускается occurrence, но с явной проверкой `digest_eligible`;
- такие посты особенно важны для будущих guide/agency pages и каталога типовых экскурсий.

### Case F: co-guide указан через `@username`, а не через полное ФИО

Контрольный пример:

- `@tanja_from_koenigsberg/3935`

Сигнал:

- occurrence extraction already materializes `guide_names=["Татьяна Удовенко","Анна Туз"]`;
- в prose есть явный `@ann_tuz`, а публичный Telegram profile даёт более точное ФИО `Анастасия Туз`.

Требование:

- public digest не должен схлопываться до одного primary guide line, если occurrence явно несёт нескольких гидов;
- pipeline должен использовать public username-resolution как support-layer поверх occurrence facts, чтобы уточнять полное ФИО co-guide без выдумки.

### Case G: один source post -> несколько future occurrences -> один и тот же media кадр

Контрольный пример:

- `@amber_fringilla/5806`

Сигнал:

- один multi-announce post материализует сразу несколько отдельных occurrences;
- naive media selection берёт `refs[0]` для каждой карточки и визуально размножает одно и то же фото.

Требование:

- digest media bundle должен распределять разные `media_refs` одного source post по разным карточкам;
- если уникальных референсов не хватает, лучше недодать иллюстрацию части карточек, чем повторить одинаковый кадр 4 раза.

### Case H: `прогулка` vs `экскурсия`

Контрольный пример:

- `@amber_fringilla/5806` с `Экопрогулка...`

Сигнал:

- source title и narrative задают walking/nature framing (`прогулка`), а downstream digest может начать называть тот же выход `экскурсией`.

Требование:

- public copy должен выбирать dominant term по source title/facts и удерживать его по всей карточке;
- если signal неустойчивый, safer wording — нейтральный `маршрут` / `выход`, а не случайная подмена одного термина другим.

## 7. Какие admin-инструменты нужны без публичных страниц

Так как публичных страниц на старте не будет, admin UX обязан компенсировать это прозрачностью.

### Минимальный набор

#### `/guide_excursions`

Главное меню фичи:

- запустить scan;
- посмотреть preview digest;
- опубликовать digest;
- открыть active occurrences;
- открыть source list;
- открыть uncertain / aggregator-only / conflicts.

#### `/guide_recent [hours]`

Rolling list:

- новые occurrences;
- patched status-updates;
- template-signals;
- aggregator-only findings.

#### `/guide_sources`

Список источников с breakdown:

- тип источника;
- source health;
- свежесть последних сигналов;
- медианы views/reactions;
- coverage по типам постов.

#### `/guide_digest`

UI выбора типа дайджеста:

- new occurrences;
- few seats / last call;
- weekend soon;
- premieres / new routes.

### Что должно попасть в `/general_stats`

Нужно добавить отдельный блок `guide_monitoring`:

- runs: `success/partial/error/skipped`;
- `sources_total`, `sources_scanned`;
- `posts_scanned`, `posts_prefiltered`, `llm_checked`, `occurrences_new`, `occurrences_updated`, `status_updates`, `template_signals`;
- breakdown по source type:
  - guide_personal
  - guide_project
  - organization_with_tours
  - aggregator
- digest activity:
  - previews built
  - digests published by family
- QA counters:
  - aggregator-only active
  - unresolved conflicts
  - low-confidence occurrences.

## 8. Основная SQLite или отдельная local SQL DB

### Рекомендация на старт

Жить в основной SQLite базе.

Причины:

- текущий проект уже работает на одной SQLite;
- канонические occurrence/template/source tables логично держать рядом с текущими admin reports;
- `/general_stats`, `/a`, scheduler и ops visibility проще строить на одной базе.

### Что не надо хранить в основной базе

В основной SQLite не стоит раздувать raw-интейк:

- полные deep dumps каналов;
- отладочные OCR outputs;
- промежуточные notebook JSON;
- временные classification traces.

Это место для `artifacts/`, а не для боевой БД.

### Когда separate local DB может понадобиться

Отдельная локальная DB может стать оправданной позже, если одновременно появятся:

- несколько сотен источников;
- частые rescans;
- большой raw cache;
- отдельный независимый worker/process для guide monitoring.

Но даже в этом сценарии лучше выносить не канонические occurrence/template данные, а именно raw intake/cache layer.

Иначе возрастёт цена joins, admin-reporting и operational visibility.

## 9. Какие дайджесты целесообразны

### Публичные digest families

#### 1. `new_occurrences`

Главный стартовый дайджест.

Когда нужен:

- появились новые экскурсии/новые даты;
- маршрут или гид ещё не публиковались в guide digest.

#### 2. `last_call`

Самый полезный после `new_occurrences`.

Когда нужен:

- осталось мало мест;
- есть waitlist / sold out / освободилось место;
- экскурсия скоро и нужна срочная коммуникация.

#### 3. `weekend_soon`

Подборка по ближайшим выходным/3-7 дням.

Когда нужен:

- для читателя важна не новизна, а ближайшая полезная выборка;
- особенно полезно при плотном потоке гидов.

#### 4. `premieres_and_new_routes`

Показывает именно свежие маршруты и первые выходы.

Когда нужен:

- у гидов часто есть повторы старых маршрутов;
- редакторски интереснее выделять премьеры отдельно.

#### 5. `popular_inside_channel`

Подборка по relative popularity.

Когда нужен:

- уже накопились зрелые metrics windows;
- хочется показывать “что у гидов сейчас реально заходит”.

### Админские digest/report families

#### `aggregator_only`

Что найдено только у агрегаторов и требует проверки origin-source.

#### `status_changes`

Переносы, sold out, точка встречи, освобождение мест.

#### `uncertain_cluster`

Сомнительные случаи merge между каналом гида, соорганизатором и агрегатором.

### Порядок внедрения

1. `new_occurrences`
2. `last_call`
3. `weekend_soon`
4. `premieres_and_new_routes`
5. `popular_inside_channel`

## 10. Updated conclusions

1. Каналы надо различать не только по trust, но и по **source archetype**.
2. Для guide monitoring нужен не binary parse, а **post taxonomy**.
3. Без `status_update` и `template_signal` фича будет наполовину слепой.
4. `excursion_operator` — это отдельный archetype, а не частный случай агрегатора или `organization_with_tours`.
5. `по запросу` / `private-group-only` предложения нужно копить как template-level knowledge, но не тащить автоматически в публичные digest’ы.
6. `Для кого эта экскурсия` — один из главных классов фактов: возраст, тип группы, уровень подготовки, tempo, “местные/туристы”, семейность, профориентация, степень интерактива.
7. На старте одна SQLite база подходит; выносить лучше только raw intake/cache, если рост реально случится.
8. Помимо дайджеста “новые экскурсии” сразу имеет смысл проектировать `last_call`.
9. Без admin surfaces и блока в `/general_stats` запускать эту фичу не стоит даже без публичных страниц.
