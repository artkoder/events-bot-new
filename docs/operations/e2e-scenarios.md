# E2E Scenario Index

Канонический реестр E2E/BDD сценариев проекта.

Правило обновления:
- При добавлении нового `.feature` файла или существенном изменении набора сценариев сначала обновляй эту страницу.
- Для сценариев с «живыми» внешними ссылками отмечай, насколько быстро они устаревают.

## Сценарии

| Feature file | Назначение | Тип ссылок/данных | Примечание по актуальности |
|---|---|---|---|
| [tests/e2e/features/telegram_monitoring.feature](../../tests/e2e/features/telegram_monitoring.feature) | Telegram monitoring + Smart Update (multi-source, facts/log/Telegraph) | Живые TG-посты | Требует периодического обновления контрольных ссылок и очистки данных перед прогоном |
| [tests/e2e/features/telegram_sources_seed.feature](../../tests/e2e/features/telegram_sources_seed.feature) | Проверка канонических Telegram-источников (seed + список + пагинация) | Локальная БД + UI | Стабильный, но зависит от UI `/tg` и `sources.yml` |
| [tests/e2e/features/smart_event_update.feature](../../tests/e2e/features/smart_event_update.feature) | Проверка merge/dedup/conflict логики Smart Update | Преимущественно локальные фикстуры + точечные живые проверки | Стабильный, но отдельные live-кейсы нужно периодически ревизовать |
| [tests/e2e/features/vk_auto_queue.feature](../../tests/e2e/features/vk_auto_queue.feature) | Авторазбор очереди VK-постов через Smart Update | Живой VK inbox + snapshot БД | Самый «временной» сценарий: перед прогоном нужна свежая snapshot БД; дополнительно проверять, что `description` не раздувается дублированием логистики («по адресу», «по телефону», «стоимость билета») |
| [tests/e2e/features/multi_source_vk_tg.feature](../../tests/e2e/features/multi_source_vk_tg.feature) | Массовый прогон VK очереди + Telegram каналов (VK+TG merge) | Живые TG-посты + snapshot БД | Требует TG_MONITORING_LIMIT>=10; live-сценарий: быстрый вариант берёт первые 15 активных VK-постов, а `@manual` вариант прогоняет всю активную VK-очередь; Telegram Monitor (Kaggle) может занимать 30–90+ минут |
| [tests/e2e/features/dom_iskusstv.feature](../../tests/e2e/features/dom_iskusstv.feature) | Поток /parse для Дом искусств | Живой источник + локальная БД | Требует проверки доступности источника и токенов |
| [tests/e2e/features/festival_parser.feature](../../tests/e2e/features/festival_parser.feature) | Фестивальный парсер и публикация | Живые сайты/данные | Периодически ломается от изменений внешних сайтов |
| [tests/e2e/features/festival_queue.feature](../../tests/e2e/features/festival_queue.feature) | Очередь фестивалей (handoff из VK auto import и Telegram Monitoring → Festival Queue → Festival Parser, индексы и иллюстрации) | Живые VK/TG/сайты | Требует актуальных ссылок на посты (VK: wall-132625599_16376, TG: @world_ocean_museum/11572) и проверку страницы «Фестивали» |
| [tests/e2e/features/festival_queue_prefilled.feature](../../tests/e2e/features/festival_queue_prefilled.feature) | Очередь фестивалей на уже заполненной `festival_queue` (накопленные данные из массовых прогонов, snapshot 20–21.02.2026) | Snapshot БД + live Telegraph | Стабильнее, чем live-ссылки: сценарий явно выставляет статус `pending` для контрольных URL; TG-прогон ограничен `--limit=2` (каноничные посты `kaliningradartmuseum/7589` и `klassster/17700`) чтобы не зависеть от нового хвоста очереди |
| [tests/e2e/features/channel_nav.feature](../../tests/e2e/features/channel_nav.feature) | Навигация и UI-кнопки бота | Локальный бот | Стабильный, низкая зависимость от внешних данных |
| [tests/e2e/features/admin_action_assistant.feature](../../tests/e2e/features/admin_action_assistant.feature) | Admin Action Assistant: RU‑запрос → команда → подтверждение (/assist,/a, кнопка меню) | Live UI + LLM (Gemma) | Стабильный smoke: использует только безопасные команды; зависит от доступности Google AI и allowlist ассистента |
| [tests/e2e/features/guide_excursions.feature](../../tests/e2e/features/guide_excursions.feature) | Guide excursions monitoring: `/guide_excursions` → full scan → preview → publish digest в `@keniggpt` | Live TG + test channel | Live manual scenario: зависит от Telethon auth, Bot API publish access в `@keniggpt`, актуальности source-каналов и того, что локальный bot runtime является единственным `getUpdates` consumer для этого токена |
| [tests/e2e/features/bot_scenarios.feature](../../tests/e2e/features/bot_scenarios.feature) | Базовые smoke-флоу бота | Локальный бот | Стабильный базовый smoke-набор |
| [tests/e2e/features/release_smoke_smart_update.feature](../../tests/e2e/features/release_smoke_smart_update.feature) | Smoke релиза: /tg (trust+festival), VK auto import (limit=10), псевдо‑фестиваль «Масленица» из holidays, Telegram Monitoring (>=10), Telegraph‑картинки (Supabase WebP / Catbox fallback), ticket-sites queue (Pyramida), /start→«Добавить событие» (Smart Update source-log), добавление фестиваля | Live UI (VK/TG/Kaggle/Telegraph) | Перед запуском обновить ENV/доступы и убедиться, что есть реальные события/посты в окне сканирования; сценарий по «Масленице» использует VK посты: wall-53460968_11095, _11053, _11099, wall-41284227_7045; часть сценариев помечена `@manual` (даты /parse, ручные флоу) |
| [tests/e2e/features/release_multisource_control.feature](../../tests/e2e/features/release_multisource_control.feature) | Контроль multi-source (VK+TG+/parse) под дату релиза | Live + DB‑проверки | Шаблон/ручной сценарий для финальной валидации “один event_id из трёх источников”; требует актуальных данных в БД |

## Live-only чеклист

- Перед прогоном сценариев на реальных постах очищать следы прошлых запусков (события/сканы/очереди).
- Для VK auto queue использовать свежий snapshot prod БД (не старше 6 часов).
- Проверять, что из отчёта доступны:
  - ссылка на Telegraph,
  - ссылка на `/log <event_id>`,
  - видимость события в нужном списке (`/events`, `/exhibitions` и т.д.).
- Для постов об отмене/переносе проверять, что событие помечается неактивным (`event.lifecycle_status=cancelled|postponed`) и исчезает из month/weekend страниц после rebuild.
