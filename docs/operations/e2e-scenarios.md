# E2E Scenario Index

Канонический реестр E2E/BDD сценариев проекта.

Правило обновления:
- При добавлении нового `.feature` файла или существенном изменении набора сценариев сначала обновляй эту страницу.
- Для сценариев с «живыми» внешними ссылками отмечай, насколько быстро они устаревают.

## Сценарии

| Feature file | Назначение | Тип ссылок/данных | Примечание по актуальности |
|---|---|---|---|
| [tests/e2e/features/telegram_monitoring.feature](../../tests/e2e/features/telegram_monitoring.feature) | Telegram monitoring + Smart Update (multi-source, facts/log/Telegraph) | Живые TG-посты | Требует периодического обновления контрольных ссылок и очистки данных перед прогоном |
| [tests/e2e/features/smart_event_update.feature](../../tests/e2e/features/smart_event_update.feature) | Проверка merge/dedup/conflict логики Smart Update | Преимущественно локальные фикстуры + точечные живые проверки | Стабильный, но отдельные live-кейсы нужно периодически ревизовать |
| [tests/e2e/features/vk_auto_queue.feature](../../tests/e2e/features/vk_auto_queue.feature) | Авторазбор очереди VK-постов через Smart Update | Живой VK inbox + snapshot БД | Самый «временной» сценарий: перед прогоном нужна свежая snapshot БД; дополнительно проверять, что `description` не раздувается дублированием логистики («по адресу», «по телефону», «стоимость билета») |
| [tests/e2e/features/multi_source_vk_tg.feature](../../tests/e2e/features/multi_source_vk_tg.feature) | Массовый прогон VK очереди + Telegram каналов (VK+TG merge) | Живые TG-посты + snapshot БД | Требует TG_MONITORING_LIMIT>=10; live-сценарий: быстрый вариант берёт первые 15 активных VK-постов, а `@manual` вариант прогоняет всю активную VK-очередь; Telegram Monitor (Kaggle) может занимать 30–90+ минут |
| [tests/e2e/features/dom_iskusstv.feature](../../tests/e2e/features/dom_iskusstv.feature) | Поток /parse для Дом искусств | Живой источник + локальная БД | Требует проверки доступности источника и токенов |
| [tests/e2e/features/festival_parser.feature](../../tests/e2e/features/festival_parser.feature) | Фестивальный парсер и публикация | Живые сайты/данные | Периодически ломается от изменений внешних сайтов |
| [tests/e2e/features/channel_nav.feature](../../tests/e2e/features/channel_nav.feature) | Навигация и UI-кнопки бота | Локальный бот | Стабильный, низкая зависимость от внешних данных |
| [tests/e2e/features/bot_scenarios.feature](../../tests/e2e/features/bot_scenarios.feature) | Базовые smoke-флоу бота | Локальный бот | Стабильный базовый smoke-набор |

## Live-only чеклист

- Перед прогоном сценариев на реальных постах очищать следы прошлых запусков (события/сканы/очереди).
- Для VK auto queue использовать свежий snapshot prod БД (не старше 6 часов).
- Проверять, что из отчёта доступны:
  - ссылка на Telegraph,
  - ссылка на `/log <event_id>`,
  - видимость события в нужном списке (`/events`, `/exhibitions` и т.д.).
- Для постов об отмене/переносе проверять, что событие помечается неактивным (`event.lifecycle_status=cancelled|postponed`) и исчезает из month/weekend страниц после rebuild.
