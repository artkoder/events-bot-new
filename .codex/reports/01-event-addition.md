**/vk добавление**
- Кнопка “VK мониторинг” → “Добавить источник” ведёт в `handle_vk_add_start` → `handle_vk_add_message`: проверка superadmin, запись сообщества в таблицу `vk_source` после `vk_resolve_group` (main_part2.py:7913-8075).
- Проверка очереди постов `/vk` идёт через `handle_vk_check`, который создаёт/возобновляет batch и открывает первый пост (`_vkrev_show_next`) (main_part2.py:8694-8721, 9854-10418).
- Импорт поста выполняется из callback `vkrev:accept*` в `handle_vk_review_cb`, вызывающем `_vkrev_import_flow` (main_part2.py:10969-11060, 10152-10722). Внутри: сбор черновиков `vk_intake.build_event_drafts` и сохранение `vk_intake.persist_event_and_pages` (vk_intake.py:1788-1962), где после upsert события запускается планировщик обновлений.
- Явных функций `vk_verify`/`vk_add`/`process_vk` в кодовой базе нет; их роль выполняют связка `handle_vk_add_*` (добавление источника) и `_vkrev_import_flow` (импорт событий из очереди VK).

**/start добавление**
- `/start` авторизует пользователя и показывает меню с кнопками “Добавить событие/фестиваль” (`send_main_menu`, `handle_start`) (main.py:7353-7405).
- Кнопка “Добавить событие” ставит пользователя в сессию `add_event_sessions` и просит текст/фото (`handle_add_event_start`, main_part2.py:7779-7791); следующий ввод уходит в `enqueue_add_event` через зарегистрированный обработчик сессии (main_part2.py:13969-13983, main.py:11188-11258).
- Очередной воркер вызывает `handle_add_event`, который парсит текст (`add_events_from_text`), сохраняет событие и уведомляет автора (main.py:10892-11119). Форс-фестиваль запускается аналогично через `handle_add_festival_start`.

**Что вызывается после записи события**
- И в VK-импорте (`vk_intake.persist_event_and_pages`, vk_intake.py:1788-1962), и в ручном добавлении (`handle_add_event` → `add_events_from_text`) события проходят через `schedule_event_update_tasks`, где в `JobOutbox` ставятся задачи `telegraph_build`, `ics_publish`, `tg_ics_post`, `month_pages`, `week_pages`, `weekend_pages`, `festival_pages`, `vk_sync` (main.py:10004-10050).
- Воркер `job_outbox_worker` выбирает задачи и запускает обработчики из `JOB_HANDLERS` (main.py:11580-11742, 13620-13635), в том числе:
  - `update_telegraph_event_page` (создание/правка Telegraph),
  - `update_month_pages_for` (main.py:12859-12927) — гарантирует наличие/патч месяца через `sync_month_page` и `patch_month_page_for_date` (вызовы там же),
  - `update_week_pages_for`, `update_weekend_pages_for`, `update_festival_pages_for_event` для навигационных страниц/VK-постов.
- При недоступности токена Telegraph `update_month_pages_for` переключается на принудительный `sync_month_page` для затронутых месяцев (main.py:12885-12927).