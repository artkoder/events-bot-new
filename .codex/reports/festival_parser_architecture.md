**Phase 1 Report**
- `source_parsing/kaggle_runner.py`: `run_kaggle_kernel()` уже умеет инжектить `run_config`, но не передаёт `dataset_sources`; это ключевая точка для подключения двух приватных Kaggle datasets с ключами.
- `video_announce/kaggle_client.py`: `KaggleClient.push_kernel(dataset_sources=...)` уже поддерживает список источников, так что расширение runner можно сделать без изменения API клиента.
- `models.py`: `Festival` уже содержит нужную «плоскую» информацию (`name`, `description`, даты, URL-ы, `photo_urls`, `activities_json`) — это хорошая база для маппинга UDS без ломки текущего текстового потока.
- `main_part2.py`: `handle_add_festival_start()` и `add_event_session_wrapper()` всегда ведут в `enqueue_add_event()` → `handle_add_event()` → `add_events_from_text()`, то есть сейчас нет развилки для URL-парсинга.
- `main_part2.py`: `build_festival_page_content()` строит страницу из `Festival` + связанных `Event`; если событий нет — показывает заглушку, значит UDS-нужна интеграция в этот рендерер, чтобы телеграф-страницы формировались и без событий.

**Phase 2 Report**
- `source_parsing/kaggle_runner.py`: добавить параметр `dataset_sources: list[str] | None` в `run_kaggle_kernel()` и передавать его в `KaggleClient.push_kernel(...)`; так вы сможете подключать 2 приватных datasets с ключами для Gemma.
- `video_announce/kaggle_client.py`: опционально расширить `deploy_kernel_update()` до `dataset_sources` (список), если планируется деплой «на лету» с обновлёнными источниками; иначе достаточно `push_kernel()`.
- `kaggle/UniversalFestivalParser/`: рекомендую новый kernel folder с `kernel-metadata.json` и ноутбуком; включить `"dataset_sources": ["owner/key-cipher", "owner/key-fernet"]`.
- Архитектура ноутбука (RDR):
- `kaggle/UniversalFestivalParser/src/config.py`: разбор `FESTIVAL_URL`, `RUN_ID`, `DEBUG`, путей `/kaggle/input/*`.
- `kaggle/UniversalFestivalParser/src/render.py`: Playwright (headless Chromium, `--no-sandbox`), `page.goto(url, wait_until="networkidle")`, сохранение `render.html`, `screenshot.png`, `network.har` (опционально).
- `kaggle/UniversalFestivalParser/src/distill.py`: `trafilatura`/`readability` + JSON-LD extraction → `distilled.json`.
- `kaggle/UniversalFestivalParser/src/reason.py`: Gemma 3-27B клиент, промпт → UDS JSON.
- `kaggle/UniversalFestivalParser/src/rate_limit.py`: лимитер 30 RPM, 15K TPM (token bucket); для 14.4K RPD — счётчик в `usage.json` (выходит вместе с артефактами; контроль на стороне бота).
- `kaggle/UniversalFestivalParser/src/uds.py`: Pydantic-схема, `uds_version`, `source_url`, `extracted_at`, `confidence`, `festival{...}`, `activities[...]`.
- Безопасность ключа:
- Dataset A: `google_api_key.enc` (Fernet ciphertext) + `meta.json` (алгоритм/версии).
- Dataset B: `fernet.key` (base64), не класть в notebook.
- Декрипт в памяти (`cryptography.fernet.Fernet`), не писать ключ на диск.
- UDS→Festival маппинг (минимально инвазивно):
- Оставить текущие поля `Festival` и заполнять их из UDS: `name`, `full_name`, `description`, `start_date`, `end_date`, `website_url`, `program_url`, `ticket_url`, `vk_url`, `tg_url`, `location_name/address/city`, `photo_url/photo_urls`.
- Новые поля, если нужны: `uds_json` (JSONB), `source_url`, `parser_version`, `parser_run_id`, `last_parsed_at` в `models.py` — все опциональные, чтобы не ломать текущий текстовый флоу.
- Альтернатива без изменения `Festival`: новая таблица `FestivalParse` (UDS + мета + ссылки на артефакты).

**Phase 3 Report**
- `main_part2.py`: в `handle_add_festival_start()` текст ответа заменить на «текст или URL»; в `add_event_session_wrapper()` добавить развилку: если `session_mode == "festival"` и сообщение содержит URL → запуск нового parser flow; иначе старый `enqueue_add_event()` сохраняется.
- `main.py`: добавить функцию `upsert_festival_from_uds(...)` (или аналог) с маппингом UDS→`Festival`, затем вызвать `sync_festival_page()` и `rebuild_fest_nav_if_changed()`; это сохранит текущий флоу и добавит новый.
- `main_part2.py`: расширить `build_festival_page_content()` — если `events` пусты и есть `fest.activities_json`/`uds_json`, рендерить расписание из UDS (иначе остаётся текущая заглушка).
- Supabase Storage:
- Добавить `SUPABASE_PARSER_BUCKET` (например `events-parser`), чтобы не смешивать с `events-ics`.
- Путь версии: `festival/{slug}/{run_id}/v1/uds.json`, `render.html`, `screenshot.png`, `distilled.json`, `metrics.json`.
- `run_id` = `YYYYMMDDTHHMMSSZ` + короткий hash URL; хранить `latest_parse_path` в `Festival` или в `FestivalParse`.
- Интеграции с nav/страницами:
- После успешного UDS-upsert вызвать `sync_festival_page()` + `rebuild_fest_nav_if_changed()` (уже инициирует `JobTask.fest_nav_update_all`).
- Если UDS создаёт реальные `Event` по дням — дополнительно `schedule_event_update_tasks()` для каждого, чтобы `festival_pages` подтянулись автоматически.
- Тестирование:
- Unit: маппинг UDS→Festival (`tests/test_festival_uds_mapping.py`).
- Unit: определение URL/текста в `handle_add_festival_start`/`add_event_session_wrapper` (`tests/test_festival_url_entry.py`).
- Integration: `source_parsing` runner инжект `dataset_sources` и `run_config` (`tests/test_kaggle_runner_config.py`).
- E2E: мок UDS JSON → построение Telegraph страницы и nav (`tests/test_festival_pages_from_uds.py`).

**Конкретные изменения (сводно)**
- `source_parsing/kaggle_runner.py`: `run_kaggle_kernel()` → добавить `dataset_sources` и прокинуть в `KaggleClient.push_kernel()`; расширить инжект `run_config` под `festival_url`.
- `video_announce/kaggle_client.py`: при необходимости — `deploy_kernel_update()` с `dataset_sources: list[str] | None`.
- `kaggle/UniversalFestivalParser/`: новый kernel folder, `kernel-metadata.json`, notebook + `src/*` (RDR, rate limiter, UDS, decrypt).
- `models.py`: `Festival` опционально расширить (`uds_json`, `source_url`, `parser_version`, `parser_run_id`, `last_parsed_at`).
- `main_part2.py`: `handle_add_festival_start()` + `add_event_session_wrapper()` — развилка URL/текст; `build_festival_page_content()` — рендер `activities_json`/`uds_json`.
- `main.py`: добавить `upsert_festival_from_uds()` и вызовы `sync_festival_page()` + `rebuild_fest_nav_if_changed()` после UDS.

Если хотите, дальше могу:
1) зафиксировать точную UDS-схему (JSON/py-представление),
2) набросать структуру нового Kaggle kernel,
3) предложить конкретные сигнатуры `upsert_festival_from_uds()` и `handle_add_festival_url()`.