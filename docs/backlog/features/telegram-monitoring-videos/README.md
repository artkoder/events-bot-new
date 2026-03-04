# Telegram Monitoring Videos

Статус: реализовано.

Каноника фичи: `docs/features/telegram-monitoring/README.md`.

Ключевые параметры реализации:

- лимит файла видео перед upload: `TG_MONITORING_VIDEO_MAX_MB=10`;
- режим загрузки видео в Kaggle: `TG_MONITORING_VIDEOS_SUPABASE_MODE=off|always` (по умолчанию `always`);
- safe-guard bucket для **видео**: `TG_MONITORING_VIDEO_BUCKET_SAFE_MB=430` (fail-closed через `SUPABASE_BUCKET_USAGE_GUARD_ON_ERROR=deny`);
- для фото/постеров сохраняется отдельный safe-порог `490MB` (общий guard helper по умолчанию).

Контракт `telegram_results.json`:

- `messages[].videos[]` с `sha256/size_bytes/mime_type/supabase_url/supabase_path`;
- `messages[].video_status` для операторского статуса (`supabase`, `skipped:too_large`, `skipped:bucket_guard`, и т.д.).

Импорт на сервере:

- видео прикрепляются в `event_media_asset` только если из сообщения импортировано ровно одно событие;
- если из одного сообщения импортировано несколько событий, видео не мапится (статус `skipped:multi_event_message`).
