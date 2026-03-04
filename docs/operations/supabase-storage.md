# Supabase Storage (buckets)

Этот документ фиксирует **текущее** использование Supabase Storage в проекте и спецификацию
планового разделения на два bucket'а: отдельно для **ICS** и отдельно для **медиа** (постеры/картинки).

Цели:

- не ломать старые ссылки (`ics_url`, `eventposter.supabase_url`);
- избежать "массовой миграции" объектов: старые остаются как есть, новые пишем в новый bucket;
- сделать rollout/rollback без простоя;
- минимизировать риски по публичности и стоимости `listing`.

## Текущее состояние (as-is)

- `SUPABASE_BUCKET` (default: `events-ics`) используется как **общий** bucket для:
  - ICS файлов календаря (`event.ics_url`);
  - legacy fallback-афиш (если `SUPABASE_MEDIA_BUCKET` не задан).
- `SUPABASE_MEDIA_BUCKET` используется для медиа (афиши/картинки):
  - Telegram Monitoring (Kaggle) пишет постеры в `SUPABASE_MEDIA_BUCKET`;
  - server-side media upload pipeline (`upload_images`, используется в `/addevent`, VK auto import и source parsers) по умолчанию работает в режиме `UPLOAD_IMAGES_SUPABASE_MODE=prefer` (сначала Supabase, затем Catbox fallback).
- `SUPABASE_PARSER_BUCKET` (default: `festival-parsing`) уже отдельный bucket для артефактов festival parser.

Важно: очистка `cleanup_old_events` сейчас удаляет объекты из **одного** bucket'а `SUPABASE_BUCKET` и тем самым
подразумевает, что ICS и постеры лежат вместе.

## Целевое состояние (to-be): 2 bucket'а

### Новые ENV (предлагаемые)

- `SUPABASE_ICS_BUCKET`
  - bucket для ICS файлов календаря.
  - default: `SUPABASE_BUCKET` (если задан) иначе `events-ics`.
- `SUPABASE_MEDIA_BUCKET`
  - bucket для медиа (постеры/картинки), в первую очередь Telegram Monitoring fallback в Supabase Storage.
  - default: `SUPABASE_BUCKET` (если задан) иначе `events-ics`.
- `SUPABASE_BUCKET`
  - legacy env; остаётся для обратной совместимости и как fallback по умолчанию.
- `SUPABASE_PARSER_BUCKET`
  - не меняется; festival parser остаётся в своём bucket'е.

Резолюция bucket'а (правило совместимости):

- Если `SUPABASE_ICS_BUCKET`/`SUPABASE_MEDIA_BUCKET` не заданы, код использует `SUPABASE_BUCKET`.
- Если и `SUPABASE_BUCKET` не задан, дефолт остаётся `events-ics` (как сейчас).

### Форматы ключей (paths)

- ICS (server):
  - key: `event-<event_id>-<YYYY-MM-DD>.ics` (или `event-<event_id>.ics` при невалидной дате)
  - хранится как публичный URL в `event.ics_url`.
- Постеры (Kaggle `TelegramMonitor`, Supabase fallback):
  - key: `<prefix>/dh16/<first2>/<dhash>.webp`, где `prefix` default `p`
  - `dhash` — перцептивный dHash16 (content‑addressed ключ): одинаковая картинка с разным разрешением/реэнкодом
    должна попадать в один и тот же ключ → Supabase не хранит дубли.
  - В Storage сохраняются только WebP (без JPEG); качество: `TG_MONITORING_POSTERS_WEBP_QUALITY` (Kaggle) /
    `SUPABASE_POSTERS_WEBP_QUALITY` (server-side), default `82`.
  - сохраняется как `eventposter.supabase_url` + `eventposter.supabase_path`.
- Видео (Kaggle `TelegramMonitor`):
  - key: `v/sha256/<first2>/<sha256>.<ext>` (content‑addressed дедуп)
  - legacy fast‑path: producer может reuse уже существующий объект `v/tg/<document.id>.<ext>` без повторной скачки.
- 3D previews (Kaggle `Preview3D`, `/3di`):
  - key: `<prefix>/event/<event_id>.webp`, где `prefix` default `p3d` (`SUPABASE_PREVIEW3D_PREFIX`)
  - хранится как публичный URL в `event.preview_3d_url` (в `SUPABASE_MEDIA_BUCKET`)
  - event-scoped key: при пересборке превью перезаписывается (`x-upsert: true`)
- Telegraph cover mirrors (server-side, best-effort):
  - зачем: Telegram не всегда генерирует `webpage.cached_page`, когда первое изображение на Telegraph‑странице — WEBP
  - key: `<prefix>/sha256/<first2>/<sha256>.jpg`, где `prefix` default `tgcover` (`SUPABASE_TELEGRAPH_COVER_PREFIX`)
  - хранится как публичный URL и используется только как **cover** (первое изображение) на Telegraph страницах событий/месяцев/выходных
- Festival parser:
  - key: `festival_parsing/<festival_slug>/<run_id>/...` в `SUPABASE_PARSER_BUCKET`.

## Публичность и доступ (RLS / public buckets)

Практическая логика проекта:

- ICS ссылки должны открываться из браузера/календаря без авторизации.
- Изображения должны загружаться внутри Telegraph страниц без авторизации.

Отсюда:

- `SUPABASE_ICS_BUCKET`: обычно нужен **public** (или эквивалентный публичный доступ на чтение).
- `SUPABASE_MEDIA_BUCKET`: тоже обычно нужен **public**, если URL вставляется в Telegraph/внешние страницы.

Если требуется **private** для медиа:

- придётся менять архитектуру: прокси-сервис/подпись URL (signed URLs) и обновление Telegraph/страниц при истечении.
  Это не совместимо с долгоживущими ссылками "как есть".

## Rollout без простоя (миграция)

1. Создать новый bucket для медиа (например `events-media`) и настроить публичность/политики.
2. Задеплоить код, который:
   - пишет ICS в bucket `SUPABASE_ICS_BUCKET`;
   - пишет постеры (Telegram Monitoring) в `SUPABASE_MEDIA_BUCKET` (producer Kaggle) и сохраняет URL/paths в БД;
   - при очистке удаляет ICS и медиа **из соответствующих bucket'ов** (не из одного общего);
   - корректно обрабатывает старые записи, где bucket можно понять только из сохранённого public URL.
3. Включить новый bucket на проде через ENV:
   - `SUPABASE_ICS_BUCKET=events-ics` (или оставить пустым, если legacy `SUPABASE_BUCKET=events-ics`);
   - `SUPABASE_MEDIA_BUCKET=events-media`.
4. Обновить доставку env в Kaggle (Telegram Monitoring):
   - прокинуть `SUPABASE_MEDIA_BUCKET` в Kaggle runtime (через encrypted datasets/config или env passthrough).
5. Проверки:
   - создание/обновление события генерирует `ICS:` ссылку и она открывается;
   - Telegram Monitoring при `TG_MONITORING_POSTERS_SUPABASE_MODE=always|fallback` пишет `eventposter.supabase_*`
     с bucket'ом `SUPABASE_MEDIA_BUCKET`;
   - очистка удаляет объекты из обоих bucket'ов (best-effort) и не падает.

## Rollback

- Снять/очистить `SUPABASE_MEDIA_BUCKET` (и при необходимости `SUPABASE_ICS_BUCKET`) чтобы вернуться к legacy `SUPABASE_BUCKET`.
- Никакой "массовой миграции" объектов не требуется:
  - старые ссылки продолжают работать;
  - новые объекты, уже записанные в новом bucket'е, остаются доступными по сохранённым URL.

Ограничение rollback: старый код очистки может не удалять объекты из нового bucket'а. Поэтому rollback лучше делать
только после деплоя совместимой версии cleanup.

## Риски и подводные камни (коротко)

- **Очистка**: удаление объектов нужно делать per-bucket (по `event.ics_url` и `eventposter.supabase_url/path`), иначе
  медиа bucket будет "утекать" по объёму.
- **Listing cost**: подсчёт размера bucket через `storage.list` рекурсивно может быть дорогим на медиа bucket'е.
  Рекомендация: считать редко, кэшировать, или считать только нужные префиксы/ICS bucket.
- **Обратная совместимость URL**: не переписывать старые `ics_url`/`supabase_url` без необходимости; хранить URL как источник истины.
- **Перцептивный хеш (dHash)**: это best-effort дедуп по «похожести». Коллизии теоретически возможны
  (разные картинки → один хеш), но при `dHash16` риск практический минимален.
- **Shared bucket (PROD/TEST)**: если один `SUPABASE_MEDIA_BUCKET` используется двумя окружениями с разными SQLite БД,
  автоматическое удаление медиа из `cleanup_old_events` может ломать второе окружение. Рекомендация: отключить удаление
  медиа на тесте (`SUPABASE_MEDIA_DELETE_ENABLED=0`) и считать bucket глобальным кэшем (объём контролируется guard + fallback).
- **Public buckets**: любой public bucket делает контент доступным всем по URL; оценить юридические/контентные риски.

## Контроль объёма (bucket usage guard)

В коде есть хелпер для проверки, что bucket “не переполнен” перед загрузкой медиа (например, коротких видео):

- модуль: `supabase_storage.py`
- API: `check_bucket_usage_limit(...)` / `require_bucket_usage_limit(...)`
- дефолтный безопасный лимит: **490MB** (чуть ниже 500MB, чтобы оставить запас под метаданные/гонки/кэш)
- кэш в процессе (in-process): по умолчанию `cache_sec=600` (10 минут)

ENV-обёртка (удобно для Kaggle/операций): `check_bucket_usage_limit_from_env(...)`.

ENV:

- `SUPABASE_BUCKET_USAGE_GUARD_MAX_USED_MB` (default `490`)
- `SUPABASE_BUCKET_USAGE_GUARD_CACHE_SEC` (default `600`)
- `SUPABASE_BUCKET_USAGE_GUARD_ON_ERROR` (`deny|allow`, default `deny`)

Для Telegram Monitoring videos в Kaggle используется отдельный (более строгий) safe-порог:

- `TG_MONITORING_VIDEO_BUCKET_SAFE_MB` (default `430`).

Для фото/постеров сохраняется базовый safe-порог `490MB` (дефолт общего helper-а).

Ограничения:

- это “best-effort” оценка по `storage.list`, то есть на больших bucket'ах может быть медленно;
- нет строгой защиты от конкурентных загрузок (возможен небольшой “перелёт” лимита).

## Режимы server-side загрузки картинок (`upload_images`)

- `UPLOAD_IMAGES_SUPABASE_MODE=prefer` (default): сначала попытка загрузки в Supabase, затем fallback в Catbox.
- `UPLOAD_IMAGES_SUPABASE_MODE=fallback`: сначала Catbox, Supabase только при неуспехе Catbox (legacy-поведение).
- `UPLOAD_IMAGES_SUPABASE_MODE=only`: только Supabase.
- `UPLOAD_IMAGES_SUPABASE_MODE=off`: Supabase отключён для этого pipeline.
- Telegraph upload fallback не поддерживается: `telegra.ph/upload` недоступен/устарел, поэтому используем только Supabase/Catbox.

## Гарантированная очистка Supabase (durable delete queue)

Исторически проблема: если Supabase временно недоступен во время `cleanup_old_events`, объекты могли “утечь” —
события уже удалены из SQLite, а ссылки на объекты потеряны.

Теперь удаление объектов **persisted**:

- таблица: `supabase_delete_queue` (SQLite)
- `cleanup_old_events` перед удалением событий кладёт туда `(bucket, path)` для:
  - ICS (`event.ics_url`)
  - постеров (`eventposter.supabase_url/supabase_path`) — если `SUPABASE_MEDIA_DELETE_ENABLED=1`
  - прочих медиа-ассетов (`event_media_asset`, включая Telegram Monitoring videos) — если `SUPABASE_MEDIA_DELETE_ENABLED=1`
- затем (best-effort) пытается удалить из Supabase и убрать строки из очереди;
- если удаление не удалось — строки остаются и будут удалены на следующем запуске cleanup’а.

Важно: при включённом удалении медиа `cleanup_old_events` дополнительно проверяет, что `supabase_path` больше не
встречается в текущей БД (иначе объект общий/дедупнутый и удалять его нельзя).

## Ручная очистка медиа (scripts)

Иногда нужно быстро очистить Supabase Storage от “старых” картинок в **тесте/локально** (без удаления событий из SQLite):

- Только 3D previews (`/3di`, prefix `p3d`):
  - dry-run: `python scripts/cleanup_preview3d_supabase.py --db /tmp/db.sqlite`
  - apply: `python scripts/cleanup_preview3d_supabase.py --db /tmp/db.sqlite --apply`
- Постеры + 3D previews (media bucket, prefixes `p` + `p3d` по умолчанию):
  - dry-run: `python scripts/cleanup_supabase_media_images.py --db /tmp/db.sqlite`
  - apply (orphans): `python scripts/cleanup_supabase_media_images.py --db /tmp/db.sqlite --apply`
  - apply (полный reset префиксов): `python scripts/cleanup_supabase_media_images.py --db /tmp/db.sqlite --mode purge --apply`

## Аудит дедупликации медиа (постеры + видео)

Чтобы проверить, что за последние сутки новые медиа‑объекты пишутся в каноничных ключах и не создаются дубли:

- DB‑аудит (быстро, без сети):
  - `python scripts/inspect/audit_media_dedup.py --db /tmp/db.sqlite --hours 24`
- DB + проверка Storage (HEAD по объектам, медленнее, но надёжнее):
  - `python scripts/inspect/audit_media_dedup.py --db /tmp/db.sqlite --hours 24 --check-storage`

Скрипт проверяет инварианты:

- Постеры: `*.webp`, каноничный key `<prefix>/dh16/<first2>/<dhash>.webp`, согласованность `eventposter.phash` и `eventposter.supabase_path`.
- Видео: отсутствие конфликтов `sha256 → path`, корректный `Content-Type` в Storage; legacy `v/tg/*` допускается как reuse, но новые `v/tg/*` за окно подсвечиваются при `--check-storage`.
