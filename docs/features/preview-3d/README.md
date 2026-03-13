# Preview 3D (`/3di`)

Фича отвечает за генерацию 3D-превью для событий через Blender в Kaggle и запись результата в `event.preview_3d_url`.

## Точки входа

- Ручной запуск супер-админом: `/3di`
- Автозапуск по расписанию: `ENABLE_3DI_SCHEDULED=1`
  - время: `THREEDI_TIMES_LOCAL` (default `05:30,15:15,17:15`)
  - таймзона: `THREEDI_TZ` (default `Europe/Kaliningrad`)

## Как работает pipeline

1. Бот отбирает события без `preview_3d_url` или по выбранному режиму `/3di`.
   - `📅 Выбрать месяц` по умолчанию тоже берёт только события выбранного месяца без `preview_3d_url`.
   - После выбора месяца бот предлагает размер батча: `25`, `50`, `100` или `все`, чтобы не отправлять в один Kaggle-run весь длинный месяц.
   - Для явной массовой пересборки текущего месяца используется отдельная кнопка `🔄 Перегенерировать все`.
2. Для Kaggle создаётся временный dataset с `payload.json`.
3. Отдельно создаются приватные runtime datasets:
   - `config.json` + `secrets.enc` в cipher dataset;
   - `fernet.key` в отдельном key dataset.
4. Перед `kernels_push` бот ждёт propagation datasets и использует тот же `KaggleClient.push_kernel(...)`,
   что и Telegram Monitoring, чтобы `dataset_sources` применялись одинаково.
5. Notebook `kaggle/Preview3D/preview_3d.ipynb` загружает `config.json` и расшифровывает секреты из `/kaggle/input` до начала рендера.
6. После рендера WebP-preview загружается в Supabase Storage по ключу
   `SUPABASE_PREVIEW3D_PREFIX/event/<event_id>.webp` в `SUPABASE_MEDIA_BUCKET`.
   Runtime payload прокидывает совместимые alias-переменные `SUPABASE_KEY`/`SUPABASE_SERVICE_KEY`
   и `SUPABASE_BUCKET`/`SUPABASE_MEDIA_BUCKET`, чтобы upload не отключался на старых notebook/runtime ветках.
7. Бот сохраняет public URL в `event.preview_3d_url` и ставит downstream-задачи на обновление страниц события.

## Обязательные ENV

- `KAGGLE_USERNAME`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY` или `SUPABASE_KEY`
- `SUPABASE_MEDIA_BUCKET` опционален
  - fallback: `SUPABASE_BUCKET`
  - default: `events-ics`
- `SUPABASE_PREVIEW3D_PREFIX` опционален
  - default: `p3d`

Если Supabase env отсутствуют, `/3di` должен падать до запуска Kaggle, а не после многоминутного рендера.

## Отладка

- `PREVIEW3D_KEEP_DATASETS=1` отключает cleanup временных Kaggle datasets после прогона.
- Хранилище и формат ключей описаны в `docs/operations/supabase-storage.md`.
- Общий паттерн доставки секретов в Kaggle описан в `docs/operations/kaggle-secrets.md`.

## UI режимы `/3di`

- `🆕 Только новые` — gap-fill для недавних событий без `preview_3d_url`.
- `🌐 All missing` — все будущие события без `preview_3d_url`.
- `⚡️ Сгенерировать (текущий мес)` — текущий месяц, только missing.
- `🔄 Перегенерировать все` — текущий месяц целиком, включая уже готовые превью.
- `📅 Выбрать месяц (без превью)` — выбор месяца и затем размера батча (`25/50/100/все`) для missing-only рендера.
