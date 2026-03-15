# Фичи

Этот раздел содержит **канонические** описания реализованного поведения по фичам.

## Реализовано

- `docs/features/digests/README.md` — дайджесты (подборки/превью/публикация)
- `docs/features/crumple-video/README.md` — видео-анонсы `/v` (CrumpleVideo)
- `docs/features/source-parsing/README.md` — извлечение/парсинг событий из внешних источников (в т.ч. `/parse`)
- `docs/features/tourist-label/README.md` — туристическая метка (ручная разметка + экспорт)
- `docs/features/llm-gateway/README.md` — управление лимитами Google AI (Gemma/Gemini) и ретраи
- `docs/features/smart-event-update/README.md` — единый Smart Update (match/merge, лог фактов, Telegraph)
- `docs/features/linked-events/README.md` — связанные события (серии “то же событие в другие даты” + UI «Другие даты»)
- `docs/features/vk-auto-queue/README.md` — авторазбор очереди VK постов
- `docs/features/exhibitions-smart-update/README.md` — поддержка выставок через Smart Update
- `docs/features/festivals/README.md` — фестивали: серии/выпуски, очередь и связь с событиями
- `docs/features/post-metrics/README.md` — статистика постов и популярность (TG/VK): медианы, уровни ⭐/👍, основа для ранжирования
- `docs/features/ticket-sites-queue/README.md` — очередь обогащения событий по ссылкам на ticket‑сайты (pyramida/dom/qtickets)
- `docs/features/admin-action-assistant/README.md` — ассистент админ‑команд: запрос «простыми словами» → команда + подтверждение
- `docs/features/recent-imports/README.md` — отчёт `/recent_imports`: список событий, недавно созданных/обновлённых из Telegram, VK и `/parse`
- `docs/features/telegraph-cache-sanitizer/README.md` — прогрев/проверка Telegram web preview (Instant View) для Telegraph страниц + очередь на пересборку “битых”
- `docs/features/preview-3d/README.md` — генерация 3D-превью (`/3di`) через Kaggle/Blender с выгрузкой в Supabase Storage
- `docs/features/guide-excursions-monitoring/README.md` — мониторинг экскурсионных анонсов гидов, отдельный digest track и публикация в тестовый канал

## Как добавлять новую фичу

1. Создай `docs/features/<feature>/README.md`.
2. Если у фичи есть “поток задач” — заведи `docs/features/<feature>/tasks/README.md` и храни в `tasks/` ссылки на backlog items/PRs/отчёты (без дублирования текста).
3. Если у фичи есть диаграммы/скриншоты — храни в `docs/features/<feature>/assets/`.
4. Добавь запись в `docs/routes.yml`.
5. Если есть протокол/спека, но фича ещё не реализована — клади её в `docs/backlog/` (а не в `docs/features/`).
