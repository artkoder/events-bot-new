# Фичи

Этот раздел содержит **канонические** описания реализованного поведения по фичам.

## Реализовано

- `docs/features/digests/README.md` — дайджесты (подборки/превью/публикация)
- `docs/features/source-parsing/README.md` — извлечение/парсинг событий из внешних источников (в т.ч. `/parse`)
- `docs/features/tourist-label/README.md` — туристическая метка (ручная разметка + экспорт)

## Как добавлять новую фичу

1. Создай `docs/features/<feature>/README.md`.
2. Если у фичи есть “поток задач” — заведи `docs/features/<feature>/tasks/README.md` и храни в `tasks/` ссылки на backlog items/PRs/отчёты (без дублирования текста).
3. Если у фичи есть диаграммы/скриншоты — храни в `docs/features/<feature>/assets/`.
4. Добавь запись в `docs/routes.yml`.
5. Если есть протокол/спека, но фича ещё не реализована — клади её в `docs/backlog/` (а не в `docs/features/`).
