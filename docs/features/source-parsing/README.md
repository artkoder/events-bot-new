# Source parsing (извлечение событий из источников)

Фича отвечает за извлечение и обновление событий из внешних источников: театры, отдельные сайты (например pyramida.info), спецпроекты Дом искусств, а также другие источники, которые будут добавляться.

## Точки входа

- Команда супер‑админа: `/parse` (и диагностический режим `/parse check`).
- Автозапуск по расписанию: `ENABLE_SOURCE_PARSING=1` (см. `source_parsing/commands.py`).
- Из VK review UI: кнопки “Извлечь …” для ссылок на поддерживаемые источники.

## Документация по источникам

- Театры (/parse): `docs/features/source-parsing/sources/theatres/README.md`
- Дом искусств: `docs/features/source-parsing/sources/dom-iskusstv/README.md`
- Pyramida: `docs/features/source-parsing/sources/pyramida/README.md`
- Третьяковка: `docs/features/source-parsing/sources/tretyakov/README.md`
- Universal Festival Parser: `docs/features/source-parsing/sources/festival-parser/README.md`

## Артефакты

Все выгрузки/логи/результаты прогонов хранить в `artifacts/` (см. `artifacts/README.md`).

## Задачи

Связанные backlog items/планы/отчёты — в `docs/features/source-parsing/tasks/README.md` (без копирования контента).
