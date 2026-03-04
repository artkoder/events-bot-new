# Geo Region Filter (Kaliningrad Oblast)

Цель: бот сейчас работает только для региона **Калининградская область**. Иногда LLM извлекает город вне региона (например, “Кисловодск”), и такие события ошибочно попадают в базу.

## Решение

Фильтрация выполняется **детерминированно** в `smart_event_update.py` (до создания/склейки события):

1) **Fast allowlist**: сверка по файлу `docs/reference/kaliningrad_oblast_places.md` (быстрый путь, не исчерпывающий).
2) **Cache**: если города нет в allowlist — проверяем `geo_city_region_cache` в SQLite.
3) **Wikidata**: если в кеше нет — делаем поиск по Wikidata и проверяем, находится ли населённый пункт в Калининградской области (через `P131*`).
4) **Gemma fallback**: если Wikidata не дала чёткого ответа (ошибка/неопределённость) — делаем “умную” проверку через Gemma и сохраняем результат в кеш.

Результат **кешируется**, чтобы не повторять запросы для одного и того же города.

## DB

Создаётся таблица (SQLite, best-effort, без Alembic):

- `geo_city_region_cache(city_norm PRIMARY KEY, is_kaliningrad_oblast, region_code, region_name, source, wikidata_qid, details, updated_at)`

## ENV

- `REGION_FILTER_ENABLED` (default: `1`) — включить/выключить фильтр.
- `REGION_FILTER_STRICT` (default: `1`) — если `1`, то “не удалось определить регион” ⇒ отклоняем (чтобы не пропускать чужие регионы).
- `REGION_FILTER_WIKIDATA_QID` (default: `Q2085`) — QID целевого региона в Wikidata (для Калининградской области).

## Где смотреть поведение

- В Telegram UI авторазбора VK, если событие отклонено фильтром: будет “Smart Update отклонил” и причина `rejected_out_of_region …`.
- Для отладки: `geo_region.py` пишет источник решения в кеш (allowlist/wikidata/gemma).
