# Kinopoisk Unofficial API (metadata provider)

Источник: Swagger UI `https://kinopoiskapiunofficial.tech/documentation/api/`

> **Status:** Research notes (для будущей имплементации в Movie Showtimes)  
> **Проверено:** 2026-02-18  
> Auth: `X-API-KEY` (API key), в окружении удобно хранить как `KINOPOISK_API_UNFFICIAL_API_KEY`.

## Базовая информация

- Base URL: `https://kinopoiskapiunofficial.tech`
- OpenAPI: `openapi.json` в Swagger (версия API в spec: `2.0.1`)
- Rate limit: в описании OpenAPI указано “20 req/sec” на пользователя + отдельные лимиты по некоторым endpoint’ам (429).

## Полезные endpoint’ы

### Поиск по названию

- `GET /api/v2.1/films/search-by-keyword?keyword=...&page=...`
  - возвращает список кандидатов с `filmId`, `nameRu`, `year`, `posterUrlPreview`, `ratingVoteCount` и т.п.

### Карточка фильма (основной источник метаданных)

- `GET /api/v2.2/films/{id}`
  - поля (замечены на реальном ответе): `nameRu/nameEn/nameOriginal`, `year`, `filmLength`, `description`,
    `ratingAgeLimits`, `countries`, `genres`, `posterUrl/posterUrlPreview/coverUrl`, `webUrl`,
    `kinopoiskHDId`, `imdbId`, разные рейтинги и vote counts, `lastSync`.

### Люди (актёры/съёмочная группа)

- `GET /api/v1/staff?filmId={id}`: список `staffId`, профессии (`professionKey`), роль/персонаж (`description`), `posterUrl`.
- `GET /api/v1/staff/{id}`: карточка персоны + фильмография.

### Изображения

- `GET /api/v2.2/films/{id}/images?type=...&page=...`
  - `type` enum в OpenAPI: `STILL|SHOOTING|POSTER|FAN_ART|PROMO|CONCEPT|WALLPAPER|COVER|SCREENSHOT`

### Премьеры (важно для “вышло в этом году”)

- `GET /api/v2.2/films/premieres?year=YYYY&month=MONTH`
  - возвращает список премьер по месяцу, включая `premiereRu` (по РФ).

## Замеченные “грабли” (важно учитывать в дизайне)

- `GET /api/v2.2/films?keyword=...` и `GET /api/v2.2/films/{id}` могут возвращать разные `year` для одного `kinopoiskId`.
- `premiere*` поля в `GET /api/v2.2/films/{id}` могут быть `null`, даже если `premiereRu` присутствует в `premieres`.
- В `images` у items поле `type` может приходить `null` (даже если фильтруем по `type=POSTER`).
- `/api/v2.1/films/{id}/sequels_and_prequels` для `id=5002802` вернул 404 (нужно быть готовым к “частично доступным” endpoint’ам).

## Пример запросов (без подстановки ключа)

```bash
curl -sS -H "X-API-KEY: $KINOPOISK_API_UNFFICIAL_API_KEY" \
  "https://kinopoiskapiunofficial.tech/api/v2.1/films/search-by-keyword?keyword=%D0%9A%D0%BE%D0%BB%D0%BE%D0%BA%D0%BE%D0%BB%20%D0%9D%D0%B0%D0%B4%D0%B5%D0%B6%D0%B4%D1%8B&page=1"
```

```bash
curl -sS -H "X-API-KEY: $KINOPOISK_API_UNFFICIAL_API_KEY" \
  "https://kinopoiskapiunofficial.tech/api/v2.2/films/5002802"
```

