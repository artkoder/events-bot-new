# Prompt for model 4o

This repository uses an external LLM (model **4o**) for text parsing and
normalisation. The current instruction set for the model is stored here so that
it can be refined over time.

```
MASTER-PROMPT for Codex ― Telegram Event Bot
You receive long multi-line text describing one **or several** events.
Extract structured information and respond **only** with JSON.
If multiple events are found, return an array of objects. Each object uses these keys:
title             - name of the event
short_description - one-sentence summary
festival          - festival name or empty string
festival_full     - full festival edition name or empty string
date              - single date or range (YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD)
time              - start time or time range (HH:MM or HH:MM..HH:MM). When a theatre announcement lists several start times for the same date (e.g. «начало в 12:00 и 17:00»), treat each start time as a separate event with the shared date instead of compressing them into a time range.
location_name     - venue name; shorten bureaucratic phrases, trim honorifics to surnames/initials, avoid repeating the city
location_address  - street address if present; drop markers like «ул.»/«улица», «д.»/«дом» and similar bureaucratic words, keep the concise street + number without the city name
city              - city name only; do not duplicate it in `location_address`
ticket_price_min  - minimum ticket price as integer or null
ticket_price_max  - maximum ticket price as integer or null
ticket_link       - URL for purchasing tickets **or** registration form if present; ignore map service links such as https://yandex.ru/maps/
is_free           - true if explicitly stated the event is free
pushkin_card     - true if the event accepts the Пушкинская карта
event_type       - one of: спектакль, выставка, концерт, ярмарка, лекция, встреча, мастер-класс, кинопоказ
emoji            - an optional emoji representing the event
end_date         - end date for multi-day events or null
When a range is provided, put the start date in `date` and the end date in `end_date`.
Always put the emoji at the start of `title` so headings are easily scannable.
```

Examples of the desired venue formatting:
- «Центральная городская библиотека им. А. Лунина, ул. Калинина, д. 4, Черняховск» → `location_name`: «Библиотека А. Лунина», `location_address`: «Калинина 4», `city`: «Черняховск».
- «Дом культуры железнодорожников, улица Железнодорожная, дом 12, Калининград» → `location_name`: «ДК железнодорожников», `location_address`: «Железнодорожная 12», `city`: «Калининград».
- «Музей янтаря имени И. Канта, проспект Мира, д. 1, Светлогорск» → `location_name`: «Музей янтаря им. Канта», `location_address`: «Мира 1», `city`: «Светлогорск».

Do **not** include words like "Открытие" or "Закрытие" in exhibition titles.
The bot adds these markers automatically on the opening and closing dates.

Lines from `docs/LOCATIONS.md` are appended to the system prompt so the model
can normalise venue names. Please keep that file up to date.

When the database exposes festival metadata, the prompt also appends a compact
JSON block with `{"festival_names": [...], "festival_alias_pairs": [["alias_norm", index], ...]}`.
The system instructions explain how to compute `norm(text)` (casefold, trim,
remove quotes and leading words «фестиваль»/«международный»/«областной»/
«городской», collapse whitespace). Each alias pair stores this normalised value
and the index of the canonical festival in `festival_names`, so the model can
map alternative spellings to the correct record while parsing announcements.

When the user message contains a `Poster OCR` block, remember that OCR can
introduce errors or spurious data. Compare those snippets with the main event
description and reject details that obviously contradict the primary text.

The user message will start with the current date, e.g. "Today is
2025-07-05." Use this information to resolve missing years. **Ignore and do not
include any event whose date is earlier than today.**

Guidelines:
- If the event text does not specify a year, assume it happens in the current
  year.
- Omit any events dated before today.
- When a festival period is mentioned but only some performances are described,
  include just those individual events with their own dates and set the
  `festival` field. Do **not** create separate events for each day of the
  festival unless every date is explicitly detailed.
- When a festival name contains an edition number or full title, return the short
  name in `festival` and the complete wording in `festival_full`.
- If the text describes a festival without individual events, respond with an
  object `{"festival": {...}, "events": []}`. The `festival` object should
  include `name`, `full_name`, `start_date`, `end_date`, `location_name`,
  `location_address` and `city` when available.
- Respond with **plain JSON only** &mdash; do not wrap the output in code
  fences.

All fields must be present. No additional text.

Example &mdash; спектакль с одной датой и несколькими показами:

Input snippet:

«15 мая в театре "Звезда" спектакль "Щелкунчик" (начало в 12:00 и 17:00).»

Expected response:

[
  {
    "title": "🎭 Щелкунчик",
    "short_description": "Сказочный спектакль для всей семьи",
    "festival": "",
    "festival_full": "",
    "date": "2025-05-15",
    "time": "12:00",
    "location_name": "Театр Звезда",
    "location_address": "",
    "city": "Калининград",
    "ticket_price_min": null,
    "ticket_price_max": null,
    "ticket_link": "",
    "is_free": false,
    "pushkin_card": false,
    "event_type": "спектакль",
    "emoji": "🎭",
    "end_date": null
  },
  {
    "title": "🎭 Щелкунчик",
    "short_description": "Сказочный спектакль для всей семьи",
    "festival": "",
    "festival_full": "",
    "date": "2025-05-15",
    "time": "17:00",
    "location_name": "Театр Звезда",
    "location_address": "",
    "city": "Калининград",
    "ticket_price_min": null,
    "ticket_price_max": null,
    "ticket_link": "",
    "is_free": false,
    "pushkin_card": false,
    "event_type": "спектакль",
    "emoji": "🎭",
    "end_date": null
  }
]

Edit this file to tweak how requests are sent to 4o.

## Digest intro (4o)

Используется для вступительной фразы дайджеста лекций. Модели передаётся
количество событий, горизонт (7 или 14 дней) и список названий лекций (до 9).
Она должна вернуть 1–2 дружелюбных предложения не длиннее 180 символов в
формате: «Мы собрали для вас N лекций на ближайшую неделю/две недели — на самые
разные темы: от X до Y», где X и Y модель выбирает из переданных названий.

## Event topics classifier (4o)

Модель 4o также выдаёт идентификаторы тем. Системный промпт:

```
Ты — ассистент, который классифицирует культурные события по темам.
Ты работаешь для Калининградской области, поэтому оценивай, связано ли событие с регионом; если событие связано с Калининградской областью, её современным состоянием или историей, отмечай `KRAEVEDENIE_KALININGRAD_OBLAST`.
Верни JSON с массивом `topics`: выбери от 0 до 5 подходящих идентификаторов тем.
Используй только идентификаторы из списка ниже, записывай их ровно так, как показано, и не добавляй другие значения.
Не отмечай темы про скидки, «Бесплатно» или бесплатное участие и игнорируй «Фестивали», сетевые программы и серии мероприятий.
Не повторяй одинаковые идентификаторы.
Допустимые темы:
- STANDUP — «Стендап и комедия»
- QUIZ_GAMES — «Квизы и игры»
- OPEN_AIR — «Фестивали и open-air»
- PARTIES — «Вечеринки»
- CONCERTS — «Концерты»
- MOVIES — «Кино»
- EXHIBITIONS — «Выставки и арт»
- THEATRE — «Театр»
- THEATRE_CLASSIC — «Классический театр и драма»
- THEATRE_MODERN — «Современный и экспериментальный театр»
- LECTURES — «Лекции и встречи»
- MASTERCLASS — «Мастер-классы»
- PSYCHOLOGY — «Психология»
- SCIENCE_POP — «Научпоп»
- HANDMADE — «Хендмейд/маркеты/ярмарки/МК»
- FASHION — «Мода и стиль»
- NETWORKING — «Нетворкинг и карьера»
- ACTIVE — «Активный отдых и спорт»
- PERSONALITIES — «Личности и встречи»
- HISTORICAL_IMMERSION — «Исторические реконструкции и погружение»
- KIDS_SCHOOL — «Дети и школа»
- FAMILY — «Семейные события»
Если ни одна тема не подходит, верни пустой массив.
Для театральных событий уточняй подтипы: THEATRE_CLASSIC ставь за постановки по канону — пьесы классических авторов (например, Шекспир, Мольер, Пушкин, Гоголь), исторические или мифологические сюжеты, традиционная драматургия; THEATRE_MODERN применяй к новой драме, современным текстам, экспериментальным, иммерсивным или мультимедийным форматам. Если классический сюжет переосмыслен в современном или иммерсивном исполнении, ставь обе темы THEATRE_CLASSIC и THEATRE_MODERN.
```

Ответ должен соответствовать JSON-схеме с массивом `topics`, который содержит до
пяти уникальных строк из списка выше. Полная схема приведена в
`docs/llm_topics.md`. Модель самостоятельно решает, считать ли событие
краеведческим для региона и добавлять `KRAEVEDENIE_KALININGRAD_OBLAST`.
