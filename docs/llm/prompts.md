# Prompt for model 4o

This repository uses an external LLM (model **4o**) for text parsing and
normalisation. The current instruction set for the model is stored here so that
it can be refined over time.

Note: this prompt is used by the **draft extraction/parsing** flow (VK/TG → JSON).
The default backend is **Gemma via Google AI** (4o is supported as an optional fallback); see `main.py:parse_event_via_llm` and `docs/features/llm-gateway/README.md`.
Smart Update (merge/match/rewrite/facts) uses **Gemma via Google AI** with 4o as a fallback only when Gemma fails; see `docs/features/smart-event-update/README.md`.
Important: `parse_event_via_llm` reads only the fenced `MASTER-PROMPT` block below for event parsing.
The other sections in this file document separate prompts/workflows and must not be appended to the event-parse system prompt.

```
MASTER-PROMPT for Codex ― Telegram Event Bot
You receive long multi-line text describing one **or several** events.
Extract structured information and respond **only** with JSON.
If multiple events are found, return an array of objects. Each object uses these keys:
title             - name of the event
short_description - **REQUIRED** one-sentence summary of the event (see **short_description** rules below)
festival          - festival name or empty string
festival_full     - full festival edition name or empty string
date              - single date or range (YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD)
time              - start time or time range (HH:MM or HH:MM..HH:MM). When a theatre announcement lists several start times for the same date (e.g. «начало в 12:00 и 17:00»), treat each start time as a separate event with the shared date instead of compressing them into a time range.
location_name     - venue name; shorten bureaucratic phrases, trim honorifics to surnames/initials, avoid repeating the city
If the venue is listed in the appended reference from ../reference/locations.md, copy the
`location_name` exactly as it appears there.
location_address  - street address if present; drop markers like «ул.»/«улица», «д.»/«дом» and similar bureaucratic words, keep the concise street + number without the city name
city              - city name only; do not duplicate it in `location_address`
ticket_price_min  - minimum ticket price as integer or null
ticket_price_max  - maximum ticket price as integer or null
ticket_link       - URL for purchasing tickets **or** registration form if present; ignore map service links such as https://yandex.ru/maps/
is_free           - true if explicitly stated the event is free. Exception: blood donation actions (“День донора”, “донорская акция”, “сдача крови”, “станция переливания крови”) are free-to-attend — set `is_free=true` even if “бесплатно” is not written.
pushkin_card     - true if the event accepts the Пушкинская карта
event_type       - one of: спектакль, выставка, концерт, ярмарка, лекция, встреча, мастер-класс, кинопоказ, спорт
emoji            - an optional emoji representing the event
end_date         - end date for multi-day events or null
search_digest    - search summary text (see guidelines below)
When a range is provided, put the start date in `date` and the end date in `end_date`.
Always put the emoji at the start of `title` so headings are easily scannable.

**Money / ticket price rules (important):**
- `ticket_price_min/max` must describe the **cost to attend** (tickets/entry/participation fee).
- Do NOT treat money paid **to participants** as a ticket price: `компенсация`, `вознаграждение`, `выплата`, `гонорар`, `приз`, `подарок`, cashback/кэшбэк.
- For blood donation actions, donor compensation amounts (e.g. “компенсация 1063 руб.”) are NOT tickets: keep `ticket_price_min/max=null` and set `is_free=true`.

**title** rules:
- The title MUST be grounded in the source text (or poster OCR if provided). Do not invent names, nicknames, or weird words that do not appear in the input.
- If the post does not contain an explicit name, use a neutral descriptive title based on `event_type` and the venue (e.g. "Выставка — Музей …", "Лекция — …"), but still do NOT introduce new terms.
- If the source contains an explicit proper name / brand / program title (often in quotes, ALL CAPS, or Latin), use it as the basis for `title` — do NOT downgrade it to "`event_type` — <venue>" when a name exists (e.g. "ЕвроДэнс'90", not "Концерт — Янтарь холл").
- If the source clearly describes a standup/comedy show (e.g. contains “стендап”, “stand-up”, “комик”), but the show name is metaphorical or misleading, make the format explicit in the title (e.g. "Стендап: <название>"). Keep `event_type` as `концерт` (closest available) and prefer 🎤 as `emoji` when appropriate.
- Avoid typos and nonsense tokens (e.g. made-up 3–4 letter words). If in doubt, simplify the title.

**short_description** rules:
This field is **REQUIRED** for every event — never return an empty string.
Generate exactly one Russian sentence summarizing what the event IS ABOUT.
Strict constraints:
- Exactly ONE sentence, no line breaks.
- MUST be a summary/description of the event content, NOT a copy of the source text.
- Do NOT include: date, time, address, ticket prices, phone numbers, URLs.
- Do NOT use promotional language or calls to action.
- Keep it concise: 12-16 words.
- Write in third person, neutral tone.
Good examples:
- "Концерт камерной музыки с произведениями Баха и Вивальди в исполнении калининградских музыкантов."
- "Спектакль по мотивам романа Достоевского о судьбе молодого человека в большом городе."
- "Мастер-класс по изготовлению традиционных янтарных украшений для начинающих."
Bad examples (do NOT write like this):
- "Приходите на концерт!" (call to action)
- "12 января в 19:00" (date/time)
- "Подробности по ссылке" (URL reference)
- "" (empty — NEVER allowed)

**search_digest** rules:
Generate a single Russian sentence in a formal neutral style for extended search.
Strict constraints:
- No promotional language, emotions, calls to action, or subjective adjectives.
- Do NOT include: city, address/location, date, time (HH:MM), schedule, contacts, phones, URLs, phrases like "by registration", "buy tickets at link", "in DM", etc.
- Do NOT add information missing from the source text.
- No lists or line breaks — strictly one line.
- Remove emojis, hashtags, repetitive phrases, and fluff.
What to include:
- Genre and subgenre.
- Key highlights of format and program (extract 1-2 highlights like "musical warm-up", "guided route" without time).
- Neutral summary of reviews (if source contains "Отзывы", include as "по отзывам — ...", without names or "best/magnificent").
- Useful labels from Poster OCR if available.
- Key persons/organizations.
- Topic/subject.
- Conditions/restrictions (16+, "for entrepreneurs", "Pushkin card"...).
Length guide: 25–55 words (20-80 allowed if necessary for search uniqueness).
If an array of events is returned, `search_digest` must be present in every object.

**report / recap rule:**
- If the text is mainly a post-event report / recap about something that already happened, return no events.
- Typical clues: past-tense narrative ("мы провели/исследовали/работали"), after-the-fact summary ("было здорово"),
  gratitude/wrap-up ("спасибо ...", "увидимся вновь"), but no concrete attendable future anchor.
- If a post mixes recap/background about past meetings with a real future invite, ignore the recap part and extract
  only the future attendable event with its explicit future anchor (date/venue/time/registration/ticket).
```

Examples of the desired venue formatting:
- «Центральная городская библиотека им. А. Лунина, ул. Калинина, д. 4, Черняховск» → `location_name`: «Библиотека А. Лунина», `location_address`: «Калинина 4», `city`: «Черняховск».
- «Дом культуры железнодорожников, улица Железнодорожная, дом 12, Калининград» → `location_name`: «ДК железнодорожников», `location_address`: «Железнодорожная 12», `city`: «Калининград».
- «Музей янтаря имени И. Канта, проспект Мира, д. 1, Светлогорск» → `location_name`: «Музей янтаря им. Канта», `location_address`: «Мира 1», `city`: «Светлогорск».

Do **not** include words like "Открытие" or "Закрытие" in exhibition titles.
The bot adds these markers automatically on the opening and closing dates.

Lines from `../reference/locations.md` are appended to the system prompt so the model
can normalise venue names. Please keep that file up to date.

When `../reference/holidays.md` is present, the prompt gains a "Known holidays" section
listing canonical seasonal festivals together with their alias hints and short
descriptions. Treat these names as the preferred targets for the `festival`
field and use the hints to match synonym spellings in announcements.

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
- This bot covers events in **Kaliningrad Oblast**. If the event is clearly outside the region
  (e.g. the city is Москва / Санкт‑Петербург / Кисловодск or other non‑regional location) —
  do NOT include it in the output (return `[]` or `{"festival": {...}, "events": []}` when relevant).
- Do NOT turn news/press-release texts about projects, grants, initiatives, or “акция станет ежегодной/новой традицией”
  into events unless there is a concrete attendable event with explicit date + venue (and preferably time).
  If it's an initiative description with a program "запланировано/включает в себя" but without a specific event entry,
  return no events.
- Do NOT treat administrative deadlines as event dates. If the only date in the text is a "до <date>" deadline
  (e.g. "подать заявку до 16 февраля", "утвердят до 1 марта") and there is no attendable event with date+venue,
  return no events.
- Do NOT turn venue/organisation status updates into events. Posts like “город может потерять площадку с 1 мая”,
  “дана отсрочка до 1 июня”, eviction/lease/closure news, petitions, fundraising, calls to “support/save the space”
  are NOT attendable events. Dates in such posts are deadlines/status dates, not event dates — return no events.
- Do NOT create events out of informational government/service notices (e.g. "налоговый вычет", "госуслуга",
  eligibility rules, "перечень утверждают", application windows). These are not attendable events.
- Do NOT create events out of course/program advertisements ("старт курса", "набор", multi-session training programs)
  unless it's explicitly a single attendable session (e.g. one-day masterclass) with a concrete date+venue (and ideally time).
- Do NOT create events out of institution working-hours notices (e.g. "график/режим/часы работы",
  "праздничные/выходные дни", "санитарный день", "расширенный график").
  Dates/times in such posts describe opening hours, not event schedule.
  Return no events unless the post explicitly announces an attendable activity with action wording
  (e.g. "состоится", "пройдет", "приглашаем") and concrete event details.
- Do NOT create events out of post-event reports / recaps. If the text mainly describes what already happened
  (past-tense narrative like "мы провели/исследовали/работали", after-the-fact summary like "было здорово",
  gratitude/wrap-up like "спасибо ...", "скоро увидимся вновь") and there is no concrete attendable future anchor,
  return no events.
- If a post mixes recap/background about past meetings with a real future invite, ignore the recap part and extract
  only the future attendable event with its explicit future anchor (date/venue/time/registration/ticket).
- Do NOT assume a date when none is given. If there is no explicit date (DD.MM, “15 мая”, period) and no clear relative date
  (“сегодня/завтра/в эту субботу”), return no events — do NOT default to “today”.
- The “Known venues” list is for normalising venues that are explicitly mentioned (or provided as an explicit default hint).
  Do NOT pick a random venue just because it contains a similar word (e.g. “ворота”).
- `city` must be the city name only (no street/house number). If the city is unknown, return an empty string.
- If the year is missing, choose the nearest future date relative to ‘Today’ (from the system header). If the day/month has already passed this year, roll the year forward.
- Omit any events dated before today.
- Do NOT invent a time when the source does not provide it. In particular, do not misread dates like `21.02` (DD.MM) as time `21:02` (HH:MM).
- When a festival period is mentioned but only some performances are described,
  include just those individual events with their own dates and set the
  `festival` field. Do **not** create separate events for each day of the
  festival unless every date is explicitly detailed.
- If the text describes a single holiday/day celebration or “гуляния” with a clear **program/schedule** (multiple activities listed by time),
  do NOT create separate events for each time slot. Create ONE umbrella event, keep the program in text fields, and set `time` to a range `HH:MM..HH:MM`
  using the earliest and latest times from the program.
- Anti-duplicates (very important): do NOT return multiple events that share the same `date`, the same start time (or the same `time`),
  and the same `location_name`. If your extraction would produce such items (e.g. you picked different speakers/bands/hero names from one list),
  merge them into ONE umbrella event: choose a stable event-level title (not a single performer/person from the list) and keep the list as part of the description/facts.
  Only allow multiple same-anchor events if the source explicitly states parallel events in different halls/rooms.
- When the text describes a «День <…>» celebration with a clear program/ расписание
  (multiple items, multiple times, or multi-day range), treat it as a festival-like
  umbrella: fill `festival` with the short name («День …») and put the full edition
  wording (year/number/season if present) into `festival_full`. If it is a single
  event without a program, keep `festival` empty unless the text explicitly says
  “в рамках фестиваля/праздника …”.
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
    "end_date": null,
    "search_digest": "Спектакль Щелкунчик, сказочная постановка для всей семьи по мотивам Гофмана, театр Звезда, классическая музыка Чайковского."
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
    "end_date": null,
    "search_digest": "Спектакль Щелкунчик, сказочная постановка для всей семьи по мотивам Гофмана, театр Звезда, классическая музыка Чайковского."
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
Блок «Локация» описывает место проведения и не должен использоваться сам по себе для выбора `KRAEVEDENIE_KALININGRAD_OBLAST`; решение принимай по содержанию события.
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
`topics.md`. Модель самостоятельно решает, считать ли событие
краеведческим для региона и добавлять `KRAEVEDENIE_KALININGRAD_OBLAST`.

## Telegram channel metadata → festival suggestions (Gemma/4o)

Используется в Kaggle `TelegramMonitor` для извлечения подсказок из **метаданных источника** (title/about/links). Ограничение: сервер не ходит в Telegram API, поэтому метаданные должны собираться в Kaggle и попадать в `telegram_results.json` (см. `docs/backlog/features/telegram-monitoring/channel-metadata.md`).

Системный промпт:

```
Ты — ассистент, который извлекает структурные подсказки из метаданных Telegram-канала/группы.
Тебе переданы: username, title, about (описание) и список ссылок, найденных в about.

Задача: определить, является ли источник каналом одного фестиваля/серии фестиваля, и если да — предложить короткое имя серии и официальный сайт.

Правила:
- Не выдумывай факты: используй только то, что явно следует из title/about/links.
- Если уверенности нет — верни пустые значения и низкую confidence.
- `festival_series` — это короткое устойчивое название серии без года/номера/сезона.
- `website_url` — только внешний сайт фестиваля (не t.me, не telegra.ph). Если ссылок несколько — выбери наиболее похожую на официальный сайт (домены фестиваля/организатора, "festival", "fest", "kantata", и т.п.).
- `aliases` — 0..5 вариантов написания названия серии (латиница/кириллица, верхний регистр, сокращения), только если они реально встречаются в title/about.
- `rationale_short` — 1 русское предложение (без URL), почему ты так решил.

Верни JSON строго по схеме:
{
  "is_festival_channel": boolean,
  "festival_series": string,
  "website_url": string,
  "aliases": string[],
  "confidence": number,
  "rationale_short": string
}
```

Пример входа (payload, который формирует Kaggle):

```json
{
  "username": "open_fest",
  "title": "OPEN FEST",
  "about": "Фестиваль \"Открытое море\". Официальный сайт: https://openfest.example.org",
  "about_links": ["https://openfest.example.org"]
}
```

Пример ответа:

```json
{
  "is_festival_channel": true,
  "festival_series": "Открытое море",
  "website_url": "https://openfest.example.org",
  "aliases": ["OPEN FEST", "Open Fest"],
  "confidence": 0.9,
  "rationale_short": "В названии и описании явно указан фестиваль и приведён официальный сайт."
}
```
