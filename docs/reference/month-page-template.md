# Month Page Template

Telegraph month pages are generated from Markdown using the variables below.
Edit this file to adjust formatting.

## Event entry

```
{title}
{description}
{ticket_info}
_{date} {time} {venue}, {location}, #{city}_
{more}
```

`ticket_info` may be "Билеты в источнике" with a price range, "Бесплатно", or
"Бесплатно [по регистрации](URL)" если требуется предварительная запись.
`more` links to the individual Telegraph page.

The first line (`{title}`) becomes an `<h4>` heading on the Telegraph page.
Recently added events (within the last 48 hours) are prefixed with the 🚩 emoji.

## Page layout

```
# События Калининграда в {month_year_prep}: полный анонс

Планируйте свой месяц заранее: интересные мероприятия Калининграда и 39 региона в {month_year_prep} — от лекций и концертов до культурных шоу. [Полюбить Калининград Анонсы](https://t.me/kenigevents)

{events}

{month_links}

`month_links` displays links to future month pages with the current month shown as plain text.

## Постоянные выставки

{exhibitions}
```

Day headers are formatted as `<h3>` elements and event titles as `<h4>`.
Day headers are formatted as:

```
🟥🟥🟥 {day} 🟥🟥🟥
```

For Saturday:

```
🟥🟥🟥 суббота 🟥🟥🟥
🟥🟥🟥 {day} 🟥🟥🟥
```

For Sunday:

```
🟥🟥 воскресенье 🟥🟥
🟥🟥🟥 {day} 🟥🟥🟥
```

If a day has no events the header is omitted.

When the generated content exceeds the configured limit (approx. 45&nbsp;kB), the bot splits the month
into multiple Telegraph pages. The first page ends with a bold link to the
continuation. Subsequent pages use a title format indicating the date range (e.g. "С 15 по 31 января...").

Splitter notes:

- day boundaries are preserved whenever a full day still fits on one page;
- if the month requires many continuation pages, the bot degrades link density first (`Добавить в календарь`, then `Подробнее`);
- if the `Постоянные выставки` section does not fit into the last page, exhibitions may spill into dedicated continuation pages instead of forcing a single oversized tail page.
- the same month/weekend rebuild helpers are used by direct sync jobs and by the operator-facing `/pages_rebuild` flow, so regressions there should be treated as production-facing page rebuild bugs.
- if ongoing exhibitions exceed `MONTH_EXHIBITIONS_PAGE_THRESHOLD` (default `10`), the month page skips the inline exhibition list and shows a footer link like `Постоянные выставки марта` to a dedicated Telegraph page for that month.
- public exhibition lists are display-deduped by `title + end_date + venue/city` heuristics, so legacy duplicate rows in the DB do not repeat on the public month/exhibitions pages.
