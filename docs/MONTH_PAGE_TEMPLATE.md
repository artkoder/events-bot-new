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
