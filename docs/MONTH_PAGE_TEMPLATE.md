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

`ticket_info` may be "Билеты в источнике" with a price range, "Бесплатно" or a
registration link. `more` links to the individual Telegraph page.

## Page layout

```
# События Калининграда в {month_year_prep}: полный анонс от [Полюбить Калининград Анонсы](https://t.me/kenigevents)

Планируйте свой месяц заранее: интересные мероприятия Калининграда и 39 региона в {month_year_prep} — от лекций и концертов до культурных шоу.

{events}

{next_month_link}

## Постоянные выставки

{exhibitions}
```

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
