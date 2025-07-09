# User Stories

| № | Role | Feature | Goal |
|--|------|---------|------|
|US-01|Anyone|`/start` registers first superadmin|get full access|
|US-02|Admin|manage moderator requests|≤10 open requests|
|US-03|Mod/Admin|forward post → parse via 4o|event created/updated|
|US-04|System|on new event|create MD template + original Telegraph|
|US-05|System|on duplicate|merge descriptions via 4o|
|US-06|System|maintain month/week/weekend pages|auto create + link|
|US-07|Admin|view "Permanent Festivals" page|actual list|
|US-08|Admin|get Telegraph stats|period report|
|US-09|System|daily announcement with buttons|auto post to channels|
|US-10|Admin|manage channels and time|CRUD channels|
|US-11|System|create festival pages|clickable links|
|US-12|Moderator|add/remove event to festival|links updated|
|US-13|Moderator|browse upcoming events by day|manage via buttons|
|US-14|Moderator|edit event|Telegraph pages updated|
|US-15|Admin|export/import data|no loss of links/token|
|US-16|Admin|view event stats|by dates|
|US-17|Admin|if source="announcement channel"|event title links to post|
|US-18|System|store MD templates|easy styling|
|US-19|System|auto cleanup old events >60d|save DB|
|US-20|Super/Admin|set timezone `/tz`|schedule correct|
|US-21|Moderator|add event via `/addevent` or `/addevent_raw`|store new events|
|US-22|Moderator|mark an event as free|hide price and show badge|
|US-23|System|replace "🆓🆓🆓🆓" with "Бесплатно" on Telegraph pages|consistent text|
|US-24|System|parse event type and emoji via 4o|categorise events|
|US-25|System|store start and end dates for multi-day events|show opening and closing|
|US-26|User|view exhibitions with `/exhibitions`|see ongoing exhibitions|
|US-27|User/Admin|add event to calendar via ICS|quick calendar save|
