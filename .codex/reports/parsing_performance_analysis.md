# Parsing Logs for Analysis

## Context
User suspects that events are being re-added (re-processed via LLM) on every `/parse` run instead of being matched as existing.
Key symptom: Events that were previously added are processed as long as new ones.

**Request to Codex**: Check `find_existing_event` logic. See if `fuzzy_title_match` or time matching is failing.

## Captured Logs (Production)
```text
2025-12-31T10:41:43Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: event_result source=muzteatr title=Лукоморье date=2026-01-03 time=17:00 result=new_updated event_id=1718 llm=1 duration=18.64s
2025-12-31T10:41:43Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: searching location=Музыкальный театр date=2026-01-04 time=13:00 title=Лукоморье
2025-12-31T10:41:43Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: found 0 candidates for location=Музыкальный театр date=2026-01-04
2025-12-31T10:41:43Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: NO MATCH for title=Лукоморье
2025-12-31T10:41:54Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: adding event 47/98 title=Лукоморье location=Музыкальный театр
2025-12-31T10:41:57Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: event created event_id=1971 title=Лукоморье (Telegraph only, no nav update)
2025-12-31T10:41:57Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: updated ticket_status event_id=1971 old=None new=available
2025-12-31T10:41:57Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: linked events event_id=1971 linked_count=6
2025-12-31T10:42:02Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: event_result source=muzteatr title=Лукоморье date=2026-01-04 time=13:00 result=new_added event_id=1971 llm=1 duration=19.26s

2025-12-31T10:42:02Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: searching location=Музыкальный театр date=2026-01-04 time=17:00 title=Лукоморье
2025-12-31T10:42:02Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: found 1 candidates for location=Музыкальный театр date=2026-01-04
2025-12-31T10:42:02Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: NO MATCH for title=Лукоморье
2025-12-31T10:42:13Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: adding event 48/98 title=Лукоморье location=Музыкальный театр
2025-12-31T10:42:17Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: event created event_id=1720 title=Лукоморье (Telegraph only, no nav update)
2025-12-31T10:42:17Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: updated ticket_status event_id=1720 old=None new=available
2025-12-31T10:42:17Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: linked events event_id=1720 linked_count=6
2025-12-31T10:42:22Z app[48e42d5b714228] iad [info]INFO:source_parsing.handlers:source_parsing: event_result source=muzteatr title=Лукоморье date=2026-01-04 time=17:00 result=new_updated event_id=1720 llm=1 duration=19.76s
2025-12-31T10:42:22Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: searching location=Музыкальный театр date=2026-01-09 time=17:00 title=Девчата
2025-12-31T10:42:22Z app[48e42d5b714228] iad [info]DEBUG:source_parsing.parser:find_existing_event: found 1 candidates for location=Музыкальный театр date=2026-01-09
2025-12-31T10:42:22Z app[48e42d5b714228] iad [info]INFO:source_parsing.parser:find_existing_event: title matches but time differs db_...
```
