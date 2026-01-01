**Findings**
- High — Заголовок парсера не фиксируется в БД: `full_description` строится из `description/scene` без `title`, а `Event.title` берется из LLM (`draft.title`) без оверрайда парсером; при наличии описания у Третьяковки название может «уплыть». `source_parsing/handlers.py:383` `source_parsing/handlers.py:393` `source_parsing/handlers.py:476`
- Medium — `scene` теряется: в JSON он есть и попадает в `TheatreEvent`, но поле отсутствует в модели БД, а строка «Сцена: …» добавляется только в `full_description`, который может быть заменен LLM-описанием; при нормализации location зал (Атриум/Кинозал) исчезает. `scripts/update_tretyakov_nb.py:315` `source_parsing/parser.py:265` `source_parsing/handlers.py:390` `source_parsing/handlers.py:478` `models.py:348`
- Medium — `theatre_event.url` не гарантированно становится `ticket_link`: при наличии любых `draft.links` сохраняется `draft.links[0]`, а последующее обновление не перезаписывает непустой `ticket_link`. Это может подменить ссылку на билеты. `source_parsing/handlers.py:456` `source_parsing/handlers.py:487` `source_parsing/handlers.py:194`
- Low/Medium — события без активных дат/тайм-слотов пишутся с пустыми `date_raw` и `parsed_date=None`, затем отбрасываются как `missing_date`. Потеря экспозиций/мероприятий без календаря. `scripts/update_tretyakov_nb.py:293` `source_parsing/parser.py:248` `source_parsing/handlers.py:931`

**Questions/Assumptions**
- Нужно ли сохранять события Третьяковки без билетов/календаря, или их действительно можно пропускать?
- Допустимо ли, что LLM генерирует заголовок вместо использования `theatre_event.title`?
- Сцену/зал важно хранить отдельным полем или достаточно всегда включать в описание?

**Recommendations**
1) Зафиксировать заголовок парсера: либо включать `theatre_event.title` в `source_text`, либо явно оверрайдить `draft.title`/`Event.title` значением из парсера.
2) Сохранить сцену: добавить поле в модель, или принудительно добавлять «Сцена: …» к итоговому `Event.description` даже при наличии LLM-описания.
3) Приоритетизировать `theatre_event.url` для `ticket_link`: использовать его напрямую в `Event.ticket_link` или синхронизировать `draft.links` с парсерным URL; при необходимости разрешить перезапись в `update_event_ticket_status`.
4) Для событий без тайм-слотов определить явную стратегию: извлекать дату из деталей/листа или явно маркировать и сохранять их с `00:00`, а не отбрасывать.

**Change Summary**
- Review only; no code changes.