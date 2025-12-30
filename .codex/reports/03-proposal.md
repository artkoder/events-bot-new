**Problem**
- Сейчас `schedule_event_update_tasks` сразу дренит nav-задачи и зовёт `sync_month_page`/`sync_weekend_page`, поэтому каждая вставка события ведёт к немедленным ребилдам. Требуется отложить их на 2 часа, с напоминанием через 10 минут и автозапуском `/pages_rebuild`, сохранив состояние при рестарте.

**Proposal**
- Dirty-флаг хранить в `Setting` (`models.py:325`) — одна запись, например ключ `pages_dirty_state` с JSON `{"since": "...", "months": ["YYYY-MM", ...], "reminded_at": "..."}`. Это переживает рестарт и не требует миграции.
- Отключать авто-вызов `sync_month_page` при добавлении событий через `schedule_event_update_tasks`:
  - для nav-задач (`month_pages`/`weekend_pages`) вместо немедленного `_drain_nav_tasks` ставить `next_run_at=now+120m` и отмечать месяц в dirty-флаге;
  - остальные задачи (telegraph/ICS/VK) оставлять без изменений.
- Авто `/pages_rebuild`: добавить в APScheduler (`scheduling.startup`) периодическую задачу (раз в 2–5 минут) `maybe_rebuild_dirty_pages`, которая:
  - читает `pages_dirty_state`; если `since >= 2h`, собирает месяцы из dirty-флага или из `JobOutbox` по `coalesce_key LIKE "month_pages:%"`/`"weekend_pages:%"`;
  - вызывает `_perform_pages_rebuild(db, months, force=True)` (как в `main_part2.py:5715+`) и после успеха очищает dirty-флаг и помечает соответствующие nav-job’ы done/superseded, чтобы они не сработали вторично.
- Reminder через 10 минут: в той же задаче, если `since >=10m` и `reminded_at` пуст, отправлять `notify_superadmin` с указанием грязных месяцев, затем писать `reminded_at=now` в Setting.

**Implementation Steps**
- Ввести хелперы `mark_pages_dirty(db, month)` / `load_pages_dirty_state` / `clear_pages_dirty_state` работающие через `Setting`.
- В `enqueue_job` или сразу после него в `schedule_event_update_tasks` для `JobTask.month_pages`/`weekend_pages` выставлять `next_run_at = now + timedelta(hours=2)` и не вызывать `_drain_nav_tasks` при включённой отсрочке; добавлять месяц в dirty-флаг.
- В `scheduling.startup` зарегистрировать APScheduler-задачу `maybe_rebuild_dirty_pages` (coalesce, max_instances=1, 2–5 min interval) использующую `_perform_pages_rebuild` и `notify_superadmin`.
- В `_perform_pages_rebuild` (или после ручного `/pages_rebuild`) очищать dirty-флаг и, при необходимости, резетить pending nav-задачи по соответствующим `coalesce_key`.
- Настроить конфиг-флаг (`DEFER_PAGE_REBUILD=1` через env/Setting) чтобы можно было включить/выключить отложенный режим без кода.

**Risks**
- Двойная работа: если не очистить/суперседить nav-job’ы после автребилда, `job_outbox_worker` может повторно пропатчить страницы.
- Потеря месяцов: нужно брать список месяцев либо из dirty-флага, либо из `JobOutbox`; при очистке следует убедиться, что pending задачи для тех же месяцев не остались со старым `next_run_at`.
- Рейсы при ручном `/pages_rebuild` и авто-джобе — нужен общий lock/флаг в Setting, чтобы не запустить rebuild параллельно.
- Напоминания могут заспамить, если не хранить `reminded_at`.
- В отложенном режиме week/weekend VK-посты (`week_pages`/`weekend_pages` задачи) останутся «горячими»; если их тоже откладывать, нужно учесть зависимость партнёрских уведомлений или внешних дедлайнов.