Проверил указанные места, нашёл 3 проблемы и поправил.

**Findings (fixed)**
- High: `source_parsing/handlers.py:435` возвращал `None` при пустом `drafts`, что ломало распаковку в вызывающем коде. Исправлено на `(None, False)`.
- Medium: `source_parsing/parser.py:278` форсировал `parsed_time="00:00"` даже при отсутствии времени, затирая более точное время из LLM/описания. Исправлено: сохраняем `None`.
- Medium: `scripts/update_tretyakov_nb.py:305` в ветке без билетов не выдавался `scene`, из‑за чего сцена терялась в описании. Исправлено: `scene` добавлен в JSON.

**OK**
- `source_parsing/handlers.py:427` `source_text` теперь содержит `Название: {title}`.
- `source_parsing/handlers.py:492` `ticket_link` берётся напрямую из `theatre_event.url`.
- `source_parsing/handlers.py:475` `scene` добавляется в `final_description`, если его нет.

**Status**
- Готово к тестовому запуску; тесты не запускал.

Если хочешь, могу запустить тестовый парсинг Третьяковки, чтобы проверить результат на выходе.