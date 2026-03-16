# Tools

Короткие шпаргалки и заметки по локальным инструментам и воркфлоу.

- Codex CLI: `docs/tools/codex-cli.md`
- Claude Code: `docs/tools/claude-code.md`
- Дедуп событий в SQLite (merge кандидатов на дубли): `python scripts/inspect/dedup_event_duplicates.py --db <path>` (dry-run), `--apply` (применить). Перед изменениями создаётся backup в `artifacts/db/`; при merge скрипт также перепривязывает `vk_inbox.imported_event_id`, `vk_inbox_import_event` и `linked_event_ids`, чтобы cleanup не оставлял битые ссылки.
