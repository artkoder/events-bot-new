# Tools

Короткие шпаргалки и заметки по локальным инструментам и воркфлоу.

- Codex CLI: `docs/tools/codex-cli.md`
- Дедуп событий в SQLite (merge кандидатов на дубли): `python scripts/inspect/dedup_event_duplicates.py --db <path>` (dry-run), `--apply` (применить). Перед изменениями создаётся backup в `artifacts/db/`.
