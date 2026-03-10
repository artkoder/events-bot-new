# Документация

Этот каталог устроен **feature‑ориентированно**: у каждой фичи есть свой “дом” в `docs/features/`.

## Быстрый роутинг (для агентов)

- Машиночитаемая карта: `docs/routes.yml`
- Список фич: `docs/features/README.md`

## Канонические разделы

- Архитектура: `docs/architecture/overview.md`
- Эксплуатация: `docs/operations/` (как запускать/поддерживать)
- LLM: `docs/llm/` (промпты, формат запросов, классификатор тем)
  Для `Smart Update lollipop`: `docs/llm/smart-update-lollipop-funnel.md`, `docs/llm/smart-update-lollipop-casebook.md`, `docs/llm/smart-update-lollipop-mixed-phase-prompts.md`
  Для downstream layout: `docs/llm/smart-update-lollipop-editorial-layout-prompts.md`
  Для downstream writer pack: `docs/llm/smart-update-lollipop-writer-pack-prompts.md`
  Для final writer: `docs/llm/smart-update-lollipop-writer-final-prompts.md`
  Последний layout run: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-editorial-layout-lab-iter2-2026-03-10.md`
  Последний layout review: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-editorial-layout-postrun-gemini-synthesis-2026-03-10.md`
  Последний writer-pack run: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-writer-pack-compose-lab-iter2-2026-03-10.md`
  Последний writer-pack review: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-writer-pack-compose-postrun-gemini-synthesis-2026-03-10.md`
  Последний writer-final run: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-writer-final-4o-lab-iter2-2026-03-10.md`
  Последний writer-final review: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-writer-final-4o-postrun-gemini-synthesis-2026-03-10.md`
  Последний baseline comparison / retune synthesis: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-writer-retune-iter2-consultation-synthesis-2026-03-10.md`
  Последний follow-up consultation synthesis: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-writer-retune-iter3-consultation-synthesis-2026-03-10.md`
  Последний quality/split consultation synthesis: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-quality-compliance-and-split-consultation-synthesis-2026-03-10.md`
  Последний pipeline profiling report: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-pipeline-profiling-2026-03-10.md`
- Справочники: `docs/reference/` (локации, праздники, шаблоны)
- Пайплайны/парсеры (каноника): `docs/features/source-parsing/` (в `docs/pipelines/` остаются redirect‑stub’ы для старых ссылок)
- Фичи: `docs/features/`
- 3D previews (`/3di`): `docs/features/preview-3d/README.md`
- Бэклог (не реализовано): `docs/backlog/`
- Отчёты и планы: `docs/reports/`
- Инструменты: `docs/tools/`

## Правила добавления/обновления доков

1. Новая фича → добавь `docs/features/<feature>/README.md` и внеси её в `docs/routes.yml`.
2. Не создавай новый документ, если можно расширить существующий канонический.
3. Если нужно сохранить старый путь (ссылки/история) — оставь короткий redirect‑stub без копирования текста.
