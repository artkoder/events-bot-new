# Agent Routes (project navigation)

Цель: чтобы **агент** и **человек** быстро находили единственный актуальный документ по фиче/задаче и не плодили дубли.

## Старт

1. Открой `docs/README.md` (человеческий индекс).
2. Для “быстрого роутинга” используй `docs/routes.yml` (машиночитаемая карта).
3. Для задач по E2E всегда сверяйся с `docs/operations/e2e-scenarios.md` и поддерживай этот индекс актуальным при изменении сценариев.

## E2E по умолчанию (важно)

- Если пользователь просит “сделай/запусти E2E” без уточнений — это **live E2E прогон**: реальные запросы и проверка через **UI в Telegram** (а не offline/фикстуры).
- При прогоне `behave` сценариев нужно **анализировать ответы из Telegram UI** (сообщения/кнопки/отчёты/логи) и результаты `behave`:
  - если в UI/логах есть ошибки/инциденты/неожиданные статусы — **расследовать и пытаться исправить** в рамках текущей задачи, а не ждать отдельного репорта от пользователя;
  - фиксировать первопричину (код/конфиг/данные), добавлять минимальные тесты/доки по месту.
- Каноническая инструкция по E2E: `docs/operations/e2e-testing.md` (подготовка/запуск/ENV) + индекс сценариев `docs/operations/e2e-scenarios.md`.
- ENV для live E2E хранится в `.env` в корне репозитория (не коммитится; шаблон — `.env.example`):
  - обязательны `TELEGRAM_BOT_TOKEN` и (`TELEGRAM_API_ID`/`TELEGRAM_API_HASH` **или** `TG_API_ID`/`TG_API_HASH`) и одна из: `TELEGRAM_AUTH_BUNDLE_E2E` или `TELEGRAM_SESSION`.
  - `behave`/`pytest` E2E подхватывают `.env` автоматически (best-effort) и **не** перетирают уже заданные переменные окружения.
  - если запускаешь бота руками из терминала, `.env` не подгружается автоматически: используй `set -a; source .env; set +a` перед `python main.py`.

## Session Boundaries (critical)

- Telegram auth bundles are **role-scoped** and must not be repurposed without explicit user permission.
- `TELEGRAM_AUTH_BUNDLE_S22` is reserved for **Kaggle / remote monitoring** runs.
- `TELEGRAM_AUTH_BUNDLE_E2E` (or `TELEGRAM_SESSION`) is reserved for **local live E2E / Telethon human client** runs.
- Never switch Kaggle guide monitoring from `TELEGRAM_AUTH_BUNDLE_S22` to `TELEGRAM_AUTH_BUNDLE_E2E` on your own, even as a temporary workaround.
- Never run the same auth bundle concurrently in multiple places when one of them is Kaggle/remote, because Telegram can invalidate the auth key with `AuthKeyDuplicatedError`.
- If the intended bundle is broken or missing, stop and report it clearly instead of borrowing another bundle.

## Правила раскладки

- **Фича** → `docs/features/<feature>/README.md` + дочерние файлы в этой же папке.
- **Операции/эксплуатация** → `docs/operations/`.
- **Архитектура** → `docs/architecture/`.
- **Пайплайны/парсеры** → `docs/pipelines/`.
- **LLM/промпты** → `docs/llm/`.
- **Справочники** (локации/праздники/шаблоны) → `docs/reference/`.
- **Бэклог/задачи** (ещё не реализовано) → `docs/backlog/`.
- **Отчёты/планы/ретроспективы** → `docs/reports/`.
- **Тулзы/шпаргалки** → `docs/tools/`.

## Обязательное сопровождение изменений

- Любое изменение кода/поведения должно сопровождаться обновлением канонической документации в `docs/` по этой фиче (без дублей, только в одном актуальном месте).
- Любое изменение кода/поведения должно сопровождаться записью в `CHANGELOG.md` в секции `[Unreleased]` (кратко и по существу: Added/Changed/Fixed).
- Задача считается незавершённой, если код изменён, а документация и `CHANGELOG.md` не синхронизированы.

## LLM‑first обработка текста (важно)

Если задача касается качества/смысла текста событий (например `title`, `description`, `search_digest`), приоритет у обработки **через LLM** (промпты в `docs/llm/` и LLM‑пасс в Smart Update).

Детерминированные функции допустимы как поддержка (санитайзеры, нормализация, извлечение дат/времени, безопасные guardrail‑проверки), но они **не должны менять смысл** текста.

Каноническая политика: `docs/llm/request-guide.md` (секция про LLM‑first).

## Claude / Opus policy

- Для Claude Code в этом репозитории используется только `Opus`.
- Effort для Claude Code должен быть только `high`.
- Для сложных консультаций, архитектурного разбора, deep-dive debugging и нетривиального redesign допускается временно повышать effort до `max`.
- Проектный shared-config хранится в `.claude/settings.json`; проектные инструкции Claude — в `CLAUDE.md`.
- Для консультаций, архитектурной критики, prompt review и нетривиальных доработок используй проектный subagent alias `Opus` из `.claude/agents/Opus.md`.
- Если задача LLM-first упирается в качество extraction/writer output, используй `Opus` прежде всего как эксперта по prompt design: проси prompt-family audit, конкретные prompt diffs, schema tightening и stage split по `lollipop`-принципу небольших self-contained запросов, а не общий абстрактный architecture advice.
- В shared-config запрещены встроенные Claude subagents, чтобы делегация не уходила в `Haiku`/`Sonnet`; для делегации оставляй только `Opus`.
- Не переключай Claude на `Sonnet`/`Haiku`, если пользователь явно не попросил изменить эту политику.

## Артефакты и временные файлы

- Любые результаты прогонов, дампы, логи, pid, локальные sqlite, выгрузки и т.п. → `artifacts/` (см. `artifacts/README.md`).
- Отчёты/черновики Codex CLI по умолчанию складывай в `artifacts/codex/` (см. `docs/tools/codex-cli.md`).
- Не коммить артефакты. Если нужно сохранить пример — клади **минимальный** fixture в `tests/fixtures/` (если такой паттерн уже есть).

## Избегаем дубликатов

- Один факт/инструкция — один “канонический” документ.
- Старые пути допускаются только как **короткие redirect-stub файлы** без повторения контента (“Актуально тут: …”).
