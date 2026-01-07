# CODEX

## Mandatory Rules
- **ALWAYS** review this CODEX before starting any task on this repository.
- **ALWAYS** keep documentation and automation artifacts synchronized with code changes.
- **ALWAYS** run and update relevant smoke tests before requesting review or completing work.
- **ALWAYS** update user-facing documentation (READMEs, help text, runbooks) when behavior changes or explicitly document why no update is required.
- **ALWAYS** append relevant entries to the CHANGELOG describing user-impacting fixes and features.
- **ALWAYS** verify new or modified functionality has appropriate logging coverage in place, consistent with the Definition of Done below.
- **NEVER** merge or submit changes without passing smoke tests and linting checks.
- **NEVER** introduce fixtures or test data that persist outside their intended scope.

## Definition of Done
Every change must satisfy **all** of the following before it is considered complete:
1. Smoke tests are executed and passing, or a justified exception is documented in the change description.
2. Test fixtures are cleaned up, scoped appropriately, and free of side effects across the suite.
3. The README, user-facing help, and CHANGELOG entries are updated (or explicitly confirmed as not needed) to reflect behavioral or interface changes.

## Critical Code Paths
The following modules and files are considered critical paths and demand extra scrutiny, regression testing, and reviewer visibility whenever touched:
- `main`
- `db`
- `imagekit_poster`
- `vk_intake`
- `vk_review`
- `markup`
- `scheduling`
- `digests`
- `sections`
- `shortlinks`
- `supabase_export`
- `safe_bot`
- `span`
- `net`
- `models`

## Agent Guidance
Future contributors and agents must treat this CODEX as required reading prior to any repository work. Confirm in your task notes that you have reviewed it.

---

# Codex CLI Cheatsheet (project defaults)

## Output paths (важно)

Чтобы не захламлять корень и не коммитить отчёты, складывай результаты Codex сюда:

- отчёты: `artifacts/codex/reports/`
- промежуточные файлы/черновики: `artifacts/codex/tasks/`

Пример:
```bash
mkdir -p artifacts/codex/reports
codex exec --sandbox workspace-write -o artifacts/codex/reports/PHASE-1.md "..."
```

## Basic commands

### Verification & login
```bash
codex login status
codex login --device-auth
printenv OPENAI_API_KEY | codex login --with-api-key
```

## Execution modes

### One-off task
```bash
codex exec --full-auto "внеси правки и обнови тесты"
codex exec --json "проанализируй репозиторий" | jq
codex exec -o artifacts/codex/reports/out.md "сгенерируй release notes"
```

### Sandbox & permissions
```bash
codex exec --sandbox workspace-write "checking changes"
codex exec --sandbox danger-full-access "сделай массовый рефакторинг"
```

### Piping input
```bash
cat task.md | codex exec -
```

### Structured output (schema)
```bash
codex exec --output-schema ./schema.json -o artifacts/codex/reports/report.json \
  "сделай краткий risk-report по изменениям"
```

## Workflow tips

1. `codex exec resume --last "исправь найденные проблемы"`
2. Всегда проверяй `git diff` перед коммитом.
