# Codex CLI Cheatsheet

## Basic Commands

### Verification & Login
```bash
# Check login status
codex login status

# Login (Headless/Codespace)
codex login --device-auth

# Login via API Key (env)
printenv OPENAI_API_KEY | codex login --with-api-key
```

## Execution Modes

### One-off Task (Non-interactive)
```bash
# Full auto execution (allows edits)
codex exec --full-auto "внеси правки и обнови тесты"

# JSON output for machine parsing
codex exec --json "проанализируй репозиторий" | jq

# Save output to file
codex exec -o ./out.txt "сгенерируй release notes"
```

### Sandbox & Permissions
```bash
# Sandbox with write permissions (default for --full-auto)
codex exec --sandbox workspace-write "checking changes"

# DANGER: Maximum access (only in isolated environments)
codex exec --sandbox danger-full-access "сделай массовый рефакторинг"
```

### Piping Input
```bash
cat task.md | codex exec -
```

### Structured Output (Schema)
generate schema.json first:
```json
{
  "type": "object",
  "properties": {
    "summary": { "type": "string" },
    "risks": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["summary", "risks"],
  "additionalProperties": false
}
```
Run:
```bash
codex exec --output-schema ./schema.json -o ./report.json \
  "сделай краткий risk-report по изменениям"
```

## Workflow Tips
1. **Resume Session**: `codex exec resume --last "исправь найденные проблемы"`
2. **Race Conditions**: `codex exec "проверь изменения на race conditions"`
3. **Diff Check**: Always check `git diff` or `/diff` in TUI before commiting to changes made by Codex.

## Advanced Workflows
Use Codex for high-level engineering tasks:
- **Architecture Planning**: `codex exec --full-auto "спроектируй архитектуру модуля X"`
- **Test Generation**: `codex exec "напиши автотесты для файла main.py с учетом граничных случаев"`
- **Code Review**: `codex exec "проведи ревью изменений в ветке, ищи проблемы безопасности"`
- **Complex Refactoring**: `codex exec --sandbox danger-full-access "выдели класс EventManager в отдельный файл и обнови импорты"`
