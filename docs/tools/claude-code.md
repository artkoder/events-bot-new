# Claude Code

Каноническая шпаргалка по Claude Code для этого репозитория.

## Политика проекта

- модель по умолчанию: `opus`
- разрешённая модель в проектном shared-config: только `opus`
- effort по умолчанию: `high`
- для сложных консультаций допускается временный переход на effort `max`
- extended thinking: включён по умолчанию
- отдельный alias для консультаций и доработок: `Opus`
- встроенные Claude subagents для делегации заблокированы shared-config, чтобы не было ухода в `haiku/sonnet`

Проектные файлы:

- shared settings: `.claude/settings.json`
- Claude instructions: `CLAUDE.md`
- Opus subagent: `.claude/agents/Opus.md`

Пользовательский default-config текущего окружения:

- `~/.claude/settings.json`

## Установка и авторизация

Установка Claude Code на Linux/WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Авторизация:

```bash
claude auth login
claude auth status
```

## Как использовать в этом репозитории

Обычная сессия в корне репозитория уже поднимется с `opus` + `high` из shared-config:

```bash
claude
```

Явный запуск с project subagent `Opus`:

```bash
claude --agent Opus
```

Проверить, что агент доступен:

```bash
claude agents
```

## Что делает alias `Opus`

`Opus` нужен для:

- консультаций и second opinion
- архитектурного review
- prompt critique
- сложных доработок и redesign

Если задача особенно сложная, для такой консультации можно временно поднять effort до `max`, после чего вернуться к базовому `high`.

Ожидаемый паттерн:

1. основная сессия формулирует задачу;
2. при необходимости делегирует её в `Opus`;
3. результат `Opus` возвращается в main thread как рекомендации или готовые правки.

## Ограничение делегации

Shared-config также ставит deny-правила на встроенные subagents:

- `Explore`
- `general-purpose`
- `Plan`
- `statusline-setup`

Это нужно, чтобы делегация внутри проекта не уходила в встроенные профили на `haiku` или `sonnet`. Для проектной делегации оставлен только `Opus`.
