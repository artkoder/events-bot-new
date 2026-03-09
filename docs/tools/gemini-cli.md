# Gemini CLI

Короткая шпаргалка по локальному `gemini` CLI для консультаций, second opinion и быстрых одноразовых запросов из терминала.

Для двухэтапного workflow по trigger phrase `комплексная консультация` см. канонику `docs/tools/complex-consultation.md`.

## Когда использовать

- Нужна внешняя консультация по архитектуре, качеству текста, отладке или trade-off.
- Нужен быстрый одноразовый ответ без ручного копирования в веб-интерфейс.
- Нужен отдельный взгляд на промпт, dry-run или код-ревью материалы.

## Установка и проверка

Установка:

```bash
npm install -g @google/gemini-cli
```

Проверка:

```bash
gemini --version
command -v gemini
```

## Базовые команды

Интерактивная сессия:

```bash
gemini
```

Один вопрос в headless-режиме:

```bash
gemini -p "Объясни, почему этот SQL-запрос может делать full scan"
```

Явный выбор модели:

```bash
gemini -m gemini-3.1-pro-preview -p "Сделай критичный review этого плана миграции"
```

Запрос из файла:

```bash
cat artifacts/codex/tasks/brief.md | gemini -m gemini-3.1-pro-preview -p "Проведи review"
```

## Рекомендуемый сценарий для консультаций

1. Подготовь краткий, проверяемый контекст.
2. Явно задай роль Gemini: review, critique, alternative design, prompt audit.
3. Если нужен более сильный ответ, укажи модель через `-m`.
4. Если ответ важен для проекта, сохрани итоговый brief/report в `artifacts/codex/`.

Пример:

```bash
gemini -m gemini-3.1-pro-preview -p "
Ты делаешь строгий технический review.
Найди риски, регрессии и слабые предположения в этом плане:
1. ...
2. ...
3. ...
"
```

## Как проверить, какая модель реально ответила

Текстовый ответ модели про саму себя может быть неточным. Для надёжной проверки смотри session log Gemini CLI.

Сессии лежат в:

- `~/.gemini/tmp/<project>/chats/`
- `~/.gemini/tmp/tmp/chats/`

Быстрый просмотр последней сессии:

```bash
LATEST=$(ls -1t ~/.gemini/tmp/tmp/chats/*.json | head -n1)
rg -n '"content":|"model":' "$LATEST"
```

Если запрос запускался из каталога репозитория, полезно проверить и project-specific логи:

```bash
LATEST=$(ls -1t ~/.gemini/tmp/events-bot-new/chats/*.json | head -n1)
rg -n '"content":|"model":' "$LATEST"
```

Важно:

- поле `"content"` показывает текст, который Gemini написала;
- поле `"model"` показывает модель, которая реально сгенерировала ответ;
- ориентируйся на `"model"`, если нужно подтвердить фактический идентификатор.

## Практические замечания

- Для коротких консультаций удобнее `gemini -p "..."`.
- Для длинных контекстов удобнее `cat file.md | gemini -p "..."`.
- Если нужна конкретная модель, всегда указывай `-m`, а не полагайся на auto-selection.
- Если Gemini отвечает о своей модели противоречиво, source of truth здесь это session log CLI, а не текст ответа.

## Артефакты

Если консультация влияет на решение по проекту:

- brief складывай в `artifacts/codex/tasks/`
- итоговый review/report складывай в `artifacts/codex/reports/`

Не коммить временные логи Gemini из `~/.gemini/`.
