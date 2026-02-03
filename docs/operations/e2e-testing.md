# E2E BDD Testing Guide

> Ретроспектива реализации тестирования Dom Iskusstv. Консолидация подходов для будущих E2E тестов.

## Архитектура тестирования

### Структура директорий

```
tests/e2e/
├── features/                    # Gherkin сценарии
│   ├── dom_iskusstv.feature     # BDD сценарии на русском
│   ├── steps/
│   │   └── bot_steps.py         # Step definitions
│   └── environment.py           # Behave hooks (before/after)
├── human_client.py              # HumanUserClient wrapper
└── conftest.py                  # Pytest fixtures
```

### Ключевые компоненты

| Компонент | Назначение |
|-----------|------------|
| `HumanUserClient` | Telethon wrapper с human-like поведением |
| `behave` | BDD framework для Gherkin сценариев |
| `bot_steps.py` | Шаги для взаимодействия с ботом |

---

## Написание сценариев

### Формат Gherkin (русский синтаксис)

```gherkin
# language: ru
Функция: Парсинг событий с Дома искусств

  Сценарий: Парсинг событий по ссылке
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом
    Когда я отправляю команду "/start"
    И я жду сообщения с текстом "Choose action"
    Тогда под сообщением должна быть кнопка "🏛 Дом искусств"
```

### Паттерны шагов

#### Отправка сообщений
```python
@when('я отправляю команду "{cmd}"')
@when('я отправляю сообщение "{text}"')
```

#### Ожидание ответа
```python
@then('я жду сообщения с текстом "{text}"')
@then('я жду долгой операции с текстом "{text}"')  # до 5 минут
```

#### Проверка кнопок
```python
@then('под сообщением должна быть кнопка "{btn}"')
@then('под сообщением должны быть кнопки: "{buttons}"')  # через запятую
```

#### Нажатие кнопок
```python
@when('я нажимаю инлайн-кнопку "{btn}"')
```

---

## Работа с долгими операциями

### Проблема
Kaggle notebook выполняется 2-5 минут. Стандартный timeout недостаточен.

### Решение
```python
@then('я жду долгой операции с текстом "{text}"')
def step_wait_long_operation(context, text):
    """Wait up to 5 minutes for long operations like Kaggle."""
    async def _wait():
        for i in range(600):  # 5 минут = 600 * 0.5s
            messages = await context.client.client.get_messages(
                context.bot_entity, limit=10
            )
            for msg in messages:
                if msg.text and text.lower() in msg.text.lower():
                    context.last_response = msg
                    return
            await asyncio.sleep(0.5)
        raise AssertionError(f"Сообщение '{text}' не получено за 5 минут")
    run_async(context, _wait())
```

### Ключевые моменты
- Polling каждые 0.5 секунды вместо блокирующего ожидания
- Проверка последних 10 сообщений (не только последнее)
- Case-insensitive поиск текста

---

## Верификация контента Telegraph

### Проблема
Нужно проверить не только наличие ссылки, но и содержимое страницы.

### Решение
```python
@then('каждая Telegraph страница должна содержать "{required_text}"')
def step_verify_telegraph_content(context, required_text):
    """Verify content on all Telegraph pages."""
    items = [x.strip() for x in required_text.split(",")]
    
    async def _verify():
        async with aiohttp.ClientSession() as session:
            for link in context.telegraph_links:
                async with session.get(link) as resp:
                    html = await resp.text()
                    html_lower = html.lower()
                    for item in items:
                        if item.lower() not in html_lower:
                            raise AssertionError(
                                f"'{item}' не найден на {link}"
                            )
    run_async(context, _verify())
```

### Использование
```gherkin
И каждая Telegraph страница должна содержать "🎟, Билеты, руб."
```

---

## Отладка ошибок

### PAGE_ACCESS_DENIED

**Симптом:** Telegraph страницы не обновляются, старые данные.

**Причина:** Страница создана с другим `access_token`.

**Решение:** Fallback — создание новой страницы при ошибке:
```python
try:
    await telegraph_edit_page(...)
except Exception as e:
    if "PAGE_ACCESS_DENIED" in str(e):
        ev.telegraph_path = None  # Clear to create new
        # Create new page...
```

### Дублирование step definitions

**Симптом:** `AmbiguousStep` ошибка при запуске behave.

**Причина:** Один шаг определён дважды (разные декораторы).

**Решение:** Объединить в один decorator:
```python
# ❌ Было:
@then("я логирую...")
def step1(ctx): ...

@when("я логирую...")  
def step1(ctx): ...  # Дубликат!

# ✅ Стало:
@then("я логирую...")
@when("я логирую...")
def step_log(ctx): ...
```

### FloodWait от Telegram

**Симптом:** `Sleeping for Xs on GetHistoryRequest flood wait`

**Причина:** Слишком частые запросы к API.

**Решение:** Увеличить интервал polling, использовать human-like delays.

---

## Чеклист для нового E2E теста

### Подготовка
- [ ] Запустить бота локально в polling режиме: `DEV_MODE=1 DB_PATH=db_prod_snapshot.sqlite python main.py`
- [ ] Убедиться что бот отвечает на `/start`
- [ ] Проверить что нет конфликтов с production ботом
- [ ] Установить `TELEGRAM_API_ID`/`TELEGRAM_API_HASH` (или `TG_API_ID`/`TG_API_HASH`) и одну из: `TELEGRAM_AUTH_BUNDLE_E2E` или `TELEGRAM_SESSION`

#### TELEGRAM_AUTH_BUNDLE_E2E (формат и расшифровка)

`TELEGRAM_AUTH_BUNDLE_E2E` — это **base64 (urlsafe) JSON** с данными Telethon‑сессии и параметрами устройства.

Обязательные ключи в JSON:
- `session`
- `device_model`
- `system_version`
- `app_version`
- `lang_code`
- `system_lang_code`

Пример расшифровки (локально, без логирования результата):
```python
import base64, json

raw = base64.urlsafe_b64decode(B64.encode("ascii")).decode("utf-8")
bundle = json.loads(raw)
session = bundle["session"]
```

Важно: **не запускайте одну и ту же session строку параллельно** в двух процессах (иначе можно словить `AuthKeyDuplicatedError`). Разные session строки для одного аккаунта допустимы.

### Написание теста
- [ ] Создать `.feature` файл с Gherkin сценариями
- [ ] Использовать русский синтаксис (`# language: ru`)
- [ ] Добавить шаги в `bot_steps.py` если нужны новые

### Запуск
```bash
# Запуск бота
DB_PATH=db_prod_snapshot.sqlite python3 main.py &

# Запуск тестов
behave tests/e2e/features/your_feature.feature --no-capture
```

### Отладка
- [ ] Проверить логи бота (`bot.log`)
- [ ] Использовать `я логирую в консоль список всех кнопок`
- [ ] Проверить HTTP доступность Telegraph ссылок

---

## Мониторинг логов во время теста

### Проблема
При долгих операциях (Kaggle) важно видеть что происходит в боте параллельно с тестом, чтобы быстрее реагировать на ошибки.

### Решение: параллельный мониторинг

Запускаем три терминала/процесса:

```bash
# Терминал 1: Бот
DB_PATH=db_prod_snapshot.sqlite python3 main.py 2>&1 | tee bot.log

# Терминал 2: Тест с записью в файл
behave tests/e2e/features/dom_iskusstv.feature --no-capture 2>&1 | tee test_output.txt

# Терминал 3: Мониторинг логов бота
tail -f bot.log | grep -E "ERROR|WARNING|Kaggle|Telegraph|PAGE_ACCESS"
```

### Ключевые паттерны для grep

```bash
# Ошибки Telegraph
tail -f bot.log | grep -E "PAGE_ACCESS_DENIED|telegraph_edit|CONTENT_TOO_BIG"

# Kaggle статус
tail -f bot.log | grep -E "Kaggle|kernel|notebook|running|complete"

# Общие ошибки
tail -f bot.log | grep -E "ERROR|Exception|Traceback"

# Dom Iskusstv специфичные
tail -f bot.log | grep -E "dom_iskusstv|parse_dom|skazka"
```

### Периодическая проверка при polling

Во время ожидания долгой операции, каждые 30-60 секунд проверяйте:

```bash
# Последние строки лога
tail -n 20 bot.log

# Статус Telegraph fallback
grep -c "PAGE_ACCESS_DENIED" bot.log

# Проверка HTTP ссылок из лога
grep -oE "https://telegra\.ph/[^ ]+" bot.log | tail -5 | xargs -I{} curl -sI {} | grep HTTP
```

### Автоматизированный мониторинг

Скрипт для параллельного мониторинга:

```bash
#!/bin/bash
# monitor_e2e.sh

# Запуск бота в фоне
DB_PATH=db_prod_snapshot.sqlite python3 main.py > bot.log 2>&1 &
BOT_PID=$!
echo "Bot started: PID=$BOT_PID"

# Ждём старта
sleep 5

# Запуск теста в фоне
behave tests/e2e/features/$1 --no-capture > test_output.txt 2>&1 &
TEST_PID=$!
echo "Test started: PID=$TEST_PID"

# Мониторинг
while kill -0 $TEST_PID 2>/dev/null; do
    echo "=== $(date) ==="
    tail -n 5 bot.log | grep -v "^$"
    echo "---"
    sleep 30
done

# Результат
echo "Test finished. Exit code: $(wait $TEST_PID; echo $?)"
tail -n 20 test_output.txt
kill $BOT_PID 2>/dev/null
```

### Что искать в логах

| Лог-паттерн | Значение | Действие |
|-------------|----------|----------|
| `PAGE_ACCESS_DENIED` | Нет доступа к Telegraph | Проверить fallback |
| `Kaggle kernel running` | Notebook запущен | Ждать завершения |
| `CONTENT_TOO_BIG` | Страница слишком большая | Проверить compact mode |
| `FloodWait` | Telegram rate limit | Увеличить delays |
| `event_id=` | Событие обработано | Проверить Telegraph URL |

## Типичные ошибки и решения

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `Timeout` | Kaggle долго выполняется | Использовать `я жду долгой операции` |
| `AmbiguousStep` | Дублирование step definitions | Убрать дубликаты |
| `PAGE_ACCESS_DENIED` | Чужой access_token | Добавить fallback создания новой страницы |
| `message is too long` | > 4096 символов | Добавить compact fallback |
| `FloodWait` | Частые API запросы | Увеличить delays |

---

## Лучшие практики

1. **Изоляция тестов** — использовать snapshot БД, не трогать production
2. **Human-like поведение** — рандомизированные задержки для Telegram
3. **Детальное логирование** — `[REPORT]` блоки для debug
4. **Верификация контента** — проверять не только наличие, но и содержимое
5. **Graceful degradation** — fallback при ошибках API
6. **Timeout для Kaggle** — минимум 5 минут для notebook операций

---

## Примеры сценариев

### Базовый сценарий
```gherkin
Сценарий: Проверка стартового меню
  Дано я авторизован в клиенте Telethon
  И я открыл чат с ботом
  Когда я отправляю команду "/start"
  И я жду сообщения с текстом "Choose action"
  Тогда под сообщением должна быть кнопка "🏛 Дом искусств"
```

### Сценарий с долгой операцией
```gherkin
Сценарий: Парсинг событий
  Дано я авторизован в клиенте Telethon
  И я открыл чат с ботом
  Когда я отправляю сообщение "https://example.com/events"
  Тогда я жду долгой операции с текстом "импорт завершён"
  И я должен найти в ответе действующую ссылку на телеграф
  И каждая Telegraph страница должна содержать "🎟, Билеты, руб."
```

### Сценарий с проверкой кнопок
```gherkin
Сценарий: Проверка VK меню
  Когда я отправляю команду "/vk"
  Тогда под сообщением должны быть кнопки: "Проверить события, 🏛 Извлечь из Дом искусств"
```
