# Работа с продакшн-данными локально

Это руководство описывает как скачать снимок продакшн базы данных и использовать её для локального тестирования и отладки.

## 🎯 Зачем это нужно

- **Реалистичное тестирование**: тестируйте на реальных данных из продакшена
- **Отладка**: воспроизводите баги с реальными данными
- **Разработка**: разрабатывайте новые функции с полной БД
- **LLM эксперименты**: тестируйте LLM операции на реальных событиях

## 📦 Методы получения данных

### Метод 1: Прямое скачивание через Fly.io SSH (рекомендуется)

Самый простой и быстрый способ - скачать БД напрямую с сервера:

```bash
# Установите flyctl если ещё не установлен
curl -L https://fly.io/install.sh | sh

# Скачайте базу данных
./scripts/sync_prod_db.sh

# База будет сохранена как ./db_prod_snapshot.sqlite
```

**Опции:**

```bash
# Указать другое приложение
./scripts/sync_prod_db.sh --app your-app-name

# Сохранить в другое место
./scripts/sync_prod_db.sh --output /path/to/db.sqlite
```

### Метод 2: Через команду бота /dumpdb

Альтернативный метод - использовать встроенную команду бота:

```bash
# Настройте переменные окружения
export TELEGRAM_BOT_TOKEN=your_token
export ADMIN_CHAT_ID=your_chat_id

# Запустите скрипт
./scripts/dump_prod_db.sh
```

Бот отправит SQL дамп в чат. Скачайте файл и импортируйте:

```bash
sqlite3 db_prod_snapshot.sqlite < dump.sql
```

## 🚀 Использование продакшн-данных

### 1. Базовое использование

После скачивания базы данных:

```bash
# Настройте окружение
export DEV_MODE=1
export DB_PATH=./db_prod_snapshot.sqlite
export DB_JOURNAL_MODE=DELETE   # рекомендуется для локальных/виртуальных FS (устраняет проблемы WAL)
export TELEGRAM_BOT_TOKEN=your_dev_bot_token
export FOUR_O_TOKEN=your_openai_token

# Запустите бота локально
python main.py
```

### 0. Рекомендуемый формат имени снимка (чтобы не путать)

Сохраняйте снимки с таймстампом:

`db_prod_snapshot_YYYY-MM-DD_HHMM.sqlite`

Пример:

`db_prod_snapshot_2026-02-03_1915.sqlite`

И используйте через `DB_PATH=...` (см. выше). Это упрощает параллельные прогоны E2E и сравнение результатов.

Рекомендуемая раскладка (локально, не коммитить):

- хранить слепки в `artifacts/db/` (см. `artifacts/README.md`);
- `db_prod_snapshot.sqlite` можно использовать как “текущий” файл (копию), но для прогонов/сравнений всегда фиксируйте конкретный таймстамп‑файл в `DB_PATH`.

### 2. Тестирование отдельных операций

Используйте утилиту для тестирования:

```bash
# Показать статистику базы данных
python scripts/test_with_prod_data.py stats

# Протестировать LLM на конкретном событии
python scripts/test_with_prod_data.py test-llm --event-id 123

# Посмотреть статистику VK review
python scripts/test_with_prod_data.py test-vk-review

# Экспортировать примеры данных для unit-тестов
python scripts/test_with_prod_data.py export-sample --output ./test_data
```

### 3. Настройка для разных сценариев

**Только чтение (анализ данных):**
```bash
export DB_PATH=./db_prod_snapshot.sqlite
export DEV_MODE=1
# Не устанавливайте TELEGRAM_BOT_TOKEN - бот не запустится
python scripts/test_with_prod_data.py stats
```

**Тестирование изменений без отправки в Telegram:**
```bash
export DB_PATH=./db_prod_snapshot.sqlite
export DEV_MODE=1
export TELEGRAM_BOT_TOKEN=test_token  # Фейковый токен
# Тестируйте логику без реальных API вызовов
```

**Полноценное тестирование с тестовым ботом:**
```bash
export DB_PATH=./db_prod_snapshot.sqlite
export DEV_MODE=1
export TELEGRAM_BOT_TOKEN=your_test_bot_token
python main.py
```

## 🔄 Регулярное обновление

Для регулярной синхронизации с продакшеном:

```bash
# Обновлять snapshot "не старше 6 часов" (рекомендуется для E2E live-сценариев):
./scripts/sync_prod_db_if_stale.sh --max-age-hours 6

# Или просто скачать прямо сейчас:
./scripts/sync_prod_db.sh

# Пример cron (опционально, раз в 6 часов):
# 0 */6 * * * cd /path/to/project && ./scripts/sync_prod_db_if_stale.sh --max-age-hours 6
```

## ⚠️ Важные моменты

### Безопасность

1. **Никогда не коммитьте продакшн-данные** в git
2. Добавьте в `.gitignore`:
   ```
   db_prod_snapshot.sqlite
   backups/
   test_data/
   ```
3. Продакшн база может содержать персональные данные - обращайтесь с ней аккуратно

### Автоматические бэкапы

Скрипт `sync_prod_db.sh` автоматически создаёт бэкапы существующих снимков в директории `./backups/` с временной меткой.

### Размер базы данных

Продакшн база может быть большой (несколько MB или больше). Учитывайте это при:
- Копировании файлов
- Работе в IDE (может быть медленно)
- Git операциях (убедитесь что база в .gitignore)

## 🧪 Примеры использования

### Тестирование LLM на реальных событиях

```python
# Создайте файл test_llm_local.py
import asyncio
from db import Database
from main import parse_event_via_llm

async def test_llm():
    db = Database("./db_prod_snapshot.sqlite")
    await db.init()
    
    # Получите реальный текст события из БД
    async with db.get_session() as session:
        # ... ваш код ...
        
        # Протестируйте LLM
        result = await parse_event_via_llm(text, source_channel="test")
        print(result)

asyncio.run(test_llm())
```

### Анализ данных VK Inbox

```python
import asyncio
from db import Database
from models import VKInbox
from sqlalchemy import select

async def analyze_vk_inbox():
    db = Database("./db_prod_snapshot.sqlite")
    await db.init()
    
    async with db.get_session() as session:
        result = await session.execute(
            select(VKInbox).where(VKInbox.status == 'pending')
        )
        
        for item in result.scalars():
            print(f"Post: {item.vk_post_id}, Text: {item.text[:100]}...")

asyncio.run(analyze_vk_inbox())
```

## 📊 Структура файлов

```
events-bot-new/
├── scripts/
│   ├── sync_prod_db.sh          # Скачивание БД с Fly.io
│   ├── dump_prod_db.sh          # Получение дампа через бота
│   └── test_with_prod_data.py   # Утилиты для тестирования
├── backups/                     # Автоматические бэкапы (gitignored)
│   └── db_snapshot_*.sqlite
├── test_data/                   # Экспортированные примеры (gitignored)
│   ├── sample_events.json
│   └── sample_festivals.json
├── db_prod_snapshot.sqlite      # Снимок продакшн БД (gitignored)
└── docs/operations/prod-data.md # Это руководство
```

## 🛠 Troubleshooting

### flyctl not found

```bash
# Установите flyctl
curl -L https://fly.io/install.sh | sh

# Добавьте в PATH
export PATH="$HOME/.fly/bin:$PATH"
```

### Permission denied

```bash
# Дайте права на исполнение скриптам
chmod +x scripts/*.sh
```

### База данных заблокирована

```bash
# Убедитесь что бот не запущен
pkill -f "python main.py"

# Проверьте нет ли других процессов работающих с БД
lsof db_prod_snapshot.sqlite
```

### Ошибка при скачивании с Fly.io

```bash
# Проверьте авторизацию
fly auth whoami

# Если не авторизованы
fly auth login

# Проверьте доступ к приложению
fly status -a events-bot-new-wngqia
```

## 🔗 См. также

- [README.md](README.md) - Основная документация проекта
- [commands.md](commands.md) - Команды бота
- [../architecture/overview.md](../architecture/overview.md) - Архитектура системы
