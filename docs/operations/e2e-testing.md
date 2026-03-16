# E2E BDD Testing Guide

> Ретроспектива реализации тестирования Dom Iskusstv. Консолидация подходов для будущих E2E тестов.

Список всех актуальных E2E сценариев и их «срока годности»: `docs/operations/e2e-scenarios.md`.

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
@then('я жду долгой операции с текстом "{text}"')  # до 35 минут (Kaggle/парсеры)
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
Kaggle notebook и парсеры могут выполняться долго (cold start, OCR/vision, сеть). Стандартный timeout недостаточен.

### Решение
```python
@then('я жду долгой операции с текстом "{text}"')
def step_wait_long_operation(context, text):
    """Wait for long operations like Kaggle (configurable, default is generous)."""
    async def _wait():
        # Timeout can be overridden via env (e.g. E2E_TG_MONITOR_TIMEOUT_SEC, E2E_PARSE_TIMEOUT_SEC).
        for i in range(600):  # пример: 5 минут = 600 * 0.5s
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
- Для тяжёлых guide runs отдельный timeout тоже должен быть generous: `Мониторинг экскурсий завершён` теперь использует `E2E_GUIDE_MONITOR_TIMEOUT_SEC` и `E2E_GUIDE_MONITOR_POLL_SEC`, а не общий 5-минутный fallback.

---

## Реакция на ошибки в Telegram UI (обязательно)

Live E2E — это не только `behave`-assert’ы, но и **операторские сообщения бота в Telegram**.

- Любое сообщение с шаблоном `Результат: ошибка …` считается **провалом прогона** и требует расследования.
- Не допускается “тихий пропуск” постов/событий из‑за ошибок: E2E должен падать как можно раньше.
- Логи для расследования: `artifacts/test-results/e2e_local_bot_*.log` + текст сообщения в Telegram UI.

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

### Проверка отсутствия «раздутой логистики»

Если инфоблок (дата/место/билеты) уже показан сверху, `description` не должен дублировать его словами вроде «по адресу», «по телефону», «стоимость билета».

```gherkin
И каждая Telegraph страница не должна содержать "по адресу, по телефону, стоимость билета"
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

## Старые контрольные посты (важно для скорости и лимитов Gemma)

Если сценарий E2E ссылается на **конкретный** Telegram‑пост для сверки (по `message_id`/URL) и этот пост уже старый, не расширяйте глобально окно перескана.

Рекомендованный подход:

- держать базовый профиль мониторинга как обычно: `TG_MONITORING_DAYS_BACK=3` (и умеренный `TG_MONITORING_LIMIT`);
- нужный конкретный пост добирать **точечно**: отдельным шагом сценария вида `И я выбираю конкретный пост "https://t.me/<channel>/<id>"`.

Почему так:

- меньше лишних запросов в Gemma (лимиты не бесконечны),
- ниже риск FloodWait,
- E2E прогон заметно быстрее.

---

## Чеклист для нового E2E теста

### Подготовка
- [ ] Обновить snapshot прод-БД (для live-сценариев): `./scripts/sync_prod_db_if_stale.sh --max-age-hours 6`
- [ ] Подготовить **изолированную** БД для E2E (чтобы прогон не мутировал snapshot):
  - `eval "$(./scripts/prepare_e2e_db_from_prod_snapshot.sh --max-age-hours 6)"`
  - (альтернатива) вручную сделать sqlite backup/copy в `artifacts/test-results/` и выставить `DB_PATH` на копию
- [ ] Запустить бота локально в polling режиме (использует `DB_PATH` из шага выше):
  - рекомендуемо: `DEV_MODE=1 python main.py`
  - или просто `python main.py` (если `WEBHOOK_URL` не задан — бот сам стартует в polling режиме)
  - если `WEBHOOK_URL` задан, но нужен polling: `FORCE_POLLING=1 python main.py`
- [ ] Если админ-команды (`/vk`, `/fest_queue`, и т.п.) отвечают `Access denied` для тестового пользователя:
  - один раз выполнить `python scripts/seed_dev_superadmin.py` (выдаёт superadmin в sqlite `DB_PATH` для Telethon‑аккаунта из `TELEGRAM_AUTH_BUNDLE_E2E`/`TELEGRAM_SESSION`)
- [ ] Если БД-снимок лежит на нестабильной FS, выставить `DB_JOURNAL_MODE=DELETE` (иначе возможны ошибки WAL)
- [ ] Убедиться что бот отвечает на `/start`
- [ ] Проверить что нет конфликтов с production ботом
- [ ] Установить `TELEGRAM_API_ID`/`TELEGRAM_API_HASH` (или `TG_API_ID`/`TG_API_HASH`) и одну из: `TELEGRAM_AUTH_BUNDLE_E2E` или `TELEGRAM_SESSION`
- [ ] (Опционально) выставить `E2E_BOT_USERNAME`, чтобы E2E не делал HTTP вызов в Bot API (`getMe`) для определения username (полезно, если `api.telegram.org` недоступен, но MTProto доступен).
- [ ] Перед каждым повторным прогоном сценария на конкретных постах выполнить предочистку: `И база очищена от событий источника "<source>"` + `И очищены отметки мониторинга для "<username>"`, иначе в проверку попадут следы прошлых прогонов.

### Границы Telegram-сессий (обязательно)

- `TELEGRAM_AUTH_BUNDLE_E2E` / `TELEGRAM_SESSION` используются только для локального live E2E и human-like Telethon клиента.
- `TELEGRAM_AUTH_BUNDLE_S22` используется только для Kaggle / remote monitoring runs.
- Нельзя без явного разрешения подменять `S22` на `E2E` в `GUIDE_MONITORING_AUTH_BUNDLE_ENV` или аналогичных Kaggle-путях.
- Если `S22` сломана, это отдельный инцидент: нужно остановиться, зафиксировать проблему и попросить новую Kaggle-сессию, а не “занимать” локальную E2E-сессию.
- Одновременное использование одного и того же auth bundle локально и в Kaggle может привести к `AuthKeyDuplicatedError`.

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
# Подготовка изолированной БД из snapshot (и выставление DB_PATH)
eval "$(./scripts/prepare_e2e_db_from_prod_snapshot.sh --max-age-hours 6)"

# Запуск бота
python3 main.py &

# Запуск тестов
behave tests/e2e/features/your_feature.feature --no-capture
```

Примечание про `@manual` сценарии:
- `@manual` — это тяжёлые/долгие live-сценарии (массовые прогоны).
- Чтобы они запускались, используйте **один** из вариантов:
  - `E2E_RUN_MANUAL=1 behave ...`
  - `behave ... --tags=manual`

Примечание:
- Если вы запускаете `behave`, `tests/e2e/features/environment.py` **автоматически** делает копию, когда `DB_PATH`
  указывает на `db_prod_snapshot*.sqlite` в корне репозитория (можно отключить `E2E_DB_ISOLATE=0`).
- Если `DB_PATH` не задан, `tests/e2e/features/environment.py` пытается выбрать snapshot автоматически:
  - сначала `db_prod_snapshot.sqlite` (если файл “здоровый”);
  - иначе — самый новый `db_prod_snapshot_*.sqlite` по времени изменения (если он “здоровый”).
  Если snapshot повреждён (например, незавершённая загрузка), он будет пропущен с warning в логах E2E.

Примечание про переменные окружения:
- `export ...` в одном терминале влияет только на текущую shell-сессию и её дочерние процессы. Если IDE открыла новый терминал/процесс, переменные “исчезнут”.
- Для E2E предпочтительнее хранить настройки в `.env` в корне репозитория: `tests/e2e/features/environment.py` подхватывает `.env` автоматически (best-effort), заполняя отсутствующие переменные.
  - Если в окружении уже задан **явно неверный** `SUPABASE_KEY`/`SUPABASE_SERVICE_KEY` (например слишком короткое значение), E2E‑раннер заменит его на значение из `.env` (иначе ломаются media upload fallback’и, где нужен доступ к Storage).
  - Если ты запускаешь бота вручную из терминала (не через `behave`), `.env` в процесс не подхватится сам — перед `python main.py` выполни:
    - `set -a; source .env; set +a`

### Быстрая диагностика “бот не отвечает на команды”

Симптом: отправляешь `/start` или `/vk`, а в Telegram тишина.

1) Проверь в логах строку режима:
   - ожидаемо для local/live E2E: `Mode: DEV_MODE | Connection: POLLING | Webhook: DISABLED`
   - если видишь `Mode: PROD_MODE | Connection: WEBHOOK | Polling: DISABLED` — бот не будет получать апдейты без корректно настроенного webhook.
2) Запусти заново в polling режиме:
   - `DEV_MODE=1 python main.py` или `FORCE_POLLING=1 python main.py`
3) Если запускаешь вручную — не забудь подхватить `.env`:
   - `set -a; source .env; set +a`

Опционально (чтобы уменьшить шум/фоновую активность при live E2E): `DISABLE_PAGE_JOBS=1`.

Если в рамках E2E вы проверяете только **страницы событий** (Telegraph build) и не хотите, чтобы фоновая очередь пыталась пересобирать агрегирующие страницы (месяц/неделя/выходные), запускайте бота с:

```bash
DISABLE_PAGE_JOBS=1
```

Это уменьшает шум и риск долгих блокирующих операций (например `PAGE_ACCESS_DENIED` на старых страницах).

Если при этом вы отключаете JobOutbox worker (чтобы не пытаться разбирать весь backlog прод‑снапшота), но всё равно хотите, чтобы Telegram Monitoring в E2E обновлял Telegraph страницы **только затронутых событий**, используйте:

```bash
ENABLE_JOB_OUTBOX_WORKER=0 TG_MONITORING_DRAIN_EVENT_JOBS=1
```

Поведение: после импорта результатов мониторинга бот в фоне “дренит” задачи только для `event_id`, которые были `created/merged` этим мониторингом (обычно `ics_publish` + `telegraph_build`). Это делает проверку “афиша появилась на Telegraph” детерминированной, не включая глобальный воркер.

### Окно сканирования для E2E (лимиты Gemma + время прогона)

- По умолчанию держите `TG_MONITORING_DAYS_BACK=3` и не расширяйте окно “на всякий случай”.
- Если нужно проверить **конкретный старый пост** (например `https://t.me/dramteatr39/3802`), не увеличивайте `DAYS_BACK` для всего источника и не “прокручивайте” историю назад.
- Правильный паттерн для E2E: обычный прогон оставляем с `TG_MONITORING_DAYS_BACK=3` (быстро, без лишних LLM вызовов).
- Правильный паттерн для E2E: конкретный пост запрашиваем **дополнительно** точечно по `message_id` (Telethon `get_messages(ids=...)`) и прогоняем импорт/проверку только для него.
- Это уменьшает нагрузку на Telegram API и снижает число лишних LLM/Gemma запросов (там лимиты не бесконечные), а также ускоряет E2E.

Пример точечной выборки поста:

```python
msg = await client.get_messages("dramteatr39", ids=3802)
```

### Offline (без сети / без Telegram)

Часть сценариев помечена тегом `@offline` и не требует Telethon/Telegram API (DB‑only регрессии).

```bash
E2E_OFFLINE=1 SMART_UPDATE_LLM=off EVENT_TOPICS_LLM=off DB_PATH=db_prod_snapshot.sqlite \\
  DB_INIT_MINIMAL=1 \\
  behave tests/e2e/features/smart_event_update.feature --no-capture
```

## Рекомендуемые сценарии (Gherkin) для регрессий мёржа

При изменениях в Telegram Monitoring / Smart Update полезно держать отдельные сценарии, которые ловят типовые регрессии:

- **Мёрж текста**: новый абзац из источника не теряется, но старые факты не дублируются.
- **Афиши**: промо‑баннеры (скидки/акции) не мёржатся как иллюстрации события; в логе источников нет дублей `Афиша в источнике`/`Добавлена афиша` для одного URL.
- **Розыгрыши**: пост “анонс + розыгрыш” импортируется/мёржится по фактам события, а “механика розыгрыша” вырезается.
- **/3di**: если у события меняется число/набор иллюстраций, `preview_3d_url` сбрасывается и событие снова попадает в “🆕 Только новые”.
- **Telegram-first двухфазно**: фиксируются два состояния одного события (`telegram-first` и `after-parse`) с отдельными snapshot-артефактами (лог источников + Telegraph текст), чтобы можно было вручную сравнить, как изменился текст и какие факты добавились.
- **Смысловые тезисы Telegram**: в `telegram-first` логе должен быть минимум один текстовый тезис события (не только `Дата/Время/Локация/Афиша`).
- **Согласованность фактов**: после `/parse` смысловые факты из лога должны присутствовать в тексте Telegraph страницы.
- **Анти-шум multi-event**: в Telegraph тексте не должно оставаться строк расписания/чужих названий из того же Telegram поста.
- **Telegram preview**: для итоговой Telegraph страницы события должен собираться `cached_page + photo` (иначе в клиенте возможен “чёрный экран” превью).
  - Частая причина отсутствия `cached_page`: на странице события есть `<img>` с битой ссылкой (например, Catbox `404`), из-за чего Telegram не может собрать Instant View.
  - Бот при сборке/обновлении event‑страницы теперь (по умолчанию) проверяет типовые источники иллюстраций (Catbox/Supabase) и **подменяет** битые URL на fallback (если есть) или **удаляет** их из набора иллюстраций. Управляется через `TELEGRAPH_VALIDATE_IMAGE_URLS=1` и `HTTP_IMAGE_UA` (User-Agent для скачивания изображений).

Канонические файлы сценариев:
- `tests/e2e/features/telegram_monitoring.feature`
- `tests/e2e/features/smart_event_update.feature`

Артефакты двухфазных snapshot-проверок:
- `artifacts/e2e/stage_snapshots/<timestamp>_<scenario>/telegram-first.json`
- `artifacts/e2e/stage_snapshots/<timestamp>_<scenario>/telegram-first.source_log.txt`
- `artifacts/e2e/stage_snapshots/<timestamp>_<scenario>/telegram-first.telegraph.txt`
- `artifacts/e2e/stage_snapshots/<timestamp>_<scenario>/after-parse.*`

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
