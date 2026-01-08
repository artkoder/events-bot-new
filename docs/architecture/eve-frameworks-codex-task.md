# EVE Frameworks Architecture Design - Задача для Codex

**Модель**: gpt-5.2 (архитектор)  
**Цель**: Спроектировать архитектуру для трёх связанных задач: EVE-11, EVE-54, EVE-55

## Задачи

### EVE-11: Глобальный фреймворк управления лимитами LLM
- Контроль лимитов на запросы к Google AI (Gemma/Gemini)
- Поддержка нескольких API ключей с единым источником правды в Supabase
- Атомарное резервирование слотов (RPM/TPM/RPD)
- Ретраи (максимум 3) с проверкой лимитов перед каждой попыткой
- Переменные окружения: GOOGLE_API_KEY, GOOGLE_API_LOCALNAME

### EVE-54: Фреймворк безопасной передачи секретов в Kaggle
- Централизованный rate limit controller для Google AI
- Пул API ключей с распределением нагрузки
- Атомарные операции через Postgres RPC
- Режимы ожидания: WAIT / NO_WAIT
- Полное логирование успехов и отказов

### EVE-55: Подключение LLM Gemma
- Основная модель: gemma-3-27b через Google API
- Использование в боте и Kaggle ноутбуках
- Увязка с фреймворками лимитов и секретов

## Лимиты Google AI (из скриншота)

| Model | RPM | TPM | RPD |
|-------|-----|-----|-----|
| gemma-3-27b | 30 | 15,000 | 14,400 |
| gemini-2.5-flash | 5 | 250,000 | 20 |

## Существующая схема БД Supabase

Таблицы:
- `google_ai_usage_counters` - счётчики использования (id, api_key_id, minute_bucket, day_bucket, rpm_used, tpm_used, rpd_used, updated_at)
- `google_ai_api_keys` - пул API ключей (id, key_alias, env_var_name, is_active, priority, created_at, notes)
- `token_usage` - расход токенов (id, ts, model, prompt_tokens, completion_tokens, total_tokens, request_id, meta)
- `google_ai_maintenance_log` - лог операций (id, operation, tables_affected, deleted_count, started_at, completed_at, status, details, created_at)
- `google_ai_model_limits` - лимиты моделей (id, model, rpm, tpm, rpd, created_at, updated_at)

## Существующие интеграции

1. **llm_logger** - используется в:
   - `kaggle/UniversalFestivalParser/universal_festival_parser.py`
   - `kaggle/UniversalFestivalParser/src/reason.py`

2. **GOOGLE_API_KEY** - используется в:
   - `main.py`
   - `kaggle/UniversalFestivalParser/src/secrets.py`
   - `kaggle/UniversalFestivalParser/src/enrich.py`

3. **Supabase** - интеграция в `main.py`, `vk_intake.py` и тестах

## Требования к архитектуре

1. **SDK vs Gateway** - выбрать подход и обосновать
2. **Атомарность** - защита от гонок при параллельных запросах
3. **Аудит** - полное логирование всех операций
4. **Расширяемость** - добавление моделей/ключей через БД без изменения кода
5. **Единая точка правды** - все лимиты в Supabase

## ЗАДАНИЕ

Работай по фазам. После каждой фазы запиши промежуточный отчёт.

### Фаза 1: Исследование
1. Изучи существующий код работы с LLM в проекте
2. Проанализируй текущую структуру БД Supabase
3. Найди все места вызова Google AI API
4. Запиши найденные интеграции и зависимости
5. Результаты запиши в `docs/architecture/eve-arch-phase-1.md`

### Фаза 2: Архитектурный анализ
1. Предложи 2-3 архитектурных варианта реализации
2. Оцени риски гонок и предложи Postgres-паттерн
3. Определи компоненты системы и их взаимодействие
4. Обоснуй выбор подхода
5. Результаты запиши в `docs/architecture/eve-arch-phase-2.md`

### Фаза 3: Подготовка вопросов
Если есть неясности, которые требуют уточнения у владельца продукта:
- Сформулируй вопросы чётко и конкретно
- Группируй по темам
- Предложи варианты ответов где возможно
- Результаты запиши в `docs/architecture/eve-arch-questions.md`

### Фаза 4: Детальный дизайн (после получения ответов на вопросы)
1. Опиши схему данных (таблицы, поля, индексы)
2. Опиши алгоритм выбора ключа и расчёта окон
3. Опиши форматы логов и примеры записей
4. Опиши edge cases и их обработку
5. Составь план тестирования
6. Результаты запиши в `docs/architecture/eve-arch-design.md`

### Выходной артефакт
Сохрани итоговый сводный отчёт в: `docs/architecture/eve-frameworks-architecture-report.md`
