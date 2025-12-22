# Диагностические логи для извлечения текста Telegraph

## Описание изменений

Добавлены подробные диагностические логи для отладки проблем с извлечением текста из Telegraph в пайплайне видеоанонса (/v).

## Файлы изменений

1. **main_part2.py** - функция `get_source_page_text()`
2. **video_announce/selection.py** - функция `_fetch_telegraph_text()`
3. **video_announce/scenario.py** - функция `_log_telegraph_token_diagnostics()`

## Новые логи

### 1. telegraph_fetch start (INFO)
Логируется при каждом запросе к Telegraph API:
- `path` - нормализованный путь к странице
- `token_source` - источник токена: `env|file|created|none|file_empty|file_error|created_failed`
- `has_token` - есть ли доступный токен
- `token_file_path` - путь к файлу токена
- `token_file_exists` - существует ли файл токена

### 2. telegraph_fetch no_token (WARNING)
Логируется когда токен недоступен:
- `path` - запрашиваемый путь
- `token_source` - причина отсутствия токена
- `token_file_exists` - существование файла токена
- `token_file_path` - путь к файлу токена

### 3. telegraph_fetch response (DEBUG)
Логируется после успешного ответа API:
- `path` - запрашиваемый путь
- `resp_type` - тип ответа (например `dict`)
- `resp_keys` - список ключей в ответе
- `len_html` - длина HTML контента
- `len_text` - длина текстового контента

### 4. telegraph_fetch empty_content (WARNING)
Логируется когда API возвращает пустой контент:
- `path` - запрашиваемый путь
- `token_source` - источник токена
- `resp_keys` - ключи ответа
- `len_html` - длина HTML (0)
- `len_text` - длина текста (0)

### 5. telegraph_fetch exception (ERROR)
Логируется при ошибках API с полным stacktrace:
- `path` - запрашиваемый путь
- `token_source` - источник токена
- `exception_type` - тип исключения

### 6. telegraph_event_fetch start (INFO)
Логируется в video_announce при начале обработки события:
- `event_id` - ID события
- `telegraph_url` - исходный URL
- `telegraph_path` - исходный path
- `resolved_path` - нормализованный path

### 7. telegraph_event_fetch no_path (WARNING)
Логируется когда у события нет telegraph_path:
- `event_id` - ID события
- `telegraph_url` - URL из базы
- `telegraph_path` - path из базы

### 8. telegraph_text_empty (WARNING)
Логируется когда извлеченный текст пустой:
- `event_id` - ID события
- `telegraph_url` - URL из базы
- `telegraph_path` - path из базы
- `resolved_path` - нормализованный path
- `text_len` - длина текста (0)

### 9. telegraph_token_diagnostics (INFO)
Логируется один раз на старте пайплайна /v:
- `env_present` - есть ли токен в переменных окружения
- `token_file_path` - путь к файлу токена
- `token_file_exists` - существует ли файл
- `token_file_readable` - доступен ли для чтения
- `token_file_empty` - пуст ли файл

## Как читать логи

### Проблема: токен недоступен
```
1. telegraph_token_diagnostics - показывает источник проблемы
2. telegraph_fetch no_token - подтверждает проблему с токеном
```

### Проблема: неправильный path
```
1. telegraph_event_fetch start - показывает исходные данные
2. telegraph_fetch start - показывает нормализованный path
3. telegraph_fetch exception или empty_content - результат
```

### Проблема: API возвращает пусто
```
1. telegraph_fetch start - показывает начало запроса
2. telegraph_fetch response - показывает len_html=0, len_text=0
3. telegraph_fetch empty_content - подтверждение проблемы
4. telegraph_text_empty - финальное предупреждение для события
```

### Успешный случай
```
1. telegraph_token_diagnostics - токен OK
2. telegraph_event_fetch start - событие обрабатывается
3. telegraph_fetch start - запрос к API
4. telegraph_fetch response - есть контент (len_html>0 или len_text>0)
```

## Безопасность

- Токены, cookies, секреты НЕ логируются
- Полный HTML/текст страниц НЕ логируется
- Логируются только метаданные и длины контента
- Preview контента ограничен 200 символами на уровне DEBUG

## Мини-тест план

### Тест 1: Событие с валидным telegraph_path
1. Запустить /v с событием имеющим telegraph_path
2. Ожидать: `telegraph_fetch response` с `len_html>0` или `len_text>0`

### Тест 2: Событие без path
1. Запустить /v с событием без telegraph_path
2. Ожидать: `telegraph_event_fetch no_path`

### Тест 3: Отсутствие токена
1. Удалить/переименовать токен файл
2. Убрать TELEGRAPH_TOKEN из env
3. Запустить /v
4. Ожидать: `telegraph_fetch no_token`

## Мониторинг production

Для отслеживания проблем в production рекомендуется настроить алерты на:

- `telegraph_fetch no_token` - критично, нет доступа к API
- `telegraph_fetch empty_content` - предупреждение, возможно проблема с конкретной страницей
- `telegraph_text_empty` - информационно, помогает выявить проблемные события
- `telegraph_fetch exception` - критично, ошибки API

Логи позволяют быстро определить первопричину проблем с пустыми telegraph_text/telegraph_full_text полями в пайплайне видеоанонса.