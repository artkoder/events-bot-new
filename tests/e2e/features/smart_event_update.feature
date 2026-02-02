# language: ru
Функция: Smart Update — пограничные сценарии

  Предыстория:
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом

  Сценарий: Идемпотентность по source_url
    Дано в базе создано тестовое событие:
      | title              | date       | time  | location_name     | source_text                 | description        | ticket_link                   | city        |
      | TEST SU idempotent | 2026-03-01 | 19:00 | Тестовая площадка | Тестовый текст.             | Тестовое описание | https://example.com/tickets/1 | Калининград |
    И для события "TEST SU idempotent" добавлен источник "test://smart-update/idempotent" типа "site"
    Когда я запускаю Smart Update на основе события "TEST SU idempotent" с правками:
      | field       | value                       |
      | source_url  | test://smart-update/idempotent |
      | source_type | site                        |
    Тогда результат Smart Update имеет статус "skipped_same_source_url"
    И для события "TEST SU idempotent" количество источников равно "1"
    И я очищаю тестовые события

  Сценарий: Якорные поля не перезаписываются при совпадении по билету
    Дано в базе создано тестовое событие:
      | title           | date       | time  | location_name     | source_text                 | description        | ticket_link                   | city        |
      | TEST SU anchors | 2026-03-02 | 19:00 | Тестовая площадка | Тестовый текст.             | Тестовое описание | https://example.com/tickets/2 | Калининград |
    И для события "TEST SU anchors" добавлен источник "test://smart-update/anchors/1" типа "site"
    Когда я запускаю Smart Update на основе события "TEST SU anchors" с правками:
      | field       | value                         |
      | source_url  | test://smart-update/anchors/2 |
      | source_type | site                          |
      | time        | 20:00                         |
      | source_text | Тестовый текст.               |
    Тогда результат Smart Update имеет статус "merged"
    И событие "TEST SU anchors" имеет поля:
      | field         | value             |
      | date          | 2026-03-02        |
      | time          | 19:00             |
      | location_name | Тестовая площадка |
    И для события "TEST SU anchors" количество источников равно "2"
    И я очищаю тестовые события

  Сценарий: Первичный импорт пишет added_facts в лог
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                         |
      | title         | TEST SU created               |
      | source_type   | manual                        |
      | source_url    | test://smart-update/created/1 |
      | date          | 2026-03-03                    |
      | time          | 18:00                         |
      | location_name | Тестовая площадка 2           |
      | source_text   | Событие в Тестовой площадке 2 |
      | raw_excerpt   | Короткое описание события     |
      | city          | Калининград                   |
    Тогда результат Smart Update имеет статус "created"
    И создано новое событие с заголовком "TEST SU created"
    И для события "TEST SU created" лог фактов содержит "Дата: 2026-03-03"
    И для события "TEST SU created" лог фактов содержит "Время: 18:00"
    И я очищаю тестовые события

  Сценарий: Хештеги удаляются при создании события
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                          |
      | title         | TEST SU hashtagged #tag        |
      | source_type   | manual                         |
      | source_url    | test://smart-update/hashtags/1 |
      | date          | 2026-03-03                     |
      | time          | 20:00                          |
      | location_name | Тестовая площадка 3            |
      | source_text   | Текст #tag                     |
      | raw_excerpt   | Описание #tag                  |
      | city          | Калининград                    |
    Тогда результат Smart Update имеет статус "created"
    И создано новое событие с заголовком "TEST SU hashtagged"
    И событие "TEST SU hashtagged" имеет поля:
      | field       | value             |
      | title       | TEST SU hashtagged |
      | description | Описание          |
    И я очищаю тестовые события

  Сценарий: manual/bot может исправлять title без LLM-мерджа и добавляет источник
    Дано в базе создано тестовое событие:
      | title             | date       | time  | location_name     | source_text   | description      | city        |
      | TEST SU title fix | 2026-03-05 | 19:00 | Тестовая площадка | Базовый текст | Базовое описание | Калининград |
    И для события "TEST SU title fix" добавлен источник "test://smart-update/title-fix/base" типа "site"
    Когда я запускаю Smart Update на основе события "TEST SU title fix" с правками:
      | field       | value                           |
      | source_type | manual                          |
      | source_url  | test://smart-update/title-fix/1 |
      | title       | TEST SU title fix updated #tag  |
      | source_text | Базовый текст                   |
    Тогда результат Smart Update имеет статус "merged"
    И событие "TEST SU title fix updated" имеет поля:
      | field | value                     |
      | title | TEST SU title fix updated |
    И для события "TEST SU title fix updated" количество источников равно "2"
    И я очищаю тестовые события

  Сценарий: allow_parallel_events + hall_hint создаёт отдельное событие
    Дано в базе создано тестовое событие:
      | title              | date       | time  | location_name      | source_text                  | description | city        |
      | TEST SU parallel A | 2026-03-04 | 18:30 | Научная библиотека  | Лекторий 1: тестовое событие | Описание A  | Калининград |
      | TEST SU parallel B | 2026-03-04 | 18:30 | Научная библиотека  | Лекторий 2: тестовое событие | Описание B  | Калининград |
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                           |
      | title         | TEST SU parallel C              |
      | source_type   | telegram                        |
      | source_url    | test://smart-update/parallel/1  |
      | date          | 2026-03-04                      |
      | time          | 18:30                           |
      | location_name | Научная библиотека              |
      | source_text   | Лекторий 3: отдельное событие   |
      | raw_excerpt   | Лекторий 3: отдельное событие   |
      | city          | Калининград                     |
    Тогда результат Smart Update имеет статус "created"
    И создано новое событие с заголовком "TEST SU parallel C"
    И я очищаю тестовые события

  Сценарий: allow_parallel_events + hall_hint матчит правильное событие (без LLM)
    Дано в базе создано тестовое событие:
      | title                | date       | time  | location_name      | source_text                  | description | city        |
      | TEST SU hall match A | 2026-03-06 | 18:30 | Научная библиотека  | Лекторий 1: тестовое событие | Описание A  | Калининград |
      | TEST SU hall match B | 2026-03-06 | 18:30 | Научная библиотека  | Лекторий 2: тестовое событие | Описание B  | Калининград |
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                          |
      | title         | TEST SU hall match A           |
      | source_type   | telegram                       |
      | source_url    | test://smart-update/hall/1     |
      | date          | 2026-03-06                     |
      | time          | 18:30                          |
      | location_name | Научная библиотека             |
      | source_text   | Лекторий 1: тестовое событие   |
      | raw_excerpt   | Лекторий 1: тестовое событие   |
      | city          | Калининград                    |
    Тогда результат Smart Update имеет статус "merged"
    И Smart Update вернул event_id как у события "TEST SU hall match A"
    И для события "TEST SU hall match A" количество источников равно "1"
    И я очищаю тестовые события
