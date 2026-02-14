# language: ru
@offline
Функция: Smart Update — пограничные сценарии

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

  Сценарий: Мёрж текста добавляет новый факт и убирает дубли
    Дано в базе создано тестовое событие:
      | title               | date       | time  | location_name       | source_text     | description                                     | city        |
      | TEST SU text merge  | 2026-03-05 | 19:00 | Драматический театр | Старый источник | Спектакль в драмтеатре. Премия Арлекин-2010. Премия Арлекин-2010. | Калининград |
    И для события "TEST SU text merge" добавлен источник "test://smart-update/site/1" типа "site"
    Когда я запускаю Smart Update на основе события "TEST SU text merge" с правками:
      | field       | value                         |
      | source_type | telegram                       |
      | source_url  | test://smart-update/tg/1        |
      | source_text | Спектакль «Лорд Фаунтлерой».\\nПрекрасный дуэт Александра Егорова и Павла Самоловова. |
      | raw_excerpt | Спектакль «Лорд Фаунтлерой».    |
    Тогда результат Smart Update имеет статус "merged"
    И описание события "TEST SU text merge" содержит "дуэт"
    И в описании события "TEST SU text merge" фрагмент "Арлекин-2010" встречается ровно "1" раз
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

  Сценарий: Поздравления не импортируются как события
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                  |
      | title         | TEST SU congrats                       |
      | source_type   | telegram                               |
      | source_url    | test://smart-update/congrats/1         |
      | date          | 2026-03-07                             |
      | time          | 19:00                                  |
      | location_name | Тестовая площадка                      |
      | source_text   | Поздравляем с днем рождения актера!    |
      | raw_excerpt   | Поздравляем с днем рождения актера!    |
      | city          | Калининград                            |
    Тогда результат Smart Update имеет статус "skipped_promo"

  Сценарий: Акции/скидки не импортируются как события
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                  |
      | title         | TEST SU promo                          |
      | source_type   | telegram                               |
      | source_url    | test://smart-update/promo/1            |
      | date          | 2026-03-08                             |
      | time          | 18:00                                  |
      | location_name | Тестовая площадка                      |
      | source_text   | Акция! Скидка 50% по промокоду         |
      | raw_excerpt   | Акция! Скидка 50% по промокоду         |
      | city          | Калининград                            |
    Тогда результат Smart Update имеет статус "skipped_promo"

  Сценарий: Розыгрыш билетов не импортируется как событие
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                      |
      | title         | TEST SU giveaway                           |
      | source_type   | telegram                                   |
      | source_url    | test://smart-update/giveaway/1             |
      | date          | 2026-03-09                                 |
      | time          | 20:00                                      |
      | location_name | Тестовая площадка                          |
      | source_text   | Розыгрыш билетов на концерт, выиграй билет |
      | raw_excerpt   | Розыгрыш билетов на концерт, выиграй билет |
      | city          | Калининград                                |
    Тогда результат Smart Update имеет статус "skipped_giveaway"

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

  Сценарий: Розыгрыш билетов — мердж фактов события без «механики»
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                                                 |
      | title         | TEST SU giveaway                                                       |
      | source_type   | telegram                                                               |
      | source_url    | test://smart-update/giveaway/1                                          |
      | date          | 2026-03-08                                                              |
      | time          | 19:00                                                                   |
      | location_name | Тестовая площадка                                                       |
      | city          | Калининград                                                             |
      | source_text   | РОЗЫГРЫШ БИЛЕТОВ!\\n\\n08.03 в 19:00 спектакль «Тестовый».\\n\\nУсловия: подпишись, сделай репост и напиши комментарий. |
      | raw_excerpt   | 08.03 в 19:00 спектакль «Тестовый».                                     |
    Тогда результат Smart Update имеет статус "created"
    И создано новое событие с заголовком "TEST SU giveaway"
    И для события "TEST SU giveaway" лог фактов содержит "Убрана механика розыгрыша"
    И я очищаю тестовые события

  Сценарий: Выставка не дублируется при новом источнике внутри периода
    Дано в базе создано тестовое событие:
      | title                 | date       | end_date   | time | location_name  | source_text                     | description         | city        | event_type |
      | TEST SU exhibition A v20260211  | 2026-01-15 | 2026-03-01 |      | Галерея TEST   | Официальный анонс выставки      | Базовое описание    | Калининград | выставка   |
    И для события "TEST SU exhibition A v20260211" добавлен источник "test://smart-update/exhibition/base-v20260211" типа "telegram"
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                      |
      | title         | TEST SU exhibition A v20260211            |
      | source_type   | telegram                                   |
      | source_url    | test://smart-update/exhibition/new-v20260211 |
      | source_text   | Выставка продлена, добавлены новые работы. |
      | raw_excerpt   | Выставка продлена.                         |
      | date          | 2026-02-20                                 |
      | end_date      | 2026-03-31                                 |
      | time          |                                            |
      | location_name | Галерея TEST                               |
      | city          | Калининград                                |
      | event_type    | выставка                                   |
      | trust_level   | medium                                     |
    Тогда результат Smart Update имеет статус "merged"
    И Smart Update вернул event_id как у события "TEST SU exhibition A v20260211"
    И для события "TEST SU exhibition A v20260211" количество источников равно "2"
    И событие "TEST SU exhibition A v20260211" имеет поля:
      | field     | value      |
      | end_date  | 2026-03-31 |
      | event_type| выставка   |
    И я очищаю тестовые события

  Сценарий: Выставка не принимает продление периода от источника с более низким trust
    Дано в базе создано тестовое событие:
      | title                    | date       | end_date   | time | location_name  | source_text                     | description         | city        | event_type |
      | TEST SU exhibition trust v20260211 | 2026-03-01 | 2026-04-30 |      | Галерея TEST   | Официальный анонс выставки      | Базовое описание    | Калининград | выставка   |
    И для события "TEST SU exhibition trust v20260211" добавлен источник "test://smart-update/exhibition/high-v20260211" типа "telegram"
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                   |
      | title         | TEST SU exhibition trust v20260211     |
      | source_type   | vk                                      |
      | source_url    | test://smart-update/exhibition/low-v20260211 |
      | source_text   | Выставка продлена до конца мая.         |
      | raw_excerpt   | Выставка продлена.                      |
      | date          | 2026-04-10                              |
      | end_date      | 2026-05-31                              |
      | time          |                                         |
      | location_name | Галерея TEST                            |
      | city          | Калининград                             |
      | event_type    | выставка                                |
      | trust_level   | low                                     |
    Тогда результат Smart Update имеет статус "merged"
    И событие "TEST SU exhibition trust v20260211" имеет поля:
      | field    | value      |
      | end_date | 2026-04-30 |
    И для события "TEST SU exhibition trust v20260211" лог фактов содержит "Дата окончания:"
    И я очищаю тестовые события

  Сценарий: Первичный импорт выставки сохраняет тип и период
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                |
      | title         | TEST SU exhibition created v20260211 |
      | source_type   | vk                                   |
      | source_url    | test://smart-update/exhibition/create-v20260211 |
      | source_text   | Выставка работает до конца июня.     |
      | raw_excerpt   | Выставка современного искусства.      |
      | date          | 2026-04-01                           |
      | end_date      | 2026-06-30                           |
      | time          |                                      |
      | location_name | E2E Выставочный зал 981              |
      | city          | Калининград                          |
      | event_type    | выставка                             |
      | trust_level   | medium                               |
    Тогда результат Smart Update имеет статус "created"
    И создано новое событие с заголовком "TEST SU exhibition created v20260211"
    И событие "TEST SU exhibition created v20260211" имеет поля:
      | field      | value      |
      | event_type | выставка   |
      | end_date   | 2026-06-30 |
    И для события "TEST SU exhibition created v20260211" лог фактов содержит "Тип: выставка"
    И для события "TEST SU exhibition created v20260211" лог фактов содержит "Дата окончания: 2026-06-30"
    И я очищаю тестовые события

  Сценарий: Parser-источник может продлить период выставки
    Дано в базе создано тестовое событие:
      | title                     | date       | end_date   | time | location_name | source_text                | description      | city        | event_type |
      | TEST SU exhibition parser v20260211 | 2026-03-01 | 2026-04-30 |      | Галерея TEST  | Telegram-анонс выставки    | Базовое описание | Калининград | выставка   |
    И для события "TEST SU exhibition parser v20260211" добавлен источник "test://smart-update/exhibition/tg-v20260211" типа "telegram"
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                   |
      | title         | TEST SU exhibition parser v20260211    |
      | source_type   | parser:tretyakov                        |
      | source_url    | test://smart-update/exhibition/parser-v20260211 |
      | source_text   | Экспозиция продлена до конца июля.      |
      | raw_excerpt   | Экспозиция продлена.                    |
      | date          | 2026-04-10                              |
      | end_date      | 2026-07-31                              |
      | time          |                                         |
      | location_name | Галерея TEST                            |
      | city          | Калининград                             |
      | event_type    | выставка                                |
      | trust_level   | high                                    |
    Тогда результат Smart Update имеет статус "merged"
    И Smart Update вернул event_id как у события "TEST SU exhibition parser v20260211"
    И событие "TEST SU exhibition parser v20260211" имеет поля:
      | field    | value      |
      | end_date | 2026-07-31 |
    И я очищаю тестовые события

  Сценарий: Выставка без даты окончания получает период по умолчанию 1 месяц
    Когда я запускаю Smart Update с кандидатом:
      | field         | value                                             |
      | title         | TEST SU exhibition default end v20260211          |
      | source_type   | telegram                                          |
      | source_url    | test://smart-update/exhibition/default-end-v20260211 |
      | source_text   | Открытие выставки современного искусства.          |
      | raw_excerpt   | Открытие выставки.                                 |
      | date          | 2026-04-15                                        |
      | time          |                                                   |
      | location_name | E2E Выставочный зал default                       |
      | city          | Калининград                                       |
      | event_type    | выставка                                          |
      | trust_level   | medium                                            |
    Тогда результат Smart Update имеет статус "created"
    И создано новое событие с заголовком "TEST SU exhibition default end v20260211"
    И событие "TEST SU exhibition default end v20260211" имеет поля:
      | field    | value      |
      | end_date | 2026-05-15 |
    И для события "TEST SU exhibition default end v20260211" лог фактов содержит "Дата окончания: 2026-05-15"
    И я очищаю тестовые события
