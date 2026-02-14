# language: ru
Функция: Mass multi-source merge (VK + Telegram)

  # Preconditions (runbook):
  # - Run with TG_MONITORING_LIMIT=10 to scan at least 10 messages per channel.
  #   (default is 50, but we want a predictable ">=10" E2E signal).
  #
  # Goal:
  # - process the whole VK inbox queue (snapshot) via Smart Update
  # - scan a curated list of Telegram channels
  # - verify there exists at least one event merged from VK + Telegram sources

  Сценарий: Массовый прогон VK очереди + Telegram каналов (VK+TG merge)
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом
    Когда я отправляю команду "/start"

    Дано в списке источников Telegram настроены:
      | username                | trust_level | default_location |
      | koihm                   | medium      | Историко-художественный музей, Клиническая 21, Калининград |
      | tretyakovka_kaliningrad | medium      | Филиал Третьяковской галереи, Парадная наб. 3, Калининград |
      | yantarholl              | medium      | Янтарь холл, Ленина 11, Светлогорск |
      | meowafisha              | low         | Калининград |
      | kulturnaya_chaika       | low         | Калининград |
      | vorotagallery           | low         | Калининград |

    И очищены отметки мониторинга для Telegram источников "koihm,tretyakovka_kaliningrad,yantarholl,meowafisha,kulturnaya_chaika,vorotagallery"

    И в VK inbox для E2E выбраны первые "15" активных постов очереди
    Когда я отправляю команду "/vk_auto_import"
    Тогда я жду сообщение прогресса VK auto import
    Тогда я жду первый отчёт VK auto import с событиями и ссылками
    И я дожидаюсь выполнения задач обновления Telegraph для событий из последнего VK отчёта
    И я жду завершение VK auto import

    Когда я отправляю команду "/tg"
    И я нажимаю инлайн-кнопку "🚀 Запустить мониторинг"
    Тогда я жду долгой операции с текстом "Telegram Monitor"

    Тогда существует событие с источниками VK и Telegram
    Когда я запрашиваю /log для события с источниками VK и Telegram
    Тогда я вижу лог источников с датой, временем, источником и фактами
    И в логе источников есть источники VK и Telegram

  @manual
  Сценарий: Полный масс-прогон всей VK очереди + Telegram каналов (VK+TG merge)
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом
    Когда я отправляю команду "/start"

    Дано в списке источников Telegram настроены:
      | username                | trust_level | default_location |
      | koihm                   | medium      | Историко-художественный музей, Клиническая 21, Калининград |
      | tretyakovka_kaliningrad | medium      | Филиал Третьяковской галереи, Парадная наб. 3, Калининград |
      | yantarholl              | medium      | Янтарь холл, Ленина 11, Светлогорск |
      | meowafisha              | low         | Калининград |
      | kulturnaya_chaika       | low         | Калининград |
      | vorotagallery           | low         | Калининград |

    И очищены отметки мониторинга для Telegram источников "koihm,tretyakovka_kaliningrad,yantarholl,meowafisha,kulturnaya_chaika,vorotagallery"

    И в VK inbox для E2E выбраны все активные посты очереди
    Когда я отправляю команду "/vk_auto_import"
    Тогда я жду сообщение прогресса VK auto import
    Тогда я жду первый отчёт VK auto import с событиями и ссылками
    И я дожидаюсь выполнения задач обновления Telegraph для событий из последнего VK отчёта
    И я жду завершение VK auto import

    Когда я отправляю команду "/tg"
    И я нажимаю инлайн-кнопку "🚀 Запустить мониторинг"
    Тогда я жду долгой операции с текстом "Telegram Monitor"

    Тогда существует событие с источниками VK и Telegram
    Когда я запрашиваю /log для события с источниками VK и Telegram
    Тогда я вижу лог источников с датой, временем, источником и фактами
    И в логе источников есть источники VK и Telegram
