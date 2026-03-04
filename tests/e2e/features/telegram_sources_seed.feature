# language: ru
Функция: Telegram Sources Seed (UI)

  Предыстория:
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом
    Когда я отправляю команду "/start"
    И я жду сообщения с текстом "Choose action"

  Сценарий: Канонические источники Telegram добавлены и список работает с пагинацией
    Когда я отправляю команду "/tg"
    И я нажимаю инлайн-кнопку "🧩 Синхронизировать источники"
    Тогда я жду сообщения с текстом "синхронизированы"
    Когда я отправляю команду "/tg"
    И я нажимаю инлайн-кнопку "📋 Список источников"
    Тогда список источников Telegram в UI использует пагинацию
    И в списке источников Telegram через UI есть источники:
      | username               |
      | tretyakovka_kaliningrad|
      | world_ocean_museum     |
      | rostec_arena           |
      | terkatalk              |
      | klassster              |
      | meowafisha             |
      | dobro39                |
      | dramteatr39            |
      | kulturnaya_chaika      |
      | kaliningradartmuseum   |
      | domkitoboya            |
      | locostandup            |
      | signalkld              |
      | kaliningradlibrary     |
      | koihm                  |
      | grezahutor             |
      | zaryakinoteatr         |
      | klgdcity               |
      | ambermuseum            |
      | telegraf_svetlogorsk   |
      | muztear39              |
      | festdir                |
      | molod_kld              |
      | minkultturism_39       |
      | zamokinsterburg        |
      | mesto_sily_bar         |
      | castleneuhausen        |
      | admyantarniy           |
      | sobor39                |
      | gumbinnen              |
      | dom_semii              |
      | kldzoo                 |
      | kozia_gorka            |
      | k_mira101              |
      | barn_kaliningrad       |
      | itcsvetlogorsk         |
      | agropark39             |
      | tastes_of_vistynets    |
      | yantarholl             |
      | open_fest              |
      | festkantata            |
      | garazhka_kld           |

