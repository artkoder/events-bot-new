# language: ru
Функция: VK auto-import error posts (regression)

  # Goal: verify previously failing VK posts (reposts/OCR/location) are processed
  # without infinite loops and produce actionable outcomes.

  Сценарий: /vk_auto_import обрабатывает проблемные посты (репост/локация/OCR)
    Дано я авторизован в клиенте Telethon
    И я открыл чат с ботом
    # Explicit group_id:post_id pairs from incident log
    И в VK inbox для E2E приоритетные посты "-78172842_7020,-211997788_2735,-39437155_16614,-212760444_4444"

    Когда я отправляю команду "/start"

    Когда я отправляю команду "/vk_auto_import"
    Тогда я жду сообщение прогресса VK auto import

    # We expect at least one Smart Update report with links during the run.
    Тогда я жду первый отчёт VK auto import с событиями и ссылками
    И я жду завершение VK auto import
