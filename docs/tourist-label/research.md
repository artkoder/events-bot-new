# Исследование: Туристическая метка (ручной режим)

## Цель исследования

Собрать корпус ручных решений по блоку «🌍 Туристам», чтобы GPT-5 Deep Research видела реальные статусы, выбранные факторы и комментарии редакторов. До завершения исследования автоматическая разметка запрещена: в корпус попадают только ручные решения неблокированных модераторов и администраторов.

## Формат выгрузки `/tourist_export`

- Команда доступна неблокированным модераторам и администраторам и принимает `--period` или `period=` с диапазонами `ГГГГ`, `ГГГГ-ММ` или `ГГГГ-ММ-ДД..ГГГГ-ММ-ДД`.
- Ответ — JSONL: каждая строка содержит исходные поля события и все столбцы `tourist_*`, включая причины и комментарий.
- Строки отсортированы по дате начала и идентификатору события. Имя файла `tourist_export_<start>_<end>_<timestamp>.jsonl`.
- При пустом наборе приходит «No events found» без вложений.
- Логи фиксируют `tourist_export.request`, `tourist_export.start`, `tourist_export.done` с `user_id` и параметрами.

### JSON-схема строки выгрузки

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "event_id",
    "event_ts",
    "tourist_label",
    "tourist_label_by",
    "tourist_label_at",
    "tourist_label_source",
    "tourist_factors",
    "tourist_note"
  ],
  "properties": {
    "event_id": { "type": "integer", "minimum": 1 },
    "event_ts": { "type": "string", "format": "date-time" },
    "tourist_label": { "type": "string", "enum": ["yes", "no", "unknown"] },
    "tourist_label_by": { "type": "integer", "minimum": 1 },
    "tourist_label_at": { "type": "string", "format": "date-time" },
    "tourist_label_source": { "type": "string", "const": "operator" },
    "tourist_factors": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "targeted_for_tourists",
          "unique_for_region",
          "major_festival",
          "nature_or_landmark",
          "photogenic",
          "local_flavor",
          "easy_logistics"
        ]
      },
      "uniqueItems": true,
      "maxItems": 7
    },
    "tourist_note": { "type": "string" },
    "tourist_message_source": {
      "type": "object",
      "properties": {
        "platform": { "type": "string", "enum": ["telegram", "vk"] },
        "chat_id": { "type": "integer" },
        "message_id": { "type": "integer" }
      },
      "required": ["platform", "chat_id", "message_id"]
    }
  }
}
```

## Промпт для GPT-5 Deep Research

```
Ты анализируешь выгрузку `/tourist_export`. Каждый объект — ручное решение редактора по блоку «🌍 Туристам».
Для каждой записи:
- Объясни, почему статус «Да», «Нет» или «—», опираясь на причины и комментарий.
- Сопоставь выбранные факторы с формулировками справочника: 🎯 Нацелен на туристов, 🧭 Уникально для региона, 🎪 Крупный фестиваль или событие, 🌊 Природа или знаковое место, 📸 Фотогенично, понравится блогерам, 🍲 Местный колорит и ремёсла, 🚆 Простая логистика.
- Если комментария нет, опиши, какие поля карточки компенсируют его отсутствие.
Ответ должен соответствовать JSON-схеме из раздела выше.
```

## Практика исследования

1. Зафиксируйте список редакторов, которые регулярно работают с кнопками «Интересно туристам», «Не интересно туристам», «Причины», «✍️ Комментарий» и «🧽 Очистить комментарий» в Telegram и VK.
2. Перед интервью напомните про тайминги: меню причин (`tourist_reason_sessions`) живёт 15 минут, окно комментария (`tourist_note_sessions`) — 10 минут.
3. На созвонах проходите экспорт с рекомендованным промптом, записывайте цитаты и сохраняйте ссылки на конкретные события.
4. После разговора выгрузите свежий JSONL через `/tourist_export`, сопоставьте `tourist_label`, `tourist_factors`, `tourist_note`, отметьте расхождения.
5. До окончания исследования не включайте авторазметку. Собранные результаты складывайте в общий репозиторий с датой, участником, ссылкой на `.jsonl` и краткими выводами для GPT-5 Deep Research.
