# Ticket Sites Queue (обогащение событий с сайтов продажи билетов)

Цель: если **в любом источнике Smart Event Update** (Telegram/VK/ручной ввод/будущие пайплайны) встречаются ссылки на сайты продажи билетов
(например `pyramida.info`, `домискусств.рф`, `qtickets`), бот добавляет их в очередь мониторинга и **раз в день** (или вручную)
запускает Kaggle‑парсинг этих URL, после чего результат импортируется через **Smart Event Update** как канонический источник
(`source_type=parser:*`, trust=`high`).

Это позволяет:
- подтянуть **фото/афиши**, **стоимость**, **описание**, **статус билетов** с более надёжного источника;
- добавить событию **второй источник с высоким доверием** и улучшить качество карточки/Telegraph.

## Как это работает

1) **Smart Event Update (универсально)**:
   - при обработке источника Smart Update извлекает ticket‑URLs из:
     - `candidate.source_text`,
     - `candidate.ticket_link`,
     - `candidate.source_url`,
     - `candidate.links_payload` (если вызывающий пайплайн передал “скрытые” ссылки/кнопки).
   - поддержанные URL кладутся в таблицу `ticket_site_queue` (идемпотентно по `url`, requeue = `next_run_at=now`).
   - при enqueue Smart Update заполняет `event_id` текущего события и source-референсы (`source_post_url`, `source_chat_*`, `source_message_id`) из `EventCandidate`.
   - в операторском отчёте Smart Update (например в `/tg`) дополнительно показывается, какие ссылки были добавлены в очередь.

   Примечание: Telegram Monitoring прокидывает `messages[].links` / event-level `links` в Smart Update через `candidate.links_payload`,
   чтобы не терять hidden URLs из entities/buttons.

2) **Очередь ticket-sites**:
   - обработчик группирует элементы по `site_kind` (`pyramida|dom_iskusstv|qtickets`);
   - запускает соответствующий Kaggle‑kernel:
     - Pyramida: `kaggle/ParsePyramida/` (URL‑scoped),
     - Дом искусств: `kaggle/ParseDomIskusstv/` (URL‑scoped),
     - Qtickets: `kaggle/ParseQtickets/` (full scan → фильтрация по URL из очереди).
   - для каждого результата вызывает Smart Update как `parser:<site_kind>` с trust=`high`.

3) После успешной обработки:
   - элемент остаётся `active`,
   - `next_run_at` сдвигается на `TICKET_SITES_QUEUE_INTERVAL_HOURS` (по умолчанию 24 часа).

## Команда (ручной запуск)

`/ticket_sites_queue [--info|-i] [--limit=N] [--source=pyramida|dom_iskusstv|qtickets] [--url=...]`

- `-i/--info` — состояние очереди (счётчики + ближайшие элементы).
- Без `--info` — запускает обработку очереди.

## Scheduler / ENV

Очередь по расписанию выключена по умолчанию.

- `ENABLE_TICKET_SITES_QUEUE=1` — включить scheduled job.
- `TICKET_SITES_QUEUE_TIME_LOCAL=11:20` — рекомендуемое время (между тяжёлыми окнами: `/parse`, `/3di`, VK crawl).
- `TICKET_SITES_QUEUE_TZ=Europe/Kaliningrad`
- `TICKET_SITES_QUEUE_LIMIT=...` — лимит элементов на один scheduled run.
- `TICKET_SITES_QUEUE_INTERVAL_HOURS=24` — период повторного сканирования URL после успеха.

## Подводные камни

- URL‑нормализация: важно обрезать хвостовую пунктуацию (`)`, `.`, `…`) — иначе в очереди окажутся разные ключи.
- Qtickets kernel по умолчанию парсит все события; очередь фильтрует результаты по URL, поэтому при большом списке watch‑URL
  стоит держать `TICKET_SITES_QUEUE_LIMIT` умеренным.
- Smart Update якорные поля (дата/время/локация) обычно не меняет для non‑parser источников; ticket‑site источники идут как
  `parser:*`, поэтому допускаются корректировки времени (и другие canonical‑обогащения) при совпадении события.
