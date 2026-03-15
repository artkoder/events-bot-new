# Guide Excursions Monitoring

Статус: implemented MVP

Канонический surface для мониторинга экскурсионных анонсов гидов в Telegram.

## Что уже работает

- отдельный guide-track в основной SQLite, без попадания в обычные `event`/`/daily`/month/weekend surfaces;
- seed-пак Telegram-источников из casebook;
- manual/live scan через `/guide_excursions`;
- preview и publish digest в тестовый канал `@keniggpt`;
- runtime semantic dedup перед render/publish digest, чтобы same-occurrence teaser/update пары не попадали в список отдельными карточками;
- отдельный блок в `/general_stats`;
- env-gated scheduler для `light` и `full` прогонов.

## Команды

- `/guide_excursions` — основное меню управления;
- `/guide_sources` — список источников и текущее покрытие;
- `/guide_recent` — preview `new_occurrences`;
- `/guide_digest` — publish текущего digest в тестовый канал.

## Формат digest-карточки

- заголовок экскурсии кликается и ведёт на исходный Telegram-пост;
- строка с каналом кликается и ведёт на сам канал;
- отдельная строка `Канал: ...` в MVP не выводится;
- booking link, если он извлечён, публикуется как кликаемая ссылка.

## Runtime boundary в текущей реализации

Текущий MVP использует локальный Telethon runtime для scan/import.

Это pragmatic delivery path для живого digest today:

- `run_guide_monitor()` читает source-каналы через Telethon;
- guide facts сохраняются в отдельные `guide_*` таблицы;
- `build_guide_digest_preview()` берёт расширенный shortlist и прогоняет его через `Route Matchmaker v1` до render/publish;
- suppressed duplicate occurrence ids всё равно помечаются опубликованными вместе с canonical row, чтобы они не всплывали в следующем digest как “новые”;
- публикация идёт через Bot API;
- media delivery сначала пытается использовать bot-side `forward -> file_id` bridge;
- для публичных source-каналов, где Bot API forward недоступен, включается Telethon download fallback и бот отправляет media как новый upload.

### Dedup status today

- текущий `Route Matchmaker v1` уже включён в live digest path;
- основной слой сейчас heuristic-first, а LLM pair judge работает best-effort;
- если Gemma pair-judge недоступен, runtime остаётся рабочим и падает обратно на conservative heuristic/fallback decisions, без остановки публикации.

Важно: это рабочая MVP-реализация для current runtime. Дизайн-пак про будущий `Kaggle notebook -> server import -> digest publish` остаётся в backlog-доках ниже.

## Related docs

- [Guide Excursions Dedup](/workspaces/events-bot-new/docs/features/guide-excursions-monitoring/dedup.md)
- [Guide Excursions Dedup Prompts](/workspaces/events-bot-new/docs/llm/guide-excursions-dedup.md)

## Live E2E note

Для полного live E2E через Telegram UI локальный бот должен быть единственным `getUpdates` consumer на токене.

Если на том же токене параллельно работает другой polling/runtime process:

- команды может обрабатывать не локальный код, а внешний процесс;
- preview/publish в UI могут смотреть в чужое состояние БД;
- для runtime-проверки релевантнее прогонять `run_guide_monitor() -> publish_guide_digest()` локально и отдельно сверять результат через Telethon в `@keniggpt`.

## Scheduler

Включается только через `ENABLE_GUIDE_EXCURSIONS_SCHEDULED=1`.

Тайминги по умолчанию:

- `GUIDE_EXCURSIONS_LIGHT_TIMES_LOCAL=09:05,13:20`
- `GUIDE_EXCURSIONS_FULL_TIME_LOCAL=20:10`
- `GUIDE_EXCURSIONS_TZ=Europe/Kaliningrad`

## Основные entrypoints

- `guide_excursions/commands.py`
- `guide_excursions/service.py`
- `guide_excursions/parser.py`
- `guide_excursions/scanner.py`
- `guide_excursions/seed.py`
- `db.py`
- `general_stats.py`
- `scheduling.py`
- `main.py`
- `main_part2.py`

## Связанные документы

- design pack: `docs/backlog/features/guide-excursions-monitoring/README.md`
- architecture: `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- digest spec: `docs/backlog/features/guide-excursions-monitoring/digest-spec.md`
- eval pack: `docs/backlog/features/guide-excursions-monitoring/eval-pack.md`
- live E2E draft/history: `docs/backlog/features/guide-excursions-monitoring/e2e.md`
