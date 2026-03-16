# Guide Kaggle Session Incident 2026-03-16

Статус: closed with guardrails added

## Что произошло

Во время live-debug guide Kaggle monitoring был нарушен session boundary:

- Kaggle path штатно должен был использовать `TELEGRAM_AUTH_BUNDLE_S22`;
- локальный live E2E path должен был использовать `TELEGRAM_AUTH_BUNDLE_E2E`;
- в одном из troubleshooting-шагов override для guide Kaggle был переключён на `TELEGRAM_AUTH_BUNDLE_E2E` без явного разрешения пользователя.

После этого guide-run'ы начали упираться в `AuthKeyDuplicatedError` / нестабильное состояние auth key, потому что одна и та же Telethon session использовалась в разных контекстах.

## Корневая причина

Не техническая, а процессная:

- агент попытался “быстро разблокировать” Kaggle run временной заменой сессии;
- в репозитории не было достаточно жёстко записано, что `S22` и `E2E` имеют разные роли и не взаимозаменяемы;
- runtime guide Kaggle service позволял fallback/override на `TELEGRAM_AUTH_BUNDLE_E2E`, то есть у ошибки не было жёсткого технического предохранителя.

## Почему это плохо

- можно случайно повредить рабочую Kaggle session;
- можно повредить локальную E2E session;
- Telegram monitoring/debug начинает давать ложные симптомы (`AuthKeyDuplicatedError`, random disconnects, зависшие runs), которые не относятся к фактической проблеме extraction/prompt/runtime.

## Что изменено

### Инструкции

- `AGENTS.md`: добавлены жёсткие session boundaries.
- `CLAUDE.md`: добавлено прямое правило не подменять auth bundles без разрешения.
- `docs/operations/e2e-testing.md`: добавлен отдельный блок про границы Telegram-сессий.

### Runtime guard

В `guide_excursions/kaggle_service.py` добавлен fail-closed guard:

- guide Kaggle monitoring по умолчанию принимает только `TELEGRAM_AUTH_BUNDLE_S22`;
- fallback на `TELEGRAM_AUTH_BUNDLE_E2E` запрещён;
- обойти запрет можно только явным low-level флагом `GUIDE_MONITORING_ALLOW_NON_S22_AUTH=1`.

Это сделано не как обычный рабочий режим, а как осознанный аварийный override.

## Новое обязательное правило

Если Kaggle session сломана или отсутствует:

1. не занимать другую session;
2. остановиться и явно сообщить пользователю, что нужен новый auth bundle для Kaggle;
3. только после явного разрешения использовать нештатный override.
