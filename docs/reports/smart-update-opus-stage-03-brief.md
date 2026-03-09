# Smart Update Opus Stage 03 Brief

Дата: 2026-03-06

Это новый, уже нормально пронумерованный пакет.

Stage 03 = следующий шаг после Stage 02 follow-up:
- casepack ещё расширен;
- локально проверен безопасный deterministic baseline;
- теперь нужен уже не обзор архитектуры, а помощь в выборе следующего рационального слоя поверх этого baseline.

## 1. Контекст Stage 03

Приоритеты не меняются:
- false merge хуже дубля;
- LLM обязателен;
- TPM важнее суммарной стоимости;
- latency можно увеличивать на спорных кейсах;
- решение должно выдерживать массовый поток recurring/schedule/multi-event постов.

## 2. Что изменилось по сравнению со Stage 02

1. В casebook и casepack добавлены ещё 6 реальных Stage 03 кейсов.
2. Собран отдельный `Stage 03 case window`.
3. Локально прогнан безопасный deterministic `production-ready layer` без LLM.
4. Теперь у нас есть не только споры с Opus, но и собственный baseline, который можно обсуждать предметно.

## 3. Stage 03 пакет данных

Ключевые файлы:
- `docs/reports/smart-update-opus-stage-index.md`
- `docs/reports/smart-update-opus-stage-03-production-layer.md`
- `docs/reports/smart-update-duplicate-casebook.md`
- `artifacts/codex/opus_session2_casepack_latest.json`
- `artifacts/codex/smart_update_stage_03_case_window_latest.json`
- `artifacts/codex/smart_update_stage_03_production_ready_dryrun_latest.json`
- `artifacts/codex/opus_consultation_bundle_latest.json`

## 4. Что показал локальный Stage 03 dry-run

На текущем expanded casepack safe-layer дал:
- `0` false merges;
- `0` false differents;
- но большой `gray` хвост.

Это важный сдвиг:
- foundation уже есть;
- теперь надо не спорить про базовые guardrails;
- а решать, какие `gray`-классы рационально тащить дальше.

## 5. Какие именно вопросы теперь важны

Opus нужен уже для следующего, более узкого шага.

### 5.1. Какие `gray`-классы пытаться закрывать deterministic rules

Например:
- `venue_noise`
- `same-source date/time anomaly`
- `semantic duplicate without strict source proof`

### 5.2. Какие `gray`-классы лучше не трогать без LLM

Например:
- `museum_holiday_program_multi_child`
- `cathedral_shared_ticket_false_friend`
- часть `hudozhnitsy`-подобных semantic alias clusters

### 5.3. Какой следующий минимальный слой правил стоит добавить

Нужен ответ в духе:
- что добавить в Stage 04;
- что пока нельзя добавлять;
- что должно идти только через pairwise Gemma triage.

## 6. Что я хочу получить от Opus на Stage 03

1. Оценку самого Stage 03 baseline.
2. Разделение `gray`-остатка по классам:
- deterministic-upgradable;
- LLM-only;
- manual-review-worthy.
3. Конкретные предложения по Stage 04:
- narrow deterministic rules;
- LLM hints;
- что не трогать.
4. Позицию по тому, как уменьшать `gray`, не открывая false merge.

## 7. Важное ограничение

Opus должен отвечать, учитывая реальный runtime:
- `SMART_UPDATE_LLM=gemma`
- JSON schema output
- `TPM≈12000`
- `RPM≈20`
- никаких fat-shortlist или жирных payload на массовом потоке

Это уже не теоретическая консультация, а Stage 03 пакет с локальным baseline.
