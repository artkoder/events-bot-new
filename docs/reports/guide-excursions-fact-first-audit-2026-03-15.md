# Guide Excursions Fact-First Audit 2026-03-15

Цель: зафиксировать расхождение между старой реализацией `guide_excursions` и канонической документацией, а также отметить текущий migration plan и уже внедрённые шаги.

Сверка проводилась против:

- `docs/backlog/features/guide-excursions-monitoring/README.md`
- `docs/backlog/features/guide-excursions-monitoring/architecture.md`
- `docs/backlog/features/guide-excursions-monitoring/mvp.md`
- `docs/features/guide-excursions-monitoring/README.md`

## Главные расхождения, которые были у старого MVP

1. Execution boundary расходилась с каноникой.
Старый path был `local Telethon -> regex/heuristic parse -> digest`, а документация требовала `Kaggle notebook -> server import -> digest publish`.

2. Extraction был не facts-first.
Основной сигнал шёл через regex/heuristics, а не через `Tier 1` Gemma extraction с явным fact contract.

3. Storage был слишком occurrence-centric.
Guide-track уже жил отдельно от `event`, но не materialize’ил полноценный inspectable слой `GuideProfile / ExcursionTemplate / ExcursionOccurrence / FactClaim`.

4. Digest был привязан к тексту поста сильнее, чем к фактам.
Это делало труднее проверку качества `title/date/location`, eligibility и same-occurrence merge.

5. У оператора не было fact inspection surface.
Нельзя было быстро ответить на вопрос “какие именно факты извлеклись по этой экскурсии”.

## Migration plan

### Phase 1. Kaggle-first intake

Статус: сделано в MVP-версии.

- добавлен guide-specific runtime: `kaggle/GuideExcursionsMonitor/guide_excursions_monitor.py`
- reuse существующего split-secrets / dataset attach / push-poll-download стека;
- guide secrets идут через `GOOGLE_API_KEY2` и Telethon auth bundle/session;
- guide LLM path в notebook зафиксирован как Gemma-only.

### Phase 2. Fact-first server import

Статус: сделано в MVP-версии.

- Kaggle result import materialize’ит `GuideProfile / GuideTemplate / GuideOccurrence / GuideFactClaim`;
- occurrence хранит `fact_pack_json`;
- claims получили `claim_role`, `provenance_json`, `observed_at`, `last_confirmed_at`;
- past occurrences в MVP не materialize’ятся.

### Phase 3. Digest from fact pack

Статус: сделано в MVP-версии.

- shortlist reader backfills empty fields из `fact_pack_json`;
- editorial prompt явно использует `fact_pack` как primary truth source;
- operator preview подсказывает `/guide_facts <id>`.

### Phase 4. Server-side merge/bind/enrich

Статус: частично сделано.

- template/profile hints уже влияют на materialized rows;
- same-occurrence digest dedup работает;
- но полноценный `Route Weaver v1` для reschedule/status-bind/cross-source bind ещё не закрыт полностью.

### Phase 5. Live E2E hardening

Статус: в процессе.

- manual scenario в `tests/e2e/features/guide_excursions.feature` теперь ждёт `transport=kaggle`;
- preview дополнительно проверяется на наличие `facts=/guide_facts <id>`;
- отдельным live smoke нужно подтверждать, что run реально уходит в Kaggle, а не живёт только на fallback.

## Текущее состояние по ключевым требованиям

- `GuideProfile / ExcursionTemplate / ExcursionOccurrence / FactClaim`: да, materialized в основной SQLite.
- `Kaggle Tier 1 Gemma extraction`: да, добавлен отдельный guide runtime на Kaggle.
- `server-side merge/bind/enrich`: частично, MVP-уровень.
- `digest generation только из fact pack`: да, это primary source для shortlist/editorial.
- `Gemma-only для guide LLM path`: да, зафиксировано в notebook и server-side guide LLM configs.
- `отсутствие past occurrences в MVP`: да, старые выходы не сохраняются как `guide_occurrence`.

## Открытые gaps после этого шага

1. OCR в guide Kaggle runtime ещё не доведён до паритета с backlog-спекой.
2. Cross-post/status/reschedule binding ещё нужно усиливать на сервере.
3. Нужен повторяемый live E2E прогон через `@keniggpt`, чтобы подтвердить качество extraction и media handling на реальных сообщениях.
