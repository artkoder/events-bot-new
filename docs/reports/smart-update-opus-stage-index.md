# Smart Update Opus Stage Index

Цель: ввести нормальную сквозную нумерацию по раундам внешней консультации и внутренней проработки, чтобы быстро понимать, к какому этапу относится документ.

Правило на будущее:
- все новые документы и артефакты для внешней консультации маркируются `stage-XX`;
- Stage считается новым, когда меняется уровень зрелости задачи:
  - новый пакет данных;
  - новый локальный прогон;
  - новый цикл вопросов к внешней модели.

## Stage 01

Первичный внешний briefing и первый набор возражений.

Ключевые материалы:
- `docs/reports/smart-update-cross-llm-brief.md`
- `docs/reports/smart-update-opus-consultation-review.md`
- `artifacts/codex/smart-update-consultation-opus-20260306.md`

## Stage 02

Глубокая консультация по реальным данным + follow-up по спорным местам.

Ключевые материалы:
- `docs/reports/smart-update-session2-deep-consultation.md`
- `docs/reports/smart-update-session2-deep-consultation-review.md`
- `docs/reports/smart-update-session2-followup-response.md`
- `artifacts/codex/opus_session2_followup_prompt_latest.md`
- `artifacts/codex/opus_session2_followup_send_manifest_latest.md`

## Stage 03

Локальная проверка безопасного `production-ready` deterministic слоя на расширенном casepack и подготовка следующего рационального пакета вопросов.

Ключевые материалы:
- `docs/reports/smart-update-opus-stage-03-brief.md`
- `docs/reports/smart-update-opus-stage-03-production-layer.md`
- `artifacts/codex/smart_update_stage_03_case_window_latest.json`
- `artifacts/codex/smart_update_stage_03_production_ready_dryrun_latest.json`
- `artifacts/codex/opus_stage_03_prompt_latest.md`
- `artifacts/codex/opus_stage_03_send_manifest_latest.md`

## Stage 04

Три локальных consensus dry-run поверх Stage 03 baseline с целью сузить набор deterministic правил до regression-safe preprod candidate и передать Opus уже narrowed dispute set.

Ключевые материалы:
- `docs/reports/smart-update-opus-stage-04-consensus-prep.md`
- `docs/reports/smart-update-stage-04-competitive-response.md`
- `docs/reports/smart-update-stage-04-competitive-response-review.md`
- `docs/reports/smart-update-stage-04-followup-response.md`
- `artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.json`
- `artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.md`
- `artifacts/codex/smart_update_stage_04_competitive_validation_latest.md`
- `artifacts/codex/opus_stage_04_prompt_latest.md`
- `artifacts/codex/opus_stage_04_send_manifest_latest.md`
- `artifacts/codex/opus_stage_04_followup_prompt_latest.md`
- `artifacts/codex/opus_stage_04_followup_send_manifest_latest.md`

## Stage 05

Финальный alignment-pass: deterministic subset почти согласован, остаются последние решения по rollout posture и residual gray LLM layer.

Ключевые материалы:
- `docs/reports/smart-update-opus-stage-05-brief.md`
- `docs/reports/smart-update-stage-04-followup-response.md`
- `docs/reports/smart-update-duplicate-casebook.md`
- `artifacts/codex/smart_update_stage_04_competitive_validation_latest.md`
- `artifacts/codex/opus_stage_05_prompt_latest.md`
- `artifacts/codex/opus_stage_05_send_manifest_latest.md`

## Stage 06

Live VK validation на очищенном casebook snapshot после локальной реализации narrowed deterministic subset. Этот этап уже выявил runtime-specific остатки: `matryoshka` false-merge risk, `oncologists` fragmentation, persistent `makovetsky` duplicate и жёсткий `TPM` pressure.

Ключевые материалы:
- `docs/reports/smart-update-opus-stage-06-live-validation.md`
- `artifacts/codex/smart_update_casebook_vk_reimport_prep_latest.md`
- `artifacts/codex/smart_update_stage_06_live_validation_latest.md`
- `artifacts/codex/smart_update_stage_06_live_validation_latest.json`
- `artifacts/codex/opus_stage_06_prompt_latest.md`
- `artifacts/codex/opus_stage_06_send_manifest_latest.md`

## Stage 07

Точечный live rerun follow-up после Stage 06: новый runtime class с false-positive giveaway/promo post, mixed zoo same-source schedule signal и узкий запрос к Opus уже не только по merge-quality, но и по prompt-level отличению `event subject` от `prize/reference event`.

Ключевые материалы:
- `docs/reports/smart-update-opus-stage-07-live-rerun-followup.md`
- `artifacts/codex/smart_update_casebook_vk_reimport_prep_v3_latest.md`
- `artifacts/codex/smart_update_stage_07_live_rerun_followup_latest.md`
- `artifacts/codex/smart_update_stage_07_live_rerun_followup_latest.json`
- `artifacts/codex/opus_stage_07_prompt_latest.md`
- `artifacts/codex/opus_stage_07_send_manifest_latest.md`

## Зачем эта схема

Она нужна, чтобы больше не было путаницы вида:
- “это ещё Session 2 или уже следующий этап?”;
- “какой из prompt-пакетов актуальный?”;
- “на каком наборе кейсов делался конкретный вывод?”.

Для следующего внешнего раунда каноническим теперь считаем `Stage 07`.
