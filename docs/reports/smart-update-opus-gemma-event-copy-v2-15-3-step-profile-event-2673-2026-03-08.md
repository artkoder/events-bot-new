# Smart Update Gemma Event Copy V2.15.3 Step Profile — Event 2673

Дата: 2026-03-08

Это step-level profiling report для одного реального события в `2.15.3` experimental pipeline.
Цель: увидеть, где именно проблема живёт в prompt behavior, а где — в deterministic / orchestration layer.

## 1. Событие

- Event ID: `2673`
- Название: `Собакусъел`
- Тип: `presentation`
- Shape: `presentation_project`
- Pattern: `scene_led`
- Branch: `fact_first_v2_15_2`
- Planner mode: `deterministic`
- Runtime total: `105.3s`

## 2. Быстрый итог

- Baseline missing: `5`
- V2.15.2 missing: `4`
- V2.15.3 missing: `2`
- V2.15.3 forbidden: `none`

## 3. Тексты

### Baseline

> Проект «Собакусъел» представляет собой социальную сеть для профессионалов креативных индустрий.

Презентация проекта «Собакусъел» – это пространство для поиска единомышленников, объединения и запуска совместных проектов. На мероприятии расскажут о задачах, устройстве и возможностях платформы, а также о том, зачем и какую проблему решает проект. 🐶

### Программа мероприятия
В программе запланированы:
- выступления артистов
- шоу дрессированных собак
- чтение стихов

### Для кого это будет интересно
Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества. Организаторы предоставят чай, а посетителям рекомендуется принести с собой печенье.

### Запуск проекта и возможности платформы
На презентации откроется предрегистрация для участников проекта. Расскажут о том, как устроена платформа «Собакусъел». ✨

### V2.15.2

> «Творчество — это когда ты делаешь что-то, что тебе нравится, и надеешься, что это понравится и другим».

«Собакусъел» – это презентация социальной сети для профессионалов креативных индустрий. Проект задуман как пространство для поиска единомышленников, объединения и запуска совместных проектов.

### Устройство и цели платформы

В центре внимания – устройство платформы «Собакусъел», задачи, которые она решает, причины её появления и проблема, которую она призвана решить.

### Формат встречи

Мероприятие включает в себя выступления артистов, шоу дрессированных собак и чтение стихов. Формат встречи предполагает возможность для нетворкинга и поиска новых партнёров. Программа собрана вокруг обмена опытом и обсуждения перспектив сотрудничества. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

### V2.15.3

На сцене — выступления артистов, шоу дрессированных собак и чтение стихов. Это презентация проекта «Собакусъел» — социальной сети для профессионалов креативных индустрий. Проект задуман как пространство для поиска единомышленников, объединения и запуска совместных проектов.

«Собакусъел» решает проблему разобщенности в креативной среде, предоставляя возможности для поиска партнеров, обмена опытом и реализации совместных идей. Платформа устроена так, чтобы упростить коммуникацию и сотрудничество между участниками. Мероприятие интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества. Также будут затронуты задачи платформы, возможности платформы, причины появления проекта «Собакусъел» и устройство платформы «Собакусъел».

## 4. Пайплайн по шагам

| Step | Kind | Runtime | Artifact |
|---|---|---:|---|
| A_event_context | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/A_event_context_state.json` |
| B_normalize_state | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/B_normalize_state_state.json` |
| C_normalize | llm_json | 82.1s | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/C_normalize_output.json` |
| D_normalized_floor | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/D_normalized_floor_state.json` |
| E_plan_seed | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/E_plan_seed_state.json` |
| G_materialized_plan | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/G_materialized_plan_state.json` |
| H_generate | llm_text | 8.1s | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/H_generate_output.txt` |
| I_pre_repair_validation | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/I_pre_repair_validation_state.json` |
| J_targeted_repair | llm_text | 8.1s | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/J_targeted_repair_output.txt` |
| K_targeted_repair_candidate | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/K_targeted_repair_candidate_state.json` |
| O_final | state | — | `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/O_final_state.json` |

## 5. Ключевые артефакты

- Full trace JSON: `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/trace.json`
- Normalize rows: `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/B_normalize_state_state.json`
- Planner seed/materialized plan: `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/E_plan_seed_state.json`, `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/G_materialized_plan_state.json`
- Generation prompt/result: `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/H_generate_prompt.txt`, `artifacts/codex/event_copy_v2_15_3_step_profile_event_2673_2026-03-08/H_generate_output.txt`

## 6. Step diagnostics

- Normalization input facts: `11`
- Normalized rows: `13`
- Final facts_text_clean: `9`
- Quote metadata: `has_verified_quote=false`
- Validation before repair: missing=`7`, policy=`0`, forbidden=`0`
- Validation final: missing=`2`, policy=`0`, forbidden=`0`

## 7. Назначение

- Этот профиль нужен как материал для следующего prompt-tuning round, а не как production-ветка.
- Он пригоден для последующей консультации с `Opus` и `Gemini`, потому что показывает не только финальный текст, но и все межшаговые решения.
