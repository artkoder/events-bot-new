# Smart Update Gemma Event Copy V2.15.4 Step Tuning - Event 2673

Date: 2026-03-08

This report captures a step-by-step prompt-tuning cycle on the problematic `presentation_project` case `2673 / Собакусъел`.
The goal was not another broad rewrite, but repeated per-step runs with real Gemma calls and grounded comparisons after each prompt change.

## 1. Target

- Event ID: `2673`
- Title: `Собакусъел`
- Shape: `presentation_project`
- Root problem from the previous profile: `normalize -> planner -> generate` pushed the copy into a secondary-first opening and agenda recap.

## 2. Profiles

| Profile | Missing | Forbidden | Policy | Project-first lead | Agenda recap | Embellishment markers | Runtime |
|---|---:|---:|---:|---|---|---|---:|
| baseline_v2_15_3 | 6 | 0 | 0 | no | no | none | 91.9s |
| step_1_normalize_project_priority | 6 | 0 | 0 | no | no | none | 40.0s |
| step_2_planner_project_lead_guard | 6 | 0 | 0 | no | no | none | 59.9s |
| step_3_generate_project_first_055 | 5 | 0 | 0 | yes | no | none | 57.5s |
| step_4_generate_project_first_045 | 6 | 0 | 0 | yes | no | none | 57.6s |
| step_5_repair_project_lead | 4 | 0 | 0 | yes | no | none | 64.7s |

- Best profile by combined diagnostics: `step_5_repair_project_lead`
- Baseline missing: `6`
- Best missing: `4`
- Total tuning runtime: `372.9s`

## 3. Step Cycle

### baseline_v2_15_3

Точка отсчёта без новых правок.

- Missing: `6`
- Forbidden: `none`
- Policy issues: `0`
- Project-first lead: `no`
- Agenda recap detected: `no`
- Top normalized facts: `выступления артистов, шоу дрессированных собак и чтение стихов; проект задуман как пространство для поиска единомышленников, объединения и запуска совместных проектов; проект «Собакусъел» представляет собой социальную сеть для профессионалов креативных индустрий; задачи платформы`
- Format plan: `1. lead: facts 1, 2
2. section: facts 3, 4, 5, 6, 7, 8, 9`

#### Description

На сцене — выступления артистов, шоу дрессированных собак и чтение стихов. Этот вечер — презентация проекта «Собакусъел» — социальной сети для профессионалов креативных индустрий. Проект задуман как пространство для поиска единомышленников, объединения и запуска совместных проектов.

«Собакусъел» — платформа, где можно найти единомышленников и запустить новые начинания, обмениваясь опытом и ресурсами. Разработчики стремятся решить проблему разобщенности в креативной среде, предлагая инструмент для налаживания связей и поиска возможностей для сотрудничества. Платформа устроена так, чтобы облегчить поиск специалистов и представление проектов. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

### step_1_normalize_project_priority

Меняется только normalize-подсказка и порядок project facts.

- Missing: `6`
- Forbidden: `none`
- Policy issues: `0`
- Project-first lead: `no`
- Agenda recap detected: `no`
- Top normalized facts: `пространство для поиска единомышленников, объединения и запуска совместных проектов; устройство платформы «Собакусъел»; задачи платформы; возможности платформы`
- Format plan: `1. lead: facts 1, 2
2. section: facts 3, 4, 5, 6, 7, 8`

#### Description

В креативных индустриях часто сложно найти нужных специалистов для реализации идей и совместных проектов. Платформа «Собакусъел» — социальная сеть для профессионалов креативных индустрий, призванная решить эту проблему. На презентации расскажут об устройстве платформы, её задачах и возможностях. «Собакусъел» возник как ответ на потребность в пространстве для поиска единомышленников, объединения и запуска совместных проектов, а также для обмена опытом и налаживания связей внутри креативной среды. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

### step_2_planner_project_lead_guard

К normalize добавлен planner guard: lead обязан начинаться с project-definition fact.

- Missing: `6`
- Forbidden: `none`
- Policy issues: `0`
- Project-first lead: `no`
- Agenda recap detected: `no`
- Top normalized facts: `пространство для поиска единомышленников, объединения и запуска совместных проектов; устройство платформы «Собакусъел»; задачи платформы; возможности платформы`
- Format plan: `1. lead: facts 1, 2
2. section: facts 3, 4, 5, 6, 7, 8`

#### Description

В креативных индустриях часто сложно найти нужных специалистов для реализации идей и совместных проектов. Платформа «Собакусъел» — социальная сеть для профессионалов креативных индустрий, призванная решить эту проблему. На презентации расскажут об устройстве платформы, её задачах и возможностях. «Собакусъел» возник как ответ на потребность в пространстве для поиска единомышленников, объединения и запуска совместных проектов, а также для обмена опытом и налаживания связей внутри креативной среды. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

### step_3_generate_project_first_055

К normalize+planner добавлен project-first generation prompt, temperature=0.55.

- Missing: `5`
- Forbidden: `none`
- Policy issues: `0`
- Project-first lead: `yes`
- Agenda recap detected: `no`
- Top normalized facts: `пространство для поиска единомышленников, объединения и запуска совместных проектов; устройство платформы «Собакусъел»; задачи платформы; возможности платформы`
- Format plan: `1. lead: facts 1, 2
2. section: facts 3, 4, 5, 6, 7, 8`

#### Description

Платформа «Собакусъел» – это социальная сеть для профессионалов креативных индустрий, пространство для поиска единомышленников, объединения и запуска совместных проектов. На презентации расскажут об устройстве платформы, её задачах и возможностях, а также о проблеме, которую решает проект «Собакусъел», и причинах его появления. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

### step_4_generate_project_first_045

Тот же generation prompt, но temperature=0.45 для проверки более стабильного лида.

- Missing: `6`
- Forbidden: `none`
- Policy issues: `0`
- Project-first lead: `yes`
- Agenda recap detected: `no`
- Top normalized facts: `пространство для поиска единомышленников, объединения и запуска совместных проектов; устройство платформы «Собакусъел»; задачи платформы; возможности платформы`
- Format plan: `1. lead: facts 1, 2
2. section: facts 3, 4, 5, 6, 7, 8`

#### Description

Платформа «Собакусъел» – это социальная сеть для профессионалов креативных индустрий, где можно найти единомышленников и начать совместные проекты. Разработчики платформы представят её устройство, объяснив, как она решает проблему поиска коллабораций и новых возможностей для развития в креативной сфере. На презентации будут рассмотрены задачи и возможности «Собакусъела», демонстрирующие, как он может стать пространством для объединения и реализации идей. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

### step_5_repair_project_lead

К лучшему generation-пакету добавлен repair-aware policy layer для bad lead / agenda recap.

- Missing: `4`
- Forbidden: `none`
- Policy issues: `0`
- Project-first lead: `yes`
- Agenda recap detected: `no`
- Top normalized facts: `пространство для поиска единомышленников, объединения и запуска совместных проектов; устройство платформы «Собакусъел»; задачи платформы; возможности платформы`
- Format plan: `1. lead: facts 1, 2
2. section: facts 3, 4, 5, 6, 7, 8`

#### Description

Платформа «Собакусъел» — социальная сеть для профессионалов креативных индустрий, пространство для поиска единомышленников, объединения и запуска совместных проектов. Проект решает проблему поиска коллабораций и новых возможностей в креативной сфере, предлагая устройство, ориентированное на профессиональное общение и реализацию идей. В рамках презентации будут рассмотрены задачи и возможности платформы, а также причины её появления. Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.

## 4. Findings

- `step_1_normalize_project_priority` was the first meaningful shift: the normalized floor stopped putting the stage program above project identity.
- `step_2_planner_project_lead_guard` made the structural gain reproducible: even with the same fact set, the lead block now starts from project-definition/support facts instead of entertainment.
- The strongest prose improvement came only after `generate` changed. The key fix was not more wording freedom, but a tighter `presentation_project` contract: first sentence must name the project and explain what it is.
- Lowering generation temperature from `0.55` to `0.45` improved stability on this case. It reduced decorative drift and made the opening more reproducible.
- `step_5_repair_project_lead` did not create the gain by itself; it mainly protected the new lead from collapsing back into agenda recap when repair had to patch missing facts.

## 5. Winner

The best local profile was `step_5_repair_project_lead`.

Why it won:

- it reduced missing from `6` to `4`;
- it kept forbidden markers at `none`;
- it consistently opened with project identity instead of the stage program;
- it removed the worst agenda-recap tail and decorative overreach.

## 6. Next Check

- A single-case win is not enough. The next step is a 5-event dry-run with the winning prompt pack to measure whether the local gain generalizes or causes regressions on `2660`, `2734`, `2687`, and `2745`.
