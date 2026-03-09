# Smart Update Gemma Event Copy V2.16.2 Lollipop Facts.Extract Provenance Map

Дата: 2026-03-09

## 1. Зачем это

- Текущий lollipop namespace использовал локальные ids вида `facts.extract_*.v1`.
- Это скрывало историческое происхождение prompt-ов и делало неочевидным, из каких именно `v2.15.x` стадий они были salvage-нуты.
- Эта карта возвращает явный provenance-layer: `current stage id -> lollipop slot/revision -> historical prompt id -> historical source round`.

## 2. Mapping

- `facts.extract_subject.v1` -> slot=`facts.extract.subject` rev=`l1` | historical=`subject_v1_strict` from `v2.15.5` | shape=`presentation_project` | lineage=`subject_v1_strict <- v2.15.5`
- `facts.extract_agenda.v1` -> slot=`facts.extract.agenda` rev=`l1` | historical=`agenda_v2_prose_ready` from `v2.15.5` | shape=`presentation_project` | lineage=`agenda_v2_prose_ready <- v2.15.5`
- `facts.extract_program.v1` -> slot=`facts.extract.program` rev=`l1` | historical=`program_v1_compact` from `v2.15.5` | shape=`presentation_project` | lineage=`program_v1_compact <- v2.15.5`
- `facts.extract_cluster.v1` -> slot=`facts.extract.cluster` rev=`l1` | historical=`cluster_v2_named_group` from `v2.15.6` | shape=`lecture_person` | lineage=`cluster_v2_named_group <- v2.15.6`
- `facts.extract_theme.v1` -> slot=`facts.extract.theme` rev=`l1` | historical=`theme_v1_compact` from `v2.15.6` | shape=`lecture_person` | lineage=`theme_v1_compact <- v2.15.6`
- `facts.extract_profiles.v1` -> slot=`facts.extract.profiles` rev=`l1` | historical=`profiles_v1_literal` from `v2.15.6` | shape=`lecture_person` | lineage=`profiles_v1_literal <- v2.15.6`
- `facts.extract_concept.v1` -> slot=`facts.extract.concept` rev=`l1` | historical=`concept_v1_compact` from `v2.15.7` | shape=`program_rich` | lineage=`concept_v1_compact <- v2.15.7`
- `facts.extract_setlist.v1` -> slot=`facts.extract.setlist` rev=`l1` | historical=`setlist_v1_grouped` from `v2.15.7` | shape=`program_rich` | lineage=`setlist_v1_grouped <- v2.15.7`
- `facts.extract_performer.v1` -> slot=`facts.extract.performer` rev=`l1` | historical=`performer_v1_awards` from `v2.15.7` | shape=`program_rich` | lineage=`performer_v1_awards <- v2.15.7`
- `facts.extract_stage.v1` -> slot=`facts.extract.stage` rev=`l1` | historical=`stage_v2_compact` from `v2.15.7` | shape=`program_rich` | lineage=`stage_v2_compact <- v2.15.7`
- `facts.extract_card.v1` -> slot=`facts.extract.card` rev=`l1` | historical=`normalize_card_v1` from `v2.15.8` | shape=`screening_card` | lineage=`normalize_card_v1 <- v2.15.8`
- `facts.extract_support.v1` -> slot=`facts.extract.support` rev=`l1` | historical=`normalize_support_v1` from `v2.15.8` | shape=`screening_card` | lineage=`normalize_support_v1 <- v2.15.8`
- `facts.extract_identity.v1` -> slot=`facts.extract.identity` rev=`l1` | historical=`normalize_identity_v2_strict` from `v2.15.8` | shape=`party_theme_program` | lineage=`normalize_identity_v2_strict <- v2.15.8`
- `facts.extract_participation.v1` -> slot=`facts.extract.participation` rev=`l1` | historical=`normalize_participation_v1` from `v2.15.8` | shape=`party_theme_program` | lineage=`normalize_participation_v1 <- v2.15.8`
- `facts.extract_program_shape.v1` -> slot=`facts.extract.program_shape` rev=`l1` | historical=`normalize_program_v1` from `v2.15.8` | shape=`party_theme_program` | lineage=`normalize_program_v1 <- v2.15.8`

## 3. Machine-readable

- [provenance json](/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_extract_family_v2_16_2_2026-03-09_provenance_map.json)
- [stage bank](/workspaces/events-bot-new/artifacts/codex/stage_bank/smart_update_lollipop_stage_bank_v2_16_2_2026-03-09.json)
- [prompt inventory](/workspaces/events-bot-new/artifacts/codex/smart_update_lollipop_facts_extract_family_v2_16_2_2026-03-09/prompt_inventory.json)
