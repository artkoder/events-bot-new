# Smart Update Lollipop Casebook

Кейсбук для `lollipop` хранит типовые классы источников, на которых проверяются stage-contracts.

## Case 1. Standard single-event post

- Один анонс, одна дата, одна площадка.
- Идёт через обычный pipeline.

## Case 2. Multi-event digest

- Один пост содержит несколько событий.
- Риск: sibling bleed.
- Проверяется `source.scope`.

## Case 3. Multi-source enrichment

- Новые facts приходят из второго или третьего источника.
- Риск: потеря secondary facts в dedup/merge.

## Case 4. Program-rich event

- Концерт, лекция, screening с насыщенной программой.
- Риск: потеря literal list items.

## Case 5. People-heavy event

- Много участников, ролей, профилей.
- Риск: схлопывание разных сущностей.

## Case 6. Screening / world-knowledge risk

- Пост про кинопоказ или произведение с сильным соблазном external knowledge.
- Риск: утечка сюжета/энциклопедии в `event_core`.

## Case 7. Mixed-phase series post

- Один пост рассказывает о уже состоявшейся фазе серии и одновременно даёт future anchor для следующей фазы.
- Канонический probe:
  - source: `https://vk.com/wall-179910542_11821`
  - packet: [vk_wall_179910542_11821_2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/source_packets/vk_wall_179910542_11821_2026-03-10.md)
- Правильная обработка:
  - past phase -> `background_context`
  - future phase -> target event
  - прошлые logistics факты не должны попадать в future card
  - если будущее событие описано только датой и phase word, это sparse future card, а missing fields идут в `not_stated`
