# Smart Update Lollipop Casebook

## Purpose

This is the canonical casebook for the experimental `Smart Update lollipop` branch.

It serves two different roles:

- `execution casebook`: local event ids that are already wired into the current family-lab harnesses;
- `source-risk probes`: external or not-yet-ingested sources that pressure-test new source classes before they enter the automated family runs.

## Current execution casebook

These `12` local events remain the current full-run execution casebook for `facts.extract -> facts.prioritize`.

Core `6`:

- `2673` `Собакусъел`
- `2687` `Лекция «Художницы»`
- `2734` `Концерт Владимира Гудожникова «Ты… моя мелодия»`
- `2659` `Посторонний`
- `2731` `Хоровая вечеринка «Праздник у девчат»`
- `2498` `Нюрнберг`

Extension `6`:

- `2747` `Киноклуб: «Последнее метро»`
- `2701` `«Татьяна танцует»`
- `2732` `Вечер в русском стиле`
- `2759` `Выставка «Королева Луиза: идеал или красивая легенда?»`
- `2657` `Коллекция украшений 1930–1960-х годов`
- `2447` `Мастер-класс «Мирное небо»`

## Source-risk probes

These probes are not yet part of the automated `12`-event execution harness. They are kept here so new source classes are fixed canonically before they disappear into ad-hoc notes.

### `probe-001` Mixed-Phase Series Post With Future Anchor

- `probe_id`: `vk_wall_179910542_11821`
- `source_url`: `https://vk.com/wall-179910542_11821`
- `source_packet`: [vk_wall_179910542_11821_2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/source_packets/vk_wall_179910542_11821_2026-03-10.md)
- source type:
  - one post describes a completed phase of a series
  - the same post also gives a future phase date for the same series

#### Cleaned source text

> Группа «Дефиле для женщин элегантного возраста» на церемонии открытия Национального конкурса красоты и таланта «Мисс и Миссис Калининград 2026».
>
> Мероприятие прошло в арт-пространстве «Заря», где участницы программы «Балтийское долголетие» и тренер Вера Трунова посетили торжественную фотосессию и вдохновились выступлениями девушек и женщин.
>
> Особое внимание привлекла Татьяна Лыкова, 69 лет — она является участницей программы «Балтийское долголетие». Татьяна приняла участие в Национальном конкурсе красоты и таланта «Мисс и Миссис Калининград 2026», доказав всем, что красота не имеет возраста.
>
> Финал конкурса состоится уже 27 марта.

#### Why this source class is risky

- It is not a clean announcement of one upcoming event.
- It is not a pure recap either.
- It contains:
  - `past-phase recap`
  - `series identity`
  - `future-phase anchor`
- A naive rewrite will often produce a misleading hybrid:
  - title about the opening, but lead about the final;
  - lead that presents a past event as upcoming;
  - future final copied without the explicit word `финал`;
  - invented venue/time for the final, borrowed from the already completed opening.

#### Safe interpretation target

- `series_identity`:
  - `Национальный конкурс красоты и таланта «Мисс и Миссис Калининград 2026»`
- `past_phase`:
  - opening ceremony
  - already happened
  - venue: `арт-пространство «Заря»`
  - recap context about participants and Tatyana Lykova
- `future_phase`:
  - final
  - date: `27 марта`
- `future_anchor_strength`:
  - weak-to-medium
  - there is a future date, but no confirmed future venue, time, ticketing, or registration

#### Architectural implication

This class belongs first to `source.scope`, not only to `facts.extract`.

The pipeline needs to detect that:

- one source mentions multiple timeline phases of the same series;
- one phase is retrospective;
- another phase is future-looking and potentially actionable.

Then extraction must split the material:

- `future_actionable_phase_facts`
- `past_recap_context_facts`
- `shared_series_identity_facts`

#### Post-consultation `v1` stage pack

After `Opus + Gemini`, the current recommended minimal pack is:

`source.scope`:

- `scope.extract.phase_map`
  - detect phases, temporal status, per-phase evidence, and shared series identity
- `scope.select.target_phase`
  - decide whether the source should update a future phase, only add recap context, or stay unresolved
  - emit `future_anchor_strength`

`facts.extract`:

- `facts.extract.phase_scoped`
  - extract target-phase facts into normal fact buckets
  - extract past recap only into `background_context`
  - emit `not_stated` for missing future logistics instead of guessing them

Guardrail:

- `phase_guard`
  - not a separate free-standing LLM family for now
  - a validation/bucketing rule that keeps `background_context` out of `title`, `lead`, `infoblock`, and `event_core`

#### Acceptance criteria for this class

- the extracted title/identity must name the contest correctly;
- if the future phase is the target, the copy must explicitly say `финал`;
- the post must not be rewritten as if the opening ceremony is still ahead;
- recap details may survive as context, but only as past context;
- weak future anchors must not justify inventing a full event card.
