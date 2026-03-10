# Smart Update Gemma Event Copy V2.16.2 Lollipop Mixed-Phase Series Post Consultation Synthesis

Date: 2026-03-10
Probe: `vk_wall_179910542_11821`
Source packet: [vk_wall_179910542_11821_2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/source_packets/vk_wall_179910542_11821_2026-03-10.md)

Consultations:

- `Opus`: [smart-update-lollipop-v2-16-2-mixed-phase-series-post-consultation-opus-2026-03-10.wrapper.md](/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-mixed-phase-series-post-consultation-opus-2026-03-10.wrapper.md)
- `Gemini`: [smart-update-lollipop-v2-16-2-mixed-phase-series-post-consultation-gemini-3.1-pro-preview-2026-03-10.md](/workspaces/events-bot-new/artifacts/codex/reports/smart-update-lollipop-v2-16-2-mixed-phase-series-post-consultation-gemini-3.1-pro-preview-2026-03-10.md)

## Probe summary

This VK post is not a clean announcement and not a pure recap.

It contains:

- a completed phase:
  - opening ceremony
  - `мероприятие прошло`
  - venue `арт-пространство «Заря»`
- a future anchor:
  - final
  - `Финал конкурса состоится уже 27 марта`
- one shared series identity:
  - `Национальный конкурс красоты и таланта «Мисс и Миссис Калининград 2026»`

The main risk is temporal bleeding:

- past-phase recap facts are rich and concrete;
- future-phase facts are sparse;
- a naive pipeline will often attach the past venue/details to the future final or drop the word `финал`.

## Shared conclusions

Both external models agreed on the following:

- this is a real separate source class;
- the right primary insertion point is `source.scope`;
- the pipeline must explicitly determine the future target phase before normal extraction proceeds;
- weak future anchors must fail closed:
  - keep only explicitly stated future facts;
  - do not inherit venue/time/logistics from the completed phase.

They also agreed that recap facts must survive:

- recap facts are not noise;
- they must remain available later as contextual material;
- but they must be structurally blocked from title, lead, and infoblock.

## Main disagreement

`Opus` direction:

- keep `phase_map -> target_phase`;
- temporary separate `future_phase / recap_context` extractors are acceptable as `v1 scaffolding`;
- later fold phase-awareness into the main extractor bank.

`Gemini` direction:

- do not create a permanent parallel extraction branch;
- this is over-split for Gemma;
- ship a smaller interceptor:
  - isolate target
  - run one strict target-aware extractor
  - keep recap in a separate background field

## Final decision

I am not adopting the fully parallel extractor pair as the canonical `v1`.

I am also not collapsing everything into a single ultra-minimal binary classifier.

The chosen `v1` is a short `3`-stage interceptor:

1. `scope.extract.phase_map.v1`
2. `scope.select.target_phase.v1`
3. `facts.extract.phase_scoped.v1`

This is the best compromise for the current `lollipop` architecture:

- short enough for Gemma;
- explicit enough to stop temporal bleeding;
- does not require immediate surgery across all existing extractors;
- does not create two permanent parallel extraction rivers that later need heavy merge logic.

## Canonical stage pack for this source class

### `scope.extract.phase_map.v1`

Responsibility:

- detect shared series identity;
- enumerate mentioned phases;
- mark each as `past|future|ongoing|uncertain`;
- capture per-phase evidence and any explicit date/venue.

### `scope.select.target_phase.v1`

Responsibility:

- choose whether the source updates a future phase, only recap context, or series context;
- emit `future_anchor_strength`;
- emit whether recap context exists.

### `facts.extract.phase_scoped.v1`

Responsibility:

- use source text plus the selected target phase;
- extract only target-phase facts into normal fact buckets;
- extract past-phase material only into `background_context`;
- emit `not_stated` for missing future fields instead of guessing them.

Expected output shape:

```json
{
  "target_phase_title": "",
  "target_facts": [
    {
      "fact_type": "subject|card|agenda|support|performer|participation|stage|theme",
      "content": "",
      "evidence": "",
      "phase_tag": "target_future"
    }
  ],
  "background_context": [
    {
      "content": "",
      "evidence": "",
      "phase_tag": "past_context"
    }
  ],
  "not_stated": []
}
```

## Guardrails

The following rules are now canonical for this source class:

1. If target phase is future, title and lead must keep the phase word if it exists:
   - `финал`
   - `следующий тур`
   - `закрытие`
   - etc.
2. Past venue/time/logistics cannot populate future target fields.
3. `background_context` facts must survive downstream.
4. `background_context` facts must not drive:
   - title
   - lead
   - infoblock
   - `event_core`
5. Weak future anchors may still produce a valid update, but only as a sparse future card:
   - series name
   - future phase word
   - explicit future date
   - no invented missing logistics

## What this means for `lollipop`

This becomes a new planned branch of `source.scope`, not a generic rewrite of all current families.

Immediate carry:

- add the probe to the canonical `lollipop` casebook;
- document the new source class in the funnel;
- keep this as a prompt-design / architecture carry until the first real stage lab is opened.

Not done yet:

- no production code change;
- no new Gemma family run yet;
- no casebook automation yet for this external source.

The current `12`-event execution casebook remains unchanged.
