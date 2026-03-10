# Smart Update Lollipop Mixed-Phase Prompts

## Purpose

This document is the canonical prompt pack for the `mixed-phase series post with future anchor` source class in `Smart Update lollipop`.

This pack is designed for sources like:

- one phase already happened;
- the post recaps that phase;
- the same post also points to a future phase of the same series.

Example probe:

- `https://vk.com/wall-179910542_11821`

Companion docs:

- casebook: [smart-update-lollipop-casebook.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-casebook.md)
- funnel: [smart-update-lollipop-funnel.md](/workspaces/events-bot-new/docs/llm/smart-update-lollipop-funnel.md)
- consultation synthesis: [smart-update-gemma-event-copy-v2-16-2-lollipop-mixed-phase-series-post-consultation-synthesis-2026-03-10.md](/workspaces/events-bot-new/docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-mixed-phase-series-post-consultation-synthesis-2026-03-10.md)

## Canonical `v1` stage pack

1. `scope.extract.phase_map.v1`
2. `scope.select.target_phase.v1`
3. `facts.extract.phase_scoped.v1`

Design carry:

- `scope.extract.phase_map.v1` and `scope.select.target_phase.v1` are derived from the stronger `Opus` rewrite.
- `facts.extract.phase_scoped.v1` is the final local synthesis after `Opus + Gemini`: one phase-aware extractor instead of two permanent parallel extractors.

## `scope.extract.phase_map.v1`

```text
ROLE
You do one small step: scope.extract.phase_map.v1.

GOAL
Read one source post. Identify the shared event series and map every mentioned phase with its temporal status.

OUTPUT
Return only JSON:
{
  "series_identity": "",
  "phases": [
    {
      "phase_label": "",
      "temporal_status": "past|future|ongoing|uncertain",
      "date_if_known": "",
      "venue_if_known": "",
      "key_facts": [],
      "evidence": []
    }
  ]
}

RULES
- series_identity = the shared name of the event series or competition across phases.
- Each phase is a distinct stage of that series (opening, semifinal, final, closing, etc.).
- temporal_status:
  - "past" = the text uses past tense or words like "прошло", "состоялось", "был".
  - "future" = the text uses future tense or words like "состоится", "пройдёт", "будет".
  - "ongoing" = happening right now.
  - "uncertain" = cannot determine from text.
- evidence = exact short quotes from the source that prove the temporal status.
- If only one phase exists, return one phase.
- Do not invent phases not mentioned in the text.
- Do not write prose.

EXAMPLE 1
Input: «Открытие фестиваля «Балтийские сезоны» прошло 5 марта в филармонии. Финал фестиваля состоится 20 апреля.»
Output:
{
  "series_identity": "фестиваль «Балтийские сезоны»",
  "phases": [
    {
      "phase_label": "открытие",
      "temporal_status": "past",
      "date_if_known": "5 марта",
      "venue_if_known": "филармония",
      "key_facts": ["opening held at philharmonic"],
      "evidence": ["прошло 5 марта в филармонии"]
    },
    {
      "phase_label": "финал",
      "temporal_status": "future",
      "date_if_known": "20 апреля",
      "venue_if_known": "",
      "key_facts": ["final scheduled"],
      "evidence": ["Финал фестиваля состоится 20 апреля"]
    }
  ]
}

EXAMPLE 2
Input: «Ежегодный забег «Янтарная миля» пройдёт 15 июня на набережной.»
Output:
{
  "series_identity": "забег «Янтарная миля»",
  "phases": [
    {
      "phase_label": "main",
      "temporal_status": "future",
      "date_if_known": "15 июня",
      "venue_if_known": "набережная",
      "key_facts": ["annual run on embankment"],
      "evidence": ["пройдёт 15 июня на набережной"]
    }
  ]
}
```

## `scope.select.target_phase.v1`

```text
ROLE
You do one small step: scope.select.target_phase.v1.

GOAL
Given a phase map, decide which phase this source should target for the event update.

INPUT
{phase_map_json}

OUTPUT
Return only JSON:
{
  "target_mode": "future_phase|past_recap_only|series_context_only|unresolved",
  "target_phase_label": "",
  "future_anchor_strength": "strong|medium|weak|none",
  "recap_available": true,
  "reason": ""
}

RULES
- Prefer "future_phase" when a future phase exists with at least a date or a named phase word (финал, тур, этап, закрытие).
- future_anchor_strength:
  - "strong" = future phase has date + venue + at least one logistics detail.
  - "medium" = future phase has date + phase word, but no venue or time.
  - "weak" = future phase is mentioned only briefly (one sentence or less), with date or phase word but not both.
  - "none" = no future phase detected.
- If future_anchor_strength is "none", choose "past_recap_only" or "series_context_only".
- "series_context_only" = the source is about the series but no specific phase is actionable.
- "unresolved" = you cannot confidently determine the target. Use sparingly.
- recap_available = true if any past phase exists that can provide background context.
- Never rewrite a past phase as if it were upcoming.

EXAMPLE
Input phase_map with: opening (past), final (future, date: 20 апреля, no venue)
Output:
{
  "target_mode": "future_phase",
  "target_phase_label": "финал",
  "future_anchor_strength": "medium",
  "recap_available": true,
  "reason": "Future final has date but no venue or time details."
}
```

## `facts.extract.phase_scoped.v1`

```text
ROLE
You do one small step: facts.extract.phase_scoped.v1.

GOAL
Extract facts for the selected target phase and keep all past-phase recap only as background context.

INPUT
- source text
- series_identity
- target_mode
- target_phase_label
- future_anchor_strength
- phase_map

OUTPUT
Return only JSON:
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

RULES
- If target_mode is not "future_phase", do not invent a future card.
- target_phase_title must keep the phase word if it exists: "финал", "тур", "этап", "закрытие".
- target_facts may contain only facts explicitly stated for the selected target phase.
- Past venue, time, price, registration, attendees, and recap details must never enter target_facts.
- If a field for the target phase is missing, add it to not_stated instead of guessing.
- background_context may contain only facts from past phases.
- background_context facts are background-only and must not drive title, lead, infoblock, or event_core later.
- Keep facts compact and literal. Do not write narrative prose.

SAFE DEFAULT
- If the future phase has only a date and a phase word, return a sparse card:
  - target title
  - explicit future date
  - everything else in not_stated

EXAMPLE
Source:
«Церемония открытия конкурса прошла в арт-пространстве „Заря“. Татьяна Лыкова приняла участие. Финал конкурса состоится 27 марта.»

Context:
- series_identity = "Национальный конкурс красоты и таланта «Мисс и Миссис Калининград 2026»"
- target_mode = "future_phase"
- target_phase_label = "финал"
- future_anchor_strength = "medium"

Output:
{
  "target_phase_title": "Финал Национального конкурса красоты и таланта «Мисс и Миссис Калининград 2026»",
  "target_facts": [
    {
      "fact_type": "subject",
      "content": "Финал Национального конкурса красоты и таланта «Мисс и Миссис Калининград 2026».",
      "evidence": "Финал конкурса",
      "phase_tag": "target_future"
    },
    {
      "fact_type": "card",
      "content": "Дата: 27 марта.",
      "evidence": "состоится уже 27 марта",
      "phase_tag": "target_future"
    }
  ],
  "background_context": [
    {
      "content": "Церемония открытия конкурса прошла в арт-пространстве «Заря».",
      "evidence": "Мероприятие прошло в арт-пространстве «Заря»",
      "phase_tag": "past_context"
    },
    {
      "content": "Татьяна Лыкова приняла участие в конкурсе.",
      "evidence": "Татьяна приняла участие в Национальном конкурсе красоты и таланта",
      "phase_tag": "past_context"
    }
  ],
  "not_stated": ["venue", "time", "price", "registration"]
}
```

## Current intended behavior

This prompt pack is meant as a narrow interceptor for the new source class.

It is not yet wired into the active `12`-event `lollipop` execution casebook.
The next logical step is a dedicated `source.scope` family-lab on a small set of similar sources.
