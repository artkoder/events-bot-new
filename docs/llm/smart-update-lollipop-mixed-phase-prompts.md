# Smart Update Lollipop Mixed-Phase Prompts

Канонический prompt-pack для класса `past-phase recap + future-phase anchor`.

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
```

## `scope.select.target_phase.v1`

```text
ROLE
You do one small step: scope.select.target_phase.v1.

GOAL
Given a phase map, decide which phase this source should target for the event update.

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
  - "weak" = future phase is mentioned only briefly, with date or phase word but not both.
  - "none" = no future phase detected.
- If future_anchor_strength is "none", choose "past_recap_only" or "series_context_only".
- "series_context_only" = the source is about the series but no specific phase is actionable.
- "unresolved" = you cannot confidently determine the target.
- recap_available = true if any past phase exists that can provide background context.
- Never rewrite a past phase as if it were upcoming.
```

## `facts.extract.phase_scoped.v1`

```text
ROLE
You do one small step: facts.extract.phase_scoped.v1.

GOAL
Extract facts for the selected target phase and keep all past-phase recap only as background context.

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
```
