# Smart Update Gemma Event Copy V2.16.2 Lollipop Mixed-Phase Interceptor Lab

- date: `2026-03-10`
- case: `vk_wall_179910542_11821`
- source: `https://vk.com/wall-179910542_11821`
- stage chain: `scope.extract.phase_map.v1 -> scope.select.target_phase.v1 -> facts.extract.phase_scoped.v1`

## Verdict

- `series_identity_ok`: `True`
- `opening_past`: `True`
- `final_future`: `True`
- `target_mode_future`: `True`
- `target_is_final`: `True`
- `phase_word_kept_in_title`: `True`
- `future_date_kept`: `True`
- `past_venue_not_in_target`: `True`
- `background_has_past_context`: `True`
- `not_stated_has_venue`: `True`
- `not_stated_has_time`: `True`

## Stage Outputs

### `scope.extract.phase_map.v1`

```json
{
  "series_identity": "Национальный конкурс красоты и таланта «Мисс и Миссис Калининград 2026»",
  "phases": [
    {
      "phase_label": "церемония открытия",
      "temporal_status": "past",
      "date_if_known": "",
      "venue_if_known": "арт-пространстве «Заря»",
      "key_facts": [
        "opening ceremony held at Zarya art space"
      ],
      "evidence": [
        "Мероприятие прошло в арт-пространстве «Заря»"
      ]
    },
    {
      "phase_label": "финал",
      "temporal_status": "future",
      "date_if_known": "27 марта",
      "venue_if_known": "",
      "key_facts": [
        "final scheduled for March 27th"
      ],
      "evidence": [
        "Финал конкурса состоится уже 27 марта"
      ]
    }
  ]
}
```

### `scope.select.target_phase.v1`

```json
{
  "target_mode": "future_phase",
  "target_phase_label": "финал",
  "future_anchor_strength": "medium",
  "recap_available": true,
  "reason": "Future final has date but no venue details."
}
```

### `facts.extract.phase_scoped.v1`

```json
{
  "target_phase_title": "Финал Национального конкурса красоты и таланта «Мисс и Миссис Калининград 2026»",
  "target_facts": [
    {
      "fact_type": "card",
      "content": "Дата: 27 марта.",
      "evidence": "Финал конкурса состоится уже 27 марта",
      "phase_tag": "target_future"
    }
  ],
  "background_context": [
    {
      "content": "Группа «Дефиле для женщин элегантного возраста» выступала на церемонии открытия конкурса.",
      "evidence": "Группа «Дефиле для женщин элегантного возраста» на церемонии открытия Национального конкурса красоты и таланта «Мисс и Миссис Калининград 2026»",
      "phase_tag": "past_context"
    },
    {
      "content": "Церемония открытия прошла в арт-пространстве «Заря».",
      "evidence": "Мероприятие прошло в арт-пространстве «Заря»",
      "phase_tag": "past_context"
    },
    {
      "content": "Татьяна Лыкова, 69 лет, участница программы «Балтийское долголетие», приняла участие в конкурсе.",
      "evidence": "Татьяна Лыкова приняла участие в Национальном конкурсе красоты и таланта «Мисс и Миссис Калининград 2026»",
      "phase_tag": "past_context"
    }
  ],
  "not_stated": [
    "venue",
    "time"
  ]
}
```
