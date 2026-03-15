# Guide Excursions Dedup Prompts

Промпт-контракт для `Route Matchmaker v1`.

Цель: semantic dedup конкретных выходов экскурсии, а не просто сравнение заголовков.

## Model role

Модель не должна решать “похожи ли тексты”.

Она должна решать более узкую задачу:

- это один и тот же конкретный выход экскурсии;
- это та же типовая экскурсия, но другой выход;
- это вообще разные продукты.

## Pair judge

Prompt family: `guide_excursions.dedup_pair_judge.v1`

### System

```text
You classify whether two guide-excursion occurrence candidates describe the same concrete outing.

Be conservative about merging.
Same date alone is not enough.
Same booking contact alone is not enough.
Different editorial angle inside the same trip can still be the same concrete outing.

Return strict JSON only.
```

### User payload

```json
{
  "task": "classify_pair",
  "left": {
    "occurrence_id": 24,
    "source_kind": "organization_with_tours",
    "source_username": "ruin_keepers",
    "source_title": "Хранители руин | Ruin Keepers",
    "title": "Два легендарных музея в одном путешествии!",
    "date": "2026-03-14",
    "time": null,
    "meeting_point": null,
    "price_text": null,
    "booking_text": "@ruin_keepers_admin",
    "booking_url": "https://t.me/ruin_keepers_admin",
    "audience_fit": ["детям", "школьным группам", "любителям истории"],
    "summary_one_liner": "Мамоновский городской музей & Музей кирпича...",
    "post_excerpt": "В ближайшую субботу, 14 марта, в рамках нашего путешествия на юго-запад области..."
  },
  "right": {
    "occurrence_id": 25,
    "source_kind": "organization_with_tours",
    "source_username": "ruin_keepers",
    "source_title": "Хранители руин | Ruin Keepers",
    "title": "юго-запад области в Мамоново и к легендарному замку Бальга",
    "date": "2026-03-14",
    "time": null,
    "meeting_point": null,
    "price_text": null,
    "booking_text": null,
    "booking_url": null,
    "audience_fit": ["детям", "любителям истории", "любителям природы"],
    "summary_one_liner": "14 марта, в субботу, приглашаем в путешествие на юго-запад области в Мамоново...",
    "post_excerpt": "В Мамоново нас ждут два отличных музея..."
  },
  "candidate_features": {
    "same_source": true,
    "same_date": true,
    "same_time": false,
    "same_booking_url": false,
    "shared_route_tokens": ["мамоново", "юго-запад", "музеи"],
    "token_overlap_score": 0.192
  }
}
```

### Expected output

```json
{
  "decision": "same_occurrence",
  "relation": "master_announce_vs_focus_teaser",
  "canonical_side": "right",
  "confidence": 0.93,
  "shared_facts": [
    "same_date",
    "same_source",
    "same_trip_to_mamonovo",
    "museum_part_explicitly_embedded_in_master_trip"
  ],
  "conflicting_facts": [],
  "reason_short": "The museum post is a focused teaser for the broader Mamonovo+Balgа trip on the same date."
}
```

### JSON schema

```json
{
  "decision": "same_occurrence | same_occurrence_update | same_template_other_occurrence | distinct | uncertain",
  "relation": "string",
  "canonical_side": "left | right | neither",
  "confidence": 0.0,
  "shared_facts": ["string"],
  "conflicting_facts": ["string"],
  "reason_short": "string"
}
```

## Cluster resolver

Prompt family: `guide_excursions.dedup_cluster_resolver.v1`

Используется, когда внутри одного candidate bucket 3+ occurrence.

### System

```text
You resolve a cluster of guide-excursion occurrence candidates into canonical and supporting members.

Choose exactly one canonical occurrence for each same-occurrence cluster.
Do not merge different outings of the same recurring tour.
Return strict JSON only.
```

### Expected output

```json
{
  "clusters": [
    {
      "member_ids": [24, 25, 31],
      "decision": "same_occurrence_cluster",
      "canonical_occurrence_id": 25,
      "member_roles": {
        "24": "focus_teaser",
        "25": "master_announce",
        "31": "aggregator_mirror"
      },
      "confidence": 0.91
    }
  ],
  "distinct_ids": [44, 45]
}
```

## Prompt rules

- Не просить модель переписывать тексты.
- Не отдавать модельке весь пост целиком без структуры, если можно дать extracted facts + excerpt.
- Не позволять merge только по одному совпавшему атрибуту.
- Обязательно передавать candidate features от deterministic prefilter.
- Для uncertain-ответов не автосклеивать, а оставлять separate occurrence.

## Canonical-side policy

Модель должна выбирать canonical сторону по смысловой полноте:

- master announce выше teaser;
- non-aggregator выше aggregator;
- post с booking/logistics выше поста без них;
- status update не должен становиться canonical, если уже есть полноценный announce.

## Negative guidance

Модель должна явно помнить:

- один и тот же телефон записи часто используется для разных экскурсий;
- один и тот же гид может вести несколько разных прогулок в один день;
- пересечение по району/городу не доказывает same occurrence;
- одна типовая экскурсия в разные даты не является дублем.

## Frozen cases to keep in prompt QA

- `ruin_keepers/5054` + `ruin_keepers/5055` => `same_occurrence`
- `vkaliningrade/4563` + `vkaliningrade/4565` => `distinct`
