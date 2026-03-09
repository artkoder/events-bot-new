# Smart Update Gemma Event Copy V2.15.10 Screening Grounding Retune

Date: 2026-03-08

Этот раунд сфокусирован только на `screening_card` после того, как `v2.15.9` улучшил lead_ok, но допустил скрытые groundedness-регрессии.

Изменения:

- lead/body разбиты на sentence-level steps;
- из body prompts убран title и усилен запрет на world knowledge;
- добавлен отдельный `grounding_audit`, и победой считается только grounded candidate без unsupported claims.

## Event 2659 — Посторонний

- Previous v2.15.8 lead_ok: `no`
- New lead_ok: `yes`
- New slot score: `5/6`
- New missing deterministic: `8`
- Unsupported claims: `0`

### Routed Facts

- lead:
  - Экранизация романа Альбера Камю.
  - Режиссёр: Франсуа Озон.
  - Действие фильма разворачивается в Алжире в 1938 году.
- plot:
  - Фильм рассказывает о мужчине по имени Мерсо, который не проявляет эмоций на похоронах своей матери и оказывается вовлеченным в событие, меняющее его жизнь.
- support:
  - Премьера на Венецианском кинофестивале (2025).
  - Продолжительность: 125 минут
  - Дублированный перевод
  - 18+
  - Кинозал филиала Третьяковской галереи

### Chosen Combo

- lead variant: `lead_v1_strict`
- plot variant: `plot_v1_strict`
- support variant: `support_v1_strict`

### Final Text

```md
«Посторонний» — это экранизация романа Альбера Камю, снятая режиссёром Франсуа Озоном, действие которой разворачивается в Алжире в 1938 году.

Мерсо, человек, не проявляющий эмоций на похоронах своей матери, оказывается вовлеченным в событие, меняющее его жизнь. - Премьера фильма состоялась на Венецианском кинофестивале в 2025 году.
- Продолжительность фильма составляет 125 минут.
```

### Audit

- unsupported_claims: `[]`
- missing_fact_ids: `[7, 8, 9]`

## Event 2747 — Киноклуб: «Последнее метро»

- Previous v2.15.8 lead_ok: `no`
- New lead_ok: `yes`
- New slot score: `6/6`
- New missing deterministic: `6`
- Unsupported claims: `0`

### Routed Facts

- lead:
  - Режиссёр: Франсуа Трюффо
  - В главных ролях: Катрин Денев, Жерар Депардье
  - Оккупированный Париж
- plot:
  - Театр под контролем нацистов
  - Любовный треугольник
- support:
  - 10 премий «Сезар»
  - Длительность: 2 ч 10 мин
  - Вход: 100 ₽

### Chosen Combo

- lead variant: `lead_v1_strict`
- plot variant: `plot_v1_strict`
- support variant: `support_v1_strict`

### Final Text

```md
«Последнее метро» — фильм Франсуа Трюффо с Катрин Денев и Жераром Депардье, действие которого происходит в оккупированном Париже.

В театре, находящемся под контролем нацистов, разворачивается любовный треугольник. - Фильм получил 10 премий «Сезар».
- Длительность фильма составляет 2 ч 10 мин.
- Вход стоит 100 ₽.
```

### Audit

- unsupported_claims: `[]`
- missing_fact_ids: `[]`

## Findings

- `screening_card` действительно нельзя считать solved только по `lead_ok` и slot coverage: известные фильмы провоцируют Gemma на world-knowledge drift.
- sentence-level split + grounding audit лучше соответствует требованию пользователя «в тексте должны быть все факты и только факты».
- Для screening финальный pipeline должен включать groundedness-gate; иначе локальный win по формальным regex-метрикам будет ложноположительным.
- Для `2659` отдельный manual follow-up на `support_sentence` показал, что missing support facts можно вернуть без hallucination, если ужесточить именно этот micro-contract и явно запретить bullets/list fallback.

## Manual Follow-up

- `2659 support_sentence` improved candidate:

```md
Премьера фильма состоялась на Венецианском кинофестивале 2025, продолжительность составляет 125 минут. Фильм имеет дублированный перевод и возрастной рейтинг 18+, показ состоится в кинозале филиала Третьяковской галереи.
```
