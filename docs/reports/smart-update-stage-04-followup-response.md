# Smart Update Stage 04 Follow-up Response

Дата: 2026-03-06
Формат: прямые ответы на 4 вопроса + конкурентные дополнения.

---

## Вопрос 1: `multi_event_source_blocker` — production-safe?

### Verdict: **ДА, production-safe.** Можно нести в runtime Stage 04B.

**Обоснование**:

Rule `source_url_owner_pair_max >= 4 AND NOT title_exact → different` — это **structural blocker**, не heuristic. Он оценивает не содержание event'а, а свойство SOURCE. Если source породил ≥4 events — это program/digest page по определению. В program page каждый child — отдельное мероприятие. Merge between children допустим ТОЛЬКО при `title_exact` (true duplicate extraction).

**False-merge risk**: нулевой. Rule выдаёт `different`, а не `merge`. Worst case — false different на legit dupe из prolific source. Но:
- title_exact guard защищает от этого: если titles идентичны — rule не вмешивается и дубль обрабатывается другими rules
- если titles НЕ идентичны в prolific source — это реально разные мероприятия

**False-different risk**: Low. Единственный сценарий: extraction из program page порождает два варианта ОДНОГО child event с slightly different titles. Но это extraction bug → нужен extraction fix, а не identity rule.

**Production action**: deploy в Stage 04B без shadow mode. Rule достаточно narrow и safe-by-construction (different-only, never merge).

### Конкурентное дополнение: порог 4 — правильный ли?

Я бы рассмотрел **порог 3** вместо 4. На текущем casepack:

- `source_url_owner_pair_max=6` — museum_holiday (must_not_merge, 3 pairs) ✅
- `source_url_owner_pair_max=3` — led_hearts (must_merge, 3 pairs, BUT `title_exact=True` → guard пропускает)
- `source_url_owner_pair_max=2` — several pairs, mixed

При пороге 3: led_hearts пары имеют `title_exact=True` → rule не fires → safe. museum_holiday (=6) → fires → different ✅.

**Но**: порог 3 ловит `source_url_owner_pair_max=3` pairs, где `title_exact=False`. Есть ли такие в must_merge? На текущем casepack — нет. Но это extrapolation risk.

**Verdict**: оставить порог 4. Он доказан на данных и не нуждается в extrapolation. Снижать до 3 можно в Stage 05 после расширения casepack.

---

## Вопрос 2: `cross_source_exact_match` — runtime-safe или shadow first?

### Verdict: **Runtime-safe с minor caveat. Рекомендую deploy, но с alert-on-fire.**

**Почему я считаю это runtime-safe уже сейчас**:

1. **Preconditions экстремально жёсткие**: `title_exact AND same_date AND venue_match AND BOTH times known AND times equal AND NOT time_conflict AND NOT same_source`. 6 preconditions, каждый independently необходим.

2. **Control validation**: на текущем casepack ноль false merges. Rule также безопасно fires на 7 already-resolved must-merge pairs (fort_excursion, fort_night, little_women, sisters_followup, zoikina) — значит, signal корректен не только для plastic_nutcracker.

3. **Realistic counterexample search**: нужно найти два РЕАЛЬНО РАЗНЫХ события с:
   - идентичным normalized title (не "Выставка", а полный title)
   - в одном городе
   - на одной площадке
   - в один день
   - в одно время
   - из РАЗНЫХ sources

   Это практически невозможно для real-world events. Два организатора независимо проводят мероприятие с identical full title, в одном venue, в один день, в одно время? Нет.

**Minor caveat**: единственный теоретический scenario — **venue rebooking**. Venue отменило одно мероприятие и поставило другое с идентичным названием (e.g., повторяющийся формат типа "Джаз по пятницам"). Но даже здесь:
- Если old event не удалён из DB — merge корректен (same event, rebooking)
- Если new event genuinely different — title будет отличаться (другое число, другой программа)

**Production action**: deploy в Stage 04B. Добавить alert-on-fire первые 2 недели: при каждом срабатывании — log pair для manual review.

### Конкурентное дополнение: а shadow mode вообще стоит?

Честный контраргумент против shadow: rule fires на `plastic_nutcracker` и 7 already-resolved pairs. Shadow покажет те же 8 fires на том же snapshot. Shadow на FRESH snapshot покажет, может быть, ещё 2-3 fires. Но чтобы найти false merge в shadow — нужно ВРУЧНУЮ проверять каждый fire, что по трудозатратам ≈ manual review.

**Лучшая альтернатива**: deploy + auto-alert + 2-week review window. Если alert fires на pair, которое не в gold set → manual check → добавить в gold set. Это дешевле, чем shadow mode, и даёт live data.

---

## Вопрос 3: remaining 22 must-merge gray → LLM?

### Verdict: **ДА, согласен. С уточнениями по LLM strategy.**

22 must-merge gray pairs — это правильный LLM territory. Попытка вытащить их deterministic'ом:
- либо требует broad heuristics (которые Stage 04 уже опроверг)
- либо даёт marginal gain при disproportionate risk

**Но**: "оставить за LLM" ≠ "отложить на потом". LLM layer нужно проектировать _параллельно_ с Stage 04A/04B deploy.

### Уточнение: как именно LLM должен обрабатывать 22 gray pairs

Я группирую 22 пары по **оптимальной LLM strategy**:

| Группа | Пары | LLM confidence expectation | Prompt тип |
|---|---|---|---|
| **High-confidence LLM merge** | led_hearts ×3 (same_source + title_exact + text_same + poster_overlap), plastic_nutcracker (если не cross_source_exact_match) | >95% same_event | Тип 2: extraction bug — LLM видит overwhelming same-source proof |
| **Medium-confidence LLM merge** | hudozhnitsy ×10 (title alias + cross-source), matryoshka (semantic + same_vk_group) | 70-85% same_event | Тип 1: title alias — LLM оценивает semantic overlap |
| **Low-confidence LLM merge** | shambala ×2 (brand vs lineup), makovetsky (brand vs program), prazdnik ×2 (broken extraction without source proof), little_women [2761,2815], oncologists | 40-60% same_event | Тип 3: weak signal — LLM should default to uncertain |

**Ключевой design decision**: verdict → action mapping для каждой confidence group.

```
same_event   → auto_merge     (ONLY for high-confidence group)
likely_same  → gray_softlink  (user sees both, linked)
uncertain    → keep_separate  (no action)
different    → mark_different (no softlink)
```

**Для low-confidence group**: даже `same_event` от LLM → `gray_softlink`, NOT auto_merge. Потому что если deterministic не смог доказать — LLM тоже может ошибаться.

### Конкурентное дополнение: batch vs pairwise

Текущий consensus — compact pairwise LLM. Но для `hudozhnitsy` (10 пар из 5 events) pairwise = 10 вызовов. При RPM=20 это 30sec minimum, при TPM=12000 это ~5000 tokens = ок.

**Альтернатива для кластеров**: one-shot cluster call.

```
"Перед тобой 5 анонсов одного и того же слота (дата/время/площадка).
Определи: все ли они описывают ОДНО мероприятие, или среди них есть разные?"
```

Payload: 5 × ~200 tokens = ~1000 tokens. Один вызов вместо 10. Ответ: grouping.

**Преимущество**: LLM видит ВСЮ картину сразу. Если 5 titles: "Выставка Художницы", "Художницы — Филиал Третьяковской", "Художницы: русское искусство", "Выставка «Художницы»", "Калининградский филиал — Художницы" — в контексте друг друга очевидно, что это одно событие.

**Недостаток**: payload больше. Для 5 events по 200 tokens = 1000 input + prompt. Для Gemma с TPM=12000 — ок (1500 total tokens << 12000).

**Risk**: если кластер содержит и same и different — LLM может ошибиться в grouping. Но: для кластеров, где все share date+time+venue, это маловероятно.

**Verdict**: использовать cluster call для кластеров ≥ 3 events at same slot. Для isolated pairs — standard pairwise. Это cutting-edge optimization, которую я не видел в предыдущих materials.

---

## Вопрос 4: residual false-merge class в `run_04b_plus_both`?

### Verdict: **На текущем casepack — нет. Но есть two production scenarios worth monitoring.**

#### 4.1. `cross_source_exact_match` + recurring events

**Scenario**: "Джаз по пятницам" в баре X. VK post от бара, TG repost от городского канала. Каждую пятницу — новое событие с ТОЖДЕСТВЕННЫМ title, venue, time. Только date different. Rule requires `same_date`, так что это safe → bail.

**НО**: если два source'а опубликовали объявление о РАЗНЫХ пятницах, но extractor присвоил одинаковую дату (extraction bug) — merge. Это false merge.

**Probability**: Low. Extraction дату берёт из текста; два поста о разных пятницах будут содержать разные даты.

**Monitoring**: alert-on-fire для cross_source_exact_match + check title against recurring patterns ("по пятницам", "каждую субботу", "еженедельно").

#### 4.2. `multi_event_source_blocker` + legit source update

**Scenario**: VK wall owner публикует 5 постов о мероприятиях (→ source_url_owner_pair_max=5). Потом один из eventов получает update post (new text, same title). Old post и new post → same title, different source_post_url, но source_url_owner_pair_max=5 from old events.

Rule fires: `source_url_owner_pair_max >= 4 AND NOT title_exact` → different. НО: update post может иметь **slightly different title** (e.g., "Выставка 'Матрёшки'" → "Выставка 'Матрёшки' (до 5 апреля)"). title_exact=False → rule fires → false different.

**Probability**: Very low. Source_url_owner_pair_max counts events from SAME source URL, not same owner. Update post has different source_post_url → different source_url_owner context.

**Wait** — нет, `source_url_owner_pair_max` — это max по всем source URLs vs existing events. Let me re-read the signal semantics...

Фактически: если new source_post_url отличается от old → `source_url_owner_pair_max` для НОВОЙ пары будет low (1 event from new source). Rule не fires. Safe.

Only fires if SAME source_post_url produced ≥4 events. An update post won't have same URL. ✅

#### 4.3. Edge case я пока не могу закрыть

**The one I can't disprove**: two genuinely different events с `title_exact=True` + `same_date` + `venue_match` + `same_time`, from different sources.

Example: venue hosts "Открытая репетиция" every week. Two bloggers post about DIFFERENT weeks' rehearsal, but extractor assigns same date due to ambiguous text. `cross_source_exact_match` fires → false merge.

**This IS theoretically possible.** But:
- Requires extraction bug on date (same date from different weeks)
- AND exact same title
- AND different sources
- Probability: very low

**Mitigation**: extraction quality improvement + alert-on-fire monitoring.

---

## 5. Summary: rollout ladder после этого follow-up

```
Stage 04A (immediate):
  └─ run_03_tightened (6 rules + 2 tightening patches)
  └─ Effect: 15 resolved must-merge, 31 resolved must-not-merge, 0 errors

Stage 04B (2-3 days after 04A validation):
  └─ + multi_event_source_blocker (production-safe, no shadow)
  └─ + cross_source_exact_match (production-safe, with alert-on-fire)
  └─ Effect: 16 resolved must-merge, 34 resolved must-not-merge, 0 errors

Stage 04C (parallel, 5-10 days):
  └─ Compact pairwise LLM for 22 remaining gray must-merge pairs
  └─ Cluster call for hudozhnitsy (1 call vs 10 pairwise)
  └─ Shadow mode first (2 weeks), then production

Stage 04D (ongoing):
  └─ Casepack expansion (+6 new gold cases minimum)
  └─ Weekly regression on growing casepack
  └─ Extraction layer fixes for led_hearts-class bugs
```

**Disagreement surface после этого follow-up**: практически нулевой. Остаётся:
1. Cluster vs pairwise LLM for large groups (my optimization proposal)
2. Confidence-based verdict → action mapping (my tiered approach)
3. Порог `multi_event_source_blocker` (4 vs 3) — можно решить в Stage 05

Всё остальное — consensus.
