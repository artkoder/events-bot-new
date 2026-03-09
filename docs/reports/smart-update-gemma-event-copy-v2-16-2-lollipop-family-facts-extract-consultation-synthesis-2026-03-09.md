# Smart Update V2.16.2 Lollipop Facts.Extract Consultation Synthesis

Дата: 2026-03-09

## 1. Scope

Этот synthesis закрывает вопрос, который должен был быть решён после broad-run family lab: по evidence-pack определить, какие prompts внутри `facts.extract` реально сильные для `Gemma`, а какие нет.

Материалы раунда:
- broad-run report: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-extract-lab-2026-03-09.md`
- evidence-pack index: `docs/reports/smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-extract-evidence-pack-2026-03-09.md`
- consultation brief: `artifacts/codex/tasks/smart-update-lollipop-v2-16-2-facts-extract-family-consultation-brief-2026-03-09.md`
- Opus report: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-extract-family-consultation-opus-2026-03-09.md`
- Gemini report: `artifacts/codex/reports/smart-update-lollipop-v2-16-2-facts-extract-family-consultation-gemini-3.1-pro-preview-2026-03-09.md`

## 2. What Both Consultants Agreed On

- Нельзя продолжать текущую family как bank из `15` stage-candidates.
- Следующий шаг должен быть не изобретение новых prompts, а жёсткий pruning текущего банка.
- `baseline` fact-extractor обязан войти в следующий раунд как контрольный кандидат; без этого нельзя честно доказывать, что decomposed extraction лучше общего baseline-extractor.
- Эти stages уже не выглядят worthwhile для немедленного carry-forward в текущем виде:
  - `facts.extract_setlist.v1`
  - `facts.extract_program.v1`
  - `facts.extract_program_shape.v1`
  - `facts.extract_profiles.v1`
  - `facts.extract_concept.v1`
  - `facts.extract_identity.v1`
  - `facts.extract_cluster.v1`
- В bank следующего раунда точно есть смысл нести:
  - `facts.extract_subject.v1`
  - `facts.extract_card.v1`
  - `facts.extract_agenda.v1`
  - `facts.extract_support.v1`
  - `facts.extract_performer.v1`
  - `facts.extract_participation.v1`
- `facts.extract_stage.v1` не стоит выбрасывать сразу, но его надо нести только в tightened form с явным abort-condition.

## 3. Main Disagreement

Главное расхождение только одно: `facts.extract_theme.v1`.

- `Opus`: считает `theme.v1` сильным stage и хочет оставить его в bank.
- `Gemini`: считает `theme.v1` семантически избыточным относительно `subject.v1` и предлагает убрать, чтобы не тащить дубли дальше в merge.

Это не мелкая stylistic disagreement, а центральный structural вопрос. Поэтому `theme.v1` нельзя считать либо чистым win, либо чистым loser без контрольного следующего раунда.

## 4. Agent Synthesis

Мой вывод после broad-run + evidence-pack + двух external reviews:

- `facts.extract` не надо сейчас дробить в subfamily как основной следующий ход.
- Сначала нужно сделать короткий aggressive-prune round на одном плоском shortlist.
- Но в этот shortlist нельзя тащить все `15` stages и нельзя тащить все внешне "неплохие" stages одновременно.

### 4.1. Strong core for next round

Это bank, который уже достаточно поддержан и внутренним run, и консультантами:

- `facts.extract_subject.v1`
- `facts.extract_card.v1`
- `facts.extract_agenda.v1`
- `facts.extract_support.v1`
- `facts.extract_performer.v1`
- `facts.extract_participation.v1`
- `facts.extract_stage.v1` -> только после tightening
- `baseline_fact_extractor` -> обязательно как control

Это даёт `8` кандидатов total, что уже выглядит operationally нормально для следующего short round.

### 4.2. Disputed challenger

- `facts.extract_theme.v1`

Его не стоит включать как безусловный permanent member следующего банка, но и выбрасывать прямо сейчас преждевременно.

Правильный статус: `challenger / tie-break candidate`.

То есть следующий раунд должен отвечать не на вопрос "theme нужен вообще или нет?" абстрактно, а на конкретный вопрос:

`subject + other core stages` лучше, чем `subject + theme + other core stages`, или нет?

### 4.3. Drop now

На текущем evidence их не надо нести дальше в main shortlist:

- `facts.extract_concept.v1`
- `facts.extract_identity.v1`
- `facts.extract_cluster.v1`
- `facts.extract_setlist.v1`
- `facts.extract_program.v1`
- `facts.extract_program_shape.v1`
- `facts.extract_profiles.v1`

Причина не в том, что они "совсем бесполезны", а в том, что на текущем шаге они не доказывают enough orthogonal value относительно будущего merge burden.

## 5. Exact Shortlist Decision

### keep now
- `facts.extract_subject.v1`
- `facts.extract_card.v1`
- `facts.extract_agenda.v1`
- `facts.extract_support.v1`
- `facts.extract_performer.v1`
- `facts.extract_participation.v1`

### keep but tighten
- `facts.extract_stage.v1`

### mandatory control
- `baseline_fact_extractor`

### challenger only
- `facts.extract_theme.v1`

### drop for now
- `facts.extract_concept.v1`
- `facts.extract_identity.v1`
- `facts.extract_cluster.v1`
- `facts.extract_setlist.v1`
- `facts.extract_program.v1`
- `facts.extract_program_shape.v1`
- `facts.extract_profiles.v1`

## 6. What This Means Practically

Следующий раунд теперь должен быть не "consult all 15 again", а такой:

1. baseline extractor
2. `subject`
3. `card`
4. `agenda`
5. `support`
6. `performer`
7. `participation`
8. `stage.tightened`
9. optional `theme` challenger

То есть не `15`, а `8 + 1 disputed challenger`.

Если нужно ужать ещё сильнее и идти в truly minimal bank, то `theme` не включать в regular run, а оставить отдельным A/B pass.

## 7. Final Answer to the Original Question

Да: именно evidence-pack и надо было отправлять консультантам, чтобы получить объективную оценку силы prompts в family. Теперь это сделано.

По текущему evidence и консультациям:
- strongest single prompt: `facts.extract_subject.v1`
- strongest practical bank: `subject, card, agenda, support, performer, participation`
- carry with guard: `stage`
- unresolved challenger: `theme`
- всё остальное сейчас не тянуть дальше в main shortlist
