# Smart Update: Opus Dry-Run Runbook

Цель: дать воспроизводимый процесс, чтобы внешняя консультация (Opus) опиралась на реальные данные и проверяемые прогоны.

Что получаем на выходе:
- единый bundle с кейсами + источниками из БД + совпадениями из `telegram_results.json`;
- dry-run отчёты по текущей логике;
- longrun отчёты по quality-first модели (с обязательным LLM/Gemma);
- материалы, которые можно передать Opus для анализа и отчёта.

---

## 1. Пререквизиты

1. Свежий snapshot прод БД (SQLite):
- пример: `artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite`
- см. `docs/operations/prod-data.md`

2. Наличие локальных результатов Telegram Monitoring:
- файлы вида `/tmp/tg-monitor-*/telegram_results.json`
- можно указать свои пути явно в командах ниже

3. Для LLM-прогонов (Gemma) должны быть доступны ENV для текущего gateway.

---

## 2. Быстрые проверки исходных данных

Проверить, что snapshot и casepack доступны:

```bash
ls -l artifacts/db/db_prod_snapshot_*.sqlite
ls -l artifacts/codex/opus_session2_casepack_latest.json
```

Проверить наличие `telegram_results.json`:

```bash
find /tmp -maxdepth 3 -type f -name telegram_results.json | head -n 20
```

---

## 3. Как вытащить тексты постов (Telegram/VK)

### 3.1. Из БД по `event_id`

```bash
sqlite3 -readonly artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite "
SELECT
  es.event_id,
  es.id AS source_id,
  es.source_type,
  es.source_url,
  es.source_chat_username,
  es.source_message_id,
  substr(replace(coalesce(es.source_text,''), char(10), ' '), 1, 400) AS source_text_preview
FROM event_source es
WHERE es.event_id IN (2793, 2810)
ORDER BY es.event_id, es.id;
"
```

### 3.2. Из БД по конкретному URL поста

```bash
sqlite3 -readonly artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite "
SELECT
  es.id AS source_id,
  es.event_id,
  es.source_type,
  es.source_url,
  substr(es.source_text, 1, 800) AS source_text
FROM event_source es
WHERE es.source_url = 'https://t.me/signalkld/9891'
ORDER BY es.id;
"
```

### 3.3. Из `telegram_results.json` по URL поста

```bash
python - <<'PY'
import json, re
from pathlib import Path

results_path = Path('/tmp/tg-monitor-56de0b4f08554992bd040e83f56a02b8/telegram_results.json')
post_url = 'https://t.me/signalkld/9891'

def parse_tg(url: str):
    m = re.match(r'https?://t\\.me/([^/]+)/([0-9]+)', url.strip())
    return (m.group(1).lower(), int(m.group(2))) if m else (None, None)

u, mid = parse_tg(post_url)
data = json.loads(results_path.read_text(encoding='utf-8'))
for msg in data.get('messages', []):
    if (msg.get('source_link') or '').split('?',1)[0].rstrip('/') == post_url.rstrip('/'):
        print('FOUND by source_link')
        print('text:', (msg.get('text') or '')[:1200])
        break
    if str(msg.get('source_username') or '').lower() == u and int(msg.get('message_id') or 0) == mid:
        print('FOUND by username/message_id')
        print('text:', (msg.get('text') or '')[:1200])
        break
else:
    print('NOT FOUND')
PY
```

---

## 4. Сбор удобной упаковки для Opus

Новый helper-скрипт:
- `scripts/inspect/prepare_opus_consultation_bundle.py`

Он собирает:
- целевые кейсы из casepack;
- все связанные `event/event_source/event_source_fact/eventposter` из snapshot;
- совпавшие сообщения из `telegram_results.json` по `source_url`;
- единый JSON + короткий MD summary в `artifacts/codex/`.

### 4.1. Автопоиск `telegram_results.json` (в `/tmp`)

```bash
python scripts/inspect/prepare_opus_consultation_bundle.py \
  --db artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --casepack artifacts/codex/opus_session2_casepack_latest.json
```

### 4.2. Явно указать конкретные файлы `telegram_results.json`

```bash
python scripts/inspect/prepare_opus_consultation_bundle.py \
  --db artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --casepack artifacts/codex/opus_session2_casepack_latest.json \
  --tg-results /tmp/tg-monitor-56de0b4f08554992bd040e83f56a02b8/telegram_results.json \
  --tg-results /tmp/tg-monitor-0a5648cc73f54d2aaddbc0502ba51286/telegram_results.json
```

После запуска скрипт печатает пути:
- `artifacts/codex/opus_consultation_bundle_<timestamp>.json`
- `artifacts/codex/opus_consultation_bundle_<timestamp>.md`

---

## 5. Dry-run прогоны для Opus

### 5.1. Deterministic quality dry-run (без LLM)

```bash
python artifacts/codex/quality_first_dedup_dryrun.py \
  --db artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --json-out artifacts/codex/quality_first_dedup_dryrun_latest.json \
  --md-out artifacts/codex/quality_first_dedup_dryrun_latest.md
```

### 5.2. Prompt профилирование Gemma на наборе кейсов

```bash
python artifacts/codex/gemma_match_prompt_eval.py \
  --db artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --json-out artifacts/codex/gemma_match_prompt_eval_latest.json
```

### 5.3. Longrun quality-first (LLM обязателен)

```bash
python artifacts/codex/smart_update_identity_longrun.py \
  --db artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --json-out artifacts/codex/smart_update_identity_longrun_latest.json \
  --md-out artifacts/codex/smart_update_identity_longrun_latest.md \
  --limit-mined 24 \
  --compare-current-prompt \
  --llm-delay-sec 8.0
```

`--llm-delay-sec` нужен, чтобы мягче проходить по TPM при длительных прогонах.

### 5.4. Stage 04 consensus dry-run (узкие deterministic гипотезы)

Когда Stage 03 baseline уже собран и нужен следующий узкий шаг без роста false merge, используй consensus dry-run:

```bash
python artifacts/codex/stage_04_consensus_dryrun.py \
  --db artifacts/db/db_prod_snapshot_2026-03-06_062238.retry3.sqlite \
  --source-json artifacts/codex/smart_update_stage_03_production_ready_dryrun_latest.json \
  --json-out artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.json \
  --md-out artifacts/codex/smart_update_stage_04_consensus_dryrun_latest.md
```

Что делает этот прогон:
- берёт Stage 03 pairwise baseline как исходный ground layer;
- прогоняет несколько узких rule bundles поверх него;
- сравнивает каждый bundle по всему текущему casepack;
- фиксирует только те изменения, которые не открывают false merge / false different на текущем пакете.

Используй этот шаг перед новым Opus-раундом, если нужно сузить disagreement surface и показать не “общие идеи”, а уже проверенные local candidates.

---

## 6. Что передавать Opus на следующий раунд

Не держи здесь вручную “вечный” список файлов для всех этапов.
Для актуального раунда используй stage-specific manifest:

- Stage 03: `artifacts/codex/opus_stage_03_send_manifest_latest.md`
- Stage 04: `artifacts/codex/opus_stage_04_send_manifest_latest.md`

Базовый принцип не меняется:
- Opus должен видеть исходные тексты/источники;
- Opus должен видеть snapshot-derived bundle;
- Opus должен видеть dry-run/longrun результаты;
- Opus должен получать уже narrowed dispute set, а не только общий пересказ.

---

## 7. Цикл итерации после рекомендаций Opus

1. Получить от Opus конкретные правки (prompt text + policy rules).
2. Внести изменения в отдельной ветке.
3. Повторить блок из раздела 5 теми же командами.
4. Сравнить `*_latest.json` до/после:
- false merge;
- must-not-merge failures;
- safe-merge misses;
- долю `gray`;
- объём LLM-вызовов и стабильность по TPM.
5. Только после этого переходить к внедрению в основной runtime.

---

## 8. Что именно смотреть в expanded sample

Файл:
- `artifacts/codex/opus_session2_sample_refresh_results_latest.md`

Он нужен как короткий индекс перед ручным анализом bundle:
- где current preview уже уверен и попадает в `merge` или `different`;
- где до сих пор остаётся `gray`;
- какие новые кейсы давят на policy больше всего:
  - `same-source triple duplicate`;
  - recurring theatre runs;
  - same-day double shows;
  - generic ticket false-friends;
  - same-slot false-merge controls.
