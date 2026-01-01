## ФАЗА 1: Поиск всех мест использования поля description
**Создание Event + description (persisted)**
- `main.py:10960-10962` `base_event = Event(... description=data.get("short_description", ""))`
- `main.py:11021-11028` `copy_e = Event(**base_event.model_dump(...))` (description копируется из base_event)
- `main.py:8271-8274` `event = Event(... description="")` (festdays)
- `main.py:11460-11463` `event = Event(... description="")` (/addevent_raw)
- `vk_intake.py:1823-1826` `event = Event(... description=(draft.description or ""))`
- `source_parsing/handlers.py:475-483` `final_description = draft.description or full_description` + `event = Event(... description=final_description)`

**Создание Event + description (in-memory clone)**
- `main_part2.py:53-56` `clone_event_with_date -> Event(**payload)` (description копируется из payload)
- `special_pages.py:48-53` `clone_event_with_date -> Event(**payload)` (description копируется из payload)

**Обновление description**
- `main.py:9660-9679` `_copy_fields` копирует `description` при апдейте существующего события
- `main.py:9921-9931` `ev.description = desc or ev.description` (LLM duplicate check)
- `source_parsing/handlers.py:250-252` `event.description = theatre_event.description` (сырой парсер)
- `main_part2.py:7905-7907` `setattr(event, field, value)` при `field == "description"` (ручное редактирование)
- `main_part2.py:13197-13205` `obj.description = desc` из `parse_event_via_4o` (LLM)

## ФАЗА 2: Анализ flow добавления событий
- `/start -> Добавить событие`: `main_part2.py:7985` → `main_part2.py:12430` → `main.py:11233` → `main.py:10598` → `parse_event_via_4o` (LLM). `description` берётся из `short_description` (`main.py:10960-10962`). LLM используется.
- `/parse` (театральные сайты): `source_parsing/commands.py:72` → `source_parsing/handlers.py:690` → `source_parsing/handlers.py:350+` → `vk_intake.py:1072` → `parse_event_via_4o` (LLM) → `draft.description`. Но далее `final_description = draft.description or full_description` + доп. строка сцены (`source_parsing/handlers.py:475-483`), т.е. LLM может быть переопределён.
- VK intake: `vk_intake.py:1072` → `parse_event_via_4o` (LLM) → `draft.description` → `event.description=(draft.description or "")` (`vk_intake.py:1823-1826`). LLM используется.

## ФАЗА 3: Выявление проблемных мест
**Проблемные места (по убыванию критичности)**
- `source_parsing/handlers.py:475-483` — `final_description = draft.description or full_description` + добавление `Сцена:`. При пустом `draft.description` в `description` уходит полный текст парсера (много предложений), а при наличии — всё равно может стать >1 предложения из‑за аппенда сцены.
- `source_parsing/handlers.py:250-252` — `event.description = theatre_event.description` заполняет из сырого парсера без LLM, что легко даёт длинный текст.
- `main.py:10960-10962`, `vk_intake.py:1823-1826`, `main.py:9921-9931`, `main_part2.py:13197-13205` — LLM‑выход `short_description` никак не валидируется/не нормализуется; если модель вернёт длинный текст (или весь `source_text`), он попадёт в `description`.
- `main_part2.py:7905-7907` — ручное редактирование позволяет задать любое количество предложений, без контроля длины.
- `main.py:8271-8274`, `main.py:11460-11463` — `description=""` при создании (festdays, /addevent_raw); без дальнейшей генерации поле остаётся пустым.

**Условия, при которых short_description пустой или длинный**
- Пустой: `short_description` отсутствует/пустой в ответе LLM (`main.py:10960-10962`, `vk_intake.py:1823-1826`), а fallback не предусмотрен.
- Пустой: события через festdays или `/addevent_raw` создаются с `description=""` (`main.py:8271-8274`, `main.py:11460-11463`).
- Длинный: /parse path подставляет `full_description` и дописывает сцену (`source_parsing/handlers.py:475-483`).
- Длинный: ручное редактирование или LLM‑ответ без нормализации (`main_part2.py:7905-7907`, `main.py:10960-10962`).

**Вопросы/предположения**
- `update_event_description` в `main_part2.py` нигде не вызывается в prod‑потоке; если это так, пустые `description` не перегенерируются автоматически.
- Нужно ли сохранять «Сцена/Возраст» в `description`, или это должно жить в `source_text`/отдельных полях?

## ФАЗА 4: Рекомендации
- Ввести единый нормализатор `normalize_short_description()` (1 предложение, без переносов, лимит по символам) и применять его в `main.py` (add_events_from_text + duplicate check), `vk_intake.py` (после LLM), `source_parsing/handlers.py` (перед записью), `main_part2.py` (update_event_description и ручное редактирование).
- В `source_parsing/handlers.py` заменить `final_description = draft.description or full_description` на: если `draft.description` пуст, вызвать LLM‑суммаризацию (или взять первое предложение) и не добавлять `Сцена:` в `description` — вынести это в `source_text` или отдельное поле.
- В `source_parsing/handlers.py` при `event.description` пустом заменить прямую запись `theatre_event.description` на генерацию short_description через LLM (аналогично `parse_event_via_4o`).
- Для festdays и `/addevent_raw` (`main.py`) добавить генерацию short_description (LLM по `source_text`/шаблонное 1‑предложение) либо запуск отдельного job на суммаризацию.
- Для ручного редактирования (`main_part2.py`) добавить валидацию/автосжатие до одного предложения, чтобы не нарушать `docs/PROMPTS.md` (short_description = one‑sentence summary).

Если хочешь, могу подготовить патч с общей функцией нормализации и точечными правками в указанных файлах.