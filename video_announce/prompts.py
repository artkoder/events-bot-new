from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).parent / "assets"
_DEFAULT_PROMPT = (
    "Собери сценарий видеоролика по списку событий. Используй живой тон,"
    " привяжи хронометраж, упомяни площадку и время."
)

def selection_response_format(candidate_count: int) -> dict:
    max_items = max(1, int(candidate_count or 0))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "VideoAnnounceRanking",
            "schema": {
                "type": "object",
                "properties": {
                    "intro_text": {"type": "string"},
                    "items": {
                        "type": "array",
                        "maxItems": max_items,
                        "minItems": max_items,
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_id": {"type": "integer"},
                                "score": {"type": "number"},
                                "reason": {"type": ["string", "null"]},
                                "selected": {"type": ["boolean", "null"]},
                                "selected_reason": {"type": ["string", "null"]},
                                "about": {"type": ["string", "null"]},
                                "description": {"type": ["string", "null"]},
                                "final_title": {"type": ["string", "null"]},
                            },
                            "required": [
                                "event_id",
                                "score",
                                "reason",
                                "selected",
                                "selected_reason",
                                "about",
                                "description",
                                "final_title",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["intro_text", "items"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

FINAL_TEXT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "VideoAnnounceFinalText",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "integer"},
                            "final_title": {"type": "string"},
                            "about": {"type": "string"},
                            "description": {"type": "string"},
                            "use_ocr": {"type": ["boolean", "null"]},
                            "poster_source": {"type": ["string", "null"]},
                        },
                        "required": [
                            "event_id",
                            "final_title",
                            "about",
                            "description",
                            "use_ocr",
                            "poster_source",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def load_prompt(name: str = "script") -> str:
    """Return prompt text for the requested name, falling back to default."""

    asset = _ASSETS_DIR / f"{name}.txt"
    if asset.exists():
        return asset.read_text(encoding="utf-8").strip()
    return _DEFAULT_PROMPT


def available_prompts() -> list[str]:
    prompts = []
    for path in _ASSETS_DIR.glob("*.txt"):
        prompts.append(path.stem)
    return sorted(set(prompts + ["script"]))


def selection_prompt() -> str:
    return (
        "Ты ассистент видеоредактора. Получишь JSON с событиями и должен"
        " выбрать порядок показа (только на русском) для короткого ролика."
        " Приоритетно выполни инструкцию оператора: если правила конфликта"
        " — следуй ей. Оцени и упомяни КАЖДОЕ событие по одному разу,"
        " выставь score 0–10 и краткую причину. Обязательно включай все"
        " promoted=true в итоговый выбор даже если это превышает лимиты."
        " Отбери 6–8 событий (максимум 8) как selected=true для ролика,"
        " остальные отмечай selected=false с коротким selected_reason."
        " Стремись к разнообразию тематик и форматов, выделяй интерес,"
        " уникальность, свежесть, семейную ценность и пригодность"
        " OCR/Telegraph контекста. Не используй количество постеров как критерий."
        " Добавь intro_text — лаконичное вступление в 1–2 предложения."
        " Для каждого события верни about (до 12 слов, без эмодзи и кавычек)"
        " и одно предложение description с узнаваемым названием +"
        " формат/место/время. Ответ строго JSON со списком items без пояснений."
    )


def finalize_prompt() -> str:
    return (
        "Ты помогаешь составить финальные заголовки и описания для афиши"
        " видеоролика. У каждого события есть базовый title/description,"
        " выдержки из OCR постера и из статьи Telegraph. Сформулируй"
        " цепляющий русский final_title из контекста постера и описания,"
        " уложенный в 12 символов (обрежь, если нужно), не копируя текст"
        " постера дословно. Поле about: до 12 слов, без эмодзи и кавычек,"
        " содержит узнаваемое название события + 1–2 уточнения"
        " (формат/аудитория/место), избегая фраз из OCR. Текст description"
        " — одно предложение о формате, времени или площадке. Ответ строго"
        " JSON с полями event_id, final_title, about, description, use_ocr,"
        " poster_source; без пояснений."
    )
