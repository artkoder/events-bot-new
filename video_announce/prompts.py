from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).parent / "assets"
_DEFAULT_PROMPT = (
    "Собери сценарий видеоролика по списку событий. Используй живой тон,"
    " привяжи хронометраж, упомяни площадку и время."
)


def selection_response_format(max_items: int = 8) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "VideoAnnounceRanking",
            "schema": {
                "type": "object",
                "properties": {
                    "intro_text": {"type": ["string", "null"], "maxLength": 60},
                    "items": {
                        "type": "array",
                        "maxItems": max_items,
                        "minItems": 0,
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_id": {"type": "integer"},
                                "score": {"type": ["number", "null"]},
                                "reason": {"type": ["string", "null"]},
                                "about": {"type": ["string", "null"]},
                            },
                            "required": [
                                "event_id",
                                "score",
                                "reason",
                                "about",
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
        " выбрать события (только на русском) для короткого ролика."
        " Приоритетно выполни инструкцию оператора: если правила конфликта"
        " — следуй ей. Выбери до 8 лучших событий, которые подходят под инструкцию и критерии."
        " Верни event_id, score (0–10) и причину выбора (reason, до 120 символов)."
        " НЕ выдумывай событий, используй только предоставленные candidates."
        " Если события не подходят, верни пустой список items или меньше 8."
        " Ответ строго JSON."
        "\n\nВАЖНО: АНАЛИЗ ПОДХОДЯЩИХ СОБЫТИЙ:"
        " 1. Внимательно проверяй 'title' И 'poster_ocr_text' на наличие ключевых слов из инструкции."
        " 2. Пример: если instruction='музыка', ищи слова 'оркестр', 'концерт', 'хор', 'dj', 'вечеринка' в title и ocr."
        " 3. Даже если категория события неочевидна (напр. 'Кинохиты'), наличие слова 'Оркестр' в OCR делает его музыкальным."
        " 4. Не игнорируй события, если keyword найден только в OCR тексте."
        "\n\nПРАВИЛА ДЛЯ intro_text (СТРОГО):"
        " 1. Это ЗАГОЛОВОК для видео для зрителей. Должен быть КРУПНЫМ, ЦЕПЛЯЮЩИМ, 'ПРОДАЮЩИМ'."
        " 2. Формат: 'ТЕМА: ДАТЫ'. Тему бери из instruction (например: 'ДЕТЯМ', 'БЕСПЛАТНО', 'ТЕАТР')."
        " 3. Если instruction общей тематики или пуста — придумай тему по составу выборки (например 'ГЛАВНОЕ', 'УИКЕНД')."
        " 4. Даты: строго КАПСОМ, без сокращений (только полные названия месяцев)."
        " 5. Примеры:"
        "    - 'ДЕТСКАЯ АФИША: 25–27 ДЕКАБРЯ'"
        "    - 'НОВОГОДНИЙ ВАЙБ: 31 ДЕКАБРЯ'"
        "    - 'ЛУЧШЕЕ В ГОРОДЕ: 1–8 ЯНВАРЯ'"
        " 6. Длина ≤ 60 символов. Без эмодзи, кавычек и лишних слов ('Афиша на...')."
        "\n\nПРАВИЛА ДЛЯ about (для каждого события):"
        " 1. Генерируется на основе title и search_digest (самое важное) за вычетом ocr_title."
        " 2. Максимум 12 слов. Лаконично, запоминаемо, отличает событие."
        " 3. Если ocr_title не пустой: НЕ дублируй слова из него. Включи ключевые слова из title, которых нет в ocr_title."
        " 4. Если ocr_title пуст или нерелевантен: опирайся на title + 1 полезная фишка из search_digest."
        " 5. Итог должен быть без кавычек и эмодзи."
    )


def finalize_prompt() -> str:
    return (
        "Ты помогаешь составить финальные заголовки и описания для афиши"
        " видеоролика. У каждого события есть title, search_digest (самое важное), description"
        " и OCR данные (ocr_title, poster_text). "
        "Поле about: максимум 12 слов, без эмодзи/кавычек. Должно быть запоминаемым и отличать событие."
        " Включает ключевые слова из title, кроме тех, что уже есть в ocr_title (зритель их и так видит)."
        " Добавляет 1–2 'фишки' из search_digest (или description), чтобы заинтересовать."
        " НЕ повторяет слова из ocr_title. Если ocr_title пустой — ориентируйся на title."
        " Поле final_title: цепляющий заголовок до 12 символов."
        " Ответ строго JSON с полями event_id, final_title, about, description, use_ocr, poster_source."
    )
