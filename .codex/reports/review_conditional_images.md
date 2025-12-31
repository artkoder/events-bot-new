**Findings**
- High: `Event` в тесте создается без обязательных полей (`time`, `location_name`, `source_text`, …), что приводит к ошибке и делает тест неисполняемым. `tests/test_conditional_images.py:8`
- Medium: `test_build_month_page_threshold` фактически пустой (`pass`) и не проверяет порог `show_images`, поэтому логика не покрыта. `tests/test_conditional_images.py:30`, `tests/test_conditional_images.py:55`
- Medium: при обработке `CONTENT_TOO_BIG` повторный вызов `patch_month_page_for_date` теряет `show_images`, из‑за чего после сплита изображения пропадают даже при включенном режиме. `main.py:13161`
- Medium (race/consistency): `show_images` считается по месячному счетчику, но патчится только один день; при переходе через порог или параллельных обновлениях страницы могут стать «смешанными» (часть дней с изображениями, часть без). `main.py:13234`, `main.py:13246`
- Medium: `main_part2.py` опирается на глобалы, определенные в `main.py` (например, `timedelta`, `LOCAL_TZ`, `format_day_pretty`), поэтому импорт `main_part2` напрямую (как в тестах) делает вызовы вроде `get_month_data` нестабильными. `main_part2.py:242`, `main_part2.py:261`
- Low (security): `event_to_nodes` вставляет `photo_urls[0]` в `img src` без валидации схемы; если источник не доверенный, возможны `file:`/`data:`/трекеры. `main_part2.py:80`

**Questions / Assumptions**
- `photo_urls` гарантированно содержат только `http/https` из доверенного источника? Если да, security‑замечание можно снять.
- Логика `show_images` должна быть глобальной для месяца? Если да, при смене порога логичнее принудительно делать полный ребилд, а не точечные патчи.

**Change summary**
- Ревью только, без правок кода.