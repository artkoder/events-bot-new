Добавил недостающие импорты SQLAlchemy в `main_part2.py` (включая `select`, чтобы убрать NameError в `get_month_data`) и сделал отдельный тест, проверяющий выборку событий и выставок за месяц.

Phase 1 - Discovery: нашёл `get_month_data` в `main_part2.py` и просмотрел использование SQLAlchemy-хелперов без локальных импортов. Interim report: обнаружены отсутствующие импорты `select`, а также `update`, `delete`, `text`, `func`, `or_`, `and_`.

Phase 2 - Implementation: добавил импорты в `main_part2.py` и создал `tests/test_get_month_data.py` с асинхронным тестом на выборку событий/выставок. Interim report: новые импорты и тест на месте.

Phase 3 - Validation: тесты не запускал. Interim report: готово к запуску `pytest tests/test_get_month_data.py`.

Следующие шаги:
1) Запустить `pytest tests/test_get_month_data.py`
2) При необходимости — прогнать полный тестовый набор