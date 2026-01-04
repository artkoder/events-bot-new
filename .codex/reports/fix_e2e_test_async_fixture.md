Привел асинхронную фикстуру в `tests/test_dom_iskusstv_e2e.py` к стандартному использованию pytest‑asyncio: добавил импорт `pytest_asyncio` и заменил декоратор `test_db` на `@pytest_asyncio.fixture`, чтобы pytest корректно ожидал async setup/teardown. Это устраняет ошибку работы с асинхронной фикстурой в тесте `test_duplicate_handling` из `tests/test_dom_iskusstv_e2e.py`.

Тесты не запускал.

Следующие шаги:
1) `pytest tests/test_dom_iskusstv_e2e.py -k duplicate_handling`
2) `pytest tests/test_dom_iskusstv_e2e.py`