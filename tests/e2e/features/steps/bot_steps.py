"""
Step definitions for Telegram bot BDD scenarios.

Maps Russian Gherkin steps to HumanUserClient actions.
"""

import re
import logging
from behave import given, when, then

logger = logging.getLogger("e2e.steps")


# =============================================================================
# Helper Functions
# =============================================================================

def run_async(context, awaitable):
    """Run async coroutine in the behave sync context."""
    return context.loop.run_until_complete(awaitable)


def get_all_buttons(message):
    """Extract all button texts from message (inline + reply keyboard)."""
    buttons = []
    
    if message and message.buttons:
        for row in message.buttons:
            for btn in row:
                buttons.append(btn.text)
    
    return buttons


def find_button(message, text):
    """Find button by text (partial match)."""
    if message and message.buttons:
        for row in message.buttons:
            for btn in row:
                if text in btn.text:
                    return btn
    return None


# =============================================================================
# Предыстория (Background)
# =============================================================================

@given("я авторизован в клиенте Telethon")
def step_authorized(context):
    """Verify client is connected and authorized."""
    assert context.client is not None, "Client not initialized"
    assert context.client._connected, "Client not connected"
    logger.info("✓ Клиент авторизован")


@given("я открыл чат с ботом")
def step_open_bot_chat(context):
    """Open chat with target bot, store entity."""
    async def _open():
        entity = await context.client.client.get_entity(context.bot_username)
        context.bot_entity = entity
        logger.info(f"✓ Открыт чат с @{context.bot_username}")
        return entity
    
    run_async(context, _open())


@given("я нахожусь в главном меню")
def step_in_main_menu(context):
    """Ensure we're in main menu (send /start if needed)."""
    if not hasattr(context, "bot_entity"):
        step_open_bot_chat(context)
    
    # Send /start to reset state
    step_send_command(context, "/start")
    logger.info("✓ Находимся в главном меню")


# =============================================================================
# Когда (When) - Actions
# =============================================================================

@when('я отправляю команду "{command}"')
def step_send_command(context, command):
    """Send command to bot using human-like behavior."""
    async def _send():
        response = await context.client.human_send_and_wait(
            context.bot_entity,
            command,
            timeout=30
        )
        context.last_response = response
        logger.info(f"→ Отправлено: {command}")
        if response and response.text:
            preview = response.text[:100].replace('\n', ' ')
            logger.info(f"← Ответ: {preview}...")
        return response
    
    run_async(context, _send())


@when('я отправляю сообщение "{text}"')
def step_send_message(context, text):
    """Send arbitrary text message."""
    async def _send():
        response = await context.client.human_send_and_wait(
            context.bot_entity,
            text,
            timeout=120  # Increased timeout for long operations
        )
        context.last_response = response
        logger.info(f"→ Отправлено сообщение: {text}")
        if response and response.text:
            preview = response.text[:100].replace('\n', ' ')
            logger.info(f"← Ответ: {preview}...")
        return response
    
    run_async(context, _send())


@when('я нажимаю инлайн-кнопку "{btn_text}"')
def step_click_inline_button(context, btn_text):
    """Click inline button by text."""
    async def _click():
        msg = context.last_response
        btn = find_button(msg, btn_text)
        
        if not btn:
            available = get_all_buttons(msg)
            raise AssertionError(
                f"Кнопка '{btn_text}' не найдена. Доступные: {available}"
            )
        
        # Human-like delay before click
        await context.client._gaussian_delay(0.5, 1.5)
        
        # Click the button
        await btn.click()
        logger.info(f"→ Нажата кнопка: {btn_text}")
        
        # Wait for response/edit
        import asyncio
        await asyncio.sleep(2)  # Wait for bot to respond
        
        # Get updated message
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]
            logger.info("← Получен обновлённый ответ")
    
    run_async(context, _click())


# =============================================================================
# Тогда (Then) - Assertions
# =============================================================================

@then('я должен увидеть сообщение, содержащее текст "{text}"')
def step_see_message_with_text(context, text):
    """Assert last response contains text."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    assert msg.text is not None, "Ответ бота пустой"
    
    # Case-insensitive search
    assert text.lower() in msg.text.lower(), (
        f"Текст '{text}' не найден в ответе: {msg.text[:200]}"
    )
    logger.info(f"✓ Найден текст: '{text}'")


@then("я должен увидеть клавиатуру с кнопками:")
def step_see_keyboard_buttons(context):
    """Assert keyboard has expected buttons from table."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    
    actual_buttons = get_all_buttons(msg)
    expected_buttons = [row["name"] for row in context.table]
    
    missing = []
    for expected in expected_buttons:
        found = any(expected in actual for actual in actual_buttons)
        if not found:
            missing.append(expected)
    
    if missing:
        raise AssertionError(
            f"Не найдены кнопки: {missing}. Доступные: {actual_buttons}"
        )
    
    logger.info(f"✓ Все ожидаемые кнопки найдены: {expected_buttons}")


@then("я логирую в консоль список всех кнопок, которые вижу")
@when("я логирую в консоль список всех кнопок, которые вижу")
def step_log_all_buttons(context):
    """Log all visible buttons to console."""
    msg = context.last_response
    buttons = get_all_buttons(msg)
    
    print("\n" + "=" * 50)
    print(f"[REPORT] Текст сообщения: {msg.text if msg else 'None'}")
    print("[REPORT] Видимые кнопки:")
    for i, btn in enumerate(buttons, 1):
        print(f"  {i}. {btn}")
    print("=" * 50 + "\n")
    
    logger.info(f"[REPORT] Всего кнопок: {len(buttons)}")


@then("бот должен прислать сообщение с блоком событий")
def step_see_events_block(context):
    """Assert response contains events block."""
    msg = context.last_response
    assert msg is not None, "Нет ответа от бота"
    assert msg.text is not None, "Ответ бота пустой"
    
    # Check for typical events indicators (dates, times, emojis)
    text = msg.text
    has_events = (
        len(text) > 50 or  # Non-trivial content
        any(char in text for char in ["📅", "🎭", "🎵", "🎪", "📍"]) or
        re.search(r'\d{1,2}[:\.]\d{2}', text)  # Time pattern
    )
    
    assert has_events, f"Не похоже на блок событий: {text[:100]}"
    logger.info("✓ Получен блок событий")


@then('под сообщением должна быть кнопка "{btn_text}"')
def step_should_have_button(context, btn_text):
    """Assert message has specific button."""
    msg = context.last_response
    btn = find_button(msg, btn_text)
    
    if not btn:
        available = get_all_buttons(msg)
        raise AssertionError(
            f"Кнопка '{btn_text}' не найдена. Доступные: {available}"
        )
    
    logger.info(f"✓ Найдена кнопка: '{btn_text}'")


@then("я жду обновления сообщения")
def step_wait_for_update(context):
    """Wait for message to be edited/updated."""
    import asyncio
    
    async def _wait():
        await asyncio.sleep(3)  # Give bot time to update
        
        # Refresh last message
        messages = await context.client.client.get_messages(
            context.bot_entity, limit=1
        )
        if messages:
            context.last_response = messages[0]
    
    run_async(context, _wait())
    logger.info("✓ Дождались обновления")


@when('я жду сообщения с текстом "{text}"')
@then('я жду сообщения с текстом "{text}"')
def step_wait_for_message_text(context, text):
    """Wait for a new message containing specific text."""
    async def _wait():
        import asyncio
        # Try for 5 seconds
        for _ in range(10):
            messages = await context.client.client.get_messages(
                context.bot_entity, limit=5
            )
            for msg in messages:
                if msg.text and text.lower() in msg.text.lower():
                    context.last_response = msg
                    logger.info(f"✓ Найдено ожидаемое сообщение: '{text}'")
                    return
            await asyncio.sleep(0.5)
        
        raise AssertionError(f"Сообщение с текстом '{text}' не получено за 5 секунд. Последние: {[m.text for m in messages]}")

    run_async(context, _wait())


@then("я пишу в лог количество отображенных событий")
def step_log_events_count(context):
    """Log estimated number of events in the message."""
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    
    # Count events by looking for patterns (dates, times, or bullets)
    date_pattern = r'\d{1,2}\s+[а-яА-Я]+(?:\s+\d{4})?'
    time_pattern = r'\d{1,2}[:\.]\d{2}'
    
    dates = len(re.findall(date_pattern, text))
    times = len(re.findall(time_pattern, text))
    
    # Rough estimate: each event typically has a date or time
    estimated_events = max(dates, times, 1)
    
    print("\n" + "=" * 50)
    print(f"[REPORT] Примерное количество событий: {estimated_events}")
    print(f"[REPORT] Найдено дат: {dates}, времён: {times}")
    print(f"[REPORT] Длина текста: {len(text)} символов")
    print("=" * 50 + "\n")
    
    logger.info(f"[REPORT] Событий: ~{estimated_events}")


@then("я логирую полный текст сообщения")
def step_log_full_message(context):
    """Log the full text of the last response."""
    msg = context.last_response
    text = msg.text if msg and msg.text else "[No text]"
    
    print("\n" + "=" * 50)
    print("[REPORT] Полный текст ответа:")
    print(text)
    print("=" * 50 + "\n")
    
    logger.info(f"[REPORT] Текст сообщения ({len(text)} chars)")


@then("я должен найти в ответе действующую ссылку на телеграф")
def step_check_telegraph_link(context):
    """Assert response contains valid and accessible Telegraph links."""
    import aiohttp
    
    msg = context.last_response
    text = msg.text if msg and msg.text else ""
    
    # Regex for Telegraph links
    link_pattern = r"https://telegra\.ph/[a-zA-Z0-9_-]+"
    links = re.findall(link_pattern, text)
    
    assert len(links) > 0, f"Не найдено ни одной ссылки на telegra.ph в тексте:\n{text}"
    
    print("\n" + "=" * 50)
    print(f"[REPORT] Найдены ссылки Telegraph ({len(links)}):")
    for link in links:
        print(f"  - {link}")
    print("=" * 50 + "\n")
    
    # Verify each link is accessible via HTTP
    async def _verify():
        async with aiohttp.ClientSession() as session:
            for link in links:
                try:
                    async with session.head(link, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            raise AssertionError(f"Telegraph ссылка {link} вернула статус {resp.status}")
                        logger.info(f"✓ Ссылка работает: {link}")
                except Exception as e:
                    raise AssertionError(f"Не удалось проверить ссылку {link}: {e}")
    
    run_async(context, _verify())
    context.telegraph_links = links
    logger.info(f"✓ Все {len(links)} Telegraph ссылок валидны")


@then('каждая Telegraph страница должна содержать "{required_text}"')
def step_verify_telegraph_content(context, required_text):
    """Verify each Telegraph page contains required content."""
    import aiohttp
    
    links = getattr(context, 'telegraph_links', [])
    if not links:
        raise AssertionError("Нет сохранённых Telegraph ссылок для проверки")
    
    required_items = [item.strip() for item in required_text.split(",")]
    
    async def _verify_content():
        async with aiohttp.ClientSession() as session:
            failed_pages = []
            
            for link in links:
                try:
                    async with session.get(link, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            failed_pages.append(f"{link}: HTTP {resp.status}")
                            continue
                        
                        html = await resp.text()
                        
                        missing = []
                        for item in required_items:
                            if item.lower() not in html.lower():
                                missing.append(item)
                        
                        if missing:
                            failed_pages.append(f"{link}: отсутствует [{', '.join(missing)}]")
                        else:
                            logger.info(f"✓ Страница {link} содержит все элементы: {required_items}")
                
                except Exception as e:
                    failed_pages.append(f"{link}: ошибка {e}")
            
            if failed_pages:
                print("\n" + "=" * 60)
                print("[ERROR] Проверка контента Telegraph страниц:")
                for fail in failed_pages:
                    print(f"  ✗ {fail}")
                print("=" * 60 + "\n")
                raise AssertionError(f"Не все страницы содержат требуемый контент: {failed_pages}")
    
    run_async(context, _verify_content())
    logger.info(f"✓ Все {len(links)} страниц содержат: {required_items}")


@then("я жду медиа-сообщения")
def step_check_media_message(context):
    """Wait for a message with media."""
    import asyncio
    async def _wait():
        for i in range(10): # 5 seconds
            messages = await context.client.client.get_messages(
                 context.bot_entity, limit=5
            )
            for msg in messages:
                if msg.media:
                    context.last_response = msg
                    logger.info("✓ Медиа-сообщение получено")
                    return
            await asyncio.sleep(0.5)
        raise AssertionError("Медиа-сообщение не получено")
    run_async(context, _wait())

@then('под сообщением должны быть кнопки: "{buttons}"')
def step_check_inline_buttons_custom(context, buttons):
    """Verify specific buttons are present (partial match)."""
    expected = [b.strip() for b in buttons.split(",")]
    msg = context.last_response
    visible = get_all_buttons(msg)
    
    missing = []
    for exp in expected:
        found = False
        for v in visible:
            if exp.strip('"').strip("'") in v:
                found = True
                break
        if not found:
            missing.append(exp)
    
    if missing:
        print(f"[ERROR] Expected: {expected}")
        print(f"[ERROR] Visible: {visible}")
        raise AssertionError(f"Не найдены кнопки: {missing}")
    logger.info(f"✓ Найдены все кнопки: {expected}")


@then('я жду долгой операции с текстом "{text}"')
def step_wait_long_operation(context, text):
    """Wait for a long operation (up to 5 minutes) for message containing text."""
    async def _wait():
        import asyncio
        # Try for 300 seconds (5 minutes - Kaggle notebook can take a while)
        for i in range(600):
            messages = await context.client.client.get_messages(
                context.bot_entity, limit=10
            )
            for msg in messages:
                if msg.text and text.lower() in msg.text.lower():
                    context.last_response = msg
                    logger.info(f"✓ Найден результат долгой операции: '{text}' (за {i*0.5:.1f}с)")
                    return
            await asyncio.sleep(0.5)
        
        last_texts = [m.text[:100] if m.text else "(no text)" for m in messages[:3]]
        raise AssertionError(f"Сообщение с текстом '{text}' не получено за 5 минут. Последние: {last_texts}")

    run_async(context, _wait())


