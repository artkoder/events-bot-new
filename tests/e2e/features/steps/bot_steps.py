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
def step_log_all_buttons(context):
    """Log all visible buttons to console."""
    msg = context.last_response
    buttons = get_all_buttons(msg)
    
    print("\n" + "=" * 50)
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
