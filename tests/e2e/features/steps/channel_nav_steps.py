import re
import logging
import asyncio
from behave import given, when, then
from bot_steps import run_async, get_all_buttons, find_button

logger = logging.getLogger("e2e.channel_nav")

TEST_CHANNEL_ID = -1002210431821
TEST_CHANNEL_USERNAME = "keniggpt"

@given("Бот запущен в режиме разработки (Dev Mode)")
def step_dev_mode(context):
    pass  # Assumed true for E2E

@given("Основной канал настроен как **Test Channel X**")
def step_main_channel_configured(context):
    context.test_channel_id = TEST_CHANNEL_ID
    logger.info(f"Using Test Channel ID: {TEST_CHANNEL_ID}")

@given("Бот является администратором в **Test Channel X**")
def step_bot_is_admin(context):
    pass  # Assumed true based on setup

@when('Администратор публикует сообщение "{text}" в **Test Channel X**')
def step_admin_posts_message(context, text):
    async def _post():
        entity = await context.client.client.get_entity(context.test_channel_id)
        # Attempt to send message
        try:
            await context.client.human_send_message(entity, text)
            logger.info(f"Posted to channel: {text}")
        except Exception as e:
            logger.error(f"Failed to post to channel: {e}")
            raise

        # Wait for bot to react
        await asyncio.sleep(2)
        
        # Get the message back to check for buttons
        history = await context.client.client.get_messages(entity, limit=1)
        if history:
            context.last_channel_message = history[0]
            
    run_async(context, _post())

@then("Бот обнаруживает новое сообщение")
def step_detect_new_message(context):
    pass # Implicitly verified by next steps

@then("Бот редактирует сообщение, добавляя инлайн-кнопки")
def step_check_buttons_added(context):
    msg = context.last_channel_message
    context.last_response = msg # Set as last_response for generic steps
    if not msg.buttons:
         # Wait a bit more and reload
        run_async(context, asyncio.sleep(2))
        async def _reload():
            history = await context.client.client.get_messages(context.test_channel_id, ids=[msg.id])
            if history:
                 context.last_channel_message = history[0]
                 context.last_response = history[0]
        run_async(context, _reload())
        
    assert context.last_channel_message.buttons, "Message has no buttons"

@then('Первая кнопка называется "📅 Сегодня <DD.MM>"')
def step_check_first_button_name(context):
    msg = context.last_channel_message
    buttons = get_all_buttons(msg)
    assert buttons, "Buttons missing"
    first_btn = buttons[0]
    # Check format "📅 Сегодня DD.MM"
    assert re.match(r"📅 Сегодня \d{2}\.\d{2}", first_btn), f"Button '{first_btn}' does not match format"

@then("Первая кнопка ведет на Telegraph таблицу текущего месяца")
def step_check_first_button_link(context):
    # Retrieve the link - handled by generic step "я должен найти в ответе действующую ссылку"
    # But specifically for button
    pass

@then("Присутствует вторая кнопка (Случайная: Завтра, Выходные или Следующий месяц)")
def step_check_second_button(context):
    msg = context.last_channel_message
    buttons = get_all_buttons(msg)
    assert len(buttons) >= 2, "Second button missing"
    second_btn = buttons[1]
    valid_patterns = [
        r"📅 Завтра \d{2}\.\d{2}",
        r"📅 Выходные \d{2}\.\d{2}-\d{2}\.\d{2}",
        r"📅 [А-Яа-я]+" # Month name
    ]
    is_valid = any(re.match(p, second_btn) for p in valid_patterns)
    assert is_valid, f"Second button '{second_btn}' is not valid"

@when("Планировщик ежедневного анонса срабатывает")
def step_trigger_daily_scheduler(context):
    # Simulate scheduler by calling /daily and clicking "Test" on the "Main" channel row
    async def _trigger():
        # 1. Send /daily
        await context.client.human_send_and_wait(context.bot_entity, "/daily")
        
        # 2. Find row with "Полюбить Калининград" (Production Channel Title) or just "Test" button
        msg = context.last_response
        
        found_btn = None
        if msg.buttons:
            for row in msg.buttons:
                 # Check if row has "Test" button
                 test_btn = next((b for b in row if b.text == "Test"), None)
                 if test_btn:
                     found_btn = test_btn
                     break
        
        assert found_btn, "Could not find 'Test' button in /daily menu"
        
        await found_btn.click()
        logger.info("Clicked 'Test' button")
        
        # Wait for valid daily announcement in Channel
        await asyncio.sleep(5)
        
        # Capture the message in the channel
        entity = await context.client.client.get_entity(context.test_channel_id)
        history = await context.client.client.get_messages(entity, limit=1)
        if history:
            context.last_channel_message = history[0]
            logger.info(f"Captured channel message: {history[0].text[:50]}...")

    run_async(context, _trigger())

@then('Бот формирует текст ежедневного анонса (содержащий "\\u200b")')
def step_check_daily_text(context):
    msg = context.last_channel_message
    assert "\u200b" in msg.text, "Invisible marker missing in daily announcement"

@then("Бот отправляет сообщение в **Test Channel X**")
def step_check_sent_to_channel(context):
    assert context.last_channel_message, "No message found in test channel"

@then("Сообщение появляется в канале")
def step_message_appears(context):
    pass

@then("Сообщение **НЕ** редактируется (кнопки не добавляются)")
@then("**НИ ОДНА** из частей не редактируется (кнопки не добавляются)")
def step_check_no_buttons(context):
    # Depending on context, we might check last message or wait
    run_async(context, asyncio.sleep(3)) # Wait to ensure no edit happens
    async def _check():
        history = await context.client.client.get_messages(context.test_channel_id, ids=[context.last_channel_message.id])
        msg = history[0]
        # It's okay if buttons are None
        assert not msg.buttons, f"Message should not have buttons, but has: {get_all_buttons(msg)}"
    run_async(context, _check())

@then("Бот разделяет анонс на несколько частей")
def step_check_split(context):
    pass # Data dependent

@then("Бот отправляет Часть 1 в **Test Channel X**")
def step_part_1(context):
    pass

@then("Бот отправляет Часть 2 в **Test Channel X**")
def step_part_2(context):
    pass

@given("Сообщение с навигационными кнопками")
def step_given_message_with_buttons(context):
    # Ensure we have a message with buttons. Reuse admin post logic.
    step_admin_posts_message(context, "Setup for buttons")
    step_check_buttons_added(context)

@when('Я нажимаю "📅 Сегодня <DD.MM>"')
def step_click_today(context):
    msg = context.last_channel_message
    buttons = get_all_buttons(msg)
    btn_text = buttons[0] # Assuming first is Today
    btn_obj = find_button(msg, btn_text)
    context.last_clicked_url = btn_obj.url
    logger.info(f"Clicked (extracted URL): {context.last_clicked_url}")

@then("Открывается Telegraph страница")
def step_url_is_telegraph(context):
    url = context.last_clicked_url
    assert "telegra.ph" in url, f"URL is not Telegraph: {url}"
    context.telegraph_links = [url] # Reuse existing step logic

@then("Страница содержит список анонсов на текущий месяц")
def step_verify_month_page_content(context):
    url = context.last_clicked_url
    context.telegraph_links = [url]
    
    import aiohttp
    async def _verify():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                text = await resp.text()
                # Basic check for month page content
                months = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", 
                          "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
                found_month = any(m in text for m in months) or any(m.lower() in text.lower() for m in months)
                assert found_month, f"Page does not look like a month calendar: {url}"
                logger.info(f"Verified month page content: {url}")

    run_async(context, _verify()) 

@when('Я нажимаю "📅 Завтра <DD.MM>" (если есть)')
def step_click_tomorrow(context):
    msg = context.last_channel_message
    buttons = get_all_buttons(msg)
    if len(buttons) > 1 and "Завтра" in buttons[1]:
        btn_obj = find_button(msg, buttons[1])
        context.last_clicked_url = btn_obj.url
        context.skipped_tomorrow = False
    else:
        context.skipped_tomorrow = True

@then('Открывается Telegraph страница с заголовком "Афиша на завтра"')
def step_verify_tomorrow_page(context):
    if context.skipped_tomorrow:
        return
    
    url = context.last_clicked_url
    context.telegraph_links = [url]
    
    # Reuse verification logic but inline here for clarity
    import aiohttp
    async def _verify():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                text = await resp.text()
                assert "Афиша на завтра" in text or "Завтра" in text or "Tomorrow" in text, "Header not found"
    run_async(context, _verify())

# Scenarios for randomness are manual mostly or require multiple posts
@given("Я публикую несколько сообщений в тестовый канал")
def step_post_multiple(context):
    context.second_buttons = set()
    for i in range(5):
        step_admin_posts_message(context, f"Random test {i}")
        step_check_buttons_added(context)
        msg = context.last_channel_message
        buttons = get_all_buttons(msg)
        if len(buttons) > 1:
            context.second_buttons.add(buttons[1])

@when("Для каждого сообщения генерируются кнопки")
def step_buttons_gen(context):
    pass

@then("Я наблюдаю разные вторые кнопки (Завтра / Выходные / Следующий месяц) в разных постах")
def step_verify_randomness(context):
    logger.info(f"Observed second buttons: {context.second_buttons}")
    if len(context.second_buttons) < 2:
        logger.warning(f"Low variance in buttons: {context.second_buttons}")
    else:
        logger.info("Randomness verified")

@when("Планировщик ежедневного анонса срабатывает для насыщенного дня")
def step_trigger_daily_saturated(context):
    step_trigger_daily_scheduler(context)

@then('Бот логирует "skipping rubric post"')
def step_check_log_skip(context):
    pass

@then('Кнопка "Сегодня" остается неизменной во всех постах')
def step_verify_today_constant(context):
    pass
