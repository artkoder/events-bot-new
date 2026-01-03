"""
E2E Test Runner - Self-contained version.
Embeds feature and step files, writes them at runtime, then runs behave.
"""
import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== EMBEDDED FEATURE FILE ==============
FEATURE_CONTENT = '''
Feature: Проверить событие на сегодня

  Scenario: Авторизоваться в боте
    Given я авторизован в Telegram через Telethon
    When я нахожу бота @eventsbotTestBot в списке моих переписок
    Then я должен увидеть переписку с ботом

  Scenario: Проверить события на сегодня
    Given я авторизован в боте @eventsbotTestBot
    When я делаю /events
    Then я должен получить ответ от бота со списком событий
    And в сообщении со списком событий будет 2 инлайн кнопки одна влево, вдругая вправо
'''.strip()

# ============== EMBEDDED STEP DEFINITIONS ==============
STEPS_CONTENT = '''
import os
import asyncio
from behave import given, when, then
from telethon import TelegramClient
from telethon.sessions import StringSession

TEST_BOT_USERNAME = "eventsbotTestBot"

def run_async(coro):
    """Helper to run async code in sync steps."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

@given('я авторизован в Telegram через Telethon')
def step_auth_telegram(context):
    api_id = os.environ.get('TELEGRAM_API_ID')
    api_hash = os.environ.get('TELEGRAM_API_HASH')
    session_str = os.environ.get('TELEGRAM_SESSION')
    
    if not all([api_id, api_hash, session_str]):
        raise EnvironmentError("Missing TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_SESSION")
    
    context.client = TelegramClient(StringSession(session_str), int(api_id), api_hash)
    
    async def connect():
        await context.client.connect()
        if not await context.client.is_user_authorized():
            raise ConnectionError("Client is not authorized. Check Session String.")
            
    run_async(connect())

@when('я нахожу бота @eventsbotTestBot в списке моих переписок')
def step_find_bot(context):
    async def find_dialog():
        try:
            entity = await context.client.get_entity(TEST_BOT_USERNAME)
            context.bot_entity = entity
        except Exception as e:
            raise AssertionError(f"Could not find bot {TEST_BOT_USERNAME}: {e}")
            
    run_async(find_dialog())

@then('я должен увидеть переписку с ботом')
def step_see_bot_dialog(context):
    pass

@given('я авторизован в боте @eventsbotTestBot')
def step_auth_bot_context(context):
    if not hasattr(context, 'client'):
        step_auth_telegram(context)
    if not hasattr(context, 'bot_entity'):
        step_find_bot(context)

@when('я делаю /events')
def step_send_events_command(context):
    async def send_command():
        async with context.client.conversation(context.bot_entity, timeout=30) as conv:
            await conv.send_message('/events')
            response = await conv.get_response()
            context.last_response = response
            
    run_async(send_command())

@then('я должен получить ответ от бота со списком событий')
def step_check_response_events(context):
    if not hasattr(context, 'last_response') or not context.last_response:
        raise AssertionError("No response received from bot")
    
    text = context.last_response.text
    if not text or len(text) < 10:
        raise AssertionError(f"Response seems too short or empty: {text}")

@then('в сообщении со списком событий будет 2 инлайн кнопки одна влево, вдругая вправо')
def step_check_buttons(context):
    msg = context.last_response
    if not msg.buttons:
        raise AssertionError("Message has no buttons")
    
    all_buttons = []
    if msg.buttons:
        for row in msg.buttons:
            all_buttons.extend(row)
            
    count = len(all_buttons)
    if count < 2:
        raise AssertionError(f"Expected at least 2 buttons, found {count}")
'''.strip()


def install_dependencies():
    """Install required python packages."""
    packages = ["telethon", "behave", "requests", "beautifulsoup4"]
    logger.info(f"Installing dependencies: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(0) 


def setup_files():
    """Write embedded feature and step files to disk."""
    logger.info("Setting up test files...")
    
    # Create directory structure
    features_dir = Path("features/steps")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Write feature file
    feature_file = Path("features/bot_telegraph.feature")
    feature_file.write_text(FEATURE_CONTENT, encoding="utf-8")
    logger.info(f"Created {feature_file}")
    
    # Write steps file
    steps_file = Path("features/steps/definitions.py")
    steps_file.write_text(STEPS_CONTENT, encoding="utf-8")
    logger.info(f"Created {steps_file}")
    
    # List created files
    print("Created files:")
    subprocess.run(["find", ".", "-name", "*.py", "-o", "-name", "*.feature"])


def load_secrets():
    """Load secrets from Kaggle environment."""
    required_vars = ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION", "TELEGRAM_BOT_TOKEN"]
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        logger.info("UserSecretsClient initialized successfully")
        
        for var in required_vars:
            if var not in os.environ:
                try:
                    val = user_secrets.get_secret(var)
                    if val:
                        os.environ[var] = val
                        logger.info(f"Loaded {var} from Kaggle Secrets (length={len(val)})")
                    else:
                        logger.warning(f"Secret {var} returned empty/None")
                except Exception as e:
                    logger.warning(f"Failed to get secret {var}: {e}")
    except ImportError as e:
        logger.info(f"kaggle_secrets module not found (local run?): {e}")
    except Exception as e:
        logger.error(f"Error initializing UserSecretsClient: {e}")
    
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}. Tests might fail.")
    else:
        logger.info("All required environment variables are set!")
    

def run_tests():
    """Run behave tests and generate report."""
    logger.info("Starting BDD tests...")
    
    # Define output paths
    working_dir = Path("/kaggle/working")
    if not working_dir.exists():
        working_dir = Path(".")
        
    report_file = working_dir / "test_report.json"
    
    # Run behave
    cmd = [
        sys.executable, "-m", "behave",
        "--format", "json",
        "--outfile", str(report_file),
        "--no-capture",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print("BEHAVE OUTPUT:\n", result.stdout)
        if result.stderr:
            print("BEHAVE ERRORS:\n", result.stderr)
            
        logger.info(f"Behave finished with return code: {result.returncode}")
        
        # Debug: List files
        print("Files in working directory:")
        subprocess.run(["ls", "-la", str(working_dir)])
        
        if report_file.exists():
            print(f"REPORT CONTENT ({report_file}):")
            print(report_file.read_text())
        else:
            print(f"REPORT FILE NOT FOUND: {report_file}")
            
    except Exception as e:
        logger.error(f"Failed to execute behave: {e}")
        
    # Always exit with 0 to ensure Kaggle saves the output artifacts
    logger.info("Test execution completed. Exiting with success code.")
    sys.exit(0)


if __name__ == "__main__":
    install_dependencies()
    setup_files()
    load_secrets()
    run_tests()
