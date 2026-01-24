
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from aiogram import Bot

# Mock Bot
class MockBot:
    async def send_message(self, chat_id, text, **kwargs):
        print(f"[MOCK BOT] send_message to {chat_id}: {text}")
        class MockMessage:
            message_id = 123
        return MockMessage()

    async def edit_message_text(self, text, chat_id, message_id, **kwargs):
        print(f"[MOCK BOT] edit_message_text {message_id}: {text}")

    async def send_document(self, chat_id, document, **kwargs):
        print(f"[MOCK BOT] send_document to {chat_id}: {document}")

async def main():
    logging.basicConfig(level=logging.INFO)
    
    from source_parsing.handlers import run_diagnostic_parse
    
    print("Starting diagnostic parse for philharmonia...")
    mock_bot = MockBot()
    
    # Run diagnostic parse
    # This will trigger run_kaggle_kernel (which is actually run_philharmonia_kaggle_kernel inside the handler? 
    # Wait, run_diagnostic_parse in handlers.py calls run_kaggle_kernel generic runner?
    # Let me check handlers.py implementation of run_diagnostic_parse again.
    
    await run_diagnostic_parse(mock_bot, 12345, "philharmonia")
    
if __name__ == "__main__":
    asyncio.run(main())
