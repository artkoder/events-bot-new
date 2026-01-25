import os
import sys
from telethon.sync import TelegramClient
from telethon.sessions import StringSession

# === Device Parameters (Samsung S22 Ultra) ===
DEVICE_MODEL = "Samsung S22 Ultra"
SYSTEM_VERSION = "13.0"
APP_VERSION = "9.6.6"

def generate_session():
    print("=== Telegram Session Generator (Samsung S22 Ultra) ===")
    
    api_id = input("Enter API ID: ").strip()
    api_hash = input("Enter API HASH: ").strip()

    if not api_id or not api_hash:
        print("Error: API ID and API HASH are required.")
        return

    print(f"\nInitializing client with device: {DEVICE_MODEL}...")
    
    try:
        # Initialize client with specific device parameters
        with TelegramClient(
            StringSession(), 
            int(api_id), 
            api_hash,
            device_model=DEVICE_MODEL,
            system_version=SYSTEM_VERSION,
            app_version=APP_VERSION
        ) as client:
            print("\nPlease follow the instructions to log in (check your Telegram app for code).")
            # This triggers the interactive login flow
            client.start()
            
            print("\n✅ Session generated successfully!")
            print("="*60)
            print(client.session.save())
            print("="*60)
            print("⚠️  KEEP THIS STRING SECRET! PASS IT TO KAGGLE VIA SECRETS ONLY! ⚠️")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    generate_session()
