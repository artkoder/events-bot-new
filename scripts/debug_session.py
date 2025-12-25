
import asyncio
import logging
import os
import sys

# Setup environment before imports
os.environ["DB_PATH"] = "./db_prod_snapshot.sqlite"
os.environ["DEV_MODE"] = "1"

# Add project root to path
sys.path.append(os.getcwd())

from db import Database
from models import VideoAnnounceSession, User
from video_announce.scenario import VideoAnnounceScenario

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_session():
    print("--- START DEBUG SESSION 34 ---")
    db = Database(os.environ["DB_PATH"])
    
    async with db.get_session() as session:
        sess = await session.get(VideoAnnounceSession, 34)
        if not sess:
            print("❌ Session 34 NOT FOUND in database!")
            return
        
        print(f"✅ Found Session 34")
        print(f"Status: {sess.status}")
        print(f"Selection Params: {sess.selection_params}")
        
    print("\n--- FETCHING CANDIDATES ---")
    
    # We need a dummy bot because VideoAnnounceScenario expects one
    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            print(f"[BOT] send_message to {chat_id}: {text[:100]}...")
            return type('obj', (object,), {'message_id': 123})
            
    bot = DummyBot()
    
    # Init scenario
    # We use a dummy chat_id/user_id, technically we should use the ones from session if possible
    # But for fetch_candidates logic, it mostly matters what's in the DB and params.
    scenario = VideoAnnounceScenario(db, bot, chat_id=123, user_id=123)
    
    # Manually reproduce steps from apply_instruction
    ctx = await scenario._build_selection_context(sess)
    print(f"Context Built:\n Target: {ctx.target_date}\n TZ: {ctx.tz}\n Limits: {ctx.candidate_limit}")
    
    from video_announce.selection import fetch_candidates, _filter_events_with_posters
    
    try:
        candidates = await fetch_candidates(db, ctx)
        print(f"✅ fetch_candidates returned {len(candidates)} raw events")
    except Exception as e:
        print(f"❌ fetch_candidates FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    filtered = _filter_events_with_posters(candidates)
    print(f"✅ _filter_events_with_posters returned {len(filtered)} events")
    
    if not filtered:
        print("⚠️ NO CANDIDATES after filtering! This explains silence (maybe?).")
    else:
        print(f"First 3 candidates: {[e.id for e in filtered[:3]]}")

    print("\n--- CHECKING LLM CONFIG ---")
    # Check tokens
    print(f"ONE_TOKEN present: {bool(os.getenv('ONE_TOKEN'))} (used for Kaggle?)")
    print(f"FOUR_O_TOKEN present: {bool(os.getenv('FOUR_O_TOKEN'))}")
    
    from main import ask_4o
    print("\n--- TESTING LLM CONNECTIVITY ---")
    try:
        response = await ask_4o("Hello, are you there?", model="gpt-4o")
        print(f"✅ LLM Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ LLM Request FAILED: {e}")

    print("\n--- SIMULATING BUILD_SELECTION ---")
    from video_announce.selection import build_selection
    try:
        # Mock bot for build_selection
        # We need it to NOT fail on send_document
        class MockBot:
            async def send_document(self, chat_id, document, caption=None, **kwargs):
                print(f"[BOT] send_document to {chat_id}: {caption}")
                return type('obj', (object,), {'message_id': 123})
        
        # We pass the real DB and context, but use our MockBot
        result = await build_selection(
            db, 
            ctx, 
            session_id=sess.id,
            candidates=filtered,
            bot=MockBot(), 
            notify_chat_id=123
        )
        print(f"✅ build_selection SUCCESS")
        print(f"Ranked items: {len(result.ranked)}")
        print(f"Intro text: {result.intro_text}")
    except Exception as e:
        print(f"❌ build_selection FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    print("--- END DEBUG ---")

if __name__ == "__main__":
    asyncio.run(debug_session())
