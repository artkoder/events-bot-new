import sys
import asyncio
import os

sys.path.append("/app")
from db import Database
from models import VideoAnnounceSession, VideoAnnounceSessionStatus

async def reset_session(session_id: int):
    db_path = os.getenv("DB_PATH", "/data/db.sqlite")
    db = Database(db_path)
    await db.init()
    async with db.get_session() as session:
        sess = await session.get(VideoAnnounceSession, session_id)
        if not sess:
            print(f"Session {session_id} not found.")
            return
        print(f"Current status: {sess.status}")
        if sess.status == VideoAnnounceSessionStatus.FAILED:
            sess.status = VideoAnnounceSessionStatus.SELECTED
            sess.started_at = None
            sess.finished_at = None
            session.add(sess)
            await session.commit()
            print(f"Session {session_id} reset to SELECTED. You can now restart it.")
        else:
            print(f"Session is not in FAILED status, cannot reset.")

if __name__ == "__main__":
    asyncio.run(reset_session(53))
