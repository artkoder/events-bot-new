import asyncio
from db import Database
from sqlalchemy import text

async def main():
    db = Database("db_prod_snapshot.sqlite")
    await db.init()
    async with db.get_session() as session:
        # Move Test Channel to dummy ID
        print("Moving Test Channel -1002210431821 to -999...")
        await session.execute(
            text("UPDATE channel SET channel_id = -999 WHERE channel_id = -1002210431821")
        )
        
        # Move Prod Channel to Test Channel ID
        print("Moving Prod Channel -1002331532485 to -1002210431821...")
        await session.execute(
            text("UPDATE channel SET channel_id = -1002210431821 WHERE channel_id = -1002331532485")
        )
        
        await session.commit()
        print("Swap complete.")

if __name__ == "__main__":
    asyncio.run(main())
