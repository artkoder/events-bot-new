import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add current dir to path
sys.path.append(os.getcwd())

from db import get_db
from models import Event
from sqlalchemy import select

async def fix_locations():
    db = get_db()
    # 3 hours just to be safe
    cutoff = datetime.now(timezone.utc) - timedelta(hours=3)
    
    # Also check if recent events might have been added without timezone awareness (though model enforces it)
    
    async with db.get_session() as session:
        # Find events from last 3 hours with duplicate info in location_address
        # Also specifically check Tretyakov events
        query = select(Event).where(
            Event.added_at > cutoff,
            Event.location_name.ilike('%Третьяков%')
        ).order_by(Event.id.desc())
        
        result = await session.execute(query)
        events = result.scalars().all()
        
        print(f"Found {len(events)} recent Tretyakov events")
        
        fixed_count = 0
        for e in events:
            # Check duplication in location_address
            addr = e.location_address or ""
            city = e.city or "Калининград"
            
            print(f"Checking {e.id}: {e.title}")
            print(f"  Address: {addr!r}")
            print(f"  City: {city!r}")
            print(f"  Location: {e.location_name!r}")
            
            new_addr = addr
            
            # Specific cleanups for Tretyakov data seen in screenshot
            # "Парадная наб. 3, #Калининград, Парадная наб. 3, Калининград"
            
            if "Парадная наб. 3, #Калининград, Парадная наб. 3, Калининград" in addr:
                new_addr = "Парадная наб. 3"
            elif "Парадная наб. 3, Парадная наб. 3" in addr.replace(", #", ", ").replace(", ", " "): 
                # Loose check for repetition
                new_addr = "Парадная наб. 3"
            elif addr == "Парадная наб. 3, #Калининград":
                 new_addr = "Парадная наб. 3"
            
            # If address contains city at the end
            # "Addres, City" + City -> "Addres, City, City"
            # We want just "Address" in location_address because formatter adds City
            
            if city and new_addr.endswith(f", {city}"):
                new_addr = new_addr[:-len(city)-2].strip()

            if new_addr != addr:
                print(f"  ✅ FIXING to: {new_addr!r}")
                e.location_address = new_addr
                fixed_count += 1
            else:
                print("  OK (no change)")
            print("-" * 20)

        if fixed_count > 0:
            await session.commit()
            print(f"Committed {fixed_count} fixes.")
            
            # Print list of fixed
            print("\nFixed Events:")
            for e in events:
                if e.location_address != addr: # Wait loop var duplication issue? No
                    pass
        else:
            print("No changes needed.")

if __name__ == "__main__":
    asyncio.run(fix_locations())
