"""
Debug script to investigate why Pianissimo cards don't have onclick attribute.
"""
import asyncio
from playwright.async_api import async_playwright

BASE_URL = "https://kaliningrad.tretyakovgallery.ru"


async def debug_card_structure():
    """Debug the HTML structure of Pianissimo cards."""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        url = f"{BASE_URL}/events/"
        print(f"🖼️ Scanning: {url}")
        
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        
        # Scroll to load all
        for _ in range(5):
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(1000)
        
        # Get first few cards with their full HTML
        cards_info = await page.evaluate("""
            () => {
                const results = [];
                document.querySelectorAll('.card').forEach((card, index) => {
                    const titleEl = card.querySelector('.card_title');
                    if (!titleEl) return;
                    
                    const title = titleEl.innerText.trim();
                    const onclick = card.getAttribute('onclick');
                    const outerHTML = card.outerHTML.substring(0, 500); // First 500 chars
                    
                    // Check all parent elements for onclick
                    let parentOnclick = null;
                    let parent = card.parentElement;
                    for (let i = 0; i < 3 && parent; i++) {
                        if (parent.getAttribute('onclick')) {
                            parentOnclick = parent.getAttribute('onclick');
                            break;
                        }
                        parent = parent.parentElement;
                    }
                    
                    // Check if card has any link
                    const allLinks = [...card.querySelectorAll('a')].map(a => a.href);
                    
                    results.push({
                        index: index,
                        title: title.substring(0, 50),
                        onclick: onclick,
                        parentOnclick: parentOnclick,
                        links: allLinks,
                        htmlPreview: outerHTML.substring(0, 200),
                    });
                });
                return results;
            }
        """)
        
        print(f"\n📋 Found {len(cards_info)} cards")
        print("=" * 80)
        
        # Show cards with and without onclick
        with_onclick = [c for c in cards_info if c['onclick']]
        without_onclick = [c for c in cards_info if not c['onclick']]
        
        print(f"\n✅ Cards WITH onclick: {len(with_onclick)}")
        for c in with_onclick[:3]:
            print(f"   [{c['index']}] {c['title']}")
            print(f"       onclick: {c['onclick'][:80] if c['onclick'] else 'None'}")
        
        print(f"\n❌ Cards WITHOUT onclick: {len(without_onclick)}")
        for c in without_onclick:
            is_pianissimo = 'PIANISSIMO' in c['title'].upper()
            marker = "🎹" if is_pianissimo else "  "
            print(f"   {marker} [{c['index']}] {c['title']}")
            print(f"       parentOnclick: {c['parentOnclick'][:80] if c['parentOnclick'] else 'None'}")
            print(f"       links: {c['links'][:2]}")
        
        # Now let's look at a specific Pianissimo card in detail
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS OF PIANISSIMO CARDS")
        print("=" * 80)
        
        pianissimo_html = await page.evaluate("""
            () => {
                const cards = document.querySelectorAll('.card');
                for (const card of cards) {
                    const title = card.querySelector('.card_title')?.innerText || '';
                    if (title.toUpperCase().includes('ТЯНЬ ЛАН')) {
                        return {
                            outerHTML: card.outerHTML,
                            title: title,
                            className: card.className,
                            tagName: card.tagName,
                        };
                    }
                }
                return null;
            }
        """)
        
        if pianissimo_html:
            print(f"\n🎹 ТЯНЬ ЛАН card structure:")
            print(f"   Tag: {pianissimo_html['tagName']}")
            print(f"   Classes: {pianissimo_html['className']}")
            print(f"\n   HTML (first 800 chars):")
            print(pianissimo_html['outerHTML'][:800])
        
        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(debug_card_structure())
