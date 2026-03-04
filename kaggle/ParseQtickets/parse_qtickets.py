
import json
import logging
import re
import time
import random
import subprocess
import sys
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Install Playwright on Kaggle if not available
def ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return True
    except ImportError:
        print("Installing Playwright...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        # Install system dependencies
        subprocess.run(["apt-get", "update"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "libglib2.0-0", "libnss3", "libnspr4", "libatk1.0-0", 
                        "libatk-bridge2.0-0", "libcups2", "libdrm2", "libdbus-1-3", "libxcb1", 
                        "libxkbcommon0", "libx11-6", "libxcomposite1", "libxdamage1", "libxext6",
                        "libxfixes3", "libxrandr2", "libgbm1", "libpango-1.0-0", "libcairo2",
                        "libasound2"], capture_output=True)
        try:
            from playwright.sync_api import sync_playwright
            return True
        except:
            return False

HAS_PLAYWRIGHT = ensure_playwright()

if HAS_PLAYWRIGHT:
    from playwright.sync_api import sync_playwright


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qtickets_parser")

BASE_URL = "https://kaliningrad.qtickets.events"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def normalize_location(name):
    """Normalize common location names to match LOCATIONS.md"""
    name = name.strip()
    mapping = {
        # Existing
        "Клуб YALTA": "YALTA",
        "Royal Park": "Royal Park",
        "МСЗ \"СОЗВЕЗДИЕ\"": "МСЗ Созвездие",
        "Дом офицеров Балтийского флота": "Дом офицеров Балтийского флота",
        "ДК Железнодорожников": "Дом железнодорожников (ДКЖ)",
        "Клуб Вагонка": "Вагонка (клуб)",
        "РК Королевская Башня": "Резиденция королей",
        # New from Qtickets
        "Клуб \"Универсал\"": "Универсал (пространство)",  # Maps to existing entry
        "Клуб Бастион": "Бастион (арт-клуб)",
        "Клуб БАСТИОН": "Бастион (арт-клуб)",
        "Бастион": "Бастион (арт-клуб)",
        "Железный Занавес": "Железный Занавес",
        "HUMAN concept": "HUMAN concept",
        "Онегин": "Онегин (ресторан)",
        "Замок Тапиау": "Замок Тапиау",
        "Океан": "Океан (караоке-клуб)",
        "B SIDE": "B SIDE (бар)",  # Калинина 4
        "Площадка Депо": "Барн",  # Same location: Каштановая аллея 1а
        "ГрандМонополь": "ГрандМонополь",
        "Концертный Зал Калининградского симфонического оркестра": "Концертный зал КСО",
    }
    return mapping.get(name, name)

def get_soup(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

def get_all_event_urls_playwright():
    """Use Playwright to scroll and capture ALL event URLs from infinite scroll."""
    if not HAS_PLAYWRIGHT:
        logger.error("Playwright not available!")
        return []
    
    event_urls = set()
    
    with sync_playwright() as p:
        logger.info("Launching browser for infinite scroll...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        page.goto(BASE_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        
        # Scroll to load all events
        prev_count = 0
        max_scrolls = 50  # Safety limit
        scroll_count = 0
        
        while scroll_count < max_scrolls:
            # Count current event links
            links = page.query_selector_all("a[href*='/']")
            current_count = len([l for l in links if re.search(r'/\d{3,}-[a-zA-Z0-9-]+', l.get_attribute("href") or "")])
            
            logger.info(f"Scroll {scroll_count}: Found {current_count} event links")
            
            if current_count == prev_count and scroll_count > 3:
                # No new content for several scrolls, we're done
                logger.info("No more new events loading, stopping scroll")
                break
            
            prev_count = current_count
            
            # Scroll down
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(1.5, 2.5))  # Human-like pause
            scroll_count += 1
        
        # Extract all event URLs from final page state
        links = page.query_selector_all("a[href*='/']")
        for link in links:
            href = link.get_attribute("href")
            if href and re.search(r'/\d{3,}-[a-zA-Z0-9-]+', href):
                full_url = href if href.startswith("http") else BASE_URL + href
                event_urls.add(full_url)
        
        browser.close()
    
    logger.info(f"Total unique event URLs found: {len(event_urls)}")
    return list(event_urls)

def get_event_urls_simple():
    """Fallback: get event URLs from initial page load only (limited)."""
    soup = get_soup(BASE_URL)
    if not soup:
        return []

    event_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r'/\d{3,}-[a-zA-Z0-9-]+', href):
            full_url = href if href.startswith("http") else BASE_URL + href
            event_urls.add(full_url)
            
    logger.info(f"Found {len(event_urls)} unique event URLs (simple mode)")
    return list(event_urls)

def parse_qtickets_events():
    # Try Playwright first for full list
    if HAS_PLAYWRIGHT:
        event_urls = get_all_event_urls_playwright()
    else:
        logger.warning("Playwright not available, using simple mode (may miss events)")
        event_urls = get_event_urls_simple()
    
    events = []
    for i, url in enumerate(event_urls):
        logger.info(f"Parsing [{i+1}/{len(event_urls)}] {url}")
        event_data = parse_event_detail(url)
        if event_data:
            events.append(event_data)
        
        # Human-like delay
        delay = random.uniform(2.0, 5.0)
        time.sleep(delay)
        
    return events

def parse_event_detail(url):
    soup = get_soup(url)
    if not soup:
        return None

    # Extract JSON-LD
    json_ld = None
    scripts = soup.find_all("script", type="application/ld+json")
    for s in scripts:
        if not s.string:
            continue
        try:
            data = json.loads(s.string)
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "Event":
                        json_ld = item
                        break
            elif isinstance(data, dict):
                if data.get("@type") == "Event":
                    json_ld = data
            
            if json_ld:
                break
        except Exception as e:
            logger.debug(f"JSON-LD parse error: {e}")
            continue
            
    # Fallback to OpenGraph
    og_title = soup.find("meta", property="og:title")
    og_desc = soup.find("meta", property="og:description")
    og_image = soup.find("meta", property="og:image")
    
    title = json_ld.get("name") if json_ld else (og_title["content"] if og_title else None)
    description = json_ld.get("description") if json_ld else (og_desc["content"] if og_desc else None)
    
    # Age restriction
    age_restriction = None
    age_span = soup.find("span", class_=lambda x: x and "age" in x)
    if not age_span:
        for t in soup.find_all(string=re.compile(r"^\d+\+$")):
            if t.parent.name in ["span", "div"]:
                age_restriction = t.strip()
                break
    else:
        age_restriction = age_span.get_text().strip()

    # Full description
    full_desc = description
    candidates = soup.find_all("div", class_=lambda x: x and ("description" in x or "text" in x))
    for c in candidates:
        text = c.get_text(separator="\n").strip()
        if len(text) > (len(description) if description else 0) + 50:
            full_desc = text
            break
            
    # Dates
    start_date = json_ld.get("startDate") if json_ld else None
    
    # Location
    location = "Калининград"
    if json_ld and "location" in json_ld:
        loc_data = json_ld["location"]
        if isinstance(loc_data, dict):
            location = loc_data.get("name", location)
    
    location = normalize_location(location)
            
    # Image
    image = json_ld.get("image") if json_ld else (og_image["content"] if og_image else None)
    if isinstance(image, list):
        image = image[0]
        
    # Price
    price_min = None
    price_max = None
    if json_ld and "offers" in json_ld:
        offers = json_ld["offers"]
        if isinstance(offers, dict):
             price_min = offers.get("price")
        elif isinstance(offers, list):
             prices = [float(o.get("price", 0)) for o in offers if o.get("price")]
             paid_prices = [p for p in prices if p > 0]
             if paid_prices:
                 price_min = min(paid_prices)
                 price_max = max(paid_prices) if len(paid_prices) > 1 and max(paid_prices) > min(paid_prices) else None
             elif prices:
                 price_min = 0

    if not title or not start_date:
        logger.warning(f"Missing mandatory fields for {url}")
        return None

    # Parsed date/time
    parsed_date = None
    parsed_time = None
    if start_date:
        try:
            dt = datetime.fromisoformat(start_date)
            parsed_date = dt.date().isoformat()
            parsed_time = dt.strftime("%H:%M")
        except:
            pass

    # Ticket Status
    ticket_status = "unknown"
    if json_ld and "offers" in json_ld:
        offers = json_ld["offers"]
        availability = None
        if isinstance(offers, dict):
            availability = offers.get("availability")
        elif isinstance(offers, list) and offers:
            for o in offers:
                if o.get("availability") == "https://schema.org/InStock":
                    availability = "https://schema.org/InStock"
                    break
        
        if availability == "https://schema.org/InStock":
            ticket_status = "available"
        elif availability in ("https://schema.org/OutOfStock", "http://schema.org/SoldOut"):
            ticket_status = "sold_out"
            
    if ticket_status == "unknown":
        if price_min is not None:
             ticket_status = "available"
        else:
             if soup.find(string=re.compile(r"Билеты проданы|Sold out", re.IGNORECASE)):
                 ticket_status = "sold_out"

    return {
        "title": title,
        "description": full_desc,
        "age_restriction": age_restriction,
        "date_raw": start_date, 
        "parsed_date": parsed_date,
        "parsed_time": parsed_time,
        "location": location,
        "url": url,
        "photos": [image] if image else [],
        "ticket_price_min": int(price_min) if price_min is not None else None,
        "ticket_price_max": int(price_max) if price_max is not None else None,
        "ticket_status": ticket_status,
        "source_type": "qtickets"
    }

if __name__ == "__main__":
    result = parse_qtickets_events()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Save to file for Kaggle output
    with open("qtickets_events.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Total events parsed: {len(result)}")
