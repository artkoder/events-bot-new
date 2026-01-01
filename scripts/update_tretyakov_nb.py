import json
import re

NOTEBOOK_PATH = "/workspaces/events-bot-new/kaggle/ParseTheatres/parse_theatres.ipynb"
OUTPUT_PATH = NOTEBOOK_PATH

NEW_TRETYAKOV_CODE = r'''
# ==========================================
# ЧАСТЬ 3: ТРЕТЬЯКОВСКАЯ ГАЛЕРЕЯ
# ==========================================

BASE_URL_TRETYAKOV = "https://kaliningrad.tretyakovgallery.ru"

def normalize_tretyakov_date(date_raw):
    """
    Парсит дату из строк вида:
    - "Уже идет 4 января в 17:00" -> ("2025-01-04", "17:00")
    - "Скоро С 25 декабря по 10 января" -> ("2024-12-25", "00:00") (берем дату начала)
    """
    import datetime
    
    if not date_raw:
        return None, None

    text = date_raw.strip().lower()
    
    # 1. Извлекаем время
    time_match = re.search(r'(\d{1,2}):(\d{2})', text)
    parsed_time = "00:00"
    if time_match:
        parsed_time = f"{int(time_match.group(1)):02d}:{int(time_match.group(2)):02d}"

    # 2. Очистка от мусора
    text = re.sub(r"уже идет|скоро", "", text).strip()
    
    # 3. Поиск диапазона или перечисления
    range_match = re.search(r"с\s+(\d{1,2})\s+([а-я]+)", text)
    
    day = None
    month_name = None
    
    if range_match:
        day = int(range_match.group(1))
        month_name = range_match.group(2)
    else:
        # "2 и 8 января" - ищем любой месяц
        ru_months = ["января", "февраля", "марта", "апреля", "мая", "июня", 
                     "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        
        found_month = None
        for m in ru_months:
            if m in text:
                found_month = m
                break
        
        if found_month:
            month_name = found_month
            # Ищем первое число
            digits = re.findall(r"\d+", text)
            if digits:
                day = int(digits[0])

    if not day or not month_name:
        simple = re.search(r"(\d{1,2})\s+([а-я]+)", text)
        if simple:
            day = int(simple.group(1))
            month_name = simple.group(2)

    # 4. Маппинг месяца
    MONTHS = {
        "января": 1, "февраля": 2, "марта", "апреля": 4, "мая": 5, "июня": 6,
        "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
    }
    month = MONTHS.get(month_name)
            
    if not month:
        return None, parsed_time

    # 5. Определение года
    today = datetime.date.today()
    year = today.year
    
    try:
        if today.month == 12 and month == 1:
             year += 1
        elif month < today.month and (today.month - month) > 4:
             year += 1
             
        event_date = datetime.date(year, month, day)
        return event_date.isoformat(), parsed_time
    except ValueError:
        return None, parsed_time


async def scrape_tretyakov_schedule(page):
    url = f"{BASE_URL_TRETYAKOV}/events/"
    print(f"\n🖼️ [Третьяковка] Этап 1: Сканируем события на {url}...")

    try:
        await page.goto(url, timeout=90000, wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)

        for _ in range(3):
            await page.mouse.wheel(0, 4000)
            await page.wait_for_timeout(random.randint(1000, 2000))

        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')

        events_list = []
        seen_ids = set()

        cards = soup.select('.card')
        for card in cards:
            title_tag = card.select_one('.card_info .card_info-top .card_title')
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)

            link_tag = card.select_one('.card_info-top a[href]')
            href = link_tag.get('href') if link_tag else ""
            if href and href.startswith("//"):
                href = f"https:{href}"
            elif href and href.startswith('/'):
                href = f"{BASE_URL_TRETYAKOV}{href}"
            
            # --- ФОТО ---
            img_url = ""
            img_tag = card.select_one('img.card_img')
            if img_tag and img_tag.get('src'):
                img_url = img_tag['src']
            if not img_url:
                img_div = card.select_one('div.card_img')
                style = img_div.get('style', '') if img_div else ""
                match = re.search(r"url\((.*?)\)", style)
                if match:
                    img_url = match.group(1).strip().strip('"').strip("'")
            if img_url and img_url.startswith('/'):
                img_url = f"{BASE_URL_TRETYAKOV}{img_url}"

            # --- ДАТА ---
            date_text = ""
            info = card.select_one('.card_info')
            if info:
                bottom = info.select_one('.card_info-bottom') or info
                date_text = bottom.get_text(" ", strip=True)

            if title:
                date_text = date_text.replace(title, "").strip()

            clean_date = re.sub(r"Купить\s*билет|Купить|Билеты|Подробнее", "", date_text, flags=re.IGNORECASE).strip()
            clean_date = re.sub(r"\b(Атриум|Кинозал)\b", "", clean_date, flags=re.IGNORECASE).strip()
            
            parsed_date, parsed_time = normalize_tretyakov_date(clean_date)

            card_text = card.get_text(" ", strip=True).upper()
            if "АТРИУМ" in card_text:
                location = "Атриум"
            elif "КИНОЗАЛ" in card_text:
                location = "Кинозал"
            else:
                location = "Филиал Третьяковской галереи, Парадная наб. 3, #Калининград"

            event_id = f"{clean_date}_{title}_{href}"
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)

            events_list.append({
                "title": title,
                "date_raw": clean_date,
                "parsed_date": parsed_date,
                "parsed_time": parsed_time,
                "ticket_status": "unknown",
                "url": href,
                "photos": [img_url] if img_url else [],
                "location": location,
            })

        print(f"✅ [Третьяковка] Найдено событий: {len(events_list)}")
        return events_list
    except Exception as e:
        print(f"❌ [Третьяковка] Ошибка расписания: {e}")
        return []

async def _tretyakov_interact_and_get_price(page, target_date=None, target_time=None):
    """
    СТРАТЕГИЯ:
    1. Check Price (если есть - выход).
    2. Try STANDARD Selectors (exact classes) -> для большинства ивентов.
    3. Try FUZZY Selectors (JS lookup) -> для 'Fairy Tale' и сложных.
    4. Click Time -> Parse Price.
    """
    try:
        # 1. Wait for widget container or price
        # Широкий вейт, чтобы не падать сразу
        try:
            await page.wait_for_selector('div[class*="calendar"], .skin-inner, app-root, .wrapper, .page-buy-container', timeout=8000)
        except:
             pass

        # === CHECK 0: Pre-existing price ===
        text_init = await page.inner_text("body")
        if re.search(r"\d+\s*(?:₽|руб)", text_init):
            # Price visible?
            pass 

        await page.wait_for_timeout(1000)

        click_success = False
        target_day = None
        if target_date:
            try:
                target_day = int(target_date.split('-')[2])
            except:
                pass

        if target_day:
            print(f"   ...Target Day: {target_day}")
            
            # --- ATTEMPT 1: STANDARD SELECTORS (Strict) ---
            # Это работало для большинства событий ранее
            # Ищем элементы с классом day/cell и точным текстом
            try:
                # Selectors used in typical calendars
                candidates = await page.locator('.cell:not(.day-header), .day, .date-item, span[class*="day"]').all()
                for cand in candidates:
                    if not await cand.is_visible(): continue
                    
                    # Check text Exact Match
                    txt = (await cand.inner_text()).strip()
                    if txt == str(target_day):
                        # Found it!
                        cls = await cand.get_attribute('class') or ""
                        if "disabled" in cls or "sold" in cls: continue
                        
                        await cand.click(force=True)
                        click_success = True
                        print(f"   >>> [Standard] Clicked date {target_day}")
                        break
            except Exception as e:
                print(f"   Standard click error: {e}")

            # --- ATTEMPT 2: FUZZY SELECTORS (JS) ---
            # Если стандарт не сработал (как в Fairy Tale)
            if not click_success:
                print(f"   ...Attempts Fuzzy Search for {target_day}")
                try:
                    candidates = await page.evaluate_handle(r"""
                        (day) => {
                            const els = Array.from(document.querySelectorAll('div, span, button, td, a'));
                            const matches = els.filter(el => {
                                if (el.offsetParent === null) return false;
                                const cls = (el.className || "").toString();
                                if (cls.includes('disabled') || cls.includes('sold')) return false;
                                
                                const txt = el.innerText.trim();
                                // Regex: word boundary check for day number
                                // e.g. matches "29" in "ПН 29" but NOT "290" or "129"
                                const dayRx = new RegExp("(^|\\D)" + day + "(\\D|$)", "i");
                                if (!dayRx.test(txt)) return false;
                                
                                // Avoid long text blocks (descriptions)
                                if (txt.length > 50) return false;
                                
                                // Avoid obvious non-date things if possible?
                                return true;
                            });
                            // Prefer shorter matches (exact button vs container)
                            matches.sort((a,b) => a.innerText.length - b.innerText.length);
                            return matches.length > 0 ? matches[0] : null;
                        }
                    """, target_day)
                    
                    if candidates.as_element():
                        await candidates.as_element().click()
                        click_success = True
                        print(f"   >>> [Fuzzy] Clicked date {target_day}")
                except Exception as e:
                    print(f"   Fuzzy click error: {e}")

        # Fallback date (Any date 1-31) if target specific failed
        if not click_success:
             print("   ...Fallback: Clicking first available date 1-31")
             try:
                 candidates = await page.evaluate_handle(r"""
                   () => {
                        const els = Array.from(document.querySelectorAll('div, span, button, td, a'));
                        const matches = els.filter(el => {
                            if (el.offsetParent === null) return false;
                            const cls = (el.className || "").toString();
                            if (cls.includes('disabled') || cls.includes('sold')) return false;
                            
                            const txt = el.innerText.trim();
                            // STRICT: Matches exactly a number 1-31 alone, or matches "Mon 29" format
                            // Checking for explicit 1-31 range match
                            const dateMatch = txt.match(/(^|\D)([1-9]|[12]\d|3[01])(\D|$)/);
                            if (!dateMatch) return false;
                            
                            if (txt.length > 40) return false; 
                            return true;
                        });
                        matches.sort((a,b) => a.innerText.length - b.innerText.length);
                        return matches.length > 0 ? matches[0] : null;
                   }
                 """)
                 if candidates.as_element():
                     txt_click = await candidates.as_element().inner_text()
                     await candidates.as_element().click()
                     print(f"   >>> [Fallback] Clicked date: {txt_click[:10]}...")
             except Exception as e:
                 print(f"   Fallback error: {e}")

        # 3. TIME SELECTION
        await page.wait_for_timeout(1500) 
        
        target_hm = target_time if target_time else None
        print("   ...Searching Time")
        
        # Standard + Fuzzy Time Search
        try:
             time_el = await page.evaluate_handle(r"""
                (tgt) => {
                    const els = Array.from(document.querySelectorAll('div, span, button, td, a'));
                    const matches = els.filter(el => {
                        if (el.offsetParent === null) return false;
                        const cls = (el.className || "").toString();
                        if (cls.includes('disabled')) return false;
                        
                        const txt = el.innerText.trim();
                        // STRICT HH:MM pattern
                        const timeMatch = txt.match(/\d{1,2}:\d{2}/);
                        if (!timeMatch) return false;
                        
                        // If target provided, must contain it
                        if (tgt && !txt.includes(tgt)) return false;
                        
                        if (txt.length > 30) return false;
                        return true;
                    });
                     matches.sort((a,b) => a.innerText.length - b.innerText.length);
                     return matches.length > 0 ? matches[0] : null;
                }
             """, target_hm)
             
             if time_el.as_element():
                 t_txt = await time_el.as_element().inner_text()
                 await time_el.as_element().click()
                 print(f"   >>> Clicked time: {t_txt}")
        except:
            pass

        # 4. PRICE EXTRACTION
        await page.wait_for_timeout(2000)

        text = await page.inner_text("body")
        lower = text.lower()
        
        if "нет билетов" in lower or "sold out" in lower or "распродано" in lower:
            return "sold_out", None, None
            
        prices = []
        matches = re.findall(r"(\d+)\s*(?:₽|руб|р\.)", text, flags=re.IGNORECASE)
        for m in matches:
            prices.append(int(m))
            
        if prices:
            prices = sorted(set(prices))
            return "available", prices[0], prices[-1]
            
        return "unknown", None, None

    except Exception as e:
        print(f"   ⚠️ Interaction error: {e}")
        return "unknown", None, None


async def scrape_tretyakov_ticket_price(context, url, parsed_date=None, parsed_time=None):
    print(f"💳 [Третьяковка] Проверяем: {url}")
    if not url: return {"ticket_status": "unknown", "ticket_price_min": None, "ticket_price_max": None}

    page = await context.new_page()
    status, p_min, p_max = "unknown", None, None

    try:
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        status, p_min, p_max = await _tretyakov_interact_and_get_price(page, parsed_date, parsed_time)
        print(f"   Result: {status} Price: {p_min}-{p_max}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await page.close()
    
    return {
        "ticket_status": status,
        "ticket_price_min": p_min,
        "ticket_price_max": p_max,
    }

async def run_tretyakov(browser):
    context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
    await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
    page = await context.new_page()

    schedule = await scrape_tretyakov_schedule(page)
    if not schedule:
        await context.close()
        return []

    for item in schedule:
        url = item.get("url", "")
        if not url or "tickets" not in url: continue
        
        res = await scrape_tretyakov_ticket_price(context, url, item.get("parsed_date"), item.get("parsed_time"))
        item.update(res)
        await page.wait_for_timeout(random.randint(500, 1000))

    await context.close()
    return schedule
'''

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb_content = json.load(f)

    code_cell = None
    for cell in nb_content.get("cells", []):
        if cell.get("cell_type") == "code":
            code_cell = cell
            break
            
    if not code_cell: return

    source = "".join(code_cell["source"])
    start_marker = "# ==========================================\n# ЧАСТЬ 3: ТРЕТЬЯКОВСКАЯ ГАЛЕРЕЯ\n# =========================================="
    end_marker = "# --- ЗАПУСК ---"
    
    if start_marker not in source or end_marker not in source: return

    parts = source.split(start_marker)
    pre = parts[0]
    rest = parts[1]
    post = rest.split(end_marker)[1]
    
    new_source = pre + NEW_TRETYAKOV_CODE + "\n\n" + end_marker + post
    new_lines = [line + '\n' for line in new_source.split('\n')]
    if new_lines[-1] == '\n': new_lines.pop()
    else: new_lines[-1] = new_lines[-1].rstrip('\n')
        
    code_cell["source"] = new_lines
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb_content, f, indent=1, ensure_ascii=False)
    
    print("✅ Notebook updated with HYBRID (Standard + Fuzzy) Logic")

if __name__ == "__main__":
    update_notebook()
