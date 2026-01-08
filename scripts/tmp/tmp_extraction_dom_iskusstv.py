
import sys
import os
sys.path.append(os.getcwd())
try:
    from source_parsing.dom_iskusstv import extract_dom_iskusstv_urls
except ImportError:
    # Maybe needs path adjustment
    sys.path.append("/workspaces/events-bot-new")
    from source_parsing.dom_iskusstv import extract_dom_iskusstv_urls

text = "Some text\n\nhttps://домискусств.рф/verified-link"
try:
    urls = extract_dom_iskusstv_urls(text)
    print(f"URLs: {urls}")
except Exception as e:
    print(f"Error: {e}")
