#!/usr/bin/env python3
"""Local test script for Festival Parser - runs Playwright locally without Kaggle.

Usage:
    python scripts/test_festival_parser_local.py https://zimafestkld.ru/

Requires:
    pip install playwright beautifulsoup4
    playwright install chromium
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
KAGGLE_SRC = PROJECT_ROOT / "kaggle" / "UniversalFestivalParser" / "src"
sys.path.insert(0, str(KAGGLE_SRC))


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_festival_parser_local.py <URL>")
        print("Example: python scripts/test_festival_parser_local.py https://zimafestkld.ru/")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = Path("test_parser_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Testing Festival Parser locally")
    print(f"📍 URL: {url}")
    print(f"📂 Output: {output_dir.absolute()}")
    print("-" * 50)
    
    # Phase 1: RENDER
    print("\n🔵 Phase 1: RENDER (Playwright)")
    try:
        from render import render_page
        render_result = await render_page(
            url=url,
            output_dir=output_dir,
            timeout_ms=30000,
        )
        
        if render_result.get("success"):
            print(f"   ✅ HTML saved: {render_result.get('html_size', 0):,} bytes")
            print(f"   ✅ Screenshot saved: {render_result.get('screenshot_path', 'N/A')}")
            print(f"   ✅ Title: {render_result.get('title', 'N/A')}")
        else:
            print(f"   ❌ Error: {render_result.get('error', 'Unknown')}")
            sys.exit(1)
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Run: pip install playwright && playwright install chromium")
        sys.exit(1)
    
    # Phase 2: DISTILL
    print("\n🟡 Phase 2: DISTILL (HTML cleaning)")
    try:
        from distill import distill_html
        
        html_path = render_result.get("html_path")
        if not html_path or not Path(html_path).exists():
            print(f"   ❌ HTML file not found: {html_path}")
            sys.exit(1)
        
        # distill_html expects (html_path, output_dir)
        distilled = distill_html(html_path, output_dir)
        
        print(f"   ✅ Main text: {len(distilled.get('main_text', '')):,} chars")
        print(f"   ✅ Links found: {len(distilled.get('links', []))}")
        print(f"   ✅ Images found: {len(distilled.get('images', []))}")
        print(f"   ✅ Distilled saved: {distilled.get('distilled_path', 'N/A')}")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Run: pip install beautifulsoup4")
        sys.exit(1)
    
    # Phase 3: REASON (requires GOOGLE_API_KEY)
    print("\n🔴 Phase 3: REASON (Gemma 3-27B)")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("   ⚠️ GOOGLE_API_KEY not set - skipping LLM extraction")
        print("   💡 Set GOOGLE_API_KEY to enable LLM extraction")
        print("\n📊 Summary: RENDER + DISTILL completed successfully!")
        print(f"   Check output in: {output_dir.absolute()}")
        return
    
    try:
        from reason import reason_with_gemma
        from llm_logger import LLMLogger
        
        llm_logger = LLMLogger("local-test")
        llm_context = distilled.get("main_text", "")[:8000]  # Limit context size
        
        print(f"   📤 Sending {len(llm_context):,} chars to Gemma...")
        uds, error = await reason_with_gemma(
            distilled_content=llm_context,
            api_key=api_key,
            llm_logger=llm_logger,
        )
        
        if error:
            print(f"   ❌ LLM Error: {error}")
        else:
            uds_path = output_dir / "uds.json"
            uds_path.write_text(json.dumps(uds, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   ✅ UDS extracted and saved: {uds_path}")
            
            # Show extracted data summary
            festival = uds.get("festival", {})
            print(f"\n📋 Extracted Festival Data:")
            print(f"   Name: {festival.get('title_short', 'N/A')}")
            print(f"   Full Name: {festival.get('title_full', 'N/A')}")
            print(f"   Dates: {festival.get('dates', {}).get('start', 'N/A')} - {festival.get('dates', {}).get('end', 'N/A')}")
            print(f"   Events in program: {len(uds.get('program', []))}")
            print(f"   Venues: {len(uds.get('venues', []))}")
            print(f"   Images: {len(uds.get('images_festival', []))}")
        
        # Save LLM log
        llm_log_path = output_dir / "llm_log.json"
        llm_logger.save(str(llm_log_path))
        print(f"   ✅ LLM log saved: {llm_log_path}")
        
    except Exception as e:
        print(f"   ❌ Reason error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 Local test completed!")
    print(f"📂 All outputs in: {output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
