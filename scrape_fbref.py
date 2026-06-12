import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from io import StringIO
import time

async def scrape_url(url, output_file):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Pretend to be a real browser
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="domcontentloaded")
        
        # Give it a few seconds to let any JS load or Cloudflare pass
        await asyncio.sleep(5)
        
        content = await page.content()
        if "cloudflare" in content.lower() or "just a moment" in content.lower():
            print("Blocked by Cloudflare!")
        else:
            print("Successfully bypassed Cloudflare!")
            
        try:
            # We look for the 'stats_standard' table
            table_html = await page.locator("table#stats_standard").evaluate("el => el.outerHTML")
            dfs = pd.read_html(StringIO(table_html))
            if dfs:
                df = dfs[0]
                # FBREF tables often have multi-level headers. Let's flatten them.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                df.to_csv(output_file, index=False)
                print(f"Saved {len(df)} rows to {output_file}")
            else:
                print("No table found!")
        except Exception as e:
            print(f"Failed to find or parse table: {e}")
            
        await browser.close()

if __name__ == "__main__":
    url = "https://fbref.com/en/comps/8/stats/Champions-League-Stats"
    asyncio.run(scrape_url(url, "data/raw/champions_league.csv"))
