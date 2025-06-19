import os
import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

EMAIL = "shrish@provusinc.com"
PASSWORD = "#Shrish16424"

base_url = "https://docs.provusinc.com"

def fetch_docs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("🔐 Logging into Provus docs...")
        page.goto("https://docs.provusinc.com/Account/Login", timeout=60000)
        page.fill("#Username", EMAIL)
        page.fill("#password", PASSWORD)
        page.get_by_role("button", name="Login ").click()

        print("⏳ Waiting and navigating to 'Documentation'...")
        page.wait_for_url("https://docs.provusinc.com/", timeout=60000)
        time.sleep(2)
        page.get_by_role("link", name="Documentation").click()
        page.wait_for_url("https://docs.provusinc.com/docs/**", timeout=60000)
        print("✅ Clicked on Documentation link.")

        visited = set()
        to_visit = [page.url]
        all_text = ""

        while to_visit:
            url = to_visit.pop()
            if url in visited or not url.startswith(base_url + "/docs"):
                continue
            print(f"📄 Visiting: {url}")
            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle", timeout=15000)
                page.wait_for_timeout(2000)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                title_tag = soup.find("title")
                page_title = title_tag.text.strip() if title_tag else "Untitled"
                text = soup.get_text(separator="\n", strip=True)

                # Save with title and metadata marker
                all_text += f"\n\n========================\nURL: {url}\nTITLE: {page_title}\n{text}\n"

                visited.add(url)

                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/docs/"):
                        full_url = base_url + href
                        if full_url not in visited:
                            to_visit.append(full_url)

            except Exception as e:
                print(f"⚠️ Failed to fetch {url}: {e}")

        os.makedirs("data/docs", exist_ok=True)
        with open("data/docs/docs_corpus.txt", "w", encoding="utf-8") as f:
            f.write(all_text)

        print(f"\n✅ Crawled {len(visited)} pages and saved to docs_corpus.txt")
        browser.close()

if __name__ == "__main__":
    fetch_docs()