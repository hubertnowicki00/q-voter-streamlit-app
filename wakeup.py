from playwright.sync_api import sync_playwright, TimeoutError
import os
import time

url = os.getenv("STREAMLIT_APP_URL", "https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/")

def run():
    with sync_playwright() as p:
        print("Launching Playwright browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        try:
            button = page.get_by_role("button", name="Yes, get this app back up!")
            button.wait_for(state="visible", timeout=15000)
            button.click()
            time.sleep(5) 
        except TimeoutError:
            print("Error 1 - app awake")
        except Exception as e:
            print(f"Error 2 - browser fail: {e}")
        finally:
            browser.close()
if __name__ == "__main__":
    run()
