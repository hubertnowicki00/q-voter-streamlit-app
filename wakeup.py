# KEEPING STREAMLIT APP AWAKE FOR IT NOT TO DISABLE ITSELF AFTER SOME TIME

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time

import traceback

url = os.getenv("STREAMLIT_APP_URL", "https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/")

chrome_options = Options()
chrome_options.binary_location = "/usr/bin/chromium-browser"
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--remote-debugging-port=922")
try:
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    wait = WebDriverWait(driver, 15)
    button_xpath = "//button[contains(text(), 'Get the app back')]"
    button = wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
    button.click()
    time.sleep(10)
except TimeoutException:
    print("Error 1 - app awake")
except Exception as e:
    print(f"\n Error 2 - browser fail : {type(e).__name__} - {str(e)}")
    print("Error log \n")
    traceback.print_exc()
    print("\n END")
finally:
    try:
        driver.quit()
    except Exception:
        pass
