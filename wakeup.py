# KEEPING STREAMLIT APP AWAKE FOR IT NOT TO DISABLE ITSELF AFTER SOME TIME

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

url = os.getenv("https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/")

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)
try:
    driver.get(url)
    wait = WebDriverWait(driver, 20)

    button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Yes, get this app back up')]")))
    button.click()
    print("Wake button clicked.")
except Exception as e:
    print(f"App likely already awake or error: {e}")
finally:
    driver.quit()
