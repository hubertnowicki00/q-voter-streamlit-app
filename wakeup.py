# KEEPING STREAMLIT APP AWAKE FOR IT NOT TO DISABLE ITSELF AFTER SOME TIME

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time

url = os.getenv("STREAMLIT_APP_URL", "https://q-voter-app-app-54xgh5gxxrr4pukoxwfmwe.streamlit.app/")

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
try:
    wait = WebDriverWait(driver, 15)
    button_xpath = "//button[contains(text(), 'Get the app back')]"
    button = wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
    button.click()
    time.sleep(10)
except TimeoutException:
    print("Error")
except Exception as e:
    print("Error")
finally:
    driver.quit()
    
