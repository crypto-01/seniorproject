
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium_test_pages import*
import sys
driverPath = "/usr/bin/chromedriver"
option = Options()
option.add_argument("--no-sandbox")
#option.add_argument("--disable-dev-shm-usage")
option.add_argument("--disable-extensions")
#option.add_argument("--headless")
option.add_argument("--mute-audio")
driver = webdriver.Chrome(driverPath ,options=option)
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
website_url = "http://172.22.87.39:5000/"
home_page = HomePage(driver,website_url)
test_page = TestPage(driver,website_url)
home_page.testLinks()
test_page.test_input_and_responce()
driver.quit()
