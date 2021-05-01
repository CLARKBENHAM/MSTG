from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import collections
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import pyautogui as pygu
import sys
import time
import re

import random

def expand_shadow_element(element):
  shadow_root = driver.execute_script('return arguments[0].shadowRoot', element)
  return shadow_root

def click_print():
    root1 = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'print-preview-app')))
    shadow_root1 = expand_shadow_element(root1)

    root2 = shadow_root1.find_element_by_css_selector('print-preview-sidebar')
    shadow_root2 = expand_shadow_element(root2)
    
    root3 = shadow_root2.find_element_by_css_selector('print-preview-button-strip')
    shadow_root3 = expand_shadow_element(root3)
    
    print_button = shadow_root3.find_element_by_css_selector("cr-button.action-button")
    print_button.click()

chromedriver_path = 'C:\Program Files\chromedriver_win32\chromedriver'
driver =  webdriver.Chrome(executable_path = chromedriver_path)
wait = WebDriverWait(driver, 10)
driver.get("https://mail.google.com/mail/u/1/?shva=1#label/IFR")
driver.find_element_by_xpath("//input[@type='email']").send_keys("cb5ye@virginia.edu\n")
driver.find_element_by_xpath("//input[@value='Log In']").send_keys("\n\n")
pygu.press('enter')

#%%Login and navigate to the sign in page
temp = ''
while 'y' not in temp.lower():
    temp = input("Enter Y once have nagivated to the IFR label: ")
#%%

pygu.PAUSE = 0.5
#change class if reload
table = driver.find_elements_by_xpath('//tbody/tr[@class="zA zE"]')
cnt = 0
# time.sleep(0.5)#give me time to change tabs
while True:
    while table[0].text == '':
        del table[0]
    while True:
        try:
            table[0].click()
            break
        except:
            del table[0]
    #%get subject, date
    while True:
        try:
            subject = wait.until(
                        EC.presence_of_element_located((By.XPATH,
                                                   "//h2[@class='hP']")))
            subject = subject.text.replace("/", "")
            recieved_date = driver.find_element_by_xpath("//span[@class='g3']").text
            day = re.findall("(Dec \d\d), 2019, \d{1,2}\:\d\d [AP]M",
                             recieved_date)[-1]
            break
        except Exception as e:
            time.sleep(random.random()*2)
            print(e)
    #% prints from email 
    print_icon = wait.until(
                    EC.element_to_be_clickable((By.XPATH,
                                               "//img[@alt='Print all']")))
    driver.execute_script("arguments[0].click();", print_icon)
    while len(driver.window_handles) < 3:
        time.sleep(0.1)
    driver.switch_to.window(driver.window_handles[2])
    click_print()
    #%
    while pygu.locateOnScreen('print_footer.PNG') is None:
        time.sleep(0.1)
    footer = pygu.locateOnScreen('print_footer.PNG')
    pygu.moveTo(footer)
    pygu.click()
    pygu.typewrite(day + "- " + subject)
    pygu.press('enter')
    #%
    driver.switch_to.window(driver.window_handles[1])
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    driver.back()
    table = wait.until(
        EC.presence_of_element_located((By.XPATH,
                                   '//tbody/tr[@class="zA zE"]')))
    table = driver.find_elements_by_xpath('//tbody/tr[@class="zA zE"]')
    cnt += 1#unnesisary
    if cnt > 63:
        break
#%%


#%%




