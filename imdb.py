# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:27:49 2020

@author: rahul
"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome(ChromeDriverManager().install())
from bs4 import BeautifulSoup as bs
page = "http://www.imdb.com/title/tt6294822/reviews?ref_=tt_urv"
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
driver.get(page)
import time
reviews = []
i=1
while (i>0):
    try:
        button = driver.find_element_by_xpath('//*[@id="load-more-trigger"]')
        button.click()
        time.sleep(5)
        ps = driver.page_source
        soup=bs(ps,"html.parser")
        rev = soup.findAll("div",attrs={"class","text show-more__control"})
        reviews.extend(rev)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break
        


len(reviews)
len(list(set(reviews)))


import re 
cleaned_reviews= re.sub('[0-9" "]+',' ', reviews)
cleaned_reviews= re.sub("[^A-Za-z" "]+"," ",reviews).lower()

f = open("reviews.txt","w")
f.write(cleaned_reviews)
f.close()

with open("The_Post.text","w") as fp:
    fp.write(str(reviews))



len(soup.find_all("p"))