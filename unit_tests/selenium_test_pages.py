from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import json
import time
import sys
import urllib.request
import requests
import os
from tqdm import tqdm
import concurrent.futures
import re


class HomePage(object):
    home = (By.ID,"home-page")
    test = (By.ID,"test-page")
    dataset = (By.ID,"dataset-page")
    project=(By.ID,"project-page")
    team = (By.ID,"team-page")

    def __init__(self,driver: webdriver.Chrome,url) ->None:
        self.driver = driver
        self.pageUrl = url

    def testLinks(self):
        self.driver.get(self.pageUrl)
        #testing home link
        home_tag = self.driver.find_element(*(self.home))
        home_tag_url = home_tag.get_attribute("href")
        self.driver.get(home_tag_url)
        time.sleep(2)
        assert self.driver.current_url == self.pageUrl + "#home","incorrect home url"
        time.sleep(2)
        #testing test link
        test_tag = self.driver.find_element(*(self.test))
        test_tag_url = test_tag.get_attribute("href")
        self.driver.get(test_tag_url)
        time.sleep(2)
        assert self.driver.current_url == self.pageUrl + "#test","incorrect test url"
        time.sleep(2)
        """
        #testing dataset
        dataset_tag = self.driver.find_element(*(self.dataset))
        dataset_tag_url = dataset_tag.get_attribute("href")
        self.driver.get(dataset_tag_url)
        time.sleep(2)
        assert self.driver.current_url == self.pageUrl + "#dataset","incorrect dataset url"
        time.sleep(2)
        """
        #testing project
        project_tag = self.driver.find_element(*(self.project))
        project_tag_url = project_tag.get_attribute("href")
        self.driver.get(project_tag_url)
        assert self.driver.current_url == self.pageUrl + "#project","incorrect project url"
        time.sleep(2)
        #testing team
        team_tag = self.driver.find_element(*(self.team))
        team_tag_url = team_tag.get_attribute("href")
        self.driver.get(team_tag_url)
        assert self.driver.current_url == self.pageUrl + "#team","incorrect team url"
        print("All links have successfully passed")
        time.sleep(2)
class TestPage(object):
    test = (By.ID,"test-page")
    tweet = (By.ID,"tweet")
    test_button = (By.ID,"testButton")
    question = (By.ID,"question")
    output = (By.ID,"output")
    tweet_text = "The best fps is call of duty"
    question_text = "What is the best fps?"

    def __init__(self,driver: webdriver.Chrome,url) ->None:
        self.driver = driver
        self.pageUrl = url

    def test_input_and_responce(self):
        self.driver.get(self.pageUrl)
        test_tag = self.driver.find_element(*(self.test))
        test_tag_url = test_tag.get_attribute("href")
        self.driver.get(test_tag_url)
        tweet_tag = self.driver.find_element(*(self.tweet))
        question_tag = self.driver.find_element(*(self.question))
        test_button_tag = self.driver.find_element(*(self.test_button))
        output_tag = self.driver.find_element(*(self.output))
        self.driver.execute_script("arguments[0].scrollIntoView();", test_button_tag)
        time.sleep(1)
        tweet_tag.send_keys(self.tweet_text)
        question_tag.send_keys(self.question_text)
        test_button_tag.submit()
        time.sleep(5)
        output_text = output_tag.text
        assert output_text == "call of duty", "incorrect response"
        print("Input and response successfully pass")
        time.sleep(3)

