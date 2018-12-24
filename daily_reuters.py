#!/usr/bin/python
import re

import urllib3
import csv
import os
import sys
import time
import datetime
import json

import numpy as np
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()



# iterate all dates
#   iterate all tickers
#     repeatDowdload
#       save to ./input/data/news_date.csv


api_key = os.getenv("NEWSAPI_APIKEY")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class news_Reuters:
    def __init__(self):
        fin = open('./input/tickerList.csv')

        filterList = set()
        try: # this is used when we restart a task
            fList = open('./input/finished.reuters')
            for l in fList:
                filterList.add(l.strip())
        except: pass

        # https://uk.reuters.com/info/disclaimer
        # e.g. http://www.reuters.com/finance/stocks/company-news/BIDU.O?date=09262017
        self.suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
        self.repeat_times = 1
        self.sleep_times = 0
        self.newsapi = NewsApiClient(api_key=api_key)
        self.iterate_by_day(fin, filterList)


    def iterate_by_day(self, fin, filterList):
        dateList = self.dateGenerator(1) # look back on the past X days
        for timestamp in dateList: # iterate all possible days
            print("%s%s%s" % (''.join(['-'] * 50), timestamp, ''.join(['-'] * 50)))
            self.iterate_by_ticker(fin, filterList, timestamp)

    def iterate_by_ticker(self, fin, filterList, timestamp):
        for line in fin: # iterate all possible tickers
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            if ticker in filterList: continue
            print("%s - %s - %s - %s" % (ticker, name, exchange, MarketCap))
            self.repeatDownload(ticker, line, timestamp, exchange)

    def repeatDownload(self, ticker, line, timestamp, exchange): 
        fout = open('./input/dates/news_' + timestamp + '.csv', 'a+')
        all_articles = self.newsapi.get_everything(q=ticker,
                                              language='en',
                                              from_param=timestamp,
                                              to=timestamp,
                                              sort_by='relevancy')
        if all_articles['articles']:
            article =  all_articles['articles'][0]
            title = article['title']
            content = article['content']
            if(content == 'None') {
                content = article['description']
            }
            print('------------------------------------News---------------------------------------')
            print('Title:', article['title'])
            fout.write(','.join([ticker, line[1], timestamp, title, body, news_type]).encode('utf-8') + '\n')
        fout.close()
        return 0
    
    def dateGenerator(self, numdays): # generate N days until now
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y-%m-%d")
        return date_list

def main():
    news_Reuters()

if __name__ == "__main__":
    main()
