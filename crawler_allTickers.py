#!/usr/bin/python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io
import sys

def preprocess(cap):
    m = {'K': 3, 'M': 6, 'B': 9, 'T': 12}
    if type(cap) == str:
        cap = cap.strip().replace('$','')
        if cap[-1] in m:
            amount = float(cap[:-1])*float(pow(10,m[cap[-1]]))
            return amount
        else:
            return float(cap)
    else:
        return np.NaN
    


def getTickers(percent):
    tot_data = None
    try:
        for exchange in ["NASDAQ", "NYSE", "AMEX"]:
            url="https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange="+exchange.lower()+"&render=download"
            s=requests.get(url).content
            c=pd.read_csv(io.StringIO(s.decode('utf-8')))
            # print(c)
            c['Exchange'] = exchange
            if tot_data is None:
                tot_data = c
            else:
                tot_data = pd.concat([tot_data, c])
    except:
        print('ERROR')
        pass
    columns = ['Symbol', 'Name', 'Exchange', 'MarketCap']
    tot_data = tot_data[columns]
    tot_data = tot_data.dropna().reset_index(drop=True)
    print('Number of Samples:',tot_data.shape[0])
    tot_data['MarketCap'] = tot_data['MarketCap'].apply(lambda x : preprocess(x))
    markets_caps = list(tot_data['MarketCap'])
    tot_data[tot_data['MarketCap']<=np.percentile(markets_caps, 99.9)].shape
    tot_data.to_csv('input/tickerList.csv')




def main():
    arg = sys.argv[1]
    s = getTickers(float(arg)) # keep the top N% market-cap companies


if __name__ == "__main__":
    main()