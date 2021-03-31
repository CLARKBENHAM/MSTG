#model impact of shorting offseting leveraged daily etfs
#break even around short borrow rates 7-10%
import numpy as np
import pandas as pd
# import beautifulsoup4
import lxml.html
import requests
# import requests_cache
import re
import math

from datetime import datetime
import time
import random

from collections import namedtuple, Counter
import pickle
import os
import sys
import win32com.client 
import matplotlib.pyplot as plt
github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\MSTG"
os.chdir(f"{github_dir}\\short_inverse")

#Note: Portfolios are not identical (long gets exposure w/ swaps & long nat gas)
long2x = pd.read_csv("UCO_hist.csv",
                     parse_dates=['Date'])
long2x.set_index("Date", inplace=True)
short2x = pd.read_csv("SCO_hist.csv",
                      parse_dates=['Date'])
short2x.set_index("Date", inplace=True)

#this varies a lot; need to improve
l_borrow_rate = pd.Series([0.0190 + 0.06]*len(long2x),
                          index = long2x.index)
#borrow rate till next price (so longer for weekends/holidays)
#last entry is 0; borrow rate for day 0 is cost to borrow till day 1
days = list((l_borrow_rate.index[:-1] - l_borrow_rate.index[1:]
             )//np.timedelta64(1, 'D')) + [0]
#rate given is APR; a periodic rate. cost till next day
long2x['Borrow'] = [(1+r)**(d/365) -1 for r,d in zip(days, l_borrow_rate)]

s_borrow_rate = pd.Series([0.0334 + 0.06]*len(short2x),
                          index = short2x.index)
days = list((s_borrow_rate.index[:-1] - s_borrow_rate.index[1:]
             )//np.timedelta64(1, 'D')) + [0]
short2x['Borrow'] = [(1+r)**(d/365) -1 for r,d in zip(days, s_borrow_rate)]

shale_start_date = datetime(year=2015, month=1, day=1)
neg_prices_date = datetime(year=2020, month=3, day=1)
long2x = long2x[(long2x.index >= shale_start_date) \
                & (long2x.index <= neg_prices_date)]
short2x= short2x[(short2x.index >= shale_start_date) \
                 & (short2x.index <= neg_prices_date)]

long2x.columns = [f"L_{i}" for i in long2x.columns ]
short2x.columns = [f"S_{i}" for i in short2x.columns]

df = long2x.join(short2x, how='inner')
df = df.iloc[::-1]#earliest dates first

#%%
initial = 100000
cash = initial
leverage = 2#total market positions n* cash buffer
acnt_value = []
#no overnight position
for index, row in df.iterrows():
    buffer = cash*leverage/2
    l_pos = -buffer//row['L_Open']
    s_pos = -buffer//row['S_Open']
    # cash = (buffer % row['L_open']) + (buffer % row['S_open']) #what not invested initially
    cash += l_pos *(row['L_Close'] - row['L_Open']) 
    cash += s_pos *(row['S_Close'] - row['S_Open'])
    cash += row['L_Borrow'] * l_pos * row['L_Open']  \
           + row['S_Borrow'] * s_pos * row['S_Open']
    if cash <= 0:
        cash = 0
    acnt_value += [cash]

acnt_value = pd.Series(acnt_value, index = df.index)
plt.plot(acnt_value, label="Close@ EOD")
plt.show()
print(acnt_value[-1]/initial, np.std(acnt_value.pct_change()))

cash = initial
acnt_value = []
#rebalance every morning: look at M2M and adjust size
l_pos, s_pos = 0,0
for index, row in df.iterrows():
    m2m = cash + l_pos*row['L_Open'] + s_pos*row['S_Open']#marked 2 market portfolio
    buffer = m2m*leverage/2
    l_t = -buffer//row['L_Open']
    s_t = -buffer//row['S_Open']
    cash += (l_pos - l_t)*row['L_Open'] \
          + (s_pos - s_t)*row['S_Open'] 
    l_pos, s_pos = l_t, s_t
    cash += row['L_Borrow'] * l_pos * row['L_Open']  \
           + row['S_Borrow'] * s_pos * row['S_Open']
    if cash <= 0:
        cash = 0
        # l_pos = 0
        # s_pos = 0
    acnt_value += [cash + l_pos*row['L_Close'] + s_pos*row['S_Close']]#evening m2m

acnt_value = pd.Series(acnt_value, index = df.index)
plt.plot(acnt_value, label = "Daily Morning Rebalance")
plt.legend()
plt.show()
print(acnt_value[-1]/initial, np.std(acnt_value.pct_change()))

#Seems like free money?