import numpy as np
import pandas as pd
# import beautifulsoup4
import lxml.html
import requests
# import requests_cache
import re

from datetime import datetime
import time
import random

from collections import namedtuple
import pickle
import os
import sys
github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\MSTG"
#%%
if __name__ == "__main__":
    base_url = "https://finance.yahoo.com/quote/GME/options"
    r = requests.get(base_url)
    root = lxml.html.fromstring(r.content)
    root.make_links_absolute(base_url)
    url_l = root.xpath('//*[@id="Col1-1-OptionContracts-Proxy"]/section/div/div[1]/select')
    url_expiry_dates = url_l[0].value_options
    urls = [base_url +f"?date={i}" for i in url_expiry_dates]
    
    header = root.xpath("(//thead)[1]/tr/th/span")
    header = [i.text.replace(" ", "_").replace("%", "Per") for i in header]
    assert header == ['Contract_Name',
             'Last_Trade_Date',
             'Strike',
             'Last_Price',
             'Bid',
             'Ask',
             'Change',
             'Per_Change',
             'Volume',
             'Open_Interest',
             'Implied_Volatility'], \
            "Columns have changed"
    row_tup = namedtuple("rows", header)
    
    d = {u:[] for u in urls}
    df = pd.DataFrame()
    #%
    def proc_row(row):
            """header = ('Contract_Name',
                 'Last_Trade_Date',
                 'Strike',
                 'Last_Price',
                 'Bid',
                 'Ask',
                 'Change',
                 'Per_Change',
                 'Volume',
                 'Open_Interest',
                 'Implied_Volatility')"""
            out = [i.text for i in row.xpath("td")]
            # Contract_name, Strike
            out[0], out[2] = [i.text for i in row.xpath("td/a")]
            for i in (2,3,4,5, 8,9,10):
                if out[i] == "-":
                    out[i] = 0
                else:
                    out[i] = float(re.sub("\%|\,", "", out[i]))
            #Change, %change
            out[6], out[7] = [float(re.sub("\%|\,", "",i.text))
                              if i.text != '-' else 0 
                              for i in row.xpath("td/span")]
            #Last Trade 
            #but hours aren't 0 padded!?
            out[1] = datetime.strptime(out[1], "%Y-%m-%d %I:%M%p EST")
            return row_tup._make(out)
    
    for url in urls[5:]:
        r2 = requests.get(url)
        d[url] += [r2]
        option_page = lxml.html.fromstring(r2.content)
        calls, puts = option_page.xpath("//tbody")    
        call_objs = [proc_row(row) for row in calls.xpath("tr")]
        put_objs = [proc_row(row) for row in puts.xpath("tr")]
        d[url] += [call_objs, put_objs]
        df = df.append(put_objs).append(call_objs)
        time.sleep(random.random()*3)
    
    print(df)
    # add a close date, from contract name
    def _close_date(name):
        return datetime(year = int('20' + name[3:5]), 
                        month = int(name[5:7]), 
                        day = int(name[7:9])) 
    df['Expiry_Date'] = [_close_date(i[0]) for i in df.values]
    df['Is_Call'] = ['C' in i[0] for i in df.values]
    
    #%
    os.chdir(f"{github_dir}\\Market_Gamma_(GME)")
    # if os.getcwd().split("\\")[-1] != "side_projects":
    #     os.chdir(".\Desktop\side_projects")
    
    df.to_pickle(f"current_option_prices {datetime.today().strftime('%b,%d %Y')}")
    # df =  pd.read_pickle(f"{github_dir}\\Market_Gamma_(GME)\\current_option_prices")
    sys.exit()

#%%:
import numpy as np
import pandas as pd
import math
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt
from matplotlib import animation
import random 
from pprint import pprint      
import itertools
from IPython import get_ipython
ipython = get_ipython()
ipython.magic(f"run -n {github_dir}\option_classes.ipynb")
 
GME = Stock(price0 = 225, exp_vol = 4)#400%
current_date = datetime.now()        
#want to find market's position as of Now, not when the trades were made. 
#Don't care about the IV from the prices, only impact on delta hedging
#Ask > bid >= 0; 0 prices bad
ask_contracts = [Euro_Option(strike = row.Strike,
                            premium = row.Ask,
                            lifetime = (row.Expiry_Date - current_date).days,
                            tp = 'call' if row.Is_Call else 'put',
                            asset = GME,
                            assigned_iv = row.Implied_Volatility)
                 for row in df.itertuples()]
#%% check vs. Schwab's greeks
check = [Euro_Option(strike = row.Strike,
                            premium = row.Ask,
                            lifetime = (row.Expiry_Date - current_date).days,
                            tp = 'call' if row.Is_Call else 'put',
                            asset = GME,
                            assigned_iv = row.Implied_Volatility)
         for row in df[(df['Strike'] == 105.0) & ~df['Is_Call']].itertuples()]
put105expMarch05 = check[5]
check[5].calc_greeks(iv=check[5].assigned_iv)
schwab = {'Delta': -0.2285,
 'Gamma': 0.002,
 'Vega': 0.1049,
 'Theta': -0.7505}
#%%
GME.price0 = 225
shares_per_contract = 100
daily_delta_change = shares_per_contract*sum(
                        [oi*(op.calc_greeks(iv = op.assigned_iv)['Delta'] \
                              - op.calc_greeks(iv = op.assigned_iv, days_elapsed = 7)['Delta'])
                          for op, iv, oi in zip(ask_contracts,
                                                df['Open_Interest'])])

#%%
GME.price0 = 100
option_delta100 = [shares_per_contract* sum(
                    [oi*(op.calc_greeks(iv = iv, days_elapsed=i)['Delta'])
                          for op, iv, oi in zip(ask_contracts,
                                                df['Open_Interest'])])
                for i in range(31)]
GME.price0 = 200
option_delta200 = [shares_per_contract* sum(
                    [oi*(op.calc_greeks(iv = iv, days_elapsed=i)['Delta'])
                          for op, iv, oi in zip(ask_contracts,
                                                df['Open_Interest'])])
                for i in range(31)]
GME.price0 = 300
option_delta300 = [shares_per_contract* sum(
                    [oi*(op.calc_greeks(iv = iv, days_elapsed=i)['Delta'])
                          for op, iv, oi in zip(ask_contracts,
                                                df['Open_Interest'])])
                for i in range(31)]

fig, ax = plt.subplots()
fig.suptitle("Delta Decay at Strikes ('Charm' or dDelta dTime)")
ax.plot(list(range(31)), option_delta100, label = "price = $100")
ax.set_xlabel("Days elapsed")
ax.set_ylabel("Total Market Delta")
ax.plot(list(range(31)), option_delta200, label = "price = $200")
ax.plot(list(range(31)), option_delta300, label = "price = $300")
plt.legend(loc=1)
plt.show()

#%%
import matplotlib.pyplot as plt

gross_contracts = df['Open_Interest'].sum()
oi_groups = df.groupby(["Is_Call", 'Expiry_Date'])['Open_Interest'].sum()

fig, ax = plt.subplots()
ax.set_title("Total Open Contracts Over Time")
ax.plot(oi_groups[True], label= "Calls")
ax.plot(oi_groups[False], label = "Puts")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Total Option Contracts")
fig.show()

#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

calls = df[df['Is_Call']]
puts = df[~df['Is_Call']]
strike_groupings_step_sz = 10
strike_groupings_max = 600
df['Strike_Groupings'] = [strike_groupings_step_sz*min(i//strike_groupings_step_sz,
                        strike_groupings_max//strike_groupings_step_sz) 
                    for i in df['Strike']]
table = df[df['Open_Interest']>2].groupby(["Is_Call", 'Expiry_Date','Strike_Groupings'])['Open_Interest'].sum()
call_t, put_t = table[True], table[False]

plt.rcParams.update({'font.size': 18})

fig, ((axt,caxt), (axb, caxb)) = plt.subplots(2,2, 
                                              gridspec_kw=dict(width_ratios=[30,1]),
                                              constrained_layout=True)
axt.set_title("Open Call Contracts by Strike Over Time")
axt.set_xlabel("Date")
axt.set_ylabel("Strikes")
# axt.set_yscale('log')
# divider = make_axes_locatable(axt2)
# cax = divider.append_axes("right", size="5%", pad=0.05)
imt = axt.scatter(call_t.index.get_level_values("Expiry_Date"),
                 call_t.index.get_level_values("Strike_Groupings"), 
                c = call_t.values, 
                norm = colors.LogNorm(),
                label= "Calls")
plt.colorbar(imt, cax=caxt, label = "Call Open Interest by Strike", format = "%1.1f", ax = axt)

axt2 = axt.twinx()
axt2.plot(oi_groups[True], 
          linewidth=3,
          c='r',
          label= "Calls")
axt2.set_ylabel("Total Call Contracts")

axb.set_title("Open Put Contracts by Strike Over Time")
axb.set_xlabel("Date")
axb.set_ylabel("Strikes")
imb = axb.scatter(put_t.index.get_level_values("Expiry_Date"),
                 put_t.index.get_level_values("Strike_Groupings"), 
                c = put_t.values, 
                norm = colors.LogNorm(),
                label= "Calls")
plt.colorbar(imb, cax=caxb, label = "Put Open Interest by Strike", format = "%1.1f", ax = axb)

axb2 = axb.twinx()
axb2.plot(oi_groups[False], 
          linewidth=3,
          c='r',
          label= "Puts")
axb2.set_ylabel("Total Put Contracts")

plt.show()
#%%










#%% Scrap
approx_value = lambda row: ((row.Expiry_date-current_date)/2*math.pi)**0.5 * row.Implied_Volatility*row.Strike


bid_contracts = [Euro_Option(strike = row.Strike,
                            premium = row.Bid,
                            lifetime = (row.Expiry_Date - current_date).days,
                            tp = 'call' if row.Is_Call else 'put',
                            asset = GME)
                 for row in df.itertuples()]


def dfs(e):
    if isinstance(e, list):
        return [dfs(e) for i in e]
    q = [e]
    out = []
    while len(q) > 0:
        e = e.pop(-1)
        try:
            kids = header[0].xpath("/*")
            q += kids
        except:
            out += [e]
    return out
