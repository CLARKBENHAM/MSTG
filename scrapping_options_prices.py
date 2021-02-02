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
#%%
base_url = "https://finance.yahoo.com/quote/GME/options"
r = requests.get(base_url)
root = lxml.html.fromstring(r.content)
root.make_links_absolute(base_url)
url_l = root.xpath('//*[@id="Col1-1-OptionContracts-Proxy"]/section/div/div[1]/select')
url_expiry_dates = url_l[0].value_options
urls = [url +f"?date={i}" for i in url_expiry_dates]

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
#%%
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

#%%
import os

os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\MSTG")
# if os.getcwd().split("\\")[-1] != "side_projects":
#     os.chdir(".\Desktop\side_projects")

df.to_pickle("current_option_prices")
df =  pd.read_pickle("current_option_prices")

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


def brownian(x0, n, sd = None, loc = None, dt = 1, out = None):
    "Computes the brownian motion with an average increase of loc and DAILY SD of sd. \
    dt is how many prices per day"
    if not out:
        out = np.ones(n)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    sd = sd or 0.16/(256**0.5)#average actualy market vol
    loc = loc or (1.08)**(1/256)
    r = norm.rvs(loc = loc, size=n, scale=sd*math.sqrt(dt))
    # This computes geometrix Brownian motion by forming the cumulative product of samples. 
    return x0*np.cumprod(r, axis=-1, out = out)

#approximating w/ 4 64 day quarters = 1 year
#GRIB!!, account for vol drag when converting daily returns to annual returns
class Market:#changed below
    def __init__(self, year_vol = 0.16, year_ret = 0.18, risk_free = 0.13, num_days = 256*30, price0 = 50):
        """Generates 30 yrs worth of prices.
        This assumes no dividends; how to incorporate?"""
        self.year_vol = year_vol #True expected volatility of ?log? returns
        self.year_ret = year_ret 
        self.risk_free = risk_free
        self.drisk_free = math.log(1 + risk_free)/256#risk free as the daily compounding rate
        self.d_vol = year_vol/math.sqrt(256)#daily vol
        
        self.ret_dist = norm(loc = self.drisk_free, scale = self.d_vol)#lognorm(loc = 0, scale = np.exp(self.drisk_free), s = self.d_vol)
        self.prices = np.exp(np.cumsum(self.ret_dist.rvs(num_days)))#brownian(price0, num_days, sd = self.year_vol, loc = self.year_ret)
        self.process = lambda x0, **kwargs: brownian(x0, num_days, *kwargs)

    #def get_prices; should yield arbitary prices and then remember them using self.process
    #add setters if want specifics returns, cor etc.  
        
class Stock:
    def __init__(self, prices = None, price0 = 50, exp_vol = 0.16, 
                 dividend = 0, market_cor = 1, exp_dvol=None, exp_ret = 0.08, 
                 name = None, const = False):
        """Takes in yearly vol as expected vol, which is distribution from which values drawn.
        True prices are the prices at the *CLOSE* of the given day. price0 is the value at CLOSE of 1st day
        The Expected and true Vol's may be off if given a series of prices
        """
        if exp_dvol is None:
            self.exp_vol = exp_vol#yearly
            self.exp_dvol = exp_vol/256**0.5# or exp_dvol or market.d_vol#daily expected vol
        else:
            self.exp_dvol = exp_dvol
            self.exp_vol = exp_dvol *256**0.5
        self.drisk_free = math.log(1.03)/256
        self.exp_dret = math.log(1 + exp_ret)/256
        #prices at end of day, assumes started just after dividend paid, 4 dividends/year
        self.dividend = dividend#same every quarter
        self.dividend_every = 64#paid on end of day
        num_days = 30*256
        self.ret_dist = norm(loc = self.exp_dret, scale = self.exp_dvol)#distribution of daily returns, lognorm(loc = 0, scale = np.exp(self.exp_dret), s = self.exp_dvol)
        if prices is None:
            self.price0 = price0
            if const:
                self.true_prices = np.full(num_days, price0)
            else:
                self.true_prices = price0*np.exp(np.cumsum(self.ret_dist.rvs(num_days))) - np.array([[0]*(64 - 1) + [dividend]]*30*4).flatten() 
#             self.true_prices = brownian(price0, n = 30*256, sd = self.exp_dvol, loc = self.exp_dret) \
#                                 - np.array([[0]*(64 - 1) + [dividend]]*30*4).flatten()  #subtract dividend on day paid 
            self.price0 = self.true_prices[0]
        else:        
            self.price0 = prices[0]
            self.true_prices = prices
#         self.market = market
        self.name = name or "".join([chr(i) for i in np.random.randint(ord('A'), ord('Z'),4)])
        
    def calc_value(self, spot = None, days_elapsed=0, iv = None):
        if spot is not None:
            return spot
        else:
            return self.true_prices[days_elapsed]
        
    def calc_greeks(self, iv = None, days_elapsed = 0, spot = None):
        delta = 1
        gamma = 0
        vega = 0
        theta = 0#don't make money with day's changing, make money with price changing
        rho = 0#unknown
        return {"Delta":delta, "Gamma":gamma, "Vega":vega/100, "Theta":theta, "Rho":rho/10000}
        
class Euro_Option:
    def __init__(self, strike = 100, premium = 0, lifetime = 256, tp = 'call', asset = None, assigned_iv = None):
        """Defines a European option;
         if asset None has constant prices as strike with a default expected vol
         assigned_iv: IV as determined by an external model, not nessisarily consistent with BS projections
         """
        self.strike = strike
        self.premium = premium
        self.lifetime = lifetime
        assert(tp in ('put', 'call'))
        self.tp = tp#type
        self.asset = asset or Stock(prices = np.full(lifetime + 1, strike))
        self.assigned_iv = assigned_iv

        
    def calc_greeks(self, iv = None, days_elapsed = 0, spot = None, ignore = None):#ignore is for extra, unnessisary
        """Calculates the greeks using the BS method.  Using **DAILY** volatility
        if the implied vol isn't given it takes expected daily for the underlying.
        Computes vol as based on 1pt change, then divides by 100; Rho divide by 10k.
        ignores option premium"""
        days_left = self.lifetime - days_elapsed
        spot = spot or self.asset.true_prices[days_elapsed]#days elapsed > 1 year?
        if days_left == 0:#issue of dividing by 0
            #calculated as if market just opened, ~6 hours before settle
            if self.tp == 'call':
                #delta,gamma are calculated based on whole dolar moves
                #theta should be changed to remaining premium?
                out = (int(spot-1 >= self.strike), int(abs(spot-self.strike) <= 2), 0, -1, 0)
            elif self.tp == 'put':
                out = (-1*int(spot <= self.strike-1), int(abs(spot-self.strike) <= 2), 0, -1, 0)
            return {k:v for k,v in zip(["Delta","Gamma","Vega","Theta", "Rho"], out)}
        if days_left < 0:
            if self.tp == 'call':
                out = (int(spot > self.strike), 0, 0, 0, 0)
            elif self.tp == 'put':
                out = (-1*int(spot < self.strike), 0, 0, 0, 0)
            return {k:v for k,v in zip(["Delta","Gamma","Vega","Theta", "Rho"], out)}

        drisk_free = self.asset.drisk_free
        if not iv:
            iv = self.asset.exp_dvol
        variance = iv**2
        d1 = (1 / (variance*days_left)**0.5) * (math.log(spot/self.strike) + (drisk_free + variance/2)*days_left)
        d2 = d1 - (variance * days_left)**0.5
        pv_k = self.strike * math.exp(-drisk_free*days_left)
        gamma = norm.pdf(d1)/(spot*(iv*days_left)**0.5)#note uses PDF, not CDF as for d1
        vega = spot*norm.pdf(d1)*(days_left)**0.5
        theta = -spot*norm.cdf(d1)*iv/(2*days_left**0.5)
        if self.tp == 'call':
            delta = norm.cdf(d1)
            theta -= drisk_free*pv_k*norm.cdf(d2)#updates theta
            rho = days_left*pv_k*norm.cdf(d2)
        elif self.tp == 'put':
            delta = norm.cdf(d1) - 1
            theta += drisk_free*pv_k*norm.cdf(-d2)
            rho = -days_left*pv_k*norm.cdf(-d2)
        return {"Delta":delta, "Gamma":gamma, "Vega":vega/100, "Theta":theta, "Rho":rho/10000}#scale down the values
        
    
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
approx_value = lambda row: ((row.Expiry_date-current_date)/2*pi)**0.5 * row.Implied_Volatility*row.Strike


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
