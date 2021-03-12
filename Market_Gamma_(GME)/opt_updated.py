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
