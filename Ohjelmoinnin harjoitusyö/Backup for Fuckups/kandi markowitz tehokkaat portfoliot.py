# laita #%% tähän, kun haluat testata ja saada kuviot
#%%

import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
import scipy.optimize as sco
import quantstats as qs
from scipy.stats import shapiro

def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate): #no need for for-loop, becayse searching for only one set of weights
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    return -portf_sharpe_ratio

def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
    efficient_portfolios = []
    n_assets = len(avg_returns)
    args = (avg_returns, cov_mat) #arguments are historical returns and covariance matrix
    bounds = tuple((0, 1) for asset in range(n_assets)) #weight of asset must be between 0-1, no shorting allowed
    initial_guess = n_assets * [1. / n_assets, ] #makes optimization faster by using initial guesses

    for ret in rtns_range:
        constraints = ({"type": "eq",
                        "fun": lambda x: get_portf_rtn(x, avg_rtns) #expected portfolio return must be equal to provided value
                        - ret},
                        {"type": "eq",
                        "fun": lambda x: np.sum(x) - 1}) #sum of asset weights must be equal to 1
        efficient_portfolio = sco.minimize(get_portf_vol,
                                            initial_guess,
                                            args = args,
                                            method = "SLSQP", #sequential least-squares programming, solves optimization problems
                                            constraints = constraints,
                                            bounds = bounds)
        efficient_portfolios.append(efficient_portfolio)
    
    return efficient_portfolios

risky_assets = ["^SP500-35", #health care
                "^SP500-40", #financials
                "^SP500-45", #IT
                "^SP500-20", #industrials
                "^SP500-55", #utilities
                "^SP500-25", #consumer discretionary
                "^SP500-30", #consumer staples
                "^SP500-15"] #materials
start_date = "2012-01-01"
end_date = "2019-12-31"
n_assets = len(risky_assets)

prices_df = yf.download(risky_assets, 
                        start = start_date, 
                        end = end_date, 
                        auto_adjust = False)

prices_df.rename(columns = {"^SP500-15":"Materials"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-20":"Industrials"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-25":"Consumer Discretionary"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-30":"Consumer Staples"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-35":"Health Care"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-40":"Financials"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-45":"IT"}, inplace = True) #rename column
prices_df.rename(columns = {"^SP500-55":"Utilities"}, inplace = True) #rename column

returns = prices_df["Adj Close"].pct_change().dropna() #simple asset returns

#returns.to_excel(r'C:\Users\Otto\Desktop\Kandi\returnshs.xlsx', index=False)

#returns.plot(title = "Daily returns of sub-indexes")

#portfolio_weights = [0, 0, 0, 0.9173, 0, 0, 0, 0.0827] #MV portfolio weights

portfolio_weights = n_assets * [1 / n_assets] #equally-weighted

portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index = returns.index) #np.dot = matrix multiplication, pd.series = stored data as pandas series
#returns.plot(kind='hist', bins = 20, column = "IT")
#portfolio_returns.plot(kind='hist', bins = 50,) #histogrammi tuotoista bins kertoo moneen ryhmään jaetaan
#print(shapiro(returns)) #testi normaalijakaumaa varten
#pf.create_simple_tear_sheet(portfolio_returns) #create data and all the fancy figures
#pf.create_simple_tear_sheet(returns)
kuvailu = returns.describe()
kuvailu.to_excel(r'C:\Users\Otto\Desktop\Kandi\kuvailu.xlsx', index=False)


avg_returns = returns.mean() * 252 #annualise average returns
cov_mat = returns.cov() * 252 #covariance-matrix and annualise it
corr_mat = returns.corr()

#print(avg_returns)

#corr_mat.head(8) #korrelaatiomatriisin printtaus

#cov_mat.head(8) #kovarianssimatriisin printtaus, ekat 8 riviä

#qs.plots.snapshot(portfolio_returns, title = "Minimum variance optimization performance", grayscale = True) #piirtelyä

#%%
rf_rate = 0 #assume risk-free rate to be 0%

risky_assets = ["Health Care", 
                "Financials", 
                "IT", 
                "Industrials",
                "Utilities", 
                "Consumer Discretionary", 
                "Consumer Staples", 
                "Materials"] #change name of risky assets to be more clear in weights

#sharpe ratio maximising portfolio

args = (avg_returns, cov_mat, rf_rate)
constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1}) #weight of assets must equal to 1
bounds = tuple((0,1) for asset in range(n_assets)) #individual asset weights cannot go below zero (or 100%), no shorting allowed
initial_guess = n_assets * [1. / n_assets]
max_sharpe_portf = sco.minimize(neg_sharpe_ratio, x0 = initial_guess,
                                args = args,
                                method = "SLSQP",
                                bounds = bounds,
                                constraints = constraints)

max_sharpe_portf_w = max_sharpe_portf["x"]
max_sharpe_portf = {"Annual Return": get_portf_rtn(max_sharpe_portf_w, 
                                            avg_returns),
                    "Annual Volatility": get_portf_vol(max_sharpe_portf_w,
                                                avg_returns,
                                                cov_mat),
                    "Sharpe ratio": -max_sharpe_portf["fun"]}

#printing the results

print("Maximum Sharpe Ratio portfolio ----")
print("Performance")

for index, value in max_sharpe_portf.items():
    print(f"{index}: {100 * value:.2f}% ", end="", flush = True)

print("\nWeights")

spw = [] #sharpe portfolio weights

for x, y in zip(risky_assets, max_sharpe_portf_w):
    print(f"{x}: {100 * y:.2f}% ", end = "", flush = True)
    spw.append(y) #adding weights into list

sharpe_portfolio_returns = pd.Series(np.dot(spw, returns.T), index = returns.index)

#pf.create_simple_tear_sheet(sharpe_portfolio_returns)

# minimum variance portfolio

rtns_range = np.linspace(-0.50, 0.50, 200) #considered range of returns, next line of code will run function with all expected returns

efficient_portfolios = get_efficient_frontier(avg_returns, cov_mat, rtns_range) #calculating the efficient frontier and placing them in a list

vols_range = [x["fun"] for x in efficient_portfolios] #extracting volatilities of efficient portfolios

#identifying minimum variance portfolio

min_vol_ind = np.argmin(vols_range)
min_vol_portf_rtn = rtns_range[min_vol_ind]
min_vol_portf_vol = efficient_portfolios[min_vol_ind]["fun"]

min_vol_portf = {"Annual Return": min_vol_portf_rtn,
                "Annual Volatility": min_vol_portf_vol,
                "Sharpe ratio": (min_vol_portf_rtn / min_vol_portf_vol)}

#printing the results

print("\nMinimum volatility portfolio ----")
print("Performance")

for index, value in min_vol_portf.items():
    print(f"{index}: {100 * value:.2f}% ", end = "", flush = True)

print("\nWeights")

mvpw = [] #min-var portfolio weights

for x, y in zip(risky_assets, efficient_portfolios[min_vol_ind]["x"]):
    print(f"{x}: {100 * y:.2f}% ", end = "", flush = True)
    mvpw.append(y)

minvar_portfolio_returns = pd.Series(np.dot(mvpw, returns.T), index = returns.index)

#pf.create_simple_tear_sheet(minvar_portfolio_returns)

# %%
