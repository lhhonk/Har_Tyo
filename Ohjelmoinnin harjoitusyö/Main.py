#pip install "xlwings[all]"            #Numpy, pandas, matplotlib, Pillow, Jinja2

import xlwings as xw #xlwings for excel manipulation
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
import pyfolio as pf

############################################################
#Excel setting
############################################################
#Excel workbook 'Main_i.xlsx' needs to be open to run the code properly
wb = xw.Book('Main_i.xlsx')

#Worksheet import
ws1 = wb.sheets['Dashboard']

#Defining the reset function for the users sheet
def reset_worksheet_dashboard():
    wb = xw.Book('Main_i.xlsx')
    ws1 = wb.sheets['Dashboard']

    # Define the range starting from "J2" to the end of the worksheet
    ws1.range('J2:XFD1048576').clear_contents()

    # Deleting charts
    for chart in ws1.charts:
        chart.delete()

# Testing the function
reset_worksheet_dashboard()

############################################################
#Portfolio testing
############################################################

def download_data(vector: list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  #filtteröidään future warning pois
        prices_df = yf.download(vector, period="1y", auto_adjust=False)

        if 'Adj Close' in prices_df.columns:
            prices_df['Adj Close'] = prices_df['Adj Close'].ffill()
            return prices_df["Adj Close"].pct_change().dropna()  #simple asset returns
        else:
            return pd.DataFrame()  #jos adj. close ei saatavilla, palautetaan tyhjä dataframe

def plottaus(returns):
    plt.figure(figsize=(10, 6))

    # Check if returns is a DataFrame or a Series
    if isinstance(returns, pd.DataFrame):
        for column in returns.columns:
            plt.plot(returns.index, returns[column], label=column)
    elif isinstance(returns, pd.Series):
        plt.plot(returns.index, returns, label=returns.name)

    plt.title('Portfolion tuotto')
    plt.xlabel('Päivämäärä')
    plt.ylabel('Tuotto')
    plt.legend()
    plt.show()

def equal_weight_returns(returns):
    portfolio_weights = n_assets * [1 / n_assets] #equally-weighted
    portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index = returns.index) #np.dot = matrix multiplication, pd.series = stored data as pandas series
    return portfolio_returns

def get_portf_rtn(weights, avg_rtns):
    return np.sum(avg_rtns * weights)

def form_min_var_portfolio():
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
    mvpw = [] #min-var portfolio weights

    for x, y in zip(vector, efficient_portfolios[min_vol_ind]["x"]):
        rounded_weight = round(100 * y, 2)  #pyöristetään
        #print(f"{x}: {rounded_weight:.2f}% ", end="", flush=True)
        mvpw.append(rounded_weight / 100)  #lisätään paino listaan ja takaisin prosenttimuotoon

    #minvar_portfolio_returns = pd.Series(np.dot(mvpw, returns.T), index = returns.index)
    return mvpw

def form_max_sharpe_portfolio():
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
                                                    cov_mat),
                        "Sharpe ratio": -max_sharpe_portf["fun"]}
    spw = [] #sharpe portfolio weights

    for x, y in zip(vector, max_sharpe_portf_w):
        rounded_weight = round(100 * y, 2)  #pyöristetään
        #print(f"{x}: {rounded_weight:.2f}% ", end="", flush=True)
        spw.append(rounded_weight / 100)  #lisätään paino listaan ja takaisin prosenttimuotoon

    #sharpe_portfolio_returns = pd.Series(np.dot(spw, returns.T), index = returns.index)
    return spw

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
    efficient_portfolios = []
    n_assets = len(avg_returns)
    args = (cov_mat,) #covariance matrix
    bounds = tuple((0, 1) for asset in range(n_assets)) #weight of asset must be between 0-1, no shorting allowed
    initial_guess = n_assets * [1. / n_assets, ] #makes optimization faster by using initial guesses

    for ret in rtns_range:
        constraints = ({"type": "eq", "fun": lambda x: get_portf_rtn(x, avg_rtns) - ret}, #expected portfolio return must be equal to provided value
                       {"type": "eq","fun": lambda x: np.sum(x) - 1}) #sum of asset weights must be equal to 1
        efficient_portfolio = sco.minimize(get_portf_vol,
                                            initial_guess,
                                            args = args,
                                            method = "SLSQP", #sequential least-squares programming, solves optimization problems
                                            constraints = constraints,
                                            bounds = bounds)
        efficient_portfolios.append(efficient_portfolio)
    
    return efficient_portfolios

def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate): #no need for for-loop, becayse searching for only one set of weights
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    return -portf_sharpe_ratio

def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def hintakaavio(weights):
    portfolio_returns = pd.Series(np.dot(weights, returns.T), index=returns.index)
    cumulative_returns = (1 + portfolio_returns).cumprod() * 100

    plottaus(cumulative_returns)

def compare_portfolios(weights1, weights2):
    prt1_ret = pd.Series(np.dot(weights1, returns.T), index=returns.index)
    returns1 = (1 + prt1_ret).cumprod() * 100

    prt2_ret = pd.Series(np.dot(weights2, returns.T), index=returns.index)
    returns2 = (1 + prt2_ret).cumprod() * 100

    plt.figure(figsize=(10, 6))

    plt.plot(returns1.index, returns1, label="Portfolio 1", color='blue')
    plt.plot(returns2.index, returns2, label="Portfolio 2", color='green')

    plt.title('Portfolioiden vertailu')
    plt.xlabel('Päivämäärä')
    plt.ylabel('Portfolion tuotto (lähtötaso 100)')
    plt.legend()

    plt.show()

def plot_return_histogram(returns, title='Tuottojakauma', xlabel='Returns'):
    plt.figure(figsize=(10, 6))
    returns.plot(kind='kde', color='blue', lw = 2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Tiheys')
    plt.show()

#TIETOJEN LATAUS
file_path = 'Ohjelmoinnin harjoitusyö/Main_i.xlsx'
df = pd.read_excel(file_path, skiprows=3, usecols=[1]) #luetaan tiedosto B-sarakkeen neljännestä rivistä alkaen
vector = df.iloc[:, 0].dropna().tolist() #muutetaan dataframe listaksi ja tiputetaan tyhjät arvot
n_assets = len(vector)

try:
    returns = download_data(vector)
    if not returns.empty:
        print(returns)
    else:
        print("No data available for the given tickers.")
except Exception as e:
    print(f"Error during download: {e}")

avg_returns = returns.mean() * 12 #average returns
cov_mat = returns.cov() * 12 #covariance-matrix 
rf_rate = 0 #oletetaan riskittömäksi 0%

#plottaus(returns)
#equal_weight_returns(returns)
#print(form_max_sharpe_portfolio())
#print(form_min_var_portfolio())

plot1 = hintakaavio(form_max_sharpe_portfolio())
plot2 = compare_portfolios(form_max_sharpe_portfolio(), form_min_var_portfolio()) #kahden plotin tekemiseen
plot2 = plot_return_histogram(equal_weight_returns(returns))

print("done")

############################################################
#Plotting and printing to excel
############################################################
def plot_and_set():
    ws1.pictures.add(plot1, name='plot1', update=True,
                     left=ws1.range('M3').left,
                     top=ws1.range('M3').top)

plot_and_set()
