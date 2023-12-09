#pip install "xlwings[all]"            #Numpy, pandas, matplotlib, Pillow, Jinja2

import xlwings as xw #xlwings for excel manipulation
import pandas as pd #for data manipulation with dataframes
import yfinance as yf #to download data from yahoo finance
import warnings #disable possible warnings
import matplotlib.pyplot as plt #plotting data
import numpy as np #data manipulation
import scipy.optimize as sco #portfolio optimisation

############################################################
#Excel setting
############################################################

#excel workbook 'Main_i.xlsx' needs to be open to run the code properly
wb = xw.Book('Ohjelmoinnin harjoitusyö/Main_i.xlsx')

#worksheet import
ws1 = wb.sheets['Dashboard']
ws2 = wb.sheets['Ticker']

#defining the reset function for the users sheet
def reset_worksheet_dashboard():
    wb = xw.Book('Main_i.xlsx')
    ws1 = wb.sheets['Dashboard']

    # define the range for deletion and deleting the content from the 'variance, mean and sharpe' table
    ws1.range('J6:M1048576').clear_contents()

#deleting the content before new content
reset_worksheet_dashboard()

#saving the workbooks, otherwise the new tickers - typed in as user desires - wont be used in the new calculation
wb.save()

############################################################
#Portfolio testing
############################################################

def download_data(vector: list): #download data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  #filter warnings
        prices_df = yf.download(vector, period="1y", auto_adjust=False)

    if 'Adj Close' in prices_df.columns:
        prices_df = prices_df['Adj Close'].ffill()  #fill possible missing data points
        
        prices_df = prices_df.dropna(axis=1, how='all') #remove columns where all values are NaN (no data for ticker)

        return prices_df.pct_change().dropna() #calculate percent change to get returns
    else:
        return pd.DataFrame()  #if 'Adj Close' not available, return empty dataframe

def plottaus(returns):
    plt.figure(figsize=(10, 6))

    if isinstance(returns, pd.DataFrame): #check if data is series or dataframe
        for column in returns.columns:
            plt.plot(returns.index, returns[column], label=column) #define plot data and index
    elif isinstance(returns, pd.Series):
        plt.plot(returns.index, returns, label=returns.name)

    plt.title('Portfolion tuotto')
    plt.xlabel('Päivämäärä')
    plt.ylabel('Tuotto')
    plt.legend()

    return plt.gcf() #return current plot

def equal_weight_returns(returns): #calculate returns of equal-weighted portfolio
    portfolio_weights = n_assets * [1 / n_assets] #equally-weighted
    portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index = returns.index) #np.dot = matrix multiplication, pd.series = stored data as pandas series
    return portfolio_returns

def get_portf_rtn(weights, avg_rtns): #calculate returns for any portfolio
    return np.sum(avg_rtns * weights) 

def form_min_var_portfolio(): #form minimum variance portfolio weights
    print("Processing...")
    rtns_range = np.linspace(-0.50, 0.50, 50) #considered range of returns, next line of code will run function with all expected returns
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
        rounded_weight = round(100 * y, 2)  #round out weights to two decimals
        #print(f"{x}: {rounded_weight:.2f}% ", end="", flush=True)
        mvpw.append(rounded_weight / 100)  #add weight to list in decimal form

    #minvar_portfolio_returns = pd.Series(np.dot(mvpw, returns.T), index = returns.index)
    return mvpw #return list of weights

def form_max_sharpe_portfolio(): #same as above but for max sharpe ratio portfolio
    args = (avg_returns, cov_mat, rf_rate)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1}) #weight of assets must equal to 1
    bounds = tuple((0,1) for asset in range(n_assets)) #individual asset weights cannot go below zero (or 100%), no shorting allowed
    initial_guess = n_assets * [1. / n_assets]
    max_sharpe_portf = sco.minimize(neg_sharpe_ratio, x0 = initial_guess, #optimise using scipy library
                                    args = args,
                                    method = "SLSQP", #sequential least-squares programming, solves optimization problems
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
        rounded_weight = round(100 * y, 2)  #round out
        #print(f"{x}: {rounded_weight:.2f}% ", end="", flush=True)
        spw.append(rounded_weight / 100)  #add weight to list in decimal form

    #sharpe_portfolio_returns = pd.Series(np.dot(spw, returns.T), index = returns.index)
    return spw

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range): #to find out the combination of efficient portfolios on different returns
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

def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate): #no need for for-loop, because searching for only one set of weights
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    return -portf_sharpe_ratio

def get_portf_vol(w, cov_mat): #get portfolio volatility
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def hintakaavio(weights): #plotting portfolio returns
    portfolio_returns = pd.Series(np.dot(weights, returns.T), index=returns.index)
    cumulative_returns = (1 + portfolio_returns).cumprod() * 100
    print(cumulative_returns)
    return plottaus(cumulative_returns)

def compare_portfolios(weights1, weights2): #plotting returns of two portfolios in same graph
    prt1_ret = pd.Series(np.dot(weights1, returns.T), index=returns.index)
    returns1 = (1 + prt1_ret).cumprod() * 100

    prt2_ret = pd.Series(np.dot(weights2, returns.T), index=returns.index)
    returns2 = (1 + prt2_ret).cumprod() * 100

    plt.figure(figsize=(10, 6))

    plt.plot(returns1.index, returns1, label="Max Sharpe Portfolio", color='blue')
    plt.plot(returns2.index, returns2, label="Minimivarianssiportfolio", color='green')

    plt.title('Portfolioiden vertailu')
    plt.xlabel('Päivämäärä')
    plt.ylabel('Portfolion tuotto (lähtötaso 100)')
    plt.legend()
    return plt.gcf()

def plot_return_histogram(returns, title='Tuottojakauma', xlabel='Returns'): #plotting histogram of return values
    plt.figure(figsize=(10, 6))
    returns.plot(kind='kde', color='blue', lw = 2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Tiheys')
    return plt.gcf()

def get_combined_returns(returns): #combine the returns of tickers and portfolios into one dataframe for easier manipulation later
    min_var_weights = form_min_var_portfolio()
    max_sharpe_weights = form_max_sharpe_portfolio()

    min_var_returns = pd.Series(np.dot(min_var_weights, returns.T), index=returns.index)
    max_sharpe_returns = pd.Series(np.dot(max_sharpe_weights, returns.T), index=returns.index)

    combined_returns = pd.DataFrame() #make empty dataframe
    combined_returns['Min Var Portfolio'] = min_var_returns
    combined_returns['Max Sharpe Portfolio'] = max_sharpe_returns

    combined_returns = pd.concat([combined_returns, returns], axis=1) #combine tickers and portfolios

    return combined_returns

def calculate_metrics(combined_returns, rf_rate=0): #calculate the metrics we want to show in excel file
    metrics = pd.DataFrame(index=combined_returns.columns, columns=['Return', 'Volatility', 'Sharpe Ratio'])

    for column in combined_returns.columns:
        annual_return = np.sum(combined_returns[column].mean()) * 252 #annualise return and volatility
        annual_volatility = combined_returns[column].std() * np.sqrt(252)
        sharpe_ratio = (annual_return - rf_rate) / annual_volatility #sharpe ratio

        metrics.loc[column] = [annual_return, annual_volatility, sharpe_ratio] #add to dataframe

    return metrics

def print_to_excel(metrics, ws): #print the metrics into the excel file starting from column J row 6
    for i, (index, row) in enumerate(metrics.iterrows(), start=5):
        ws.range(f'J{i+1}').value = index  # Ticker/Portfolio name
        ws.range(f'K{i+1}').value = row['Return']
        ws.range(f'L{i+1}').value = row['Volatility']
        ws.range(f'M{i+1}').value = row['Sharpe Ratio']

############################################################
#Downloading the data from excel
############################################################

file_path = 'Ohjelmoinnin harjoitusyö/Main_i.xlsx'
df = pd.read_excel(file_path, skiprows=3, usecols=[1]) #read file starting from b-column's 4th row
vector = df.iloc[:, 0].dropna().tolist() #make dataframe into a list and drop empty values
n_assets = len(vector)

try:
    returns = download_data(vector)
    if not returns.empty: #if no data is available for ticker
        print(returns)
    else:
        print("No data available for the given tickers.")
except Exception as e:
    print(f"Error during download: {e}")


avg_returns = returns.mean() * 12 #average returns
cov_mat = returns.cov() * 12 #covariance-matrix 
rf_rate = 0 #assume risk-free rate to be 0%

combined_returns = get_combined_returns(returns)
metrics = calculate_metrics(combined_returns)
print_to_excel(metrics, ws1)  

#defining the plot variables with a 'Figure' variable type
plot1 = plt.figure(hintakaavio(form_max_sharpe_portfolio())) #(Max sharpe portfolio, plot)
plot2 = plt.figure(compare_portfolios(form_max_sharpe_portfolio(), form_min_var_portfolio())) #(Price, doubleplot)
plot3 = plt.figure(plot_return_histogram(equal_weight_returns(returns))) #(Return, histogram)

############################################################
#Plotting and printing to excel
############################################################

#moving the plots to the excel 'update=True -> as we want to update the plot every time the script is run'
ws1.pictures.add(plot1, name='plot1', update=True)
ws1.pictures.add(plot2, name='plot2', update=True)
ws1.pictures.add(plot3, name='plot3', update=True)

wb.save()

print("done")