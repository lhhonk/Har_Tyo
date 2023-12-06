#xlwings for excel manipulation
#Panda for dataframes
#ggplot for visualization
#NumPy for numerical calculations

#pip install "xlwings[all]"            #Numpy, pandas, matplotlib, Pillow, Jinja2

import xlwings as xw
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

############################################################
#Excel setting
############################################################

path = r'C:\Users\Testi\OneDrive - O365 Turun yliopisto\Työpöytä\Harjoitustyö\Main_i.xlsx'
#Path pitää laittaa käyttäjän mukaan... Pitää keksiä parempi vaihtoehto
#xw.Workbook.caller jos mainia kutsutaan excelistä
wb = xw.Book(path)

#Worksheet import
ws1 = wb.sheets['Dashboard']
ws2 = wb.sheets['Price']
ws3 = wb.sheets['Returns']
ws4 = wb.sheets['Log_returns']

#Data to dataframes
df2 = ws2.range('A1').options(pd.DataFrame, expand='table').value
df3 = ws3.range('A1').options(pd.DataFrame, expand='table').value
df4 = ws4.range('A1').options(pd.DataFrame, expand='table').value
print(df2.head())
print(df3.head())
print(df4.head())

############################################################
#Global value weighted portfolio, calculation
############################################################
daily_returns = df2.pct_change()
total_market_cap = df2.sum(axis=1)
weights = df2.divide(total_market_cap, axis=0)
weighted_returns = daily_returns.multiply(weights, axis='columns')
portfolio_daily_returns = weighted_returns.sum(axis='columns')
portfolio_price = (1 + portfolio_daily_returns).cumprod()

#plotting
portfolio_price_df = portfolio_price.to_frame(name='Portfolio Price')
portfolio_price_df.reset_index(inplace=True)

print(portfolio_price_df.head())

#fig2 = px.line(portfolio_price_df, x='Date', y='Portfolio Price', title='Portfolio Price Over Time')
#fig2 = px.line(portfolio_price_df, x='Date', y='Portfolio Price', title='Portfolio Price Over Time')
#fig2.show

#ws1.pictures.add(fig, name='PortfolioPricePlot', update=True, left=ws1.range('N1').left, top=ws1.range('N3').top)
#Kaleido package needed for this
fig = plt.figure()
plt.plot([1, 2, 3])
ws1.pictures.add(fig, name='MyPlot', update=True, left=ws1.range('N1').left, top=ws1.range('N3').top)

############################################################
#
############################################################
