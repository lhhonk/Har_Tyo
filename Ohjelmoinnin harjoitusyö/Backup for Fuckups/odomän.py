# Ohjelmoinnin harjoitustyö 

# Stock screener

import yfinance as yf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import finplot as fplt

def load_tickers_from_csv(file_path):
    with open(file_path, 'r', encoding = "utf-8") as file:
        reader = csv.reader(file)
        next(reader) # skip header row
        ticker_data = {row[0]: {"nimi": row[1], "maa": row[4]} for row in reader}
    return ticker_data

def is_valid_ticker(ticker, valid_tickers):
    return ticker in valid_tickers

def hintadata(ticker, period):
    prices_df = yf.download(ticker, 
                        period = period)
    return prices_df["Adj Close"]

def plot_tuottoluvut(tuottoluvut, ticker, nimi):
    plt.figure(figsize=(14, 7))
    plt.plot(tuottoluvut.index, tuottoluvut.values)
    plt.title(f'Yhtiön {nimi} ({ticker}) kurssi väliltä {tuottoluvut.index[0].strftime("%d-%m-%Y")} — {tuottoluvut.index[-1].strftime("%d-%m-%Y")}')
    plt.xlabel("Päivämäärä")
    plt.ylabel("Hinta")
    plt.grid(True)
    plt.show()

def kynttila(ticker, interval, period):
    ticker = yf.Ticker(ticker)
    data = ticker.history(interval = interval, period = period)
    fplt.candlestick_ochl(data[['Open', 'Close', 'High', 'Low']])
    fplt.show()

def tuloslaskelma(ticker):
    ticker = yf.Ticker(ticker)
    print("Tuloslaskelma:")
    print(ticker.income_stmt)
    print()

def tase(ticker):
    ticker = yf.Ticker(ticker)
    print("Tase:")
    print(ticker.balance_sheet)
    print()

def kassavirta(ticker):
    ticker = yf.Ticker(ticker)
    print("Kassavirtalaskelma:")
    print(ticker.cashflow)
    print()

def omistajat(ticker):
    ticker = yf.Ticker(ticker)
    print(ticker.major_holders)
    print()
    print(ticker.institutional_holders)
    print()
    print(ticker.mutualfund_holders)
    print()

csv_path = 'tickers.csv'
valid_tickers = load_tickers_from_csv(csv_path)

print("Mistä yrityksestä haluat tietoja?")
ticker = input("")

while True:
    if ticker in valid_tickers:
        print(f"\nYrityksen nimi: {valid_tickers[ticker]['nimi']}")
        print(f"Maa: {valid_tickers[ticker]['maa']}\n")
        break
    else:
        print("Tarkista tickerin oikeinkirjoitus!")
        ticker = input("")

print("Mitä tietoja haluat? Kirjoita kaikkien haluamiesi tietojen numerot välilyönnillä erotettuna. ")
print("1) Historiallista hintadataa viivadiagrammina")
print("2) Historiallista hintadataa kynttilädiagrammina")
print("3) Tuloslaskelma")
print("4) Tase")
print("5) Kassavirtalskelma")
print("6) Omistajatietoja")

mita_haluaa = input("")
print()
halutut_tiedot = mita_haluaa.split() # tekee listan

if "1" in halutut_tiedot:
    period = input("Kuinka pitkältä ajalta haluat dataa? \n1y = vuosi \n2y = kaksi vuotta \n5y = viisi vuotta \n10y = kymmenen vuotta \nytd = vuoden alusta \nmax = maksimi\n")
    tuottoluvut = hintadata(ticker, period)
    yrityksen_nimi = valid_tickers[ticker]['nimi']  # Haetaan yrityksen nimi csv:stä
    plot_tuottoluvut(tuottoluvut, ticker, yrityksen_nimi)

if "2" in halutut_tiedot:
    period = input("Kuinka pitkältä ajalta haluat dataa? \n1mo = yksi kuukausi \n3mo = kolme kuukautta \n6mo = kuusi kuukautta \n1y = yksi vuosi \n2y = kaksi vuotta \n5y = viisi vuotta \n10y = kymmenen vuotta \nytd = vuoden alusta \nmax = maksimi\n")
    interval = input("Millä tiheydellä haluat kynttilöitä? \n1d = yksi päivä \n5d = viisi päivää \n1wk = yksi viikko \n1m = yksi kuukausi \n3m = kolme kuukautta\n")
    kynttila(ticker, interval, period)

if "3" in halutut_tiedot:
    tuloslaskelma(ticker)

if "4" in halutut_tiedot:
    tase(ticker)

if "5" in halutut_tiedot:
    kassavirta(ticker)

if "6" in halutut_tiedot:
    omistajat(ticker)
