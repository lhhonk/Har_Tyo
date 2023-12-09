import xlwings as xw
import warnings
import yfinance as yf
import pandas as pd

# Function for downloading data
def download_data(vector: list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # Filter out future warnings
        try:
            prices_df = yf.download(vector, period="1y", auto_adjust=False)
            if 'Adj Close' in prices_df.columns:
                prices_df['Adj Close'] = prices_df['Adj Close'].ffill()
                return prices_df["Adj Close"].pct_change().dropna()  # Simple asset returns
        except Exception as e:
            print(f"An error occurred while downloading data for {vector}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if download fails or if 'Adj Close' is not present

# Function to check if the ETF data is downloadable
def is_data_downloadable(ticker):
    # Attempt to fetch the historical data for the ticker
    data = download_data([ticker])
    # If the data is empty, assume the ticker data is not downloadable
    return not data.empty

# Open the workbook and select the sheet
wb = xw.Book('Ohjelmoinnin harjoitusyö/ticker.xlsx')  # Replace with the path to your workbook
sheet = wb.sheets['Sheet1']

# Get the list of tickers from the Excel sheet, excluding the header
tickers = sheet.range('A2:A' + str(sheet.cells.last_cell.row)).value

# Make sure tickers is a list even if there's only one ticker
if not isinstance(tickers, list):
    tickers = [tickers]

# Collect all the row indices for tickers with no data
rows_to_delete = []

# Check if there is data in the cell
for i, ticker in enumerate(tickers, start=2):  # Starting from row 2 to account for header
    if ticker is None or ticker == "":  # Check if the cell is empty
        print(f"No ticker in row {i}, exiting the loop.")
        break  # Exit the loop if a blank cell is encountered

    # Ensure the ticker is a string
    if not isinstance(ticker, str):
        continue

    try:
        if not is_data_downloadable(ticker):
            rows_to_delete.append(i)
            print(f"Data not downloadable for ticker {ticker}. Marking row {i} for deletion.")
    except Exception as e:
        print(f"An error occurred while checking ticker {ticker} at row {i}: {e}")

# Delete the rows in reverse order to avoid index shifting
for row in reversed(rows_to_delete):
    sheet.range(f'A{row}').api.EntireRow.Delete()