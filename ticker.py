import xlwings as xw
import yfinance as yf

def is_ticker_delisted(ticker):
    """ Check if the ticker is delisted by trying to download its data. """
    try:
        data = yf.download(ticker, period="1d")
        if data.empty:
            return True
    except Exception as e:
        return True
    return False

# Open your Excel workbook
wb = xw.Book('path_to_your_excel_file.xlsx')
sheet = wb.sheets['Sheet1']  # Replace with your actual sheet name

# Assuming tickers are in the first column, starting from row 1
tickers = sheet.range('A1').expand('down').value

# Iterate over tickers and check each one
for i, ticker in enumerate(tickers, start=1):
    if is_ticker_delisted(ticker):
        sheet.api.Rows(i).Delete()
        print(f"Deleted row {i} for delisted ticker: {ticker}")