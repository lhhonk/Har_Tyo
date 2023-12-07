#pip install "xlwings[all]"            #Numpy, pandas, matplotlib, Pillow, Jinja2

import xlwings as xw #xlwings for excel manipulation
import pandas as pd #Panda for dataframes
import numpy as np #NumPy for numerical calculations
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import time

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
