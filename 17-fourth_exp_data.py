import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
pd.set_option('display.max_columns', 500)

# Report the market data
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "Different Asset/"
vix = pd.read_csv(directory + folder + "vix.csv")

# Sort the dataframe in chronological order
vix["Date"] = pd.to_datetime(vix.Date, format="%d/%m/%Y")
vix.sort_values(by="Date", inplace=True)

# Plot the closing price time series 
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(vix["Date"], vix["VIX Close"])
ax.set_title("VIX Close from Dec13 to Apr19", fontsize=16)
monthyearFmt = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(monthyearFmt)
ax.xaxis.set_major_locator(mdates.MonthLocator((6,12)))
plt.xticks(rotation=90)
fig.savefig(directory + folder + "vix.png")
plt.legend()
plt.show()

# Calculate the daily change between open and close
vix["Daily_Change"] = vix["VIX Close"] - vix["VIX Open"]

# Add the daily direction column
vix["Daily_Direction"] = (vix["Daily_Change"] >= 0) * 1

# Define a function for adding in the cut-off time given dates
def cut_off_time(Date, Hour, Minute):
    cutoff = datetime(Date.year,Date.month,Date.day,Hour,Minute)
    return cutoff

# Add in the cutoff time column
vix["Cutoff"] = vix["Date"].apply(lambda x: cut_off_time(x,Hour=9,Minute=25))

# Add a column to indicate if a trading day falls into training or test set
N = len(vix)
Train = int(N * 0.9)
vix["Split"] = ""
vix["Split"][:Train] = "train"
vix["Split"][Train:] = "test"

# Export this S&P 500 df as csv
vix.to_csv(directory + folder + "vix1.csv", index=False) 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "Different Asset/"
file = "us_10yr.csv"

# Import the SP500 time series
treasury = pd.read_csv(directory + folder + file)
treasury["Date"] = pd.to_datetime(treasury.Date, format="%d/%m/%Y")

# Sort the dataframe in chronological order
treasury.sort_values(by ="Date", inplace=True)

# Plot the closing price time series 
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(treasury["Date"], treasury["10 Yr"])
ax.set_title("US Treasury 10 Year Yield Close from Dec13 to Apr19", fontsize=16)
monthyearFmt = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(monthyearFmt)
ax.xaxis.set_major_locator(mdates.MonthLocator((6,12)))
plt.xticks(rotation=90)
fig.savefig(directory + folder + "us_10yr.png")
plt.legend()
plt.show()

# Add the daily direction column
treasury["Daily_Direction"] = (treasury["Daily_Change"] >= 0) * 1

# Define a function for adding in the cut-off time given dates
def cut_off_time(Date, Hour, Minute):
    cutoff = datetime(Date.year,Date.month,Date.day,Hour,Minute)
    return cutoff

# Add in the cutoff time column
treasury["Cutoff"] = treasury["Date"].apply(lambda x: cut_off_time(x, Hour=15, Minute=0))

# Add a column to indicate if a trading day falls into training or test set
N = len(treasury)
Train = int(N * 0.9)
treasury["Split"] = ""
treasury["Split"][:Train] = "train"
treasury["Split"][Train:] = "test"

# Export this S&P 500 df as csv
treasury.to_csv(directory + folder + "us_10yr_1.csv", index=False) 


