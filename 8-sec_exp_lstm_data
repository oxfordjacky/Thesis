import pandas as pd
from datetime import datetime

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2.csv")

# Make the cutoff time date time format
sp500["Cutoff"] = pd.to_datetime(sp500["Cutoff"])

# Define a df for prior trading day with cutoff times
prior_day = sp500["Cutoff"][:-1]

# Reset the index of prior_day to start from 1
prior_day.index = range(1, sp500.index[-1]+1)

# Create a column to host the prior trading day for sp500
sp500["Prior_Cutoff"] = prior_day

# Assign a value to the first trading day in sp500
sp500["Prior_Cutoff"][0] = datetime(2013, 12, 17, 9, 25)

# Export the updated df to csv
sp500.to_csv(directory + folder + "sp5001_sd2_t1.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2_t1.csv")

# Make the cutoff columns date time formate
sp500["Cutoff"] = pd.to_datetime(sp500["Cutoff"]) 
sp500["Prior_Cutoff"] = pd.to_datetime(sp500["Prior_Cutoff"])

# Zip the prior cutoff and cutoff columns to form a new combined one
sp500["Prior_Current"] = list(zip(sp500.Prior_Cutoff, sp500.Cutoff))

# Define a function to calculate the intermediate time periods
def intermediate(prior, current, period):
    diff = current - prior
    diff_24 = diff / 24
    inter_prior = prior + diff_24 * period
    return inter_prior

# Create 24 new columns to hold intermediate time periods
for i in range(1,25):
    sp500[i] = sp500["Prior_Current"].apply(lambda x: intermediate(x[0], x[1], i))
    print(i)

# Drop the zipped prior and current cutoff columns
sp500.drop(columns="Prior_Current", inplace=True)

# Export the updated df as csv
sp500.to_csv(directory + folder + "sp5001_sd2_t1_1.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2_t1_1.csv")

# Make all cutoff related columns as date time in the sp500 df
sp500["Cutoff"] = pd.to_datetime(sp500["Cutoff"])
sp500["Prior_Cutoff"] = pd.to_datetime(sp500["Prior_Cutoff"])
sp500["Date"] = pd.to_datetime(sp500["Date"])
for i in range(1, 25):
    sp500["{}".format(i)] = pd.to_datetime(sp500["{}".format(i)])

 # Import the title csv
df = pd.read_csv(directory + folder + "df_sent_ner.csv")

# Make the us_eastern_time date time format in the title csv
df["us_eastern_time"] = pd.to_datetime(df["us_eastern_time"])
df["trading_day"] = pd.to_datetime(df["trading_day"])

# Sort the df in ascending order by US Eastern time
df.sort_values(by="us_eastern_time", inplace=True)

# Create a new column that will contain the following trading day
df["time_period"] = ""

# Define a mechanism to align dates of the two dataframes
outer_loop = len(sp500)
inner_loop = len(df)
#Current index in the news dataframe
current_step = 0

#Loop the price dataframe
for outer in range(outer_loop):
    #Current date and next date in the price dataframe
    if sp500["No_of_News"][outer] != 0:
        trade_date = sp500["Date"][outer]
        j = 1
        prior_cutoff = sp500["Prior_Cutoff"][outer]
        while (df["trading_day"][current_step] == trade_date):
            current_cutoff = sp500["{}".format(j)][outer]
            while (df["us_eastern_time"][current_step] > prior_cutoff and df["us_eastern_time"][current_step] <= current_cutoff):
                df["time_period"][current_step] = j
                print("Time Period = ", df["time_period"][current_step])
                current_step += 1
                if current_step == inner_loop:
                   break
            if current_step == inner_loop:
                   break
            j += 1
            prior_cutoff = sp500["{}".format(j-1)][outer]
        print("Current Outer Loop:", outer)
        
# Export the updated title df as csv
df.to_csv(directory + folder + "df_sent_ner1.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2.csv")

# Re-import the titles csv
df = pd.read_csv(directory + folder + "df_sent_ner1.csv")

# Create 24 new columns calculating the no of articles
for i in range(1, 25):
    sp500["No_of_News_{}".format(i)] = ""

# Set the date columns to datetime format
df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Populate these 24 columns
N = len(sp500)
for i in range(N):
    current_date = sp500["Date"][i]
    dummy = df[df["trading_day"]==current_date]
    for j in range(1, 25):
        dummy_time = dummy[dummy["time_period"]==j]
        sp500["No_of_News_{}".format(j)][i] = len(dummy_time)
    print(i)

# Export the updated price df as csv
sp500.to_csv(directory + folder + "sp5001_sd2_t1_2.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2_t1_2.csv")

# Re-import the titles csv
df = pd.read_csv(directory + folder + "df_sent_ner1.csv")

# Set the trading date columns as date time format
df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Add 24 columns in the price csv that contain the concatenated titles for that time period
for i in range(1, 25):
    sp500["NER_Titles_{}".format(i)] = ""

# Populate this column
outer_loop = len(sp500)
separator = " "
#Loop the price dataframe
for outer in range(outer_loop):
   no_news = sp500["No_of_News"][outer]
   if no_news > 0:
      current_date = sp500["Date"][outer]
      # Define a dummy news df
      dummy = df[df["trading_day"]==current_date]
      # Reset the index from 0
      dummy.reset_index(inplace=True, drop=True)
      # Get the unique time period for this trade date
      time_unique = dummy.time_period.unique()
      # Current step in the dummy df
      current_step = 0
      # Define the length of dummy
      inner_loop = len(dummy)
      for j in time_unique:
          daily_titles = []
          while (dummy["time_period"][current_step] == j):
              title = dummy["ner_title"][current_step]
              daily_titles.append(title)
              current_step += 1
              if current_step == inner_loop:
                 break
          sp500["NER_Titles_{}".format(j)][outer] = separator.join(daily_titles)
   print(outer)
   
# Add 24 columns that check total length of concatenated titles
for i in range(1, 25):
    sp500["NER_Title_Check_{}".format(i)] = sp500["NER_Titles_{}".format(i)].str.split().str.len()

# Export this dataframe as csv
sp500.to_csv(directory + folder + "sp5001_sd2_t1_3.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2_t1_3.csv")

# Replace NaN with empty space
sp500.fillna(value="", inplace=True)

