# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:29:49 2019

@author: oxfor
"""

import pandas as pd
pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/"
folder = "EDA/"
file = "df_all.csv"

# Define the columns required
columns = ["url","title","section","published"]

# Import the csv with all the sections
df = pd.read_csv(directory + folder + file, usecols=columns)
print(df.head())

# Export this as csv
folder = "Data/SP500/"
df.to_csv(directory + folder + "df_title.csv", index=False)