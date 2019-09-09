import pandas as pd
pd.set_option('display.max_columns', 500)

# Import the full df with main text 
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/"
folder_eda = "EDA/"
df_main = pd.read_csv(directory + folder_eda + "df_all.csv")

# Set its index as the url
df_main.set_index("url", inplace=True)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/"
folder_data = "Data/SP500/"
df = pd.read_csv(directory + folder_data + "df_topic_1.csv")
# Set its index as the url
df.set_index("url", inplace=True)

# Add main text to the df with topic 1 and titles only
df["text"] = df_main["text"]

# Check there is no null value
print(df.isnull().sum())

# Export the updated df as csv
folder_output = "Data/Different Asset/"
df.to_csv(directory + folder_output + "df_t1_1.csv")

import pandas as pd
pd.set_option('display.max_columns', 500)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/"
folder_lda = "Data/SP500/Model/"
df_t1 = pd.read_csv(directory + folder_lda + "df_sent_ner2.csv")
# Set its index as the url
df_t1.set_index("url", inplace=True)


# Add main text to the df with topic 1 and titles only
df_t1["text"] = df_main["text"]

# Export this updated df as csv
folder_output = "Data/SP500/Main Text/"
df_t1.to_csv(directory + folder_output + "df_sentner_t1.csv", index=True)

import pandas as pd
import re
import string
pd.set_option('display.max_columns', 500)

# Import the updated df 
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentner_t1.csv")

# Define the translation mapping to remove punctuation
punct = dict.fromkeys(string.punctuation, " ")
punct["“"] = " "
punct["”"] = " "
punct["‘"] = " "
punct["’"] = " "
translator = str.maketrans(punct)

# Define a function to clean the text for sentiment analysis
def clean_text(my_string):
    
   # Clean the text
   my_string = re.sub(r"Amazon\.com", " Amazon ", my_string)
   my_string = re.sub(r"AT&T-Time Warner", " AT&T Time Warner ", my_string)
   my_string = re.sub(r"Conoco", " ConocoPhillips ", my_string)
   my_string = re.sub(r"Exxon Mobil", " Exxon ", my_string)
   my_string = re.sub(r"ExxonMobil", " Exxon ", my_string)
   my_string = re.sub(r"Democrats", " Democrats ", my_string)
   my_string = re.sub(r"Federal Reserve", " Fed ", my_string)
   my_string = re.sub(r"Citi", " Citigroup ", my_string)
   my_string = re.sub(r"GE", " General Electric ", my_string)
   my_string = re.sub(r"Goldman Sachs", " Goldman ", my_string)
   my_string = re.sub(r"Alphabet", " Google ", my_string)
   my_string = re.sub(r"JP Morgan", " JPMorgan ", my_string)
   my_string = re.sub(r"McDonald's", " McDonald ", my_string)
   my_string = re.sub(r"McDonald’s", " McDonald ", my_string)
   my_string = re.sub(r"Coca-Cola", " CocaCola ", my_string)
   my_string = re.sub(r"Moody's", " Moody ", my_string)
   my_string = re.sub(r"Moody’s", " Moody ", my_string)
   my_string = re.sub(r"N\.Y\.", " New York ", my_string)
   my_string = re.sub(r"NY", " New York ", my_string)
   my_string = re.sub(r"New York Times", " NYT ", my_string)
   my_string = re.sub(r"Republicans", " Republican ", my_string)
   my_string = re.sub(r"Treasury's", " Treasury ", my_string)
   my_string = re.sub(r"Treasury’s", " Treasury ", my_string)
   my_string = re.sub(r"Treasuries", " Treasury ", my_string)
   my_string = re.sub(r"Wall St", " Wall Street ", my_string)
   my_string = re.sub(r"Wal-Mart", " Walmart ", my_string)
   my_string = re.sub(r"Wells Fargo's", " Wells Fargo ", my_string)
   my_string = re.sub(r"Wells Fargo’s", " Wells Fargo ", my_string)
   my_string = re.sub(r"WSJ", " Wall Street Journal ", my_string)
   
   # Convert all US related terms to "American"
   my_string = re.sub(r"U\.S\.", " American ", my_string)
   my_string = re.sub(r"America", " American ", my_string)
   my_string = re.sub(r"American n", " American ", my_string)
   my_string = re.sub(r"United States", " American ", my_string)
   my_string = re.sub(r"U\.S", " American ", my_string)
   my_string = re.sub(r"US", " American ", my_string)
   
   # Convert words to lower case
   my_string = my_string.lower()
   
   # Remove punctuations
   my_string = my_string.translate(translator)
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   
   # Remove one "American" if it is duplicated
   my_string = re.sub(r"american american", " american ", my_string)
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   return my_string

# Clean the titles
df["ner_text"] = df["text"].apply(lambda x: clean_text(x))

# Calculate the length of the cleaned titles
df["ner_text_len"] = df["ner_text"].str.split().str.len()

# Export the dataframe as csv
df.to_csv(directory + folder + "df_sentner_t1_1.csv", index=False)

