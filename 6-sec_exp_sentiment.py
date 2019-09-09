import pandas as pd
import re

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"

# Import the csv for LDA topic 1
df = pd.read_csv(directory + "df_topic_1.csv")

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
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   
   # Remove one "American" if it is duplicated
   my_string = re.sub(r"american american", " american ", my_string)
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   return my_string

# Clean the titles
df["ner_title"] = df["title"].apply(lambda x: clean_text(x))

# Calculate the length of the cleaned titles
df["ner_title_length"] = df["ner_title"].str.split().str.len()

# Export the dataframe as csv
df.to_csv(directory + folder + "df_sent_ner.csv", index=False)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"

# Import the csv files for different types of vocab
fin_neg = pd.read_csv(directory + folder + "fin_neg.csv")
fin_pos = pd.read_csv(directory + folder + "fin_pos.csv")
us_entity = pd.read_csv(directory + folder + "us_entity.csv")

# Concatenate all the vocab together
df_vocab = pd.concat([fin_neg, fin_pos, us_entity], axis=0)
df_vocab.reset_index(inplace=True, drop=True)

# Export the vocab as csv
df_vocab.to_csv(directory + folder + "vocab.csv", index=False)

