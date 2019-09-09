import string
import re
import pandas as pd

pd.set_option('display.max_columns', 500)

# Import the df with only LDA topic 1 and titles
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
def clean_text_sent(my_string):
   # Expand some of the contraction words
   my_string = re.sub(r"No\.", "number", my_string)
    
   # Convert words to lower case and split them
   my_string = my_string.lower()
   
   # Clean the text
   my_string = re.sub(r"\'s", "  ", my_string)
   my_string = re.sub(r"\’s", "  ", my_string)
   my_string = re.sub(r"\'ve", " have ", my_string)
   my_string = re.sub(r"n\'t", " not ", my_string)
   my_string = re.sub(r"i\'m", "i am ", my_string)
   my_string = re.sub(r"\'re", " are ", my_string)
   my_string = re.sub(r"\'d", " would ", my_string)
   my_string = re.sub(r"\'ll", " will ", my_string)
   my_string = re.sub(r"'t", " not ", my_string)
   my_string = re.sub(r"'cause", " because ", my_string)
   my_string = re.sub(r",000","000", my_string)
   my_string = re.sub(r"u\.s\.", " american ", my_string)
   my_string = re.sub(r"u\.n\.", " unitednation ", my_string)
   my_string = re.sub(r"\.\d", r"\.", my_string)
   my_string = re.sub(r"\+\d", r" positive \d", my_string)
   my_string = re.sub(r"un-", "un", my_string)
   my_string = re.sub(r"write-", "write", my_string)
   my_string = re.sub(r"over-", "over", my_string)

   # Remove punctuations
   my_string = my_string.translate(translator)

   # Clean the text further
   #Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}"," ",my_string)
   #Replace k by 000
   my_string = re.sub(r"(\d+)(k)", r"\g<1> thousand", my_string)
   return my_string

# Clean the text
df["sent_text"] = df["text"].apply(lambda x: clean_text_sent(x))

# Calculate the length of the cleaned titles
df["sent_text_len"] = df["sent_text"].str.split().str.len()

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1.csv", index=False)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1.csv")

# Import the fin_neg and fin_pos sentiment lists
fin_neg = pd.read_csv(directory + folder + "fin_neg.csv")
fin_pos = pd.read_csv(directory + folder + "fin_pos.csv")

# Add columns that need to be populated as part of sentiment analysis
df["count_pos_text"] = ""
df["count_neg_text"] = ""
df["unique_pos_text"] = ""
df["unique_neg_text"] = ""

# Define 3 arrays tracking the counts for each word in each list
count_pos = np.zeros(len(fin_pos))
count_neg = np.zeros(len(fin_neg))

# Define a procedure to count sentiment/negation words for each article
N = len(df)
for i in range(N):
    # Extract the text from df
    text = df["sent_text"].loc[i].split()
    
    # Define the respective dictionaries
    dummy = list(np.zeros(len(fin_pos)).astype(int))
    dict_pos = dict(zip(fin_pos["Vocab"], dummy))
    dummy = list(np.zeros(len(fin_neg)).astype(int))
    dict_neg = dict(zip(fin_neg["Vocab"], dummy))
    # Count the occurrences for each dictionary in text  
    for w in text:
        if w in dict_pos:
           dict_pos[w] += 1
        if w in dict_neg:
           dict_neg[w] += 1
    
    # Calculate the gross numbers of sentiment words
    df["count_pos_text"].loc[i] = np.sum(list(dict_pos.values()))
    df["count_neg_text"].loc[i] = np.sum(list(dict_neg.values()))
    count_pos += np.array(list(dict_pos.values()))
    count_neg += np.array(list(dict_neg.values()))
        
    # Calculate the unique number of sentiment words
    df["unique_pos_text"].loc[i] = np.sum((np.array(list(dict_pos.values())) > 0))
    df["unique_neg_text"].loc[i] = np.sum((np.array(list(dict_neg.values())) > 0))
    if i % 100 == 0:
        print(i)
     
# Convert the count arrays into dictionaries
dict_pos = dict(zip(fin_pos["Vocab"], count_pos))
dict_neg = dict(zip(fin_neg["Vocab"], count_neg))

# Convert the dictionaries into lists of tuples of unique words and frequencies
pos_count_list = [(key,value) for key, value in dict_pos.items()]
neg_count_list = [(key,value) for key, value in dict_neg.items()]

# Sort both lists of tuples
pos_count_list = sorted(pos_count_list, key=lambda x: x[1], reverse=True)
neg_count_list = sorted(neg_count_list, key=lambda x: x[1], reverse=True)

# Print the top 100 words in the list
print(pos_count_list[:100])
print(neg_count_list[:100])

# Unzip the positive count list and negative count list
unzip_pos = list(zip(*pos_count_list))
unzip_neg = list(zip(*neg_count_list))

# Create a df for the unzipped list
pos = {"word": unzip_pos[0], "count": unzip_pos[1]}
df_pos = pd.DataFrame(pos)
neg = {"word": unzip_neg[0], "count": unzip_neg[1]}
df_neg = pd.DataFrame(neg)

# Export the df as csv
df.to_csv(directory + folder + "df_sentscore_t1_1.csv", index=False)
df_pos.to_csv(directory + folder + "df_pos_text_t1.csv", index=False)
df_neg.to_csv(directory + folder + "df_neg_text_t1.csv", index=False)

import string
import pandas as pd
import re
pd.set_option('display.max_columns', 500)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_1.csv")

# Define the translation mapping to remove punctuation
punct = dict.fromkeys(string.punctuation, " ")
punct["“"] = " "
punct["”"] = " "
punct["‘"] = " "
punct["’"] = " "
# Remove the & from the dictionary of punctuation
del punct["&"]
translator = str.maketrans(punct)

# Define a function to clean the text for sentiment analysis
def clean_title_sent(my_string):
    
   # Remove punctuations
   my_string = my_string.translate(translator)
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}"," ", my_string)
   return my_string

# Clean the text
df["sent_title"] = df["ner_title"].apply(lambda x: clean_title_sent(x))

# Calculate the length of the cleaned titles
df["ner_title_len"] = df["sent_title"].str.split().str.len()

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_2.csv", index=False)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_2.csv")

# Import the fin_neg and fin_pos sentiment lists
fin_neg = pd.read_csv(directory + folder + "fin_neg.csv")
fin_pos = pd.read_csv(directory + folder + "fin_pos.csv")

# Add columns that need to be populated as part of sentiment analysis
df["count_pos_title"] = ""
df["count_neg_title"] = ""
df["unique_pos_title"] = ""
df["unique_neg_title"] = ""

# Define 3 arrays tracking the counts for each word in each list
count_pos = np.zeros(len(fin_pos))
count_neg = np.zeros(len(fin_neg))

# Define a procedure to count sentiment/negation words for each article
N = len(df)
for i in range(N):
    # Extract the text from df
    text = df["sent_title"].loc[i].split()
    
    # Define the respective dictionaries
    dummy = list(np.zeros(len(fin_pos)).astype(int))
    dict_pos = dict(zip(fin_pos["Vocab"], dummy))
    dummy = list(np.zeros(len(fin_neg)).astype(int))
    dict_neg = dict(zip(fin_neg["Vocab"], dummy))
    # Count the occurrences for each dictionary in text  
    for w in text:
        if w in dict_pos:
           dict_pos[w] += 1
        if w in dict_neg:
           dict_neg[w] += 1
    
    # Calculate the gross numbers of sentiment words
    df["count_pos_title"].loc[i] = np.sum(list(dict_pos.values()))
    df["count_neg_title"].loc[i] = np.sum(list(dict_neg.values()))
    count_pos += np.array(list(dict_pos.values()))
    count_neg += np.array(list(dict_neg.values()))
        
    # Calculate the unique number of sentiment words
    df["unique_pos_title"].loc[i] = np.sum((np.array(list(dict_pos.values())) > 0))
    df["unique_neg_title"].loc[i] = np.sum((np.array(list(dict_neg.values())) > 0))
    if i % 100 == 0:
        print(i)
     
# Convert the count arrays into dictionaries
dict_pos = dict(zip(fin_pos["Vocab"], count_pos))
dict_neg = dict(zip(fin_neg["Vocab"], count_neg))

# Convert the dictionaries into lists of tuples of unique words and frequencies
pos_count_list = [(key,value) for key, value in dict_pos.items()]
neg_count_list = [(key,value) for key, value in dict_neg.items()]

# Sort both lists of tuples
pos_count_list = sorted(pos_count_list, key=lambda x: x[1], reverse=True)
neg_count_list = sorted(neg_count_list, key=lambda x: x[1], reverse=True)

# Print the top 100 words in the list
print(pos_count_list[:100])
print(neg_count_list[:100])

# Unzip the positive count list and negative count list
unzip_pos = list(zip(*pos_count_list))
unzip_neg = list(zip(*neg_count_list))

# Create a df for the unzipped list
pos = {"word": unzip_pos[0], "count": unzip_pos[1]}
df_pos = pd.DataFrame(pos)
neg = {"word": unzip_neg[0], "count": unzip_neg[1]}
df_neg = pd.DataFrame(neg)

# Export the df as csv
df.to_csv(directory + folder + "df_sentscore_t1_3.csv", index=False)
df_pos.to_csv(directory + folder + "df_pos_title_t1.csv", index=False)
df_neg.to_csv(directory + folder + "df_neg_title_t1.csv", index=False)

import pandas as pd
import re
import string

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"

# Import the csv for LDA topic 1
df = pd.read_csv(directory + folder + "df_sentscore_t1_3.csv")

# Define the translation mapping to remove punctuation
punct = dict.fromkeys(string.punctuation, " ")
punct["“"] = " "
punct["”"] = " "
punct["‘"] = " "
punct["’"] = " "
# Remove the & from the dictionary of punctuation
del punct["&"]
translator = str.maketrans(punct)

# Define a function to clean the text for sentiment analysis
def clean_ner_text(my_string):
    
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
   my_string = re.sub(r"\&", "and", my_string)
   
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
df["ner_text"] = df["text"].apply(lambda x: clean_ner_text(x))

# Export the dataframe as csv
df.to_csv(directory + folder + "df_sentscore_t1_4.csv", index=False)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_4.csv")

# Import the US entities list
vocab = pd.read_csv(directory + folder + "us_entities.csv")

# Convert the vocab df to a dictionary
keys = list(vocab["word"])
values = list(vocab.index)
dictionary = dict(zip(keys, values))

# Try out one example
count_vector = CountVectorizer(vocabulary=dictionary, ngram_range=(1,3))
X = count_vector.fit_transform(df["ner_text"])

import numpy as np
import pandas as pd

# Perform sum along the batch dimension
X_tf = np.squeeze(np.array(X.sum(axis=0)))

# Transform the tf into a df
tf = {"word": list(dictionary.keys()), "term_freq": list(X_tf)}
df_tf = pd.DataFrame(tf)

# Export the tf df as csv
df_tf.to_csv(directory + folder + "df_us_ent_text_t1.csv", index=False)

# Perform sum along the vocab dimension
X_count = np.squeeze(np.array(X.sum(axis=1)))

# Assign these counts to a new column in the df
df["us_entity_count"] = X_count

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_5.csv", index=False)

import pandas as pd
import re

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"

# Import the csv for LDA topic 1
df = pd.read_csv(directory + folder + "df_sentscore_t1_5.csv")

# Define a function to clean the text for sentiment analysis
def clean_ner_title(my_string):
    
   # Clean the text
   my_string = re.sub(r"\&", "and", my_string)
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   return my_string

# Clean the titles
df["sent_title"] = df["sent_title"].apply(lambda x: clean_ner_title(x))

# Export the dataframe as csv
df.to_csv(directory + folder + "df_sentscore_t1_6.csv", index=False)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_6.csv")

# Import the US entities list
vocab = pd.read_csv(directory + folder + "us_entities.csv")

# Convert the vocab df to a dictionary
keys = list(vocab["word"])
values = list(vocab.index)
dictionary = dict(zip(keys, values))

# Try out one example
count_vector = CountVectorizer(vocabulary=dictionary, ngram_range=(1,3))
X = count_vector.fit_transform(df["sent_title"])

import numpy as np
import pandas as pd

# Perform sum along the batch dimension
X_tf = np.squeeze(np.array(X.sum(axis=0)))

# Transform the tf into a df
tf = {"word": list(dictionary.keys()), "term_freq": list(X_tf)}
df_tf = pd.DataFrame(tf)

# Export the tf df as csv
df_tf.to_csv(directory + folder + "df_us_ent_title_t1.csv", index=False)

# Perform sum along the vocab dimension
X_count = np.squeeze(np.array(X.sum(axis=1)))

# Assign these counts to a new column in the df
df["us_entity_title"] = X_count

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_7.csv", index=False)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_7.csv")

# Add the combined count columns of pos and neg
df["count_title"] = df["count_pos_title"] + df["count_neg_title"]
df["unique_title"] = df["unique_pos_title"] + df["unique_neg_title"]
df["count_text"] = df["count_pos_text"] + df["count_neg_text"]
df["unique_text"] = df["unique_pos_text"] + df["unique_neg_text"]

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_8.csv", index=False)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_8.csv")

# Define the criteria for US entities for an article to be included
title_threshold = 1
text_threshold = 5

# Define the two columns for these thresholds
df["us_entity_title_check"] = (df["us_entity_title"] >= title_threshold) * 1
df["us_entity_text_check"] = (df["us_entity_count"] >= text_threshold) * 1  
df["us_entity_check"] = df["us_entity_title_check"] + df["us_entity_text_check"]
df["us_entity_check"] = (df["us_entity_check"] > 0) * 1

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_9.csv", index=False)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_9.csv")

# Define the criteria for US entities for an article to be included
title_threshold = 1
unique_threshold = 3
text_threshold = 6

# Define the two columns for these thresholds
df["unique_title_check"] = (df["unique_title"] >= title_threshold) * 1
df["unique_text_check"] = (df["unique_text"] >= unique_threshold) * 1  
df["count_text_check"] = (df["count_text"] >= text_threshold) * 1
df["count_check"] = df["unique_title_check"] + df["unique_text_check"] + df["count_text_check"]
df["count_check"] = (df["count_check"] > 0) * 1

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_10.csv", index=False)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_10.csv")

# Define the two columns for these thresholds
df["final_check"] = df["us_entity_check"] * df["count_check"] 

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_11.csv", index=False)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentscore_t1_11.csv")

# Define the two columns for these thresholds
df = df[df["final_check"]==1]

# Export the updated df as csv
df.to_csv(directory + folder + "df_sentscore_t1_12.csv", index=False)

