import pandas as pd
pd.set_option('display.max_columns', 1000)

directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
file = "df_final.csv"

df = pd.read_csv(directory+file)

publish = df["published"]
publish1 = df[publish.isin(publish[publish.duplicated()])]

# Re-export the section_null dataframe as csv
publish1.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/pub_dup.csv",
                index=False)
                
title = df["title"]

title1 = df[title.isin(title[title.duplicated()])]

# Re-export the section_null dataframe as csv
title1.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/title_dup.csv",
              index=True)

title2 = title[title.duplicated()]

title2.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/title_dup1.csv",
              index=False)

text = df["text"]
text1 = df[text.isin(text[text.duplicated()])]

# Re-export the section_null dataframe as csv
text1.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/text_dup.csv",
             index=True)

misstitle = df[df["title"].isnull()]
misstitle.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/misstitle.csv",
                 index=True)
 
df1 = df[df["title"].notnull()]
df1.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/removetextdup.csv",
           index=True)

# Check null values across the sections
print(df.isnull().sum())

"""
print(df.isnull().sum()/len(df)*100)
print(len(df))
"""

# Remove the rows with null values except those under Section
directory1 = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/EDA/"
column_list = ["title","text","published"]
df_filter = df[column_list]
df_filter["Filter"] = np.sum(df_filter.isnull(),axis=1)
df_clean = df[df_filter["Filter"]==0]
df_clean.reset_index(inplace=True, drop=True)
print(df_clean.head())
print(len(df_clean))

# Define a new df with the remaining null values only
section = df_clean["section"]
df_clean_null = df_clean[section.isnull()]
df_clean_null.reset_index(inplace=True, drop=True)

# Export the df as csv
df_clean.to_csv(directory1 + "df_clean.csv", index=False)
df_clean_null.to_csv(directory1 + "df_clean_null.csv", index=False)

# Split the df into null and non-null
section = df["section"]
df_notnull = df[section.notnull()]
df_null = df[section.isnull()]

# Reset the index of both df
df_notnull.reset_index(inplace=True, drop=True)
df_null.reset_index(inplace=True, drop=True)

import numpy as np         
import string
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer

# Define a cleaning procedure for the titles
def clean_text(my_string):
    
   # Convert words to lower case and split them
   my_string = my_string.lower()
   
   # Clean the text
   my_string = re.sub(r"\'s", "  ", my_string)
   my_string = re.sub(r"\'ve", " have ", my_string)
   my_string = re.sub(r"n't", " not ", my_string)
   my_string = re.sub(r"i'm", "i am ", my_string)
   my_string = re.sub(r"\'re", " are ", my_string)
   my_string = re.sub(r"\'d", " would ", my_string)
   my_string = re.sub(r"\'ll", " will ", my_string)
   my_string = re.sub(r"'t", " not ", my_string)
   my_string = re.sub(r"'cause", " because ", my_string)
   my_string = re.sub(r",000","000", my_string)
   my_string = re.sub(r"u\.s\.", " american ", my_string)
   my_string = re.sub(r"u\.n\.", " un ", my_string)
   my_string = re.sub(r"\.\d", r"\.", my_string)
   my_string = re.sub(r"\+\d", r" positive \d", my_string)

   # Remove punctuations
   translator = str.maketrans(dict.fromkeys(string.punctuation, " "))
   my_string = my_string.translate(translator)
   
   # Remove stop words
   my_string = my_string.split()
   stops = set(stopwords.words("english"))
   my_string = [w for w in my_string if not w in stops]
   my_string = " ".join(my_string)

   # Clean the text further
   #Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}"," ",my_string)
   #Replace k by 000
   my_string = re.sub(r"(\d+)(k)", r"\g<1>000", my_string)

   # Stemming
   stemmer = SnowballStemmer("english")
   my_string = my_string.split()
   stemmed_words = [stemmer.stem(word) for word in my_string]
   my_string = " ".join(stemmed_words)
   
   return my_string

# Clean the titles for both df
df_notnull["clean_title"] = df_notnull["title"].apply(lambda x: clean_text(x))
df_null["clean_title"] = df_null["title"].apply(lambda x: clean_text(x))

# Export both df as csv
df_notnull.to_csv(directory + "df_notnull.csv", index=False)
df_null.to_csv(directory + "df_null.csv", index=False)

# Re-import clean titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/EDA/"
file1 = "df_null.csv"
file2 = "df_notnull.csv"

df_null = pd.read_csv(directory + file1)
df_notnull = pd.read_csv(directory + file2)
print(df_null.head())

# Define the data input in the training/test set for 1-NN
df_notnull_filter = df_notnull[df_notnull["section"] != "Special Reports"]
df_notnull_filter.reset_index(inplace=True, drop=True)
X_train = df_notnull_filter["clean_title"]
X_test = df_null["clean_title"]

# Featurise the data input with TF-IDF
tfidf = TfidfVectorizer(max_features = 10000, sublinear_tf=True)
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)
print(X_train.shape)
print(X_test.shape)
print(df_notnull.head())

# Define the labels in the training set for 1-NN
# section_train = df_notnull["section"]
section_train = df_notnull_filter["section"]
section_unique = section_train.unique()
N = len(section_unique)
section_index = list(range(N))
dict_section = dict(zip(section_unique,section_index))
y_train = np.zeros(len(section_train))

for i in range(len(section_train)):
    y_train[i] = dict_section[section_train.loc[i]]

# Reverse the keys and values in the dictionary
inverted_dict = dict([value,key] for key,value in dict_section.items())

from sklearn.neighbors import KNeighborsClassifier
hyperparameter = [1]

for h in hyperparameter:
    print("h =", h)
    knn = KNeighborsClassifier(n_neighbors=h)
    knn.fit(X_train,y_train)
    prediction = knn.predict(X_test)
    prediction_int = prediction.astype(int)
    for i in range(len(prediction_int)):
        df_null["section"].loc[i] = inverted_dict[prediction_int[i]]
        if i % 100 == 0:
           print(i)
    df_null.to_csv(directory + "df_null_fill_{}_filter.csv".format(h), index=False)

# Re-import data
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/EDA/"
file1 = "df_null_fill_1_filter.csv"
file2 = "df_notnull.csv"

df_null_fill = pd.read_csv(directory + file1)
df_notnull = pd.read_csv(directory + file2)

# Stack both df together
df = pd.concat([df_notnull,df_null_fill], axis=0)

# Convert published to datetime format
df["published"] = pd.to_datetime(df["published"])

# Sort the dataframe by published
df.sort_values(by="published", inplace=True)
df.reset_index(inplace=True, drop=True)
df.to_csv(directory + "df_all.csv", index=False)

# Find the duplication in the title column
title = df["title"]
df_title_dup = df[title.duplicated()]

# Find the duplication in the text column
text = df["text"]
df_text_dup = df[text.duplicated()]

# Export both df as csv
df_title_dup.to_csv(directory + "df_title_dup.csv", index=True)
df_text_dup.to_csv(directory + "df_text_dup.csv", index=True)

# Export all duplications as csv
df_title_dup_all = df[title.isin(title[title.duplicated()])]
df_text_dup_all = df[text.isin(text[text.duplicated()])]
df_title_dup_all.to_csv(directory + "df_title_dup_all.csv", index=True)
df_text_dup_all.to_csv(directory + "df_text_dup_all.csv", index=True)

# Reimport the all version of csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/EDA/"
file = "df_all.csv"
df = pd.read_csv(directory + file)

# Find the duplication in the title column
title = df["title"]
text = df["text"]
mask = title.duplicated()*1 + text.duplicated()*1

# Remove all the duplication 
df_removedup = df[mask == 0]
df_removedup.reset_index(inplace=True, drop=True)

# Export this df as csv
df_removedup.to_csv(directory + "df_removedup.csv", index=False)

