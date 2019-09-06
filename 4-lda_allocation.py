import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import LatentDirichletAllocation as LDA

from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

# Import the latest df that contains news titles only
df = pd.read_csv("/kaggle/input/df_title7/df_title7.csv")

# Define the df for training


df_train = df[df["split"]=="train"]

# Define the df for test
df_test = df[df["split"]=="test"]

# Join the different processed titles together.
long_string = ','.join(list(df_train['clean_title'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, 
                      contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_file("title_cloud.png")
wordcloud.to_image()

# Initialise the count vectorizer
count_vectorizer = CountVectorizer(max_features=15000)

# Fit and transform the processed titles
count_vectorizer.fit(df_train['clean_title'])
count_data = count_vectorizer.transform(df_train['clean_title'])
count_test = count_vectorizer.transform(df_test['clean_title'])

sns.set_style('whitegrid')
%matplotlib inline

# Define a list to contain the dictionary
words = count_vectorizer.get_feature_names()

# Count the number of occurrence for each word 
total_counts = np.zeros(len(words))
for t in count_data:
    total_counts+=t.toarray()[0]

# Create a dictionary where words are keys and total counts are values
count_dict = zip(words, total_counts)

# Sort this in descending order and get the top 10 words only
count_dict_10 = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
words = [w[0] for w in count_dict_10]
counts = [w[1] for w in count_dict_10]
x_pos = np.arange(len(words)) 

# Plot the occurrences of the 10 most common words
fig = plt.figure(2, figsize=(10, 10/1.6180))
plt.subplot(title='10 most common words')
sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, counts, palette='husl')
plt.xticks(x_pos, words, rotation=90) 
plt.xlabel('words')
plt.ylabel('counts')
plt.show()
fig.savefig("common_word.png")

# Suppress the warning signs
warnings.simplefilter("ignore", DeprecationWarning)

# Tweak the two parameters below (use int values below 15)
number_topics = 3
random_seed = 2

# Create and fit the LDA model
lda = LDA(n_components=number_topics, random_state=random_seed, verbose=1)
lda.fit(count_data)

# Define the word list
number_words = 10
words = count_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("\nTopic #%d" % topic_idx)
    print(" ".join([words[i] for i in topic.argsort()[:-number_words:-1]]))
    
pyLDAvis.enable_notebook()
sklearn_lda.prepare(lda, count_data, count_vectorizer)

# Calculate the topic distributions for all articles in the training and test sets
X_train = lda.transform(count_data)
X_test = lda.transform(count_test)

# Find the topic given the topic distribution in the training set
Topic_train = np.argmax(X_train, axis=1)
Topic_train_df = pd.DataFrame(Topic_train, columns=["Topic"])
# Find the topic given the topic distribution in the test set
Topic_test = np.argmax(X_test, axis=1)
Topic_test_df = pd.DataFrame(Topic_test, columns=["Topic"])
# Reset the index of the df for Topic_test
Topic_test_df.index += df_test.index[0]

# Allocate the topic to the original dataframe
Topic = pd.concat([Topic_train_df, Topic_test_df], axis=0) + 1
df["Topic"] = Topic

# Export the df as csv
df.to_csv("df_topic.csv", index=False)
