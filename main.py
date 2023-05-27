## import pckgs
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import neattext.functions as nfx
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nltk.download('punkt')


## upload data
clean_tweets_df = pd.read_excel("JWC_raw_data.xlsx")
print(clean_tweets_df.head())

## task
# + text
#    - text preprocessing
#    - sentiment analysis
#    - keyword extraction
#    - entity extraction

# get info
clean_tweets_df.info()

# check columns
print(clean_tweets_df.columns)

# unique locations and counts
print(clean_tweets_df['Username'].value_counts())

# plot largest value counts
plt.figure(figsize=(15,10))
clean_tweets_df['Username'].value_counts().nlargest(15).plot(kind='bar')
plt.xticks(rotation='vertical')
plt.title('Users Who Tweeted the Most')
# tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.55)


# create df of text data
text_df = clean_tweets_df.drop(['Datetime', 'Tweet Id', 'Username', 'User Location',
       'Reply Count', 'Retweet Count', 'Like Count', 'Quote Count', 'Date',
       'compound', 'Month', 'Text'], axis=1)

print(text_df.head())

# perform  stemming
stemmer = PorterStemmer()
def stemming(data):
    words = word_tokenize(data)  # Tokenize the input text into words
    filtered_text = [w for w in words if not w in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]  # Perform stemming on each word
    return ' '.join(stemmed_words)  # Join the stemmed words back into a single string

text_df['Clean Tweet'] = text_df['Clean Tweet'].apply(stemming)

def polarity(filtered_text):
    return TextBlob(filtered_text).sentiment.polarity

text_df['polarity'] = clean_tweets_df['Clean Tweet'].apply(polarity)
print(text_df['polarity'].head())

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"
    
text_df['sentiment'] = text_df['polarity'].apply(sentiment)
print(text_df.head())