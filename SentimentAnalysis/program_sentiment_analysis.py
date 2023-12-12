import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input
from sklearn.utils import shuffle
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
DATASET_ENCODING = "ISO-8859-1"

df = pd.read_csv("archive.zip",encoding=DATASET_ENCODING)
df= df.iloc[:,[0,-1]]
df.columns = ['sentiment','tweet']
df = pd.concat([df.query("sentiment==0").sample(20000),df.query("sentiment==4").sample(20000)])
df.sentiment = df.sentiment.map({0:0,4:1})
df =  shuffle(df).reset_index(drop=True)

df,df_test = train_test_split(df,test_size=0.2)

print(df.head(5))

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

print(embed(['hi samuels, this is our project']).numpy().shape)

def vectorize(df):
    embeded_tweets = embed(df['tweet'].values.tolist()).numpy()
    targets = df.sentiment.values
    return embeded_tweets,targets

embeded_tweets,targets = vectorize(df_test)

from sklearn.metrics import accuracy_score


#TextBlob
from textblob import TextBlob

def text_sentiment(text):
    testimonial = TextBlob(text)
    return int(testimonial.sentiment.polarity>0.5)

predictions = df_test.tweet.map(lambda x :  text_sentiment(x))
result = accuracy_score(predictions,targets)
print('TextBlob')
print('Accuracy: ', result)

#Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def text_sentiment_vader(text):
    vs = analyzer.polarity_scores(text)
    return int(vs.get("compound")>0)
 
predictions = df_test.tweet.map(lambda x : text_sentiment_vader(x))
result = accuracy_score(predictions.values,targets)
print('Vader')
print('Accuracy: ', result)

#Flair
from flair.models import TextClassifier
from flair.data import Sentence
#classifier = TextClassifier.load('en-sentiment')

def text_sentiment_flair(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    return np.round(sentence.labels[0].score)

predictions = df_test.tweet.map(lambda x : text_sentiment_flair(x))
result = accuracy_score(predictions.values,targets)
print('Flair')
print('Accuracy: ', result)



