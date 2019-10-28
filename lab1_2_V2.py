# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:14:22 2019

@author: Charles
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Librairies 
"""""""""""""""""""""""""""""""""""""""""""""""""""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Data Obtention
"""""""""""""""""""""""""""""""""""""""""""""""""""

tweets_train = pd.read_csv('Train.csv', engine = 'python')
tweets_test = pd.read_csv('Test.csv', engine = 'python')
tweets_train.columns = ['Topics', 'Sentiments', 'TweetDate', 'TweetText']
tweets_test.columns = ['Topics', 'Sentiments', 'TweetDate', 'TweetText']

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Training Data Treatment 
"""""""""""""""""""""""""""""""""""""""""""""""""""

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

train_text =[]

for X in range(0,len(tweets_train.TweetText)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(tweets_train.TweetText[X]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
        
    train_text.append(document) 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
train_tweet_set = vectorizer.fit_transform(train_text).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
train_tweet_set = tfidfconverter.fit_transform(train_tweet_set).toarray()


"""""""""""""""""""""""""""""""""""""""""""""""""""
#Test Data Treatment 
"""""""""""""""""""""""""""""""""""""""""""""""""""

test_text =[]

for X in range(0,len(tweets_test.TweetText)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(tweets_test.TweetText[X]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
        
    test_text.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
test_tweet_set = vectorizer.fit_transform(test_text).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
test_tweet_set = tfidfconverter.fit_transform(test_tweet_set).toarray()

  
    
"""""""""""""""""""""""""""""""""""""""""""""""""""
#Classifier Training 
"""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(train_tweet_set,tweets_train.Sentiments) 

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Classifier Prediction
"""""""""""""""""""""""""""""""""""""""""""""""""""

test_pred = classifier.predict(test_tweet_set)

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Classifier Evaluation
"""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(tweets_test.Sentiments,test_pred))
print(classification_report(tweets_test.Sentiments,test_pred))
print(accuracy_score(tweets_test.Sentiments,test_pred))
