# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:22:11 2019

@author: Charles
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Librairies 
"""""""""""""""""""""""""""""""""""""""""""""""""""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Training Data Obtention
"""""""""""""""""""""""""""""""""""""""""""""""""""

train_source = open("text.txt","r", errors = "ignore")
lignes = train_source.readlines()
train_source.close()

train_liste_topic = []
train_liste_sentiment = []
train_liste_date = []
train_liste_text = []

train_doc_travail = []

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Training Data Treatment 
"""""""""""""""""""""""""""""""""""""""""""""""""""

for ligne in lignes:
    liste = ligne.split(',""')   
   
    train_liste_topic.append(liste[0])
    train_liste_sentiment.append(liste[1])
    train_liste_date.append(liste[2])
    train_liste_text.append(liste[3])
     
train_doc_travail.append(train_liste_topic)
train_doc_travail.append(train_liste_sentiment)
train_doc_travail.append(train_liste_date)
train_doc_travail.append(train_liste_text)

del(train_liste_topic[0])
del(train_liste_sentiment[0])
del(train_liste_date[0])
del(train_liste_text[0])

#print(train_liste_topic)
#print(train_liste_sentiment)
#print(train_liste_date)
#print(train_liste_text)


from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

train_liste_text_util =[]

for X in train_liste_text:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X))
    
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
        
        train_liste_text_util.append(document)

#print(train_liste_text_util)

train_liste_sentiment_util =[]
        
for X in train_liste_sentiment:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X))
    
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
        
        train_liste_sentiment_util.append(document)



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
train_tweet_set = vectorizer.fit_transform(train_liste_text_util).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
train_tweet_set = tfidfconverter.fit_transform(train_tweet_set).toarray()


#print(train_tweet_set)




"""""""""""""""""""""""""""""""""""""""""""""""""""
#Test Data Obtention 
"""""""""""""""""""""""""""""""""""""""""""""""""""

test_source = open("zeub.txt","r", errors = "ignore")
lignes = test_source.readlines()
test_source.close()

test_liste_topic = []
test_liste_sentiment = []
test_liste_date = []
test_liste_text = []

test_doc_travail = []

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Test Data Treatment 
"""""""""""""""""""""""""""""""""""""""""""""""""""

for ligne in lignes:
    liste = ligne.split(',""')
    test_liste_topic.append(liste[0])
    test_liste_sentiment.append(liste[1])
    test_liste_date.append(liste[2])
    test_liste_text.append(liste[3])
        
test_doc_travail.append(test_liste_topic)
test_doc_travail.append(test_liste_sentiment)
test_doc_travail.append(test_liste_date)
test_doc_travail.append(test_liste_text)

del(test_liste_topic[0])
del(test_liste_sentiment[0])
del(test_liste_date[0])
del(test_liste_text[0])

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

test_liste_text_util =[]

for X in test_liste_text:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X))
    
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
        
        test_liste_text_util.append(document)


#print(test_liste_text_util)

test_liste_sentiment_util =[]
        
for X in test_liste_sentiment:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X))
    
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
        
        test_liste_sentiment_util.append(document)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
test_tweet_set = vectorizer.fit_transform(test_liste_text_util).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
test_tweet_set = tfidfconverter.fit_transform(test_tweet_set).toarray()

#print(test_tweet_set)


#print(len(train_tweet_set))
#print(len(train_liste_sentiment_util))
#print(len(test_tweet_set))
#print(len(test_liste_sentiment_util))



"""""""""""""""""""""""""""""""""""""""""""""""""""
#Classifier Training 
"""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(train_tweet_set,train_liste_sentiment_util ) 

"""""""""""""""""""""""""""""""""""""""""""""""""""
#Classifier Prediction
"""""""""""""""""""""""""""""""""""""""""""""""""""

test_pred = classifier.predict(test_tweet_set)
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""
#Classifier Evaluation
"""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_liste_sentiment_util,test_pred))
print(classification_report(test_liste_sentiment_util,test_pred))
print(accuracy_score(test_liste_sentiment_util,test_pred))


"""