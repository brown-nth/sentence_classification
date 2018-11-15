#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:41:04 2018

@author: kiyoshitaro
"""

import os
from six.moves.urllib.request import urlretrieve 
#import zipfile
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer,WordNetLemmatizer



#url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3971/labeledTrainData.tsv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1542091665&Signature=n0tPbpvA%2FuBU4IUPbnIazMM37dHtHpF7Za%2Bai8xAGJSBoqkpceLp1t2eEOvK0ejWS4SYnp3oHQXfbJbB0SiSjOIo6wZcrFXSUVyRxDYzGbeeNUSaBi%2Ff8qytR66OTV8hFWO%2FDPUrc7rr6NbJNtH5rteBC3BJahNyi%2F01COpfsq1iZqii3wTi7WrDmyWgh%2BT%2BaUk%2F2rG80AhiFKN5iqqyrLzG1q22I1RWeT%2BHr4Qf8kpzXhfe8hkNJ%2F9Xws9uCxLgYKAFCrYEjUTyypeuhoXkoJPK5akbV6OiEUfAvYCarHW%2FLr34ZbASsohHoahdaoqQpaeq8n2FTQTA%2F%2Fzun%2BqGkw%3D%3D"
#
#def maybe_download(filename):
#    if not os.path.exists('data/' + filename):
#        filename , _ = urlretrieve(url,filename)
#    
#    return 'data' + filename
#
#        
#filename = maybe_download("data/labeledTrainData.tsv.zip")

train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','labeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)

test_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv'),header=0,
                    delimiter="\t", quoting=3)
print("Review : ", train_data["review"][0])


def preprocess_review( review_text, remove_stopwords=False ):
        # 1. Remove HTML
        review_text = BeautifulSoup(review_text,features="html5lib").get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        words = review_text.lower()
        if remove_stopwords:
#            "This is a king." --> "this" , "is" , "a" , "king" , "."
            word_tokens = word_tokenize(words) 
#            remove a , an , the , with ,...
            stops = set(stopwords.words("english"))
            words_not_stop = [w for w in word_tokens if not w in stops]
            # programmer, program ,..... --> program (heuristic)
#            ps = PorterStemmer() 
#            words_stem = [ps.stem(w) for w in words_not_stop]
            
            lemmatizer = WordNetLemmatizer()
            words_lemma = [lemmatizer.lemmatize(w) for w in words_not_stop]

        return(" ".join(words_lemma))
    
vectorizer = CountVectorizer(analyzer = "word",   
                     tokenizer = None,    
                     preprocessor = None, 
                     stop_words = None,   
                     max_features = 5000)

train_review = []
for i in range(len(train_data["review"])):
    train_review.append(preprocess_review(train_data["review"][i],True))

test_review = []
for i in range(len(test_data["review"])):
    test_review.append(preprocess_review(test_data["review"][i],True))

train_data_features = vectorizer.fit_transform(train_review)
test_data_features = vectorizer.fit_transform(test_review)

np.asarray(train_data_features)
np.asarray(test_data_features)

print("Take a while ...")
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, train_data["sentiment"])
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)



