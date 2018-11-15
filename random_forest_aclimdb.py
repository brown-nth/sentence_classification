#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 22:20:28 2018

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
import time


test_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv'),header=0,
                    delimiter="\t", quoting=3)
#print("Review : ", train_data["review"][0])

begin_load = time.time()
#   IMDB
train_review_raw = []
train_label= []
def get_train_data_to_array(path_data, label):
    for path , subdirs , files in os.walk(path_data):
        for name in files:
             with open(os.path.join(path,name)) as f:
                 train_review_raw.append(f.read())
                 train_label.append(label)

get_train_data_to_array("./data/aclImdb_v1/aclImdb/train/neg",0)
get_train_data_to_array("./data/aclImdb_v1/aclImdb/train/pos",1)
print("Loaded all raw data !")
print("Begin clean raw data ... ")

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


train_review_clear = []
for i in range(len(train_review_raw)):
    train_review_clear.append(preprocess_review(train_review_raw[i],True))
#
test_review_clear = []
for i in range(len(test_data["review"])):
    test_review_clear.append(preprocess_review(test_data["review"][i],True))


train_data_features = vectorizer.fit_transform(train_review_clear)
test_data_features = vectorizer.fit_transform(test_review_clear)

np.asarray(train_data_features)
np.asarray(test_data_features)
end_load = time.time()
print("Clean all data !")

print("Take a while to compute model ...")
forest = RandomForestClassifier(n_estimators = 100)
#forest = forest.fit(train_data_features, train_data["sentiment"])

forest = forest.fit(train_data_features, train_label)
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)

end_model = time.time()
print("Done !")
print("Time to load data : ", end_load - begin_load)
print("Time to run model : ", end_model - end_load)

