#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:24:01 2018

@author: kiyoshitaro
"""

import os
from six.moves.urllib.request import urlretrieve 
#import zipfile
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import collections
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer,WordNetLemmatizer

        

train_data_raw = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','labeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)

test_data_raw = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv'),header=0,
                    delimiter="\t", quoting=3)
print("Review : ", train_data_raw["review"][0])

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
#        return(" ".join(words_lemma))
        return words_lemma


train_review_clear = []
for i in range(len(train_data_raw["review"])):
    train_review_clear.append(preprocess_review(train_data_raw["review"][i],True))

test_review_clear = []
for i in range(len(test_data_raw["review"])):
    test_review_clear.append(preprocess_review(test_data_raw["review"][i],True))
max_sent_length = 0

all_review_clear = train_review_clear+ test_review_clear
for i in range(len(all_review_clear)):
    if(max_sent_length < len(all_review_clear[i])):
        max_sent_length = len(all_review_clear[i])

for i in range(len(train_review_clear)):
    for _ in range(max_sent_length - len(train_review_clear[i])):
        train_review_clear[i].append("pad")

for i in range(len(test_review_clear)):
    for _ in range(max_sent_length - len(test_review_clear[i])):
        test_review_clear[i].append("pad")
        
def build_dataset(data):
    words = []
    data_list = []
    count = []
    
    for sent in data:
        words.extend(sent)
    print("Length words: ", len(words))
    
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for w, _ in count:
        dictionary[w] = len(dictionary)
    for sent in data:
        tmp = list()
        for word in sent:
            index = dictionary[word]
            rmp.append(index)
        
        data_list.append(tmp)
    
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    
    return data_list, count , dictionary, reverse_dictionary

review_index , count , dictionary , reverse_dictionary = build_dataset(all_review_clear)

    