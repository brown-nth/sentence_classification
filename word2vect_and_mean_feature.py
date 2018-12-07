#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 02:04:25 2018

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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import time
import logging
import pickle


from gensim.models import word2vec, Word2Vec

begin = time.time()
unlabeled_train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','unlabeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)
train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','labeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)

test_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv'),header=0,
                    delimiter="\t", quoting=3)
print("Review : ", train_data["review"][0])

##   IMDB
#train_review_raw = []
#train_label= []
#def get_train_data_to_array(path_data, label):
#    for path , subdirs , files in os.walk(path_data):
#        for name in files:
#             with open(os.path.join(path,name)) as f:
#                 train_review_raw.append(f.read())
#                 train_label.append(label)
#
#get_train_data_to_array("./data/aclImdb_v1/aclImdb/train/neg",0)
#get_train_data_to_array("./data/aclImdb_v1/aclImdb/train/pos",1)
#print("Loaded all raw data !")
#print("Begin clean raw data ... ")


def preprocess_review( review_text, remove_stopwords=False ):
        # 1. Remove HTML
        review_text = BeautifulSoup(review_text,features="html5lib").get_text()
        #
        review_text = review_text.lower()

        if(remove_stopwords):
            review_text = re.sub("[^a-zA-Z]"," ", review_text)
            word_token = word_tokenize(review_text)
            stops = set(stopwords.words("english"))
            words_not_stop = [word for word in word_token if not word in stops]
            lemmatizer = WordNetLemmatizer()
            words_lemma = [lemmatizer.lemmatize(w) for w in words_not_stop]
            return words_lemma
        
        sent_tokens = sent_tokenize(review_text)
        sentences = []
        for sent in sent_tokens:
            sent = re.sub("[^a-zA-Z]"," ", sent)
            sentences.append(sent.split())
        return sentences
      
begin_preprocess_data = time.time()
sentences = []
labeled_train_review_raw = train_data["review"]
unlabeled_train_review_raw = unlabeled_train_data["review"]
test_review_raw = test_data["review"] 
test_review_clean = []
train_review_clean = []

for review in labeled_train_review_raw:
    sentences.extend(preprocess_review(review, False))
    train_review_clean.append(preprocess_review(review, True))
    
for review in unlabeled_train_review_raw:
    sentences.extend(preprocess_review(review))
for review in test_review_raw:
    test_review_clean.append(preprocess_review(review, True))
       
end_preprocess_data = time.time()
    
#len(sentences) = 75000
#sentences : [["this", "is" ,... "cat."],....]
#len(train_review_clean) = 25000 , same as sentences but review not a sentence
     

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print("Training word2vec model...")
begin_train_word2vec = time.time()
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
end_train_word2vec = time.time()
    
#model = Word2Vec.load("300features_40minwords_10context")

def meanFeatureVect(words , model , num_features):
    index2word = set(model.wv.index2word)
    review2vec = [model.wv[word] for word in words if word in index2word]
    review2vec = np.array(review2vec)
    return np.mean(review2vec, axis = 0)

train_review_vect = [meanFeatureVect(review, model, num_features) for review in train_review_clean]
test_review_vect = [meanFeatureVect(review, model, num_features) for review in test_review_clean]
#train_review_vect = [[0.2,0.4,....],...] 25000*300 
forest = RandomForestClassifier( n_estimators = 100 )

print("Fitting a random forest to labeled training data...")
begin_random_forest = time.time()
forest = forest.fit( train_review_vect, train_data["sentiment"] )
# Test & extract results 
result = forest.predict( test_review_vect )
end_random_forest = time.time()
# Write the test results 
output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Word2Vec_AverageVectors.csv'), index=False, quoting=3 )
end = time.time()

print("Save file , done! ")
print("Time to preprocess data: ", end_preprocess_data - begin_preprocess_data)
print("Time to train word2vec :", end_train_word2vec - begin_train_word2vec)
print("Time to train random forest: ", end_random_forest - begin_random_forest)
print("All time : ", end - begin)
save_model = pickle.dumps(forest)
pd.to_pickle(forest,"data/word2vect_and_mean_feature.pickle")
print("Save !")
model = pd.read_pickle("data/word2vect_and_mean_feature.pickle")
#model.predict


