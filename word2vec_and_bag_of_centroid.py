#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:33:41 2018

@author: kiyoshitaro
"""

import os
#import zipfile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import time
import logging
from sklearn.cluster import KMeans



from gensim.models import word2vec

begin = time.time()
unlabeled_train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','unlabeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)
train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','labeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)

test_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv'),header=0,
                    delimiter="\t", quoting=3)
print("Review : ", train_data["review"][0])

begin_load = time.time()

def preprocess_review( review_text, remove_stopwords=False ):
        # 1. Remove HTML
        review_text = BeautifulSoup(review_text,features="html5lib").get_text()
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        words = review_text.lower()
        if(remove_stopwords):
            word_token = word_tokenize(words)
            stops = set(stopwords.words("english"))
            words_not_stop = [word for word in word_token if not word in stops]
            lemmatizer = WordNetLemmatizer()
            words_lemma = [lemmatizer.lemmatize(w) for w in words_not_stop]
            return words_lemma
        
        sent_tokens = sent_tokenize(words)
        sentences = []
        for sent in sent_tokens:
            sentences.append(sent.split())
        return sentences
      
    
sentences = []
labeled_train_review_raw = train_data["review"]
unlabeled_train_review_raw = unlabeled_train_data["review"]
test_review_raw = test_data["review"] 
test_review_clean = []
train_review_clean = []

begin_preprocess_data = time.time()


for review in labeled_train_review_raw:
    sentences.extend(preprocess_review(review, False))
    train_review_clean.append(preprocess_review(review, True))
    
for review in unlabeled_train_review_raw:
    sentences.extend(preprocess_review(review))
for review in test_review_raw:
    test_review_clean.append(preprocess_review(review, True))
      
#len(sentences) = 75000
#sentences : [["this", "is" ,... "cat."],....]
#len(train_review_clean) = 25000 , same as sentences but review not a sentence
end_preprocess_data = time.time()


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

model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)
end_train_word2vec = time.time()
    
#model = Word2Vec.load("300features_40minwords_10context")
begin_cluster = time.time()
word_vectors = model.wv.syn0

num_clusters = int(word_vectors.shape[0] / 5)
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )
word_centroid_map = dict(zip( model.wv.index2word, idx ))
def bag_of_centroid(review, word_centroid_map): 
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    for w in review: 
        if w in word_centroid_map:
            bag_of_centroids[word_centroid_map[w]] += 1 
    
    return bag_of_centroids
    
train_centroids = [bag_of_centroid(review, word_centroid_map) for review in train_review_clean]
test_centroids = [bag_of_centroid(review, word_centroid_map) for review in test_review_clean]

end_cluster = time.time()


forest = RandomForestClassifier(n_estimators = 100)
print("Fitting a random forest to labeled training data...")
begin_random_forest = time.time()
forest = forest.fit(train_centroids,train_data["sentiment"])
result = forest.predict(test_centroids)
end_random_forest = time.time()

# Write the test results 
output = pd.DataFrame(data={"id":test_data["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
end = time.time()

print("Save file , done! ")
print("Time to preprocess data: ", end_preprocess_data - begin_preprocess_data)
print("Time to train word2vec :", end_train_word2vec - begin_train_word2vec)
print("Time to train random forest: ", end_random_forest - begin_random_forest)
print("Time to train KMean: ", end_cluster - begin_cluster)
print("All time: ", end - begin)