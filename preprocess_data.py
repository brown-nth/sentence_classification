import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import time
import logging
from gensim.models import word2vec
#nltk.download('punkt')
#import nltk
#nltk.download('stopwords')
#
#nltk.download('wordnet')
def preprocess_review( review_text, remove_stopwords=False ):
        # 1. Remove HTML
        review_text = BeautifulSoup(review_text,features="html.parser").get_text()
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
      
        
begin = time.time()
begin_preprocess_data = time.time()
sentences = []

unlabeled_train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','unlabeledTrainData.tsv'),
                                   header=0,
                                   delimiter="\t", 
                                   quoting=3
                                   )
unlabeled_train_review_raw = unlabeled_train_data["review"]
for review in unlabeled_train_review_raw:
    sentences.extend(preprocess_review(review))
del unlabeled_train_review_raw

del unlabeled_train_data




train_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','labeledTrainData.tsv'),
                         header=0,
                         delimiter="\t", 
                         quoting=3
                         )

labeled_train_review_raw = train_data["review"]
train_review_clean = []
for review in labeled_train_review_raw:
    sentences.extend(preprocess_review(review, False))
    train_review_clean.append(preprocess_review(review, True))
del labeled_train_review_raw

del train_data





test_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv'),
                        header=0,
                        delimiter="\t", 
                        quoting=3
                        )
test_review_raw = test_data["review"] 
test_review_clean = []
for review in test_review_raw:
    test_review_clean.append(preprocess_review(review, True))
end_preprocess_data = time.time()
del test_review_raw

del test_data


#len(sentences) = 75000
#sentences : [["this", "is" ,... "cat."],....]
#len(train_review_clean) = 25000 , same as sentences but review not a sentence
     

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
num_features = 80    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print("Training word2vec model...")
begin_train_word2vec = time.time()
model = word2vec.Word2Vec(sentences, workers=num_workers, 
                          size=num_features, min_count = min_word_count, 
                          window = context, sample = downsampling
                          )
del sentences
# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
model_name = str(num_features) + "features_40minwords_10context"
model.save(model_name)
end_train_word2vec = time.time()
    
#model = Word2Vec.load("300features_40minwords_10context")





def featureVect(words , model , num_features):
    index2word = set(model.wv.index2word)
    review2vec = [model.wv[word] for word in words if word in index2word]
    review2vec = np.array(review2vec)
    return review2vec
def meanFeatureVect(words , model , num_features):
    index2word = set(model.wv.index2word)
    review2vec = [model.wv[word] for word in words if word in index2word]
    review2vec = np.array(review2vec)
    return np.mean(review2vec, axis = 0)



#
max_sent_length = 100
#
for i in range(len(test_review_clean)):
    test_review_clean[i] = test_review_clean[i][:max_sent_length]
test_review_vect = [featureVect(review, model, num_features) for review in test_review_clean]
del test_review_clean
for i in range(len(test_review_vect)):
   if(max_sent_length-len(test_review_vect[i])):
       test_review_vect[i] = np.append(test_review_vect[i],[[0]*num_features]*(max_sent_length-len(test_review_vect[i])), axis = 0)
#print([len(test_review_vect[i]) for i in range(len(test_review_vect))])
#test_review_vect = np.array([np.array(i) for i in test_review_vect])
#print(test_review_vect.shape)
pd.to_pickle(test_review_vect, "data/test_review_vect_"+str(max_sent_length)+".pickle")
del test_review_vect
##
##
for i in range(len(train_review_clean)):
    train_review_clean[i] = train_review_clean[i][:max_sent_length]
train_review_vect = [featureVect(review, model, num_features) for review in train_review_clean]
del train_review_clean
for i in range(len(train_review_vect)):
   if(max_sent_length-len(train_review_vect[i])):
       train_review_vect[i] = np.append(train_review_vect[i],[[0]*num_features]*(max_sent_length-len(train_review_vect[i])), axis = 0)
pd.to_pickle(train_review_vect, "data/train_review_vect_"+str(max_sent_length)+".pickle")
del train_review_vect



