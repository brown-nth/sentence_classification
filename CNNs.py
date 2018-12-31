#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 02:04:25 2018

@author: kiyoshitaro
"""

# install env in drive 

# !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
# !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
# !apt-get update -qq 2>&1 > /dev/null
# !apt-get -y install -qq google-drive-ocamlfuse fuse
# from google.colab import auth
# auth.authenticate_user()
# from oauth2client.client import GoogleCredentials
# creds = GoogleCredentials.get_application_default()
# import getpass
# !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
# vcode = getpass.getpass()
# !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


import os
from six.moves.urllib.request import urlretrieve 
import pandas as pd
import numpy as np
import logging
import pickle
import tensorflow as tf


# !mkdir -p data
# !google-drive-ocamlfuse data


# READ DATA
train_data = pd.read_csv(os.path.join('data','labeledTrainData.tsv'),header=0,
                    delimiter="\t", quoting=3)
train_labels = train_data["sentiment"]
del train_data
train_review_vect = pd.read_pickle("data/train_review_vect_100.pickle")
train_review_vect = np.array(train_review_vect)
test_review_vect = pd.read_pickle("data/test_review_vect_100.pickle")
test_review_vect = np.array(test_review_vect)


# REFORMAT DATA
num_features = train_review_vect.shape[2]    # Word vector dimensionality       
sent_length = train_review_vect.shape[1]
num_test_data = test_review_vect.shape[0]
num_classes = 2
def reformat(labels):
  label = (np.arange(num_classes) == labels[:,None]).astype(np.float32)
  return label
print('Training set', train_review_vect.shape, train_labels.shape)
train_labels = reformat(train_labels)
print('Testing set', test_review_vect.shape)
print('Label', train_labels.shape)


# BUILD GRAPH
batch_size = 32
graph = tf.Graph()
with graph.as_default():
  # Different filter sizes we use in a single convolution layer
  filter_sizes = [3,5,7] 
  # inputs and labels
  tf_train_sents = tf.placeholder(shape=[batch_size,sent_length,num_features],dtype=tf.float32,name='sentence_inputs')
  tf_train_labels = tf.placeholder(shape=[batch_size,num_classes],dtype=tf.float32,name='sentence_labels')
  tf_test_sents =  tf.placeholder(shape=[num_test_data,sent_length,num_features],dtype=tf.float32,name='sentence_inputs')

  # Weights of the first parallel layer
  con_w1 = tf.Variable(tf.truncated_normal([filter_sizes[0],num_features,1],stddev=0.02,dtype=tf.float32),name='weights_1')
  con_b1 = tf.Variable(tf.random_uniform([1],0,0.01,dtype=tf.float32),name='bias_1')

  # Weights of the second parallel layer
  con_w2 = tf.Variable(tf.truncated_normal([filter_sizes[1],num_features,1],stddev=0.02,dtype=tf.float32),name='weights_2')
  con_b2 = tf.Variable(tf.random_uniform([1],0,0.01,dtype=tf.float32),name='bias_2')

  # Weights of the third parallel layer
  con_w3 = tf.Variable(tf.truncated_normal([filter_sizes[2],num_features,1],stddev=0.02,dtype=tf.float32),name='weights_3')
  con_b3 = tf.Variable(tf.random_uniform([1],0,0.01,dtype=tf.float32),name='bias_3')

  # Fully connected layer
  fc_w1 = tf.Variable(tf.truncated_normal([len(filter_sizes),num_classes],stddev=0.5,dtype=tf.float32),name='weights_fulcon_1')
  fc_b1 = tf.Variable(tf.random_uniform([num_classes],0,0.01,dtype=tf.float32),name='bias_fulcon_1')

  def model(data):

    # Calculate the output for all the filters with a stride 1
    # We use relu activation as the activation function
    conv1 =tf.nn.conv1d(data,con_w1,stride=1,padding='SAME')
    hidden1_1 = tf.nn.relu(conv1 + con_b1)
    conv2 =tf.nn.conv1d(data,con_w2,stride=1,padding='SAME')
    hidden1_2 = tf.nn.relu(conv2 + con_b2)
    conv3 =tf.nn.conv1d(data,con_w3,stride=1,padding='SAME')
    hidden1_3 = tf.nn.relu(conv3 + con_b3)
   
    # Pooling over time operation

    # This is doing the max pooling. Thereare two options to do the max pooling
    # 1. Use tf.nn.max_pool operation on a tensor made by concatenating h1_1,h1_2,h1_3 and converting that tensor to 4D
    # (Because max_pool takes a tensor of rank >= 4 )
    # 2. Do the max pooling separately for each filter output and combine them using tf.concat 
    # (this is the one used in the code)

    hidden2_1 = tf.reduce_mean(hidden1_1,axis=1)
    hidden2_2 = tf.reduce_mean(hidden1_2,axis=1)
    hidden2_3 = tf.reduce_mean(hidden1_3,axis=1)

    hidden2 = tf.concat([hidden2_1,hidden2_2,hidden2_3],axis=1)
    return tf.matmul(hidden2,fc_w1) + fc_b1

  # Calculate the fully connected layer output (no activation)
  # Note: since h2 is 2d [batch_size,number of parallel filters] 
  # reshaping the output is not required as it usually do in CNNs
  logits = model(tf_train_sents)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels,logits=logits))

  # Momentum Optimizer
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(loss)
#   predictions = tf.nn.softmax(model(tf_test_sents))
  predictions = tf.argmax(tf.nn.softmax(model(tf_test_sents)),axis=1)



# RUN SESSION
num_steps = 4001
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_review_vect[offset:(offset + batch_size), :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_sents : batch_data, tf_train_labels : batch_labels, tf_test_sents : test_review_vect}
    _, l = session.run(
      [optimizer, loss], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
    if(l < 0.2):
      print(l)
      break
  _, l,predictions = session.run(
      [optimizer, loss,predictions], feed_dict=feed_dict)

# SAVE RESULT
del test_review_vect
del train_review_vect
test_data = pd.read_csv(os.path.join(os.path.dirname('__file__'),'data','testData.tsv' ),header=0,
                    delimiter="\t", quoting=3)

output = pd.DataFrame(data={"id":test_data["id"], "sentiment":predictions})
output.to_csv( "data/CNN_mean.csv", index=False, quoting=3)
