# -*- coding: utf-8 -*-
#import pandas as pd
from __future__ import division
import os
import re
import codecs
import logging
import numpy as np
import gensim
from gensim.models import word2vec
from gensim.models import doc2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time

#from pymystem3 import Mystem

# def lemmatize_text(old_text):
#     m = Mystem()
#     text = re.sub(u'[^А-Яа-яA-Za-z\s]*', u'', old_text)
#     lemmas=m.lemmatize(text)
#     new_text = ''.join(lemmas)
#     return new_text
#
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids
    
def get_data(filename):
    f = codecs.open(filename,'r','utf-8')
    sentences = []
    for i in f:
        sentences.append(i.split())
    return sentences


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
    
num_features = 100 
min_word_count = 10
num_workers = 4    
context = 10       
downsampling = 1e-3

file = 'lemmatized_text.txt'
output = get_data(file)

print output[7]
print len(output)

# print "Training model..."
#
# model = word2vec.Word2Vec(output, workers=num_workers, \
#             size=num_features, min_count = min_word_count, \
#             window = context, sample = downsampling)
# model.init_sims(replace=True)
# model_name = "test_enjoyme"
# model.save(model_name)
#
# print "Hurray! we trained this sh*t"
model = gensim.models.Word2Vec.load('')

start = time.time() 
word_vectors = model.syn0

#print model.shape()
num_clusters = 100 #это можно регулировать, для получения наилучших результатов
w2v_dict = {}

print 'Creating W2V dict for speed'

t_s = time.time()

for i in model.index2word:  #для скорости
     w2v_dict[i] = model[i]

print 'It took ' + str(time.time() - t_s) + 'seconds'

queriesNP = np.zeros((1,100), dtype='float32') 
c = 0
numb = len(output)
print numb
queriesNP = np.zeros((numb,100))

st = time.time()
for i in output:
    tnp = np.zeros((1,100), dtype='float32')
    # c+=1
    # print i
    for j in i:
        try:
            tnp += w2v_dict[j]
            # print 'hurray'
        except:
            continue
            # print j +  ' not in vocab'
    # c+= 1
    if c%10000 == 0:
        print (c/numb * 100, time.time() - st ) 
    queriesNP[c] = tnp
    c +=1

np.save('vect_sentences', queriesNP)
# np.load('outfile')

print queriesNP.shape

start = time.time() 

print 'clustering...'
        
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( queriesNP[1:] )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# word_centroid_map = dict(zip( model.index2word, idx ))
text_out = []

for i in output:
    tmp_str = ''
    for j in i:
        tmp_str += j + ' '
    text_out.append(tmp_str.rstrip(' '))
        
    
word_centroid_map = dict(zip( text_out, idx ))

print 'Saving clusters...'
o = codecs.open('queries_clusters.csv','w','utf-8')

for cluster in xrange(0,999):
    print "\nCluster %d" % cluster
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    for i in words: 
        o.write('Cluster ' + str(cluster) + ';' + i +'\n')
        print i

o.close()
# modeld2v = doc2vec.load_word2vec_format(fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict')

# for i in output[9]:
#     print i



        
