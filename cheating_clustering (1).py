#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import re
import string
import time
import os
import collections
from pymystem3 import Mystem
from sklearn.cluster import KMeans
# from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import MeanShift
# from scipy.spatial import distance
from sklearn.decomposition import PCA
# from sklearn.cluster import DBSCAN
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def parse_queries(filename):
    import pysftp
    from aparser import AParser

    sftp = pysftp.Connection('#####', username='root', password='###')
    sftp.chdir('/home/aparser/queries/auto_input/')
    sftp.put(filename, preserve_mtime=True)
    ap = AParser("#####", "###")
    print "Ping result:", ap.ping()
    print "Total proxies:", len(ap.getProxies())
    #print "Parsing with:", preset_name
    task_id = ap.addTask('100 Threads', 'Y+G: Keywords Clustering (only URL)', 'file', 'auto_input/' + filename, )


    if task_id:
        print "Add task:", task_id
        ap.waitForTask(task_id)
        print "Task state:", ap.getTaskState(task_id)['status'];
        results = ap.getTaskConf(task_id)['resultsFileName']
        print 'File name:', results;
    else:
        print 'error'

    sftp.chdir('/home/aparser/results/')
    sftp.get(results, preserve_mtime=True)
    return results

def lemmatize_text(old_text):
    # m = Mystem()
    lemmas=mystem.lemmatize(old_text)
    new_text = ''.join(lemmas)
    return new_text
    
def clear_special(old_text):
    re_clr = re.compile(u"[^a-zа-яё|0-9]", re.I+re.U)
    new_text = re_clr.sub(u' ',  old_text)
    return new_text

def lemmatize_text(old_text):
    # Инициализировать mystem в корне проги
    lemmas=mystem.lemmatize(old_text)
    new_text = ''.join(lemmas)
    return new_text[:-1]
    
def get_list_of_words(queries):
    words = []
    for query in queries:
        for word in query.split():
            words.append(word)
    words = list(set(words))#.difference(set(stop_words)))
    print len(words)
    return words

    
def get_weight(pos):
    position = int(pos)
    return 31 - position


def find(lst, str1):
    result = []
    for i, x in enumerate(lst):
        if x == str1:
            result.append(i)
    return result
    
def get_border(cluster,n=10):
    len_cluster = len(cluster)
    tens = len_cluster/10
    if tens == 0:
        return get_weight(n)
    else:
        return tens*get_weight(n)


def get_num_clusters(len_urls):
    return len_urls/200
    
def get_best_len(len_clusters):
    len_clusters = sorted(len_clusters)
    best_len = 0
    median = sum(len_clusters)/len(len_clusters)
    
    distanse = []
    for i in len_clusters:
        distanse.append(abs(median - i))
    min_d = min(distanse)
    
    for i in xrange(len(len_clusters)):
        if distanse[i] == min_d:
            return len_clusters[i]
    print 'Error'
    return len_clusters[0]

    
def select_optimal_cluster(clusters):
    if len(clusters) == 0:
        return clusters
        
    len_clusters = []
    for i in clusters:
        len_clusters.append(len(i))
        
    best_len_clusters = []
    best_len = get_best_len(len_clusters)
    for i in clusters:
        if len(i) == best_len:
            best_len_clusters.append(sorted(i))
            # return i
        
    blc_freq = []
    tmp_c = []
    for i in best_len_clusters:
        if i not in tmp_c:
            tmp_c.append(i)
            blc_freq.append(1)
        else:
            ind = tmp_c.index(i)
            blc_freq[ind] += 1 
    
    max_fr = max(blc_freq)
    print blc_freq
    for i in xrange(len(blc_freq)):
        if blc_freq[i] == max_fr:
            return tmp_c[i]
            
    print 'WTFWTFTFWTFTTWFTWFTWFTWFTWFWTFTWf'
    return clusters[0]
    
    
# Данные

### Если нужно запустить парсинг - расскоментить эти 3 строчки и закомментить столку под ними
#
# query_for_clustering = 'trigrams.txt'
# data_filename = parse_queries(query_for_clustering)
# data_file = codecs.open(data_filename,'r','utf-8')

### Закоментить, если идет парсинг

# data_file = codecs.open('Jul-08_13-04-42.csv','r','utf-8')
data_file = codecs.open('Jul-07_14-49-00.csv','r','utf-8')

# 'Jul-08_13-04-42.csv'
# Jul-07_14-49-00.csv
mystem = Mystem()
print 'Getting the data'
res1 = codecs.open('clusters.txt','w','utf-8')
# data_lines = data_file.readlines()
data = [[],[],[],[],[]]
# Создаем массив с данными
for line in data_file:
    line_parts = line.split('";"')
    if len(line_parts) >= 4:
        # data[0].append(line_parts[0][1:])
        data[1].append(line_parts[1])
        data[2].append(line_parts[2])
        data[3].append(line_parts[3])
        # data[4].append(line_parts[4])
    else:
        continue

keywords = [[],[],[],[],[]]
print len(data[1])
keywords[0] = list(set(data[1]))
print "Lemmatizing queries"
lem_keywords = []
for i in xrange(len(keywords[0])):
    lem_keywords.append(lemmatize_text(keywords[0][i]))

print len(keywords[0])

print "k"

p = 0
max_postion = 30

print 'Preparing arrays'
urls = []
for key in keywords[0]:
    indexes = find(data[1], key)
    se_literals = []
    se_positions = []
    se_websites = []
    se_anchors = []
    for index in indexes:
        if int(data[2][index]) <= max_postion:
            # se_literals.append(data[0][index])
            se_positions.append(data[2][index])
            # se_websites.append(data[3][index]+data[4][index])
            clean_url = data[3][index].lower()
            clean_url = re.sub('www\.','',clean_url)
            se_websites.append(clean_url)
            urls.append(clean_url)
            # se_anchors.append(data[4][index])
    # keywords[1].append(se_literals)
    keywords[2].append(se_positions)
    keywords[3].append(se_websites)
    # keywords[4].append(se_anchors)
    p += 1

# url_freq = collections.Counter(urls)
# u_freq = codecs.open('url_frequency.csv','w','utf-8')
# for i in url_freq.keys():
#     u_freq.write(i + ';' + str(url_freq[i]) + '\n')
# u_freq.close()

ignored_urls = [u'google.com',u'yandex.ru',u'market.yandex.ru']

urls = list(set(urls).difference(set(ignored_urls))) 
q_data = np.zeros((len(keywords[0]),len(urls)), dtype='float32')
# url_data = np.zeros((len(urls),len(keywords[0])), dtype='float32')

print q_data.shape 
print len(urls)
print "Filling NP.array"
for i in xrange(len(urls)):
    for j in xrange(len(keywords[0])):
        if urls[i] in keywords[3][j]:
            q_data[j,i] = 1
            # url_data[i,j] = 1
            
start = time.time()

print 'Decomposing values via PCA'
pca = PCA(n_components = 100)

red_data = pca.fit_transform(q_data)
print red_data.shape
# url_data = pca.fit_transform(url_data)

end = time.time()
elapsed = end - start
print "Time taken for decomposing: ", elapsed, "seconds."


start = time.time()

print 'Clustering queries...'


num_clusters = 20#get_num_clusters(len(urls))

kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict(red_data)
n_clusters_ = num_clusters




end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."




word_centroid_map = dict(zip( keywords[0], idx ))

print 'Saving clusters...'
o = codecs.open('queries_clusters.csv','w','utf-8')
# num_clusters = max(idx)
for cluster in xrange(n_clusters_):
    print "\nCluster %d" % cluster
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    for i in words:
        o.write('Cluster ' + str(cluster) + ';' + i +'\n')
        print i

o.close()
