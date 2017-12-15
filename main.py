# -*- coding: utf-8 -*-
#import pandas as pd
from __future__ import division
import os
import re
import logging
import numpy as np
import gensim
from gensim.models import word2vec
import time
from gensim.models.keyedvectors import KeyedVectors

def testCorpus(path_, split = 100):
    print('loading corpus')
    result, spl = [], 0
    file = open(path_, 'r', encoding='utf8')
    for line in file:
        if spl != split:
            element = line.split(',')
            result.append([element[0], element[1], element[2], int(element[3]), int(element[4]), int(element[5])])
        else:
            break
    print('test corpus loaded!')
    return(result)

if __name__ == '__main__':
    test_corpus = testCorpus('validation_wiki.txt')
    
    path_ = 'glove.6B.100d.txt'
    word_vectors = KeyedVectors.load_word2vec_format(path_, binary=False)

    X, y = [], []
    for group in test_corpus:
        try:
            sim_01 = word_vectors.similarity(group[0], group[1])
            sim_12 = word_vectors.similarity(group[1], group[2])
            sim_02 = word_vectors.similarity(group[0], group[2])
            wiki_3_in_1 = group[4]
            wiki_3_in_2 = group[5]
            if word_vectors.doesnt_match(group[:3]) == group[2]:
                third_wheel = 1
            else:
                third_wheel = 0
            
            X.append((sim_01, sim_12, sim_02, wiki_3_in_1, wiki_3_in_2, third_wheel))
            y.append(group[3])
            
        except Exception as e:
            pass
            print('things went bad due to this little rat: %s'%e)        
        


