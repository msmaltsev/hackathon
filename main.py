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
# word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format


def testCorpus(path_, split = 100):
    print('loading corpus')
    a = []
    spl = 0
    f = open(path_, 'r', encoding='utf8')
    for line in f:
        if spl != split:
            l = line.split(',')
            a.append([l[0],l[1],l[2],int(l[3])])
        else:
            break
    print('test corpus loaded!')
    return a


if __name__ == '__main__':
    path_ = 'E:\hackaton\word2vec_models\GoogleNews-vectors-negative300.bin'
    word_vectors = KeyedVectors.load_word2vec_format(path_, binary=True)
    # word_vectors.init_sims(replace=True)
    # print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
    a = testCorpus('E:/hackaton/hackathon/train.txt')
    # print(a[0])
    for i in a:
        try:
            sim_0 = word_vectors.similarity(i[0], i[1])
            sim_1 = word_vectors.similarity(i[1], i[2])
            sim_2 = word_vectors.similarity(i[0], i[2])
            print(sim_0, sim_1, sim_2)
            print(i)
            print(word_vectors.doesnt_match(i[:3]), i[3])
            print('')
        except Exception as e:
            print('things went bad due to: %s'%e)

