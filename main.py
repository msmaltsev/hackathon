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


def testCorpus(path_):
    return None


if __name__ == '__main__':
    path_ = 'E:\hackaton\word2vec_models\GoogleNews-vectors-negative300.bin'
    word_vectors = KeyedVectors.load_word2vec_format(path_, binary=True)
    print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))