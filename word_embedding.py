import sys
import re
from gensim.models import Word2Vec
from gensim.test.utils import datapath


def contains_non_ascii(input_string):
    for c in input_string:
        if not 0 <= ord(c) <= 127:
            return True
    return False

doc = []
with open('AllMerge_normalize.txt', 'r', encoding='UTF-8') as file:
    for line in file.readlines():
        if contains_non_ascii(line):
            continue
        words = line.split(' ')
        doc.append(words)

model = Word2Vec(doc, size=300, window=7, min_count=0, workers=4, iter=100)
#model.save("reference.model")
model.wv.save_word2vec_format("reference_Dim300.kv", binary=False)

'''
result = model.wv.similar_by_word("Control")
for x in result:
        print("{}: {:.4f}".format(*x))
'''
