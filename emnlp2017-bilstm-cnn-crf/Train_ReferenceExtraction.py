# This script trains the BiLSTM-CNN-CRF architecture for NER in German using
# the GermEval 2014 dataset (https://sites.google.com/site/germeval2014ner/).
# The code use the embeddings by Reimers et al. (https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/)
from __future__ import print_function
import os
import logging
import sys
import re
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'ReferenceExtraction':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'INITCAP',2:'ALLCAPS',3:'CONTAINSDIGITS',4:'ALLDIGITS',5:'CONTAINSDOTS',6:'CONTAINSDASH',7:'LONELYINITIAL',8:'SINGLECHAR',
        			 9:'lineStart',10:'lineIn',11:'lineEnd',12:'CoraTag'},    #format for the input data.
         'label': 'CoraTag',                      #Which column we like to predict
         'evaluate': True,                        #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                   #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}
# :: Path on your computer to the word embeddings. Embeddings by Reimers et al. will be downloaded automatically ::
# embeddingsPath = 'reference_Dim300.kv.model'
embeddingsPath = '../reference_Dim300.kv'
# Preprocessing.
with open(embeddingsPath, 'r+', encoding='utf-8') as f:
    data = f.readlines()
    # print(len(data))
    if len(data[0].split(' ')) == 2:
        data = data[1:]
    # print(len(data))
    # Remove white spaces start sentence.
    fout = []
    buf = ''
    for d in data:
        clean_d = d.strip()
        d_len = len(clean_d.split(' '))
        '''
        if d_len != 301:
            print(d_len)
        '''
        if d_len == 0:
            continue
        elif d_len == 300 and len(buf) == 1:
            fout.append(buf + ' ' + clean_d + '\n')
        elif d_len == 301:
            fout.append(clean_d + '\n') 
        buf = clean_d
    '''
    for x in fout:
        print(x)
        if len(x.split(' ')) != 301:
            print(len(x.split(' ')))
    '''
    f.seek(0)
    f.writelines(fout)
    f.truncate()

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Hyperparameters for the network
defaultParams = {'dropout': (0.5,0.5), 'classifier': ['Softmax'], 'LSTM-Size': (100,), 'customClassifier': {},
                         'optimizer': 'adam',
                         'charEmbeddings': None, 'charEmbeddingsSize': 30, 'charFilterSize': 30, 'charFilterLength': 3, 'charLSTMSize': 25, 'maxCharLength': 25,
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 5, 'miniBatchSize': 32,
                         'featureNames': ['tokens', 'casing'], 'addFeatureDimensions': 10}


# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, ], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN', 'maxCharLength': 50}


model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)
