import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import itertools
from collections import Counter
from New_Utils import *
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model

# train=[]
# train_Text='./test.txt'
# with open(train_Text) as f:
#     for line in f:
#     	train.append(line)
#     f.close()
train = []
def intent_module(string):
    train = [string]



    sequence_length=46
    f=open('New_Tokenizer.tkn','r')
    tokenizer=pickle.load(f)
    f.close()
    f=open('Emb_Mat.mat','r')
    embedding_matrix=pickle.load(f)
    f.close()
    f=open('dict.pickle','r')
    lbl_dict=pickle.load(f)
    f.close()
    X_train = load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')

    model=load_model("cnn1dlstm.h5")
    predicted=model.predict(X_train)

    for i in predicted:
        pos=i.argmax()

    print(lbl_dict.keys()[lbl_dict.values().index(pos)])
    return lbl_dict.keys()[lbl_dict.values().index(pos)].rstrip()
# filename = 'intent_identified.txt'
# with open(filename,'w') as fil:
# 	fil.write(lbl_dict.keys()[lbl_dict.values().index(pos)])
