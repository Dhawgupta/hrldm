import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import itertools
from collections import Counter
from new_utils import *
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

# first load all the text and get the sequence lenfgth

test=[]
train=[]

train_Text='./ATIS_TRAIN.txt'
test_Text='./ATIS_TEST.txt'

intent_file = 'cnn_lstm_multiintent.h5'

with open(train_Text) as f:
    for line in f:
    	train.append(line)

with open(test_Text) as f:
    for line in f:
        test.append(line)

text = train + test
text = [s.split(" ") for s in text]
sequence_length = max(len(x) for x in text) # somehow this over here is coming to 47 , whereas the value is supposed to be 46
sequence_length = 46
Y_train=[]
train_labels='./ATIS_TRAIN_LABEL.txt'


with open(train_labels) as f:
    for line in f:
    	Y_train.append(line)

# this will give us the max sequqnce length of the text

lbl_dict={}
index=0
for dial_lbls in Y_train:
	label=dial_lbls.split("+")
	if(len(label)>=2):
		for i in label:
			i = i.strip()
			if i not in lbl_dict:
				lbl_dict[i]=index
				index=index+1
	else:
		dial_lbls = dial_lbls.strip()
		if dial_lbls not in lbl_dict:
			lbl_dict[dial_lbls]=index
			index=index+1

print(f"{lbl_dict}\nThe Total number of intents are : {len(lbl_dict)}\n\n")
# get the label dictionaries

def intent_module(string):
    train = [string]
    global sequence_length
    tokenizer = []
    with open("./New_Tokenizer.tkn",'rb') as f:
        # print("Going to read the file")
        tokenizer=pickle.load(f)
    # embedding_matrix = []
    # with open('./Emb_Mat.mat','rb') as f:
    #     embedding_matrix=pickle.load(f)
    # create the sentence into teh embedding format
    X_train=load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
    print("{}".format(X_train))
    model = load_model(intent_file)
    predict = model.predict(X_train)
    return predict, lbl_dict

if __name__ == "__main__":
    print("{}".format(intent_module("Show me flights")))
