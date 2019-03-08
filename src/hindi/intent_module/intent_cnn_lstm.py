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
import datetime
import argparse
import csv
import tensorflow as tf
import codecs
from keras import backend as k
from keras.backend.tensorflow_backend import set_session
ap = argparse.ArgumentParser()
# add the training parameters
ap.add_argument('-xt','--train-text',required = False, help = "Path and filename to the file which contains Text for training data", type = str, default = '/Users/dhawgupta/Desktop/Semester-8/BTP/Codes/hrldm/src/hindi/misc/atis_hindi_train_text.csv')
ap.add_argument('-yt','--test-text',required = False, help = "Path and filename to the file containing the test text", type = str, default = '/Users/dhawgupta/Desktop/Semester-8/BTP/Codes/hrldm/src/hindi/misc/atis_hindi_test_text.csv')
args  = vars(ap.parse_args())

ap.add_argument('-xl','--train-label',required = False, help = "Path and filename to the file which contains label for training data", type = str, default = '/Users/dhawgupta/Desktop/Semester-8/BTP/Codes/hrldm/src/hindi/misc/atis_hindi_intent_label_train.txt')
ap.add_argument('-yl','--test-label',required = False, help = "Path and filename to the file containing the test labels", type = str, default = '/Users/dhawgupta/Desktop/Semester-8/BTP/Codes/hrldm/src/hindi/misc/atis_hindi_intent_label_test.txt')
ap.add_argument('-g','--gpu',required = False, help = "Give the GPU number, put -1 to not use the GPU", type = int, default = -1)

args  = vars(ap.parse_args())


def setup_gpu(gpu_id):
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # don't show any messages
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

if args['gpu'] < 0:
	pass
else:
	setup_gpu(str(args['gpu']))

test = []
train = []
train_Text = args['train_text']
test_Text = args['test_text']

train = list(csv.reader(codecs.open(train_Text, encoding='utf-8')))
train = [i[0] for i in train]
test = list(csv.reader(codecs.open(test_Text, encoding='utf-8')))
test = [i[0] for i in test]
'''
# with open(train_Text) as f:
# 	for line in f:
# 		train.append(line)

# f.close() # not required

# with open(test_Text) as f:
# 	for line in f:
# 		test.append(line)
#
# # f.close() # not required
'''

train = [s.strip("\n") for s in train]
test = [s.strip("\n") for s in test]

text = train + test
text = [s.split(" ") for s in text]
sequence_length = max(len(x) for x in text)


tokenizer = load_create_tokenizer(train,None,True)
X_train = load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
X_test = load_create_padded_data(X_train=test,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
word_index = tokenizer.word_index
embedding_matrix = load_create_embedding_matrix(word_index,len(word_index)+1,300,'./cc.hi.300.vec',False,True,'./Emb_Mat.mat')


Y_test=[]
Y_train=[]
train_Text = args['train_label']
test_Text = args['test_label']

with open(train_Text) as f:
	for line in f:
		Y_train.append(line)

# f.close()
with open(test_Text) as f:
	for line in f:
		Y_test.append(line)

# f.close()


lbl_dict = {}
index=0
for dial_lbls in Y_train:
	if dial_lbls not in lbl_dict:
		lbl_dict[dial_lbls]=index
		index=index+1

def create_label(label):
	
    Y=[]
    for i in label:
    	xxx=np.zeros(int(len(lbl_dict)))
    	j=lbl_dict.get(i)
    	xxx[j]=1
    	Y.append(xxx)
    return Y

label = Y_train
Y_train = create_label(label)
label = Y_test
Y_test = create_label(label)

y_train=np.array([i for i in Y_train])
y_test=np.array([i for i in Y_test])

embedding_dim = 300
a = str(datetime.datetime.now())
a = a.split('.')[0]
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))
model.add(LSTM(100))
model.add(Dense(23, activation='softmax'))
checkpoint = ModelCheckpoint('./CNN/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=100, callbacks=[checkpoint], batch_size=50, validation_data=(X_test, y_test))
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")
model.save("{}_cnn_lstm.h5".format(a))

