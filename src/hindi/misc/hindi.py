import pandas as pd
import numpy as np
import os
import pickle
from HindiTokenizer import Tokenizer
import codecs

train=[]
'''
t=Tokenizer()
train=t.read_from_file('./ATIS_data/ATIS_SLOT_TRAIN_HINDI.txt')
print(train)
'''
#train=[]
train_Text='./ATIS_data/ATIS_SLOT_TRAIN_HINDI.txt'

i=0

with codecs.open(train_Text, 'r', encoding='utf-8') as f:
    for line in f:
    	#print(len(line))
    	if(len(line)>1):
    		i=0
    		train.append(line.strip("\n"))
    	elif(len(line)<=1):
    		i=i+1
    		if(i==1):
    			train.append("\n")
    		
f.close()
print(train)
'''
for i in train:
	print(i)
'''
with open("./ATIS_data/ATIS_INTENT_TRAIN_HINDI.txt", "w") as f:
	for i in train:
		if(i=="\n"):
			f.write("\n")
		else:
			f.write(i.encode('utf-8'))
			f.write(" ")
		
f.close()
'''
for i in train:
	print(i)
'''

