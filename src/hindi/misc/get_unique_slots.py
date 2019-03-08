'''
@author : Dhawal Gupta
This code is meant to extract the unique intents from the data
'''

import random
import numpy as np
import argparse

ap = argparse.ArgumentParser()
# add the training parameters
ap.add_argument('-p','--file-path',required = True, help = "Path and filename to the file which contains the label for the data", type = str, default = 'atis_hindi_label_train.txt')
ap.add_argument('-w','--writefile-path',required = True, help = "Path and filename to the file will save the unique slots in the respective files", type = str, default = 'atis_unqiue_hindi_intent.txt')

args  = vars(ap.parse_args())

filename = args['file_path']

all_slots = []

with open(filename) as fil:
    line = fil.readline()
    while line :
        line = line.strip('\n')
        line = line.strip('\t')
        all_slots.append(line)
        line = fil.readline()

# print(all_slots)

with open(args['writefile_path'], 'w') as fil:
    for i in set(all_slots):
        fil.write( "'" + i + "'" + '\n')

