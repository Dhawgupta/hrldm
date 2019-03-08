'''
This program is writtent to clean and extract the hindi data from the given CSV files
'''
import csv
import pandas as pd
import random
import numpy as np
import argparse
import codecs
import argparse

ap = argparse.ArgumentParser()
# add the training parameters
ap.add_argument('-tt','--file-path',required = True, help = "Path and filename to the file which contains the label for the data", type = str, default = 'atis_hindi_label_train.txt')
ap.add_argument('-w','--writefile-path',required = False, help = "Path and filename to the file will save the unique slots in the respective files", type = str, default = 'atis_unqiue_hindi_intent.txt')

args  = vars(ap.parse_args())
filename = args['file_path']

train = []
txt = ''
data = csv.reader(codecs.open(filename, encoding='utf-8'))
# with codecs.open(filename, encoding='utf-8') as fil:
#     # print('The string is:', string)
#     #
#     # # default encoding to utf-8
#     # string_utf = string.encode()
#     #
#     # # print result
#     # print('The encoded version is:', string_utf)
#     txt = fil.readline().encode(encoding='utf-8')
#     while txt:
#         train.append(txt)
#         txt = fil.readline()
data = list(data)




