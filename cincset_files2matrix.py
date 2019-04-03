#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import scipy.io
import numpy as np
import glob

# Parameters
dataDir = '/some_path/' # <---- change!!
dataDir = 'training2017/'
FS = 300
WINDOW_SIZE = 60*FS


## Loading time serie signals
files = sorted(glob.glob(dataDir+"*.mat"))
trainset = np.zeros((len(files),WINDOW_SIZE))
count = 0
for f in files:
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    print('Loading record {}'.format(record))    
    data = mat_data['val'].squeeze()
    # Preprocessing
    print('Preprocessing record {}'.format(record))       
    data = np.nan_to_num(data) # removing NaNs and Infs
    data = data - np.mean(data)
    data = data/np.std(data)
    trainset[count,:min(WINDOW_SIZE,len(data))] = data[:min(WINDOW_SIZE,len(data))].T # padding sequence
    count += 1
    
## Loading labels    
import csv
csvfile = list(csv.reader(open(dataDir+'REFERENCE.csv')))
traintarget = np.zeros((trainset.shape[0],4))
classes = ['A','N','O','~']
for row in range(len(csvfile)):
    traintarget[row,classes.index(csvfile[row][1])] = 1
            
# Saving both
scipy.io.savemat('trainingset.mat',mdict={'trainset': trainset,'traintarget': traintarget})
