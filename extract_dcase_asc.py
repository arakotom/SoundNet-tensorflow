#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:57:25 2018

@author: alain
"""

import os

start_layer = 16
end_layer = 22

year = 2013
data_path = '/media/alain/9C33-6BBD2/recherche/dcase2013/'

year = 2016
data_path = '/media/alain/9C33-6BBD2/recherche/dcase2016/data/data/'




script = "python3 extract_feat.py"
optionlist = " -m " + str(start_layer) + " -x " + str(end_layer) + " -s -p extract "
data = 'dcase' + str(year)

for train in [True,False]:
    if train :
        train_test = 'train'
        data_path_full = data_path + 'train/'
    else:
        train_test = 'test'
        data_path_full = data_path + 'test/'
    
    output = "".join([data,'_',train_test])
    file_wav = output + '.txt'
    #-t dcase2013_train.txt -o dcase2013_test -i /media/alain/9C33-6BBD2/recherche/dcase2013/test/"
    
    command = "".join([script, optionlist, " -t ", file_wav, " -o ", output, " -i ", data_path_full])
    print(command)
    os.system(command)