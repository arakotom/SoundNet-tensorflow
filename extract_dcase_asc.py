#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script for creating Soundnet feature for dcase 2013 and 2016

you need to define the path of the data and provide in that directory
files listing all the .wav to be processed.  They should be named train.txt and test.txt

I typically obtain them through a  ls | cat > training_set.txt  on the appropriate directory.


"""

import os

start_layer = 16
end_layer = 22
sample_rate = 22050 # target sampling rate for soundnet
year = 2013


if year == 2013:
    data_path = '/media/alain/9C33-6BBD2/recherche/dcase2013/'
    load_size = 30*sample_rate #
elif year == 2016:
    load_size = 10*sample_rate #
    data_path = '/media/alain/9C33-6BBD2/recherche/dcase2016/data/data/'
elif year == 2017:
    load_size = 10*sample_rate
    data_path = '/media/alain/9C33-6BBD2/recherche/dcase2017/'


script = "python3 -u extract_feat.py"
optionlist = " -m " + str(start_layer) + " -x " + str(end_layer) + " -s -p extract "

for train in [True,False]:
    if train :
        train_test ='train'
    else:
        train_test = 'test'
    
    
    data_path_output = data_path + 'feat_' + train_test + '/'
    file_wav = data_path + train_test + '.txt'
    if year == 2017:
        data_path_input = data_path
    else:
        data_path_input = data_path + train_test + '/'

    command = "".join([script, optionlist, " -t ", file_wav, " -o ", data_path_output, " -i ", data_path_input])
    command = "".join([command," -l ", str(load_size), " -r ",  str(sample_rate)])
    print(command)
    os.system(command)