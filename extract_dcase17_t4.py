#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:57:25 2018

Script for feature extraction on for dcase 2017  T4 dataset

in your data directory, you need to files listing all the .wav for which
features are needed.  I typically obtain them through a 
ls | cat > training_set.txt  on the appropriate directory.





@author: alain 16/01/2018
"""
import os

# setup for the feature extraction
start_layer = 16  # first layer to consider
end_layer = 22    # last layer to consider
sample_rate = 22050 # target sampling rate for soundnet
load_size = 10*sample_rate #


# where your data are located
if os.getcwd().find('alain') > 0:
    data_path = '/media/alain/Dell\ Portable\ Hard\ Drive/dataaudio/dcase2017-Task4/'
else:
    data_path = '/mnt/hdd1/audio_data_2017/task_4/' # on server

# set to process
data_path_train = 'unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/'
data_path_test = 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads/'
data_path_eval = 'evaluation_set_formatted_audio_segments/'

data_list = [ data_path_test, data_path_eval, data_path_train]
file_wav_list = ['testing_set.txt','evaluation_set.txt','training_set.txt']




script = "python3 -u extract_feat.py"
optionlist = " -m " + str(start_layer) + " -x " + str(end_layer) + " -s -p extract "
for (i,dataset) in enumerate(data_list):
    # creating all the filename and directory path
    file_wav = data_path + file_wav_list[i]
    output = data_path + file_wav_list[i] + '/'
    dataset_dir = data_path + data_list[i]
    
    command = "".join([script, optionlist, " -t ", file_wav, " -o ", output, " -i ", dataset_dir, " -l ", str(load_size), " -r ",  str(sample_rate)])
    
    
    print(command)
    os.system(command)