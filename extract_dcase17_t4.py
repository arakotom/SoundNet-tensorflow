#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:57:25 2018

@author: alain
"""
import os

start_layer = 18
end_layer = 19
sample_rate = 22050
load_size = 10*sample_rate
data = 'dcase17_T4_'

if os.getcwd().find('alain') > 0:
    data_path = '/media/alain/Dell\ Portable\ Hard\ Drive/dataaudio/dcase2017-Task4/'
else:
    data_path = '/mnt/hdd1/audio_data_2017/task_4/' # on server


data_path_train = 'unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/'
data_path_test = 'unbalanced_train_segments_test_set_audio_formatted_and_segmented_downloads/'
data_path_eval = 'evaluation_set_formatted_audio_segments/'

data_list = [ data_path_test, data_path_eval, data_path_train]
file_wav_list = ['testing_set','evaluation_set','training_set']




script = "python3 extract_feat.py"
optionlist = " -m " + str(start_layer) + " -x " + str(end_layer) + " -s -p extract "
for (i,dataset) in enumerate(data_list):
    # creating all the filename and directory path
    file_wav = data_path + data + file_wav_list[i] + '.txt'
    output = data_path + file_wav_list[i] + '/'
    dataset_dir = data_path + data_list[i]
    
    command = "".join([script, optionlist, " -t ", file_wav, " -o ", output, " -i ", dataset_dir, " -l ", str(load_size), " -r ",  str(sample_rate)])
    
    
    print(command)
    os.system(command)