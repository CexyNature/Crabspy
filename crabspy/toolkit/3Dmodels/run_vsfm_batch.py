#!/usr/bin/python

"""
This script recursively runs VisualSFM feature detection, image matching and spare and dense reconstruction in all folder inside the selected directory.
"""

import os
import subprocess
from datetime import datetime

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

start_time = datetime.now()

current_path = os.getcwd()
vsfm_path = '/VisualSFM'

data_path = os.path.join(current_path, 'Data_frames')
print('Listing directories in following path: {}'.format(data_path))

# Get all sub directories in data directory
data_directories = os.listdir(data_path)

# Create empty list to store full path for each data directory
list_dirs = []
list_dirs_m = []

# Append each full path to list
for i in range(len(data_directories)):
    sub_directory = os.path.join(data_path, data_directories[i])
    list_dirs.append(sub_directory)
    sub_directory_model = os.path.join(sub_directory, 'model_vsfm')
    list_dirs_m.append(sub_directory_model)
    print('Sub-directory: {}'.format(sub_directory))

# Create directory named model in each data directory, and result path name place holder
models_paths = []
for i in range(len(list_dirs_m)):

    model_res_path = os.path.join(list_dirs_m[i], 'model.nvm')
    models_paths.append(model_res_path)
    print('Result path name holder created: {}'.format(model_res_path))

    if not os.path.exists(list_dirs_m[i]):
        os.makedirs(list_dirs_m[i])
        print('Folder model created in: {}'.format(list_dirs_m[i]))
    else:
        print('Folder model already exists')

# Run VisualSfM in each data directory
os.chdir(vsfm_path)
for i, j in zip(list_dirs, models_paths):
    print(i)
    print(j)
    subprocess.call(['VisualSFM', 'sfm+pairs+pmvs', i, j], shell = True)

print('Total time {}'.format(datetime.now() - start_time))
