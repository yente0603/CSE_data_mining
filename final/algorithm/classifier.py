import csv
import numpy as np
import dataProcess
datset_path = 'final/data/Arrhythmia Data Set/'

train_data, test_data, train_label, test_label = dataProcess.loadFile(datset_path)
# print(train_label.shape)
