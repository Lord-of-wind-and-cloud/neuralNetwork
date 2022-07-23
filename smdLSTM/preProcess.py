import ast
import csv
import os
import sys
from pickle import dump

import numpy as np

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    with open(os.path.join('output/', dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data():
    dataset_folder = 'ServerMachineDataset'
    file_list = os.listdir(os.path.join(dataset_folder, "train"))
    for filename in file_list:
        if filename.endswith('.txt'):
            load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
            load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
            load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder)


if __name__ == '__main__':
    load_data()