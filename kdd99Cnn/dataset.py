# -*- coding:utf-8 -*-

from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import csv, os, random
import torch.nn as nn


class BasicDataset(Dataset):
    """create the BasiceDataset"""

    def __init__(self, data_path='/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/cnn/train_data.csv', train_flag=True):
        # initialize
        super(BasicDataset, self).__init__()
        self.train_flag = train_flag
        self.data_path = data_path
        print('the data path: ', self.data_path)
        self.data, self.data_length = self.preprocess(data_path=self.data_path, train_flag=self.train_flag)

    def __len__(self):
        return self.data_length

    def preprocess(cls, data_path, train_flag):
        csv_reader = csv.reader(open(data_path))
        datasets = []
        if train_flag:
            # if the data have not been trained
            label0_data = []
            label1_data = []
            label2_data = []
            label3_data = []
            label4_data = []
            label_status = {}
            for row in csv_reader:
                data = []
                for char in row:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))  # transform data from format of string to float32

                # if the last of the data point to a type
                # add the data to the corresponding class
                if data[-1] == 0:
                    label0_data.append(data)
                if data[-1] == 1:
                    label1_data.append(data)
                if data[-1] == 2:
                    label2_data.append(data)
                if data[-1] == 3:
                    label3_data.append(data)
                if data[-1] == 4:
                    label4_data.append(data)
                # record the number of different labels
                if label_status.get(str(int(data[-1])), 0) > 0:
                    label_status[str(int(data[-1]))] += 1
                else:
                    label_status[str(int(data[-1]))] = 1
            while len(label0_data) < 10000:
                label0_data = label0_data + label0_data
            # make sure that the length of label_data are more than 10000
            # label0_data = random.sample(label0_data, 10000)
            while len(label1_data) < 10000:
                label1_data = label1_data + label1_data
            # label1_data = random.sample(label1_data, 10000)
            while len(label2_data) < 10000:
                label2_data = label2_data + label2_data
            # label2_data = random.sample(label2_data, 10000)
            while len(label3_data) < 10000:
                label3_data = label3_data + label3_data
            # label3_data = random.sample(label3_data, 10000)
            while len(label4_data) < 10000:
                label4_data = label4_data + label4_data
            # label4_data = random.sample(label4_data, 10000)'''
            datasets = label0_data + label1_data + label2_data + label3_data + label4_data
        else:
            for row in csv_reader:
                data = []
                for char in row:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))  # transform data from format of string to float32
                datasets.append(data)
        # delete ignored 9-21
        ignored_col = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        datasets = np.delete(datasets, ignored_col, axis=1)
        data_length = len(datasets)

        minibatch = datasets
        # arr:输入向量 (需要处理的矩阵)
        # obj:表明哪一个子向量应该被移除。可以为整数或一个int型的向量
        # axis:表明删除哪个轴的子向量，若默认，则返回一个被拉平的向量
        data = np.delete(minibatch, -1, axis=1)
        # get the last info of every row
        labels = np.array(minibatch, dtype=np.int32)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            # avoid getting devided by 0
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001
        res = (data - mmin) / (mmax - mmin)
        # print('res.shape:', res.shape)
        res = np.c_[res, labels]
        # 随机打乱数据集实例
        np.random.shuffle(datasets)
        print('init data completed!')
        return datasets, data_length

    def __getitem__(self, i):
        data = np.array(np.delete(self.data[i], -1))
        data = np.concatenate([data, np.array([0] * 8)])
        data = data.reshape(6, -1)
        data = np.expand_dims(data, axis=2)
        # HWC to CHW
        # 转置函数 transepose()完成高维数组的转置，返回的是源数据的视图，不进行任何复制操作
        data = data.transpose((2, 0, 1))
        label = np.array(self.data[i], dtype=np.float32)[-1]
        return {'data': torch.from_numpy(data), 'label': label}



