# -*- coding: utf-8 -*-
from sklearn import model_selection
from sklearn import svm


def read_data(filename):
    train_set = []
    label_set = []
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip('\n').split(',')
            train_set.append(line[:-1])
            # line[:-1]其实就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
            label_set.extend([line[-1]])
    return train_set,label_set


# 创建分类器
if __name__ == "__main__":
    clf = svm.SVC(kernel='linear', C=100)
    train_set,label_set = read_data("/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/second.kddcup.data.corrected.csv")
    train_set.remove(['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_serror_rate'] )
    train_set.remove(['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_serror_rate'])
    train_set.remove(['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_serror_rate'])
    label_set.pop(0)
    label_set.pop(0)
    label_set.pop(0)
    train_set = [[float(y) for y in x] for x in train_set]
    label_set = [float(m) for m in label_set]
    score = model_selection.cross_val_score(clf, train_set, label_set, cv=5, scoring='accuracy')

    print(score)

    with open('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/crossvalidation.txt','a') as f:
        f.write(str(score))
