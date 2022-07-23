# -*- coding: utf-8 -*-
import csv

# order the feature
def transalte_label(filename):
    out = open('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/second.kddcup.data.corrected.csv', 'a', newline="")
    csv_write = csv.writer(out, dialect="excel")
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(',')
            data = []
            data.extend(line[22:24])
            data.extend(line[31:33])
            data.append(line[28])
            data.append(line[35])
            data.append(line[37])
            data.append(line[41])
            csv_write.writerow(data)

if __name__ == "__main__":
    transalte_label('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/dos.kddcup.data.corrected.csv')
