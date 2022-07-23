# -*- coding: utf-8 -*-
import csv

#extract needy feature
def transalte_label(filename):
    out = open('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/first.kddcup.data.corrected.csv', 'a', newline="")
    csv_write = csv.writer(out, dialect="excel")
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(',')
            data = []
            # only reserve line[22, 42]
            data.extend(line[22:42])
            csv_write.writerow(data)

if __name__ == "__main__":
    transalte_label('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/dos.kddcup.data.corrected.csv')
