"""
数据清洗 获取dos流量和normal的混合流量 且是网络协议类型是TCP类型的
"""
# -*- coding: utf-8 -*-
import csv
# dos攻击的类型和正常类型
dos_label = ['back.', 'land.', 'neptune.',
             'smurf.', 'teardrop.', 'pod.', 'normal.']

# count dos攻击的类型数量和正常类型数量
def get_train_dataset(filename):
    normal = 0
    attack = 0
    columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']
    # newline=''这一参数使写入的文件没有多余的空行
    out = open('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/dos.kddcup.data.corrected.csv', 'a', newline="")
    # csv文件中的格式有很多：逗号分隔 冒号分隔等等。自己设置一个dialect类，规定格式来读取或写入
    csv_write = csv.writer(out, dialect="excel")
    csv_write.writerow(columns) # 写入属性列
    with open(filename, "r") as f:
        for line in f:
            # strip()表示删除掉数据中的换行符
            line = line.strip('\n')
            # split（‘，’）则是数据中遇到‘,’ 就隔开。
            line = line.split(',')
            # line[41] : 'label'; : line[1] : 'protocol_type'
            if line[41] in dos_label and line[1] == 'tcp':  # 挑选出是dos攻击的流量和正常流量
                if line[41] == 'normal.':
                    line[41] = '1'  # 正常流量设置为 1
                    normal += 1
                else:
                    attack += 1
                    line[41] = '-1'  # 异常流量设置为 -1
                csv_write.writerow(line)
    print("get train dataset success")
    print("normal:", normal)  # 记录筛选之后normal的数量
    print("attack:", attack)  # 记录筛选之后attack的数量


if __name__ == "__main__":
    get_train_dataset("/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/codeOutput/svm/kddcup.data.corrected.csv")
