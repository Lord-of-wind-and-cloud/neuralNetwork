# -*- coding: utf-8 -*-

import csv
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch import nn
import torch.utils.data.dataloader as Dataloader
import torch.utils.data.dataset as Dataset
import matplotlib.pyplot as plt  #绘图

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE=2048    # 训练数据大小 与显存有关 越大速度越快
EPOCHS=30         # 总训练批次
LRATE = 0.02  # 学习率

class myData(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, filename):
        mydata = []
        mylabel =[]
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i == 0:
                    i = i + 1
                    continue
                mylabel.append(int(row.pop(0)))
                temp = list(map(float, row))
                mydata.append(temp)
                i = i + 1
        temp=np.array(mylabel)
        self.Label=torch.from_numpy(temp)
        self.Data=torch.from_numpy(np.array(mydata)).float()
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor([self.Label[index]]).squeeze().long()
        data = torch.div(data, 255).reshape(784)
        return data, label

class Batch_Net(nn.Module):
    """
    增加了一个加快收敛速度的方法——批标准化
    """

    def __init__(self):
        super(Batch_Net, self).__init__()
        # BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。
        # ①不仅仅极大提升了训练速度，收敛过程大大加快；
        # ②还能增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式，所以不用Dropout也能达到相当的效果；
        # ③另外调参过程也简单多了，对于初始化要求没那么高，而且可以使用大的学习率等。
        self.layer1 = nn.Sequential(nn.Linear(28*28, 1000), nn.BatchNorm1d(1000), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(1000, 100), nn.BatchNorm1d(100), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(100, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/mytrain.csv'
train_data=myData(filename)
train_loader = Dataloader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/mytest.csv'
test_data = myData(filename)
test_loader = Dataloader.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
loss_f = nn.CrossEntropyLoss()
#use_gpu = torch.cuda.is_available()
model = Batch_Net()
#model=CNNNet()
#if (use_gpu):
# model = model.cuda()
# loss_f = loss_f.cuda()
model = model.to(device)
loss_f = loss_f.to(device)
print(model)
#optimizer = torch.optim.Adam(model.parameters(),lr=LRATE)

#optimizer  = torch.optim.SGD(model.parameters(), lr=LRATE)
# momentum 动量加速,在SGD函数里指定momentum的值即可
#optimizer = torch.optim.SGD(model.parameters(), lr=LRATE, momentum=0.8)
# RMSprop 指定参数alpha
optimizer  = torch.optim.RMSprop(model.parameters(), lr=LRATE, alpha=0.9)
# Adam 参数betas=(0.9, 0.99)
#optimizer = torch.optim.Adam(model.parameters(), lr=LRATE, betas=(0.9, 0.99))

t=np.arange(EPOCHS)
trainY=np.zeros((EPOCHS))
testY=np.zeros((EPOCHS))
i=0
for epoch in range(EPOCHS):
    print('epoch {}'.format(epoch + 1))
    # train
    model.train()
    train_loss = 0.
    train_acc = 0.
    for x, y in train_loader:
        #if (use_gpu):
        # x = x.cuda()
        # y = y.cuda()
        x = x.to(device)
        y = y.to(device)
        x, y = Variable(x), Variable(y)
        out = model(x)
        loss = loss_f(out, y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
    trainY[i]=train_acc / len(train_data)
    # test
    model.eval()
    test_loss = 0.
    test_acc = 0.
    for x, y in test_loader:
        #if (use_gpu):
        # x = x.cuda()
        # y = y.cuda()
        x = x.to(device)
        y = y.to(device)
        x, y = Variable(x), Variable(y)
        out = model(x)
        loss = loss_f(out, y)
        test_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == y).sum()
        test_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / (len(test_data)), test_acc / (len(test_data))))
    testY[i]=test_acc / len(test_data)
    i=i+1
    if trainY[i-1]>=0.99999:
        break
plt.plot(t, trainY,label="train")
plt.plot(t, testY,label="test")
plt.show()

filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/test.csv'
outflie="/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/codeOutput/mlpResult.csv"
csvout = open(outflie,'w',encoding='utf-8',newline='')
csv_writer = csv.writer(csvout)
csv_writer.writerow(['ImageId', 'Label'])
with open(filename, 'r') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
        if i == 0:
            i = i + 1
            continue
        temp = list(map(float, row))
        x = torch.Tensor(temp)
        x = torch.div(x, 255).reshape(1,784)
        x = Variable(x)
        #x=x.cuda()
        x=x.to(device)
        out = model(x)
        pred = torch.max(out, 1)[1]
        lable=pred.cpu().numpy()
        csv_writer.writerow([i,lable[0]])
        i = i + 1
csvout.close()
