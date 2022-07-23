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
EPOCHS=150        # 总训练批次
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
        data = torch.div(data, 255).reshape(28, 28)
        return data, label

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.rnn=nn.LSTM(
            # features
            input_size=28,
            #features
            hidden_size=64,
            # one layer
            num_layers=1,
            # see bath as first parameter in the tensor
            batch_first=True
        )
        # input_features = 64
        # output_fearture = 64
        # output has 10 classes
        self.out=torch.nn.Linear(64,10)
    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        # h_n for output last step
        # h_c for cell state last step
        # None 表示 hidden state 会用全0的 state
        # 比如第一层没有hidden state时就是none
        r_out, (h_n, h_c) = self.rnn(x, None)
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out


filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/mytrain.csv'
train_data=myData(filename)
train_loader = Dataloader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/mytest.csv'
test_data = myData(filename)
test_loader = Dataloader.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
loss_f = nn.CrossEntropyLoss()
#use_gpu = torch.cuda.is_available()
model = LSTMNet()
#if (use_gpu):
# model = model.cuda()
# loss_f = loss_f.cuda()
model = model.to(device)
loss_f = loss_f.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr=LRATE)

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
    for step, (x, y) in enumerate(train_loader):
        # if (use_gpu):
        #     x = x.cuda()
        #     y = y.cuda()
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
        # if (use_gpu):
        #     x = x.cuda()
        #     y = y.cuda()
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

    if trainY[i-1]>=0.999999999:
        break
plt.plot(t, trainY,label="train")
plt.plot(t, testY,label="test")
plt.show()

filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/test.csv'
outflie="/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/codeOutput/lstmResult.csv"
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
        x = torch.div(x, 255).reshape(1,28, 28)
        #x=x.cuda()
        x=x.to(device)
        out = model(x)
        pred = torch.max(out, 1)[1]
        lable=pred.cpu().numpy()
        csv_writer.writerow([i,lable[0]])
        i = i + 1
csvout.close()
