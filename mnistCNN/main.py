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

BATCH_SIZE=512 # batch size
EPOCHS=100         # batch epoches
LRATE = 0.001  # learning rate

class myData(Dataset.Dataset):
    # intitialize the dataset
    def __init__(self, filename):
        mydata = []
        mylabel = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i == 0:
                    i = i + 1
                    continue
                # add the the second row of the csv to my label
                mylabel.append(int(row.pop(0)))
                # convert row into float and transform to be a list
                temp = list(map(float, row))
                # add the list to mydata
                mydata.append(temp)
                i = i + 1
        # turns into ndarray
        temp=np.array(mylabel)
        # turns into tensor
        self.Label=torch.from_numpy(temp)
        self.Data=torch.from_numpy(np.array(mydata)).float()
    # get the length of data
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        # turns into a tensor
        data = torch.Tensor(self.Data[index])
        # squeeze(): drop the data  whose dimension is 1
        label = torch.Tensor([self.Label[index]]).squeeze().long()
        #long(): turns into a long type
        data = torch.div(data, 255).reshape(1,28, 28)
        return data, label

class CNNNet(nn.Module):
    def __init__(self):
        # subclass inherit nn's attribute and method
        super(CNNNet, self).__init__()
        self.conv1 = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            # stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(1,81, 3, 1, 1),
            nn.ReLU(),
            # class torch.nn.MaxPool2d(kernel_size, stride=None,
            # padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(81, 144, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(144, 100, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # full connected
        self.dense = nn.Sequential(
            # input: [batch_size, in_features]
            # output: [batch_size, out_features]
            # nn.Linear(in_features , out_features),
            # 1st fc layer
            nn.Linear(100*3*3 , 100),
            nn.ReLU(),
            # classify to 10 classes
            #2nd fc layer
            nn.Linear(100, 10)
        )

    # forward propagation
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        # convert conv_3 to one dimension output
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/mytrain.csv'
train_data=myData(filename)
# multi-processing
train_loader = Dataloader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/mytest.csv'
test_data = myData(filename)
test_loader = Dataloader.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

# loss function
loss_f = nn.CrossEntropyLoss()
#use_gpu = torch.cuda.is_available()
model=CNNNet()

#if (use_gpu):
    # true
    # model = model.cuda()
    # loss_f = loss_f.cuda()
model = model.to(device)
loss_f = loss_f.to(device)
print(model)
# model.parameters(): parameters to optimize
optimizer = torch.optim.Adam(model.parameters(),lr=LRATE)

# initial = 0 ; stride = 1
t=np.arange(EPOCHS)
# return a object which is full of zeros
trainY=np.zeros((EPOCHS))
testY=np.zeros((EPOCHS))
i=0
for epoch in range(EPOCHS):
    # epoch start from 0
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
        # convert tensor to float
        train_loss += loss.item()
        # torch.max(input_tensor, dimension)
        # dimension: 0 for column; 1 for row
        # output: [max, max_index](both are tensors)
        pred = torch.max(out, 1)[1]
        # the count of correct times
        train_correct = (pred == y).sum()
        train_acc += train_correct.item()
        # zero the parameter gradients
        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        # update(optimize) the parameters
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
    trainY[i]=train_acc / len(train_data)
    # test
    # 训练完train样本后，生成的模型model要用来测试样本。
    # 在model(test)之前，需要加上model.eval()，
    # 否则的话，有输入数据，即使不训练，它也会改变权值。
    # 这是model中含有BN层和Dropout所带来的的性质。
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
    # if accuracy is greatly high
    if trainY[i-1]>=0.99999:
        break
plt.plot(t, trainY,label="train")
plt.plot(t, testY,label="test")
plt.show()
#输出
filename = '/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/test.csv'
outflie="/Users/university/learningBoardly/scientificResearch/RSpartII/mnist/codeOutput/cnnResult.csv"
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
        # x/255 =》  [1, 1, 28, 28]
        x = torch.div(x, 255).reshape(1, 1, 28, 28)
        # x=x.cuda()
        x=x.to(device)
        out = model(x)
        pred = torch.max(out, 1)[1]
        # convert cuda tensor to cpu tensor and convert to numpy type
        label=pred.cpu().numpy()
        # predict the class for each image
        csv_writer.writerow([i,label[0]])
        i = i + 1
csvout.close()

