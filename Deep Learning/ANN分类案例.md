```

# 导包
import torch #Pytorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset #数据集对象，数据->Tensor->数据集->数据加载器
from torch.utis.data import DataLoader #数据加载器
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import time

# 1.定义函数，构建数据集
def creat_dataset():
    #1.加载csv文件数据集
    data=pd.read_csv('./data/手机价格预测.csv')
    #print（'data:{data.head()}'）
    #print('data:{data.shape}')#(2000,21)

    #2.获取x特征列和y标签列
    x,y=data.iloc[:,:-1],dataa.iloc[:,-1]
  
    #3.把特征列转换成浮点型
    x=x.astype(np.float32)
  
    #4.切分训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test+size,random_state=42)

    #5.把数据封装成 张量数据。数据->张量Tensor->数据集TensorDataSet->数据加载器DataLoader
    train_dataset=TensorDataset（torch.tensor(x_train.values),torch.tensor(y_train.values)）
    test_dataset=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))

    #6.返回结果
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))


# 2. 搭建神经网络
class PhonePriceModel(nn.Module):
    #1.在init方法中初始化父类对象，以及搭建神经网络
    def __init__(self,input_dim,out_dim):
        #1.1初始化父类成员
        super().__init__
        #1.2搭建神经网络
        #隐藏层1
        self.linear1=nn.Linear(20,128)
        #隐藏层2
        self.linear2=nn.Linear(128,256)
        #输出层
        self.output=nn.Linear(256,4)
    #定义前向传播方法forward（）
    def forward(self,x):
        #2.1隐藏层1:加权求和+激活函数（relu）
        x=torch.relu(self.linear1(x))
        #2.2隐藏层2:加权求和+激活函数（relu）
        x=torch.relu(self.linear2(x))
        #2.3输出层：加权求和+激活函数（softmax）
        # x=torch.softmax(self.output(x),dim=1)
        x=self.output(x)
        #2.4返回处理结果
        return x

#3.模型训练
def train(train_dataset,input_dim,output_dim):
    #1.创建数据加载器
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #2.创建神经网络模型
    model=PhonePriceModel(input_dim,output_dim)
    3.定义损失函数，因为是多分类，这里用的是多分类交叉熵损失函数
    criterion=nn.CrossEntropLoss（）
    #4.创建优化器对象
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    #5.模型训练
    #5.1定义变量，记录训练总轮数
    epochs=50
    #5.2开始（每轮的）训练
    for epoch in range(epochs):
        #5.2.1定义变量，记录每次训练的损失值，训练批次数
        total_loss,batch_num=0.0,0
        #5.2.2定义变量，表示训练开始时间
        start=time.time()
        #5.2.3开始本轮的各个批次的训练
        for x,y in train_loader:
            #5.2.4切换模型（状态）
            model.train()
            #5.2.5模型预测
            y_pred=model(x)
            #5.2.6计算损失
            loss=criterion(y_pred,y)
            #5.2.7梯度清零，方向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #5.2.8累加损失值
            total_loss+=loss.item()
            batch_num+=1
        #5.2.4至此，本轮训练结束，打印训练信息
        print(f'epoch:{epoch+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start:.2f}s')
    #6.多轮训练结束，保存模型
    print(f'模型的参数信息：{model.state_dict()}')
    torch.save(model.state_dict(),'./model/phone.pth')
#4.模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #1.创建神经网络分类对象
    model=PhonePriceModel(input_dim,output_dim)
    #2.加载模型参数
    model.load_state_dict(torch.load('./model/phone.pth'))
    #3.创建测试集的数据加载器对象
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False）
    #4.定义变量，记录预测正确的样本数
    correct=0
    #5.从数据加载器中，获取到每批次的数据
    for x,y in test_loader:
        #5.1切换模型状态
        model.eval()
        #5.2模型预测
        y_pred=model(x)
        print(f'y_pred:{y_pred}')
        #5.3根据加权求和，得到类别，用argmax（）获取最大值对应的下标，就是类别
        y_pred=torch.argmax(y_pred,dim=1)
        print(f'y_pred:{y_pred}')
    #6.打印正确率
    print(f'Accuracy:{correct/len(test_dataset):.4f}')


# 测试
if __name__=='__main__':
    train_dataset,test_dataset,input_dim,out_dim=creat_dataset()
    print(f'训练集 数据集对象：{train_dataset}')
    print(f'测试集 数据集对象：{test_dataset}')
    print(f'输入特征数：{input_dim}')#20
    print(f'输出标签数:{output_dim}')#4
    #2.构建神经网络模型
    model=PhonePriceModel(input_dim,out_dim)
    #计算模型参数
    summary(model,input_size(16,input_dim))
    #3.模型训练
    train(train_dataset,input_dim,output_dim)
    #4.模型测试
    evaluate(test_dataset,input_dim,output_dim)
```
