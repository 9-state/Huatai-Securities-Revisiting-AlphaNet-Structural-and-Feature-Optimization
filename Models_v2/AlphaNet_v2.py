#!/usr/bin/env python
# coding: utf-8

# # AlphaNet_v2

# 1. 扩充了6个比率类特征，“数据图片”维度变为15*30
# 2. 将池化层和全连接层替换为LSTM层，从而更好地学习特征的时序信息

# In[1]:


import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


# # 自定义卷积核(特征提取)

# In[2]:


'''
过去d天X值构成的时序数列和Y值构成的时序数列的相关系数
'''

class ts_corr(nn.Module):
    def __init__(self, d=10, stride=10):
        super(ts_corr, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X): #X:3D
        B, n, T = X.shape # 批量大小，特征数量，时间窗口长度
        
        w = (T - self.d) // self.stride + 1  # 窗口数量,例如T=10，d=3，stride=2时，w=4
        h = n * (n - 1) // 2  # 特征对的数量 C(n, 2)

        # 使用 unfold 提取滑动窗口 [形状: (B, n, w, d)]
        unfolded_X = X.unfold(2, self.d, self.stride)
        
        #生成C(n,2)组合数
        #例如当n=3时，rows = tensor([0,0,1]), cols = tensor([1,2,2])
        rows, cols = torch.triu_indices(n, n, offset=1)

        # 提取特征对数据得到x和y [形状: (B, h, w, d)]，分别对应batch维度，特征维度，窗口维度，时间维度
        #x为([[:,0,:,:],[:,0,:,:],[:,1,:,:])
        #y为([[:,1,:,:],[:,2,:,:],[:,2,:,:])
        
        x = unfolded_X[:, rows, :, :]
        y = unfolded_X[:, cols, :, :]
        
        x_mean = torch.mean(x, dim=3, keepdim=True) #keepdim维持原本维度不变,在维度3做mean
        y_mean = torch.mean(y, dim=3, keepdim=True)
        
        cov = ((x-x_mean)*(y-y_mean)).sum(dim=3) #(B, h, w)
        corr = cov / (torch.std(x, dim=3) * torch.std(y, dim=3)+ 1e-8) #分母添加极小值防止除零错误

        return corr


# In[3]:


'''
过去d天X值构成的时序数列和Y值构成的时序数列的协方差
'''

class ts_cov(nn.Module):
    def __init__(self, d=10, stride=10):
        super(ts_cov, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        B, n, T = X.shape 
        
        w = (T - self.d) // self.stride + 1  
        h = n * (n - 1) // 2  

        unfolded_X = X.unfold(2, self.d, self.stride)
        
        rows, cols = torch.triu_indices(n, n, offset=1)

        x = unfolded_X[:, rows, :, :]
        y = unfolded_X[:, cols, :, :]
        
        x_mean = torch.mean(x, dim=3, keepdim=True)
        y_mean = torch.mean(y, dim=3, keepdim=True)
        
        cov = ((x-x_mean)*(y-y_mean)).sum(dim=3) 

        return cov


# In[4]:


'''
过去d天X值构成的时序数列的标准差
'''

class ts_stddev(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_stddev,self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        #input:(B,n,T)，在T维度用unfold展开窗口，变为(B,n,w,d),w为窗口数量会自动计算
        unfolded_X = X.unfold(2, self.d, self.stride)
        #在每个窗口，即d维度上进行std计算
        std = torch.std(unfolded_X, dim=3) #输出形状为(B,n,w)
        
        return std


# In[5]:


'''
过去d天X值构成的时序数列的平均值除以标准差
'''

class ts_zscore(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_zscore,self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        
        unfolded_X = X.unfold(2, self.d, self.stride)
        
        mean = torch.mean(unfolded_X, dim=3)
        std = torch.std(unfolded_X, dim=3)
        zscore = mean / (std + 1e-8)
        
        return zscore


# In[6]:


'''
研报原话为：
(X - delay(X, d))/delay(X, d)-1, delay(X, d)为 X 在 d 天前的取值
这里可能有误，return为“收益率“，应该是误加了-1
为了保持代码一致性，这里计算的是(X - delay(X, d-1))/delay(X, d-1),  delay(X, d-1)为 X 在 d-1 天前的取值
在构造卷积核的逻辑上是相似的
'''

class ts_return(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_return,self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        
        unfolded_X = X.unfold(2, self.d, self.stride)
        return1 = unfolded_X[:,:,:,-1] /(unfolded_X[:,:,:,0] + 1e-8) - 1
        
        return return1


# In[7]:


'''
过去d天X值构成的时序数列的加权平均值，权数为d, d – 1, …, 1(权数之和应为1，需进行归一化处理）
其中离现在越近的日子权数越大。 
'''

class ts_decaylinear(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_decaylinear,self).__init__()
        self.d = d
        self.stride = stride
        #如下设计的权重系数满足离现在越近的日子权重越大
        weights = torch.arange(d, 0, -1, dtype = torch.float32) 
        weights = weights / weights.sum()
        #注册权重，不用在前向传播函数中重复计算
        #注册了一个形状为(d,)的一维张量，存放权重系数，以便在forward函数中使用
        self.register_buffer('weights', weights) 
        
    def forward(self, X):
        
        unfolded_X = X.unfold(2, self.d, self.stride)
        #view将一维张量广播为4D张量，并在时间维度上，将weights与unfoled_X相乘
        decaylinear = torch.sum(unfolded_X * self.weights.view(1,1,1,-1), dim=-1)
        
        return decaylinear


# # v2神经网络结构设计 

# 路径(Path)：特征提取层→BN→LSTM层→BN→预测目标

# In[8]:


'''
路径前半部分：特征提取层→BN
'''
class Path1(nn.Module):
    def __init__(self, extractor, bn_dim): #传入参数：卷积核，特征维度
        super().__init__()
        self.extractor = extractor
        self.bn = nn.BatchNorm1d(bn_dim)
        
    def forward(self, X):
        x = self.extractor(X) #extract
        x = self.bn(x) #BN

        return x


# In[9]:


'''
路径后半部分：LSTM→BN→Output

帮助理解LSTM：https://blog.csdn.net/mary19831/article/details/129570030
LSTM的三个参数：
input_size: 输入特征的维度
hidden_size：隐藏状态的数量
output_size：输出的维度
num_layers：LSTM堆叠的层数

'''
class Path2(nn.Module): 
    #研报中定义，LSTM中，输出神经元数 = 30，time_step = 3
    def __init__(self, input_size, hidden_size = 30, num_layers = 1):
        super().__init__()
        
        '''
        若batch_first = True，则lstm模型的输入维度为（batch_size,seq_len,input_size)，
        输出维度为（batch_size,seq_len,input_size)。若默认，则batch_size与seq_len位置互换
        研报中指定time_step = 3(seq_len = 3)，这刚好跟卷积核输出维度中的w是一样的，因此可以直接变换维度带入
        '''
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, X):
        x = X.permute(0, 2, 1) #将（B，n，w）变为lstm需要的（B，w，n）
        x, _ = self.lstm(x) #形状（B，3，hidden_size）
        x = x[:,-1,:] # 取最后一个时间步作为输出 形状(B，hidden_size)
        x = self.bn(x) #(B,hidden_size)

        return x 


# In[10]:


class AlphaNet_v2(nn.Module):

    def __init__(self, d=10, stride=10,n=15, T=30): #池化层窗口d=3，步长stride=3
        super(AlphaNet_v2, self).__init__()
        
        self.d = d 
        self.stride = stride 
        h = n * (n - 1) // 2 #手动计算cov和corr特征提取后的特征维度大小
        w = (T - d) // stride + 1 #手动计算特征提取层窗口数
        
        #特征提取层列表，共6个
        self.extractors = nn.ModuleList([
            ts_corr(d,stride),
            ts_cov(d,stride),
            ts_stddev(d,stride),
            ts_zscore(d,stride),
            ts_return(d,stride),
            ts_decaylinear(d,stride),
        ])
        

        # 初始化
        self.Path1s = nn.ModuleList()
        
        # 前2个特征提取器使用h维BN
        for i in range(2):
            self.Path1s.append(Path1(self.extractors[i], h))
            
        # 后4个特征提取器使用n维BN
        for i in range(2, 6):
            self.Path1s.append(Path1(self.extractors[i], n))
            
        self.head = nn.Sequential(
            Path2(input_size = h*2 + n*4),
            nn.Linear(30, 1)
        )

        # 初始化线性层和输出层
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                
                #截断正态初始化
                fan_in = m.weight.size(1)
                std = math.sqrt(1.0 / fan_in)  # Xavier方差标准
                nn.init.trunc_normal_(m.weight, std=std, a=-2*std, b=2*std)
                nn.init.normal_(m.bias, std=1e-6)
                
                #Kaiming初始化
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                #Xavier初始化
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.normal_(m.bias, std=1e-6)
        
    def forward(self, X):
        
        features = [path(X) for path in self.Path1s] 
        features = torch.cat(features, dim=1)
        
        return self.head(features)






