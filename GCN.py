import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

'''图卷积层'''
'''
nn.Module： PyTorch 中所有神经网络模块的基类，实现了图卷积层的基本结构
weight：定义权重矩阵，维度为[input_dim, output_dim]，并将其注册为模型的参数
bias：定义偏置，维度为[output_dim]，并将其注册为模型的参数；否则为None
reset_parameters()：初始化权重和偏置
'''
class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        '''
        :param input_dim: int, 输入特征维度
        :param output_dim: int, 输出特征维度
        :param use_bias: bool,optional, 是否使用偏置
        '''
        super(GraphConvolution,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    '''
    通过 Kaiming 均匀分布初始化权重，并将偏置初始化为零
    以确保模型在训练开始时有一个合理的初始状态
    '''
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight) #Kaiming 均匀分布初始化权重矩阵
        if self.use_bias:
            init.zeros_(self.bias) #将偏置向量 self.bias 初始化为零
    
    def forward(self,adjacency,input_feature):
        '''
        :param adjacency: torch.Tensor, 邻接矩阵
        :param input_feature: torch.Tensor, 输入特征
        :return: torch.Tensor, 输出特征
        '''
        '''邻接矩阵是稀疏矩阵'''
        support=torch.mm(input_feature,self.weight)
        output=torch.sparse.mm(adjacency,support) #稀疏矩阵乘法
        if self.use_bias:
            output+=self.bias
        return output
    
    def __repr__(self):
        return self.__class__.__name__+'('+str(self.input_dim)+'->'+str(self.output_dim)+')'
    
'''将邻接矩阵标准化'''
def normalization(adjacency):
    '''
    L=D^-0.5 * (A+I) * D^-0.5

    归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    '''
    adjacency+=sp.eye(adjacency.shape[0]) # 邻接矩阵加上自连接
    degree=np.array(adjacency.sum(1)) #每行的和
    d_hat=sp.diags(np.power(degree,-0.5).flatten())
    L=d_hat.dot(adjacency).dot(d_hat).tocoo()
    '''将稀疏矩阵转换为torch.Tensor (torch.sparse.FloatTensor)'''
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long() #获取稀疏矩阵的行和列索引，并转换为 PyTorch 张量
    values = torch.from_numpy(L.data.astype(np.float32)) #获取稀疏矩阵的非零值，并转换为 PyTorch 张量
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape) #创建一个 PyTorch 的稀疏张量 
    return tensor_adjacency
