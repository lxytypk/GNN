import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer,self).__init__()
        self.in_features = in_features #节点向量的特征维度
        self.out_features = out_features #经过GAT后的特征维度
        self.dropout = dropout #dropout参数
        self.alpha = alpha #LeakyReLU参数

        #定义可训练参数：W和a
        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #xavier初始化
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414) #xavier初始化

        #定义LeakyReLU激活函数
        self.leakyrelu=nn.LeakyReLU(self.alpha)

    def forward(self,input_h,adj):
        '''
        input_h: 输入节点特征，维度为[N,in_features]
        adj: 邻接矩阵，维度为[N,N]
        self.W: [input_features,out_features]
        input_h*W: [N,out_features]
        '''

        #[N, out_features]
        h=torch.mm(input_h,self.W)

        N=h.size()[0] #节点个数
        input_concat=torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)

        #[N,N,2*out_features] * [2*out_features,1] = [N,N,1]
        #[N,N,1] => [N,N] 图注意力的相关系数（未归一化）
        e=self.leakyrelu(torch.matmul(input_concat,self.a).squeeze(2))

        zero_vec=-1e12 * torch.ones_like(e) #没有连接的边设置为负无穷
        attention=torch.where(adj>0,e,zero_vec)

        attention=F.softmax(attention,dim=1)
        attention=F.dropout(attention,self.dropout,training=self.training) #dropout,防止过拟合
        #[N, N] * [N, out_features] = [N, out_features]
        output_h=torch.matmul(attention,h)
        
        #得到由周围节点通过注意力权重进行更新的表示
        return output_h


if __name__=="__main__":
    x=torch.randn(6,10)
    adj=torch.tensor([[0,1,1,0,0,0],
                      [1,0,1,0,0,0],
                      [1,1,0,1,0,0],
                      [0,0,1,0,1,1],
                      [0,0,0,1,0,1],
                      [0,0,0,1,1,0]])
    
    my_gat=GATLayer(10,5,0.2,0.2)
    print(my_gat(x,adj))
