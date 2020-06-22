


import torch
import torch.nn as nn
import torch.nn.init as init



class ChebConv(nn.Module):
    def __init__(self,in_c,out_c,K,normalization= True):
        super(ChebConv,self).__init__()
        self.weights = nn.Parameter(torch.Tensor(K,1,in_c,out_c))
        init.xavier_normal_(self.weights)

        
        self.bias = nn.Parameter(torch.Tensor(1,1,out_c)) 
        init.zeros_(self.bias)
        self.K = K
        self.norm = normalization
        
        



    def laplacian_matrix(self,graph):

        degree_mat = torch.sum(graph,dim=1)

        if self.norm:
            degree_mat_inverse = degree_mat ** (-0.5)
            D = torch.diag(degree_mat_inverse)
            laplacian_mat=torch.eye(graph.size(0),dtype=graph.dtype ,device = graph.device)- torch.mm(torch.mm(D,graph),D)
        else:
            
            laplacian_mat = torch.diag(degree_mat) - graph
        return laplacian_mat





    def chebp_laplacian(self,laplacian_mat):
        
        N = laplacian_mat.size(0)
        laplacian_mat_s = torch.zeros([self.K,N,N],dtype=torch.float,device=laplacian_mat.device)
        laplacian_mat_s[0] = torch.eye(N,dtype=torch.float,device=laplacian_mat.device)
        if self.K == 0:
            return laplacian_mat_s

        else:
            laplacian_mat_s[1] = laplacian_mat
            if self.K  ==1:
                return laplacian_mat_s

            else:
                for k in range(2,self.K):
                    laplacian_mat_s[k] = 2.0* torch.mm(laplacian_mat,laplacian_mat_s[k-1])- \
                                            laplacian_mat_s[k-2]
        return laplacian_mat_s.unsqueeze(1)   # [K,1,N,N]

    def forward(self,graph,data):
        """
        data :   # [B,N,H*D]
        """

        laplacian_mat = self.laplacian_matrix(graph)

        T_L = self.chebp_laplacian(laplacian_mat)    # [K,1,N,N]

        _ = torch.matmul(T_L,data)    # [K,B,N,H*D]

        _ = torch.matmul(_,self.weights) # [K,B,N,out_c]

        _ = torch.sum(_,dim=0) + self.bias #[B,N,out_c]

        return _