# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/4

@Author : Shen Fang
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from traffic_dataset import LoadData
from ChebConv import ChebConv
from GATConv import GraphAttentionLayer
from utils import MAE_per_batch,MAPE_per_batch,RMSE_per_batch
import numpy as np
class chebNet(nn.Module):
    def __init__(self,in_c,hid_c,out_c):
        super(chebNet,self).__init__()
        self.conv1 = ChebConv(in_c,hid_c,K=5)
        self.conv2 = ChebConv(hid_c,out_c,K=5)
        self.act  = nn.ReLU()

        

    def forward(self,data,device):

        graph = data['graph'][0].to(device)

        flow_x = data['flow_x'].to(device)

        B,N,H,D = flow_x.size()
        
        flow_x = flow_x.view(B,N,H*D)

        _ = self.act(self.conv1(graph,flow_x))

        _ = self.act(self.conv2(graph,_))

        return _.unsqueeze(2)

        

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, data, device):
        adj  = data["graph"][0].to(device)
        flow_x = data["flow_x"].to(device)
        B,N,H,D = flow_x.size()
        x = flow_x.view(B,N,H*D)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x.unsqueeze(2)











def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=32)

    # Loading Model
    # TODO:  Construct the GAT (must) and DCRNN (optional) Model



    
    my_net = GAT(6,6,1,0.6,0.2,8)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Epoch = 100

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

            loss = criterion(predict_value, data["flow_y"])

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))

    # Test Model
    # TODO: Visualize the Prediction Result
    # TODO: Measure the results with metrics MAE, MAPE, and RMSE
    my_net.eval()
    with torch.no_grad():

        MAE_total_loss  = 0.0
        MAPE_total_loss  = 0.0
        RMSE_total_loss  = 0.0

        for data in test_loader:

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]

            mae_loss = MAE_per_batch(predict_value, data["flow_y"])
            mape_loss = MAPE_per_batch(predict_value, data["flow_y"])
            rmse_loss = RMSE_per_batch(predict_value, data["flow_y"])


            MAE_total_loss += mae_loss
            MAPE_total_loss += mape_loss
            RMSE_total_loss += rmse_loss

        print("Test MAE Loss: {:02.4f}".format(1000 * MAE_total_loss / len(test_data)))
        print("Test MAPE Loss: {:02.4f}".format(1000 * MAPE_total_loss / len(test_data)))
        print("Test RMSE Loss: {:02.4f}".format(1000 * RMSE_total_loss / len(test_data)))
    results_v(my_net)

def results_v(my_net):
    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=32)
    
    max_data, min_data = test_data.flow_norm   # [N, T, D] , norm_dim=1, [N, 1, D]
    max_data = LoadData.to_tensor(max_data).unsqueeze(dim=0)
    min_data = LoadData.to_tensor(min_data).unsqueeze(dim=0)  # [1,N, 1, D]



    
    my_net.eval()
    full_pred=[]
    full_truth=[]




    with torch.no_grad():


        for data in test_loader:

            predict_value = my_net(data, "cuda").to("cpu")  # [1, N, 1, D]
            truth_data = data["flow_y"]    # [1, N, 1, D]
            _,N,_,_ = truth_data.size()

            recover_pred_value = LoadData.recover_data(max_data,min_data,predict_value).reshape([N,1])
            recover_truth_value = LoadData.recover_data(max_data,min_data,truth_data).reshape([N,1])


            full_pred.append(recover_pred_value)
            full_truth.append(recover_truth_value)
        
        full_pred= torch.cat(full_pred,dim=1)
        full_truth= torch.cat(full_truth,dim=1)

        np.save("full_pred.npy",full_pred)
        np.save("full_truth.npy",full_truth)





if __name__ == '__main__':
    main()
