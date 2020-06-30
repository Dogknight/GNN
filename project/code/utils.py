import torch

def MAE_per_batch(y_pred,y_truth):
    """
        y_pred :[B,N,1,D]
            
        y_truth :[B,N,1,D]
                    
    """
    nodes_loss = torch.mean(torch.abs(y_pred-y_truth),dim=1)
    per_batch_loss = torch.mean(node_loss,dim = 0).item()
    
    return per_batch_loss


def MAPE_per_batch(y_pred,y_truth):
     """
     y_pred :[B,N,1,D]
     y_truth :[B,N,1,D]
     """
                                
    nodes_loss = torch.mean(torch.abs(y_pred-y_truth)/torch.abs(y_pred+ 1e-8),dim=1)
    per_batch_loss = torch.mean(node_loss,dim = 0).item()
    return per_batch_loss

                                           

def RMSE_per_batch(y_pred,y_truth):
    """
        y_pred :[B,N,1,D]
                
        y_truth :[B,N,1,D]
                    
    """
                                
    nodes_loss = torch.mean(torch.abs(y_pred-y_truth) ** 2.0,dim=1)
    per_batch_loss = torch.mean(node_loss ** 0.5,dim = 0).item()
    return per_batch_loss

                                            
