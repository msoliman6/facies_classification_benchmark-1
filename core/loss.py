import torch
import torch.nn.functional as F
import numpy as np

torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)

def focal_loss2d(input, target, alpha=None , gamma=None, size_average=False,ignore_index=0):
    N,c,w,h=target.shape
    one_hot = torch.cuda.FloatTensor(input.shape).zero_()
    target = one_hot.scatter_(1, target.data, 1)#create one hot representation for the target

    logpt = F.log_softmax(input,dim=1) #log softmax of network output (softmax across first dimension)
    pt = logpt.exp() #Confidence of the network in the output
   #confidence values
    if gamma == None:
        raise ValueError("Gamma Not Defined")


    # alpha is used for class imbalance:
    if alpha is not None:
        at = torch.cuda.FloatTensor(pt.shape).fill_(1) * alpha.float().view(1, -1, 1, 1)
    else:
        at = torch.cuda.FloatTensor(pt.shape).fill_(1)


    loss = -1 * (1-pt)**gamma * at*target*logpt


    if size_average:
        return loss.mean()
    else:
        return loss.sum()