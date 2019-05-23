import torch
import torch.nn as nn
import torch.nn.functional as F



def getting_pic(predicted_labels, target, criterion):
    cross_entropy_loss = criterion(predicted_labels, target).squeeze()
    cross_entropy_loss = (-1) * cross_entropy_loss
    p_i_c = torch.exp(cross_entropy_loss)
    return p_i_c



