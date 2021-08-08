import numpy as np
import torch.nn as nn
import torch

class TripletLoss(nn.Module):
    def __init__(self, alpha = 0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, anchor, positive, negative):
        alpha = self.alpha
        pos_dist = torch.pow(anchor - positive, 2).sum(dim=1)
        neg_dist = torch.pow(anchor - negative, 2).sum(dim=1)
        base_loss = pos_dist - neg_dist + alpha
        loss = torch.clamp(base_loss, min=0).sum()
        return loss



def triplet_loss(anchor, positive, negative, alpha=0.2):
    TL = TripletLoss(alpha)
    return TL(anchor, positive, negative)