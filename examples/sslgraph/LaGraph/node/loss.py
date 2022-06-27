import torch
import torch.nn as nn
import numpy as np


class LaGraphLoss(torch.nn.Module):
    def __init__(self):
        super(LaGraphLoss, self).__init__()

    def forward(self, embed_orig, rep_orig, dec_orig, rep_aug, mask, alpha):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # mmode == 'whole':
        num_nodes = embed_orig.size()[0]
        m = mask.to(device)

        loss_rec = nn.MSELoss(reduction='mean')(dec_orig, embed_orig[0])
        loss_inv = nn.MSELoss(reduction='none')(rep_orig, rep_aug)
        loss_inv_masked = torch.sqrt(torch.sum(torch.mean(loss_inv, dim=1) * m) / torch.clamp(torch.sum(m), min=1))

#         # mmode == 'partial':
#         m = mask.to(device)
#         loss_rec = nn.MSELoss(reduction='mean')(dec_orig, embed_orig[0])
#         loss_inv = nn.MSELoss(reduction='none')(rep_orig, rep_aug)
#         loss_inv_masked = torch.sqrt(torch.sum(loss_inv * m) / torch.clamp(torch.sum(m), min=1))
        
        loss = loss_rec + alpha * loss_inv_masked
        return loss, loss_rec, loss_inv_masked



