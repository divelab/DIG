import torch
import torch.nn as nn
import numpy as np


class LaGraphLoss(torch.nn.Module):
    def __init__(self):
        super(LaGraphLoss, self).__init__()

    def forward(self, embed_orig, rep_orig, dec_orig, rep_aug, mask, loss, alpha, num_features):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_nodes = embed_orig.size()[0]
        m = mask.to(device)
        if loss == 'mse':
            loss_rec = nn.MSELoss(reduction='mean')(dec_orig, embed_orig)
            loss_inv_masked = torch.sqrt(nn.MSELoss(reduction='mean')(rep_orig, rep_aug))

        elif loss == 'ce':
            loss_rec = nn.CrossEntropyLoss(reduction='mean')(dec_orig, torch.argmax(embed_orig, axis=1))
            loss_inv_masked = torch.sqrt(nn.MSELoss(reduction='mean')(rep_orig, rep_aug))
        
        elif loss == 'ce2':
            if num_features < 2:
                num_features = 2
            num_features = torch.tensor(num_features).to(device)
            loss_rec = nn.CrossEntropyLoss(reduction='mean')(dec_orig, torch.argmax(embed_orig, axis=1))
            loss_inv = nn.KLDivLoss(reduction='none')(rep_orig, rep_aug)
            loss_inv_masked = torch.sqrt(2 * (num_features-1) / num_features) * torch.sum(torch.mean(loss_inv, dim=1) * m) / torch.clamp(torch.sum(m), min=1)

        
        loss = loss_rec + alpha * loss_inv_masked
        
        return loss, loss_rec, loss_inv_masked



