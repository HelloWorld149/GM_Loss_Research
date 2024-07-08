import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class GM_Loss(nn.Module):
    '''
    GM loss, Geometric Mean loss, is the loss function based on the Continuous Geometric Mean loss.
    '''
    def __init__(self, 
                 with_logits,
                 smooth=1):
        self.with_logits = with_logits
        self.smooth = smooth
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input  #if logit, use sigmoid
        
        loss = self._compute_gm_loss_(flat_input, flat_target)
        
        return loss
        
    
    def _compute_gm_loss_(self, input, target):
        soft_tp = torch.sum(target * input)
        soft_fp = torch.sum((1 - target) * input)
        soft_fn = torch.sum(target * (1 - input))
        soft_tn = torch.sum((1 - target) * (1 - input))

        soft_sensitivity = soft_tp / (soft_tp + soft_fn + self.smooth)
        soft_specificity = soft_tn / (soft_tn + soft_fp + self.smooth)
        
        gm = torch.sqrt(soft_sensitivity * soft_specificity)
        loss = 1 - gm
        
        return loss
        

        