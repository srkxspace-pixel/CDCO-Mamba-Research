import torch.nn as nn
import math
from omegaconf import DictConfig
import os
import torch.nn.functional as F
from .encoder.mamba_fuse import Mamba_fusion

class CDCo_MambaNet(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.fuse_encoder = Mamba_fusion(config, config.model.fuse_channel, config.model.fuse_shape)
        self.contrastive_loss = None
        self.config = config
        self.rate1 = config.model.rate1
        self.rate2 = config.model.rate2
        self.ct_loss = config.model.ct_loss
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, fmri_feature, fmri_con, smri_feature):
        cls, cls1, contrastive_loss = self.fuse_encoder(smri_feature, fmri_feature, fmri_con)
        self.contrastive_loss = contrastive_loss
        return cls, cls1
    
    def loss_function(self, pred1, label, pred2):
        label = label.float()
        
        l1 = F.cross_entropy(pred1, label)
        l2 = F.cross_entropy(pred2, label)

        pred1 = pred1 * self.rate2
        pred2 = pred2 * self.rate2
        
        l3 = F.cross_entropy(pred1, F.softmax(pred2, dim=-1))
        l4 = F.cross_entropy(pred2, F.softmax(pred1, dim=-1))

        return self.rate1*(l1+l2)+(1-self.rate1)*(l3+l4)+self.contrastive_loss*self.ct_loss

