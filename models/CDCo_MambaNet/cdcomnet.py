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
        self.distill_detach = config.model.get('distill_detach', False)
        self.edl_weight = config.model.get('edl_weight', 0.5)
        self.edl_anneal_epochs = config.model.get('edl_anneal_epochs', 10)
        self.epoch = 0  # set by the trainer; anneals the evidential KL term
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, fmri_feature, fmri_con, smri_feature, avail=None):
        cls, cls1, contrastive_loss = self.fuse_encoder(smri_feature, fmri_feature, fmri_con, avail)
        self.contrastive_loss = contrastive_loss
        return cls, cls1
    
    def loss_function(self, pred1, label, pred2):
        label = label.float()
        
        l1 = F.cross_entropy(pred1, label)
        l2 = F.cross_entropy(pred2, label)

        pred1 = pred1 * self.rate2
        pred2 = pred2 * self.rate2
        
        t2, t1 = F.softmax(pred2, dim=-1), F.softmax(pred1, dim=-1)
        if self.distill_detach:  # peer prediction as a fixed target; upstream's open gradient collapses both heads
            t1, t2 = t1.detach(), t2.detach()
        l3 = F.cross_entropy(pred1, t2)
        l4 = F.cross_entropy(pred2, t1)

        loss = self.rate1*(l1+l2)+(1-self.rate1)*(l3+l4)+self.contrastive_loss*self.ct_loss
        gate = self.fuse_encoder.gate
        if gate is not None:
            loss = loss + self.edl_weight * gate.loss(label, min(1.0, self.epoch / max(1, self.edl_anneal_epochs)))
        return loss

