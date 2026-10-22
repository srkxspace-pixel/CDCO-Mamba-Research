import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from torch.utils.checkpoint import checkpoint
from .smri_encoder import generate_model
from .fmri_encoder import ConvNeXtV2_Mamba
from ..patch_ssm.mamba_vision import MambaVisionMixer_Fuse, MambaVisionMixer_Cls
from .reliability import ReliabilityGate

def checkpoint_blocks(blocks):
    """Recompute each block's activations in backward instead of storing them (same fp32 grads).

    Recomputation runs BatchNorm forward twice per step, so running stats would update twice;
    momentum m' = 1 - sqrt(1 - m) makes the two updates equal one update with the original m.
    State-dict keys are unchanged. DropPath masks are replayed (checkpoint preserves RNG state).
    """
    for blk in blocks:
        fwd = blk.forward
        blk.forward = lambda x, fwd=fwd, blk=blk: (
            checkpoint(fwd, x, use_reentrant=False) if blk.training and torch.is_grad_enabled() else fwd(x))
        for bn in blk.modules():
            if isinstance(bn, nn.modules.batchnorm._BatchNorm):
                bn.momentum = 1 - (1 - bn.momentum) ** 0.5


class Mamba_fusion(nn.Module):
    def __init__(self, config, in_features, in_features_ssm): 
        super().__init__()
        self.in_features = in_features
        self.cls_token_s = nn.Parameter(torch.zeros(1, 1, in_features_ssm))
        self.cls_token_f = nn.Parameter(torch.zeros(1, 1, in_features_ssm))
        # Upstream switched between these by commenting lines; pick <dataset>_fmri_feature_shape instead.
        self.fmri_change = nn.Linear(config.model[f"{config.dataset.name}_fmri_feature_shape"], in_features_ssm)
        self.smri_change = nn.Linear(config.model.smri_feature_shape, in_features_ssm)
        self.fmri_encoder = ConvNeXtV2_Mamba(depths=config.model.fmri_encoder_deepth, dims=config.model.fmri_encoder_dims)
        self.smri_encoder = generate_model(50)

        self.mamba_r2 = MambaVisionMixer_Fuse(d_model=in_features_ssm,
                                            d_state=config.model.d_state,
                                            d_conv=config.model.d_conv,
                                            expand=config.model.expand,
                                            output_style='two')
        self.mamba_r1 = MambaVisionMixer_Fuse(d_model=in_features_ssm,
                                            d_state=config.model.d_state,
                                            d_conv=config.model.d_conv,
                                            expand=config.model.expand,
                                            output_style='two')
        self.mamba_r3 = MambaVisionMixer_Cls(d_model=in_features_ssm,
                                            d_state=config.model.d_state,
                                            d_conv=config.model.d_conv,
                                            expand=config.model.expand)
        self.predictor = MLPHead(in_features_ssm, in_features_ssm, in_features)
        self.sl1 = ContrastiveLossModel()
        self.cls_linear = nn.Linear(in_features_ssm*2, config.model.cls_num)
        self.mlp_cls = Mlp_cls(in_features=in_features_ssm,act_layer=nn.GELU,cls_num=config.model.cls_num,cls_in=in_features)
        raf, dfd = config.model.get('reliability_fusion', False), config.model.get('dynamic_dim', False)
        self.gate = ReliabilityGate(in_features_ssm, config.model.cls_num, raf=raf, dfd=dfd,
                                    min_dim=config.model.get('min_dim', 12),
                                    nested_dropout=config.model.get('nested_dropout', 0.3)) if raf or dfd else None
        self.apply(self._init_weights)
        if config.model.get('grad_checkpoint', False):
            blocks = [b for layer in (self.smri_encoder.layer1, self.smri_encoder.layer2,
                                      self.smri_encoder.layer3, self.smri_encoder.layer4) for b in layer]
            checkpoint_blocks(blocks + [b for stage in self.fmri_encoder.stages for b in stage])

    def _init_weights(self, m):
        trunc_normal_(self.cls_token_s, std=.02)
        trunc_normal_(self.cls_token_f, std=.02)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.))
            if  m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, smri, fmri, fmri_con, avail=None):
        """avail: (b, 2) bool, which of (sMRI, fMRI) each patient has; None = both."""
        sMRI_f = self.smri_encoder(smri)
        fMRI_f = self.fmri_encoder(fmri.unsqueeze(1))

        b, c, h, w, z = sMRI_f.shape
        sMRI_f = self.smri_change(sMRI_f.view(b,c,-1))
        fMRI_f = self.fmri_change(fMRI_f.view(b,c,-1))
        if self.gate is not None:
            if avail is None:
                avail = torch.ones(b, 2, dtype=torch.bool, device=sMRI_f.device)
            sMRI_f, fMRI_f = self.gate([sMRI_f, fMRI_f], avail)

        cls_token_s = self.cls_token_s.expand(b, -1, -1)
        sMRI_c = torch.cat((cls_token_s, sMRI_f), dim=1)
        cls_token_f = self.cls_token_f.expand(b, -1, -1)
        fMRI_c = torch.cat((cls_token_f, fMRI_f), dim=1)

        fuse_out, r_loss = self.fuse_method(fMRI_c, sMRI_c)
        
        fmri_age = fuse_out[:, 0, :]
        smri_age = fuse_out[:, c+1, :]
        age_out_linear = torch.cat((fmri_age, smri_age), dim=-1)
        age_out = self.cls_linear(age_out_linear)
        
        mlp_out, r_loss1 = self.fuse_method1(fMRI_f, sMRI_f)
        mlp_out = self.mlp_cls(mlp_out)

        return age_out, mlp_out, (r_loss+r_loss1)
    

    def fuse_method(self, fmri, smri):
        fuse_feature = torch.cat((smri, fmri), dim=1)
        fuse_feature = self.mamba_r3(fuse_feature)

        smri_y, smri_z = self.mamba_r1(smri)
        fmri_y, fmri_z = self.mamba_r2(fmri)
        fmri_fuse = (fmri_y * smri_z)
        smri_fuse = (smri_y * fmri_z)
        sf_feature_o = torch.cat((fmri_fuse, smri_fuse), dim=1)

        # 对比扰动不考虑cls_token
        fmri_cls_token = fmri[:,0,:].unsqueeze(1)
        fmri_feature = fmri[:,1:,:]
        random_noise_f = torch.rand_like(fmri_feature)
        fmri_feature = fmri_feature + torch.sign(fmri_feature) * F.normalize(random_noise_f)
        fmri_rc = torch.cat((fmri_cls_token, fmri_feature), dim=1)

        smri_cls_token = smri[:,0,:].unsqueeze(1)
        smri_feature = smri[:,1:,:]
        random_noise_s = torch.rand_like(smri_feature)
        smri_feature = smri_feature + torch.sign(smri_feature) * F.normalize(random_noise_s)
        smri_rc = torch.cat((smri_cls_token, smri_feature), dim=1)

        smri_y, smri_z = self.mamba_r1(smri_rc)
        fmri_y, fmri_z = self.mamba_r2(fmri_rc)
        fmri_fuse = (fmri_y * smri_z)
        smri_fuse = (smri_y * fmri_z)
        sf_feature = torch.cat((fmri_fuse, smri_fuse), dim=1)

        # 对比扰动添加cls_token
        # random_noise_f = torch.rand_like(fmri).cuda()
        # fmri_r = fmri + torch.sign(fmri) * F.normalize(random_noise_f)
        # random_noise_s = torch.rand_like(smri).cuda()
        # smri_r = smri + torch.sign(smri) * F.normalize(random_noise_s)

        # smri_y, smri_z = self.mamba_r1(smri_r)
        # fmri_y, fmri_z = self.mamba_r2(fmri_r)
        # fmri_fuse = (fmri_y * smri_z)
        # smri_fuse = (smri_y * fmri_z)
        # sf_feature = torch.cat((fmri_fuse, smri_fuse), dim=1)

        r_loss = self.sl1.sf_loss(sf_feature_o, sf_feature)
        # r_loss = 0
        fuse = self.predictor(sf_feature_o + fuse_feature)

        return fuse, r_loss

    def fuse_method1(self, fmri, smri):
        fuse_feature = torch.cat((smri, fmri), dim=1)
        fuse_feature = self.mamba_r3(fuse_feature)

        smri_y, smri_z = self.mamba_r1(smri)
        fmri_y, fmri_z = self.mamba_r2(fmri)
        fmri_fuse = (fmri_y * smri_z)
        smri_fuse = (smri_y * fmri_z)
        sf_feature_o = torch.cat((fmri_fuse, smri_fuse), dim=1)

        random_noise_f = torch.rand_like(fmri)
        fmri = fmri + torch.sign(fmri) * F.normalize(random_noise_f)
        random_noise_s = torch.rand_like(smri)
        smri = smri + torch.sign(smri) * F.normalize(random_noise_s)
        smri_y, smri_z = self.mamba_r1(smri)
        fmri_y, fmri_z = self.mamba_r2(fmri)
        fmri_fuse = (fmri_y * smri_z)
        smri_fuse = (smri_y * fmri_z)
        sf_feature = torch.cat((fmri_fuse, smri_fuse), dim=1)
        r_loss = self.sl1.sf_loss(sf_feature_o, sf_feature)
        # r_loss = 0
        fuse = self.predictor(sf_feature + fuse_feature)

        return fuse, r_loss

class ContrastiveLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.1)
    
    def contrastive_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels)

    def clip_loss(self, similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity, labels)
        return caption_loss

    def sf_loss(self, smri, fmri):
        b, x, y = smri.shape
        smri_embeds = smri.reshape(b, -1)
        fmri_embeds = fmri.reshape(b, -1)
        # 正则化嵌入
        smri_embeds = F.normalize(smri_embeds)
        fmri_embeds = F.normalize(fmri_embeds)
        # 计算余弦相似度作为 logits
        logit_scale = self.logit_scale.exp()
        logits_per_smri = torch.matmul(smri_embeds, fmri_embeds.t()) * logit_scale

        # 动态负样本挖掘：获取每个样本的 hardest negative sample
        # hard_negatives = torch.topk(logits_per_smri, k=1, dim=1, largest=True, sorted=True).indices
        # for i in range(b):
        #     logits_per_smri[i, hard_negatives[i]] *= 0.9  # 减小 hardest negative 的相似度权重
        # 将 one-hot 标签转换为 dense 格式
        # dense_labels = torch.argmax(labels, dim=1)
        dense_labels = torch.arange(b).to(logits_per_smri.device)

        # 计算对比损失
        loss = self.clip_loss(logits_per_smri, dense_labels)
        return loss

class Mlp_cls(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0.5, cls_num=4, cls_in=None):
        super().__init__()
        out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.cls = nn.Linear(cls_in*2, cls_num)
        self.norm = nn.LayerNorm(cls_in*2)

    def forward(self, x):
        x = self.fc1(x).squeeze(-1)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        cls = self.cls(x)
        return cls


class MLPHead(nn.Module):
    def __init__(self, in_channels, projection_size, channel):
        super().__init__()
        in_channels = int(4 * in_channels // 3)
        mlp_hidden_size = in_channels * 2

        self.l1 = nn.Linear(in_channels, mlp_hidden_size)
        self.bn1 = nn.LayerNorm(mlp_hidden_size)
        self.act1 = nn.GELU()
        self.ln2 = nn.Linear(mlp_hidden_size, in_channels)
        self.bn2 = nn.LayerNorm(in_channels)
        self.act2 = nn.GELU()
        self.ln3 = nn.Linear(in_channels, projection_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.ln2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.ln3(x)
        return x
