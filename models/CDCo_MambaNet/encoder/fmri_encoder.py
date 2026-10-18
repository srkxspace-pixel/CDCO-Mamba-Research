import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm_2d, GRN_2d
import math

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = LayerNorm_2d(dim, eps=1e-6)
        hidden_dim = int(4 * dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN_2d(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.ssm = MambaVisionMixer_Cls(d_model=hidden_dim, d_state=8, d_conv=3, expand=4)
        # self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x) + input
        return x

class ConvNeXtV2_Mamba(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0.1, head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm_2d(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=1, padding=1),
                LayerNorm_2d(dims[i+1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i+1], dims[i+1], kernel_size=2, stride=2),  
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norm = nn.BatchNorm2d(dims[-2], eps=1e-6)
        # self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(x.mean([-2, -1]))
        # return self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        # return self.norm(x)
        return x
    
