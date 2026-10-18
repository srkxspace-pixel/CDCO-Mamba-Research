# import torch
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from torch import nn

# MIN_NUM_PATCHES = 16


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim ** -0.5

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, mask=None):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         mask_value = -torch.finfo(dots.dtype).max

#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value=True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, mask_value)
#             del mask

#         attn = dots.softmax(dim=-1)

#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(PreNorm(dim, Attention(dim, heads=heads,
#                                                 dim_head=dim_head, dropout=dropout))),
#                 Residual(PreNorm(dim, FeedForward(
#                     dim, mlp_dim, dropout=dropout)))
#             ]))

#     def forward(self, x, mask=None):
#         for attn, ff in self.layers:
#             x = attn(x, mask=mask)
#             x = ff(x)
#         return x


# class ViT3D(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=32, dropout=0., emb_dropout=0.):
#         super().__init__()
#         assert all([each_dimension % patch_size ==
#                     0 for each_dimension in image_size])
#         num_patches = (image_size[0] // patch_size) * \
#             (image_size[1] // patch_size)*(image_size[2] // patch_size)
#         patch_dim = channels * patch_size ** 3
#         assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
#         assert pool in {
#             'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.patch_size = patch_size

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(
#             dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

#         self.change_mlp = nn.Linear(1729, 64)

#     def forward(self, img, mask=None):
#         p = self.patch_size
#         x = rearrange(
#             img, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=p, p2=p, p3=p)
#         x = self.patch_to_embedding(x)
#         b, n, _ = x.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)
#         x = self.transformer(x, mask)
#         # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#         # x = self.to_latent(x)
#         # return self.mlp_head(x)
#         x = x.permute(0, 2, 1)
#         x = self.change_mlp(x)
#         return x

        


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, DropPath
# from .utils import LayerNorm, GRN
# import math

# class Block(nn.Module):
#     """ ConvNeXtV2 Block.
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#     """
#     def __init__(self, dim, drop_path=0.):
#         super().__init__()
#         self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1)
#          # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         # self.norm = nn.BatchNorm3d(dim, eps=1e-6)
#         hidden_dim = int(4 * dim / 3)
#         self.pwconv1 = nn.Linear(dim, hidden_dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.grn = GRN(hidden_dim)
#         self.pwconv2 = nn.Linear(hidden_dim, dim)
#         self.dwconv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 4, 1)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.grn(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 4, 1, 2, 3)
#         # x = self.dwconv1(x)
#         # x = input + self.drop_path(x)
#         x = self.drop_path(input) + x
#         return x

# class ConvNeXtV2(nn.Module):
#     """ ConvNeXt V2
        
#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#     def __init__(self, in_chans=1, num_classes=2, 
#                  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
#                  drop_path_rate=0.1, head_init_scale=1.
#                  ):
#         super().__init__()
#         self.depths = depths
#         self.downsample_layers = nn.ModuleList()
#         stem = nn.Sequential(
#             nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=1),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
#             nn.MaxPool3d(kernel_size=3, stride=2)

#         )
#         self.downsample_layers.append(stem)
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                     nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=1),
#                     LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
#                     nn.Conv3d(dims[i+1], dims[i+1], kernel_size=3, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)

#         self.stages = nn.ModuleList()
#         dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]

#         # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
#         self.norm = nn.BatchNorm3d(dims[-1], eps=1e-6)
#         # self.head = nn.Linear(dims[-1], num_classes)

#         self.apply(self._init_weights)
#         # self.head.weight.data.mul_(head_init_scale)
#         # self.head.bias.data.mul_(head_init_scale)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv3d):
#             torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         # elif isinstance(m, nn.Linear):
#         #     nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.))
#         #     if m.bias is not None:
#         #         nn.init.constant_(m.bias, 0)  

#     def forward_features(self, x):
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#         # return self.norm(x.mean([-3, -2, -1]))
#         return self.norm(x)

#     def forward(self, x):
#         x = self.forward_features(x)
#         # x = self.head(x)
#         # return self.norm(x)
#         return x

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [16, 32, 64, 128]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model