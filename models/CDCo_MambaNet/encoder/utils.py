import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.autograd import Variable

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class LayerNorm_2d(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class GRN_2d(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

def pairwise_l2_distance(embs1, embs2):
    norm1 = torch.sum(embs1**2, dim=1)
    norm1 = norm1.view(-1, 1)
    norm2 = torch.sum(embs2**2, dim=1)
    norm2 = norm2.view(1, -1)
    dist = torch.max(norm1 + norm2 - 2.0 * torch.matmul(embs1, embs2.t()), torch.tensor(0.0))
    return dist
    
def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    channels = embs1.shape[1]
    # Go from embs1 to embs2.
    if similarity_type == 'cosine':
        similarity = torch.matmul(embs1, embs2.t())
    elif similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature
    return similarity

def align_pair_of_sequences(embs1, embs2, similarity_type,
                temperature):
    max_num_steps = embs1.shape[0]

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, dim=1)

    # Calculate soft-nearest neighbors.
    nn_embs = torch.matmul(softmaxed_sim_12, embs2)

    # Find distances between nn_embs and embs1.
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

    logits = sim_21
    labels = torch.eye(max_num_steps)
    # print(labels)
    return logits, labels

def first_align(embs1, embs2, similarity_type,
                temperature):
    max_num_steps = embs1.shape[0]

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    logits = sim_12
    labels = torch.eye(max_num_steps)
    # print(labels)
    return logits, labels
