"""
Implement:
- sin-cos position embedding too
- function to get position embedding
"""
import math
import torch
import torch.nn as nn
from einops import repeat

from model.ssd.embeddings.base import Embedding


# class PositionEmbedding(Embedding):
#     def __init__(self, embedding_type, n_positions, output_dim):
#         if embedding_type == 'learn_1d':


def get_position_embedding(embedding_type, **kwargs):
    if embedding_type == 'learn_1d':
        return LearnedPositionEmbedding(**kwargs)
    elif embedding_type == 'sinusoid_1d':
        return SinusoidPositionEmbedding(**kwargs)
    else:
        raise NotImplementedError
        

class LearnedPositionEmbedding(Embedding):
    """
    n_positions is abused to be self.input_dim
    """
    def __init__(self, n_positions, output_dim):
        self.n_positions = n_positions
        super().__init__(n_positions, output_dim)
        
    def initialize_layers(self):
        # self.num_positions should be number of patches
        # self.output_dim should be unrolled patch len <- maybe not
        self.layers = nn.Embedding(self.n_positions, self.output_dim)
        
    def forward(self, x):
        # Assume input is shape B x C x N x T x L x D
        b, c, n, t, l, d = x.shape
        # assert n == self.n_positions  # N can be number of patches or number of pixels
        
        # This should be torch.arange(self.n_positions)? 
        x = repeat(torch.arange(n), 'n -> b c n t l', b=b, c=c, t=t, l=l).to(x.device)
        self.layers.to(x.device)
        x = self.layers(x)
        # Output is shape B x N x T x L x D, D is output dim
        return x
    
    
class SinusoidPositionEmbedding(Embedding):
    """
    1D sine-cosine embeddings like in Transformer
    """
    def __init__(self, n_positions, output_dim, max_seq_len=None):
        assert output_dim % 2 == 0
        self.n_positions = n_positions
        self.max_seq_len = max_seq_len
        super().__init__(n_positions, output_dim)
        
    def initialize_layers(self):
        if self.max_seq_len is None:
            pass
        else:
            layers = torch.zeros(self.max_seq_len, self.output_dim)
            position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
            denom = (torch.exp(torch.arange(0, self.output_dim, 2).float() * 
                               -math.log(10000.) / self.output_dim))
            layers[:, 0::2] = torch.sin(position * denom)
            layers[:, 1::2] = torch.cos(position * denom)
                 
            # layers is shape L x D
            self.register_buffer("layers", layers)
        
    def forward(self, x):
        # Assume input is shape B x C x N x T x L
        with torch.no_grad():
            b, c, n, t, l, d = x.shape
            if self.max_seq_len is None:
                self.max_seq_len = n  # n_positions
                self.initialize_layers()  # initialize positional embeddings
            
            # print('sinusoid input.shape:', x.shape)
            # print('sinusoid self.layers.shape:', self.layers.shape)
            p = self.layers[:n]  # N x D
            # print('sinusoid self.layers[n].shape:', p.shape)
            p = repeat(p, 'n d -> b c n t l d', b=b, c=c, t=t, l=l).to(x.device)
            return p