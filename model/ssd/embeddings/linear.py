import torch
import torch.nn as nn
from einops import repeat

from model.ssd.embeddings.base import Embedding


class LinearEmbedding(Embedding):
    """
    Fully-connected linear layer encoder
    - Assumes input is size B x ... x input_dim
    - input_dim should be patch_len, output_dim should be patch_len
    """
    def __init__(self, 
                 input_dim, 
                 output_dim,  # tuple? 
                 identity_init, 
                 identity_val=1,
                 repeat=True):
        self.identity_init = identity_init
        self.identity_val  = identity_val  # Testing 
        self.repeat        = repeat
        super().__init__(input_dim, output_dim)
        
    def _initialize_1d_layers(self, output_dim):
        self.layers = nn.Linear(self.input_dim, output_dim, bias=True)
        if self.identity_init:
            assert self.input_dim == output_dim, print(f'input dim ({self.input_dim}) != output dim ({output_dim})')
            w = torch.diag(torch.ones(self.input_dim) * self.identity_val)
            b = torch.zeros(self.input_dim)
            with torch.no_grad():
                self.layers.weight.copy_(w)
                self.layers.bias.copy_(b)
         
    def initialize_layers(self):
        if len(self.output_dim) == 1:
            self.repeat_dim = 1
            self._initialize_1d_layers(output_dim=self.output_dim[0])
        else:
            self.repeat_dim = self.output_dim[1]
            self._initialize_1d_layers(output_dim=self.output_dim[0])
                
    def forward(self, x):
        # Assume input is B x C x N x T x L
        x = self.layers(x)
        # if len(self.output_dim) > 1:
        #     # assume repeat is true
        #     x = repeat(x, 'b c n t l -> b c n t l d', d=self.repeat_dim)  # may break
        # else:
        #     x = x.unsqueeze(-1)
        x = repeat(x, 'b c n t l -> b c n t l d', d=self.repeat_dim)  # may break
        return x