"""
Classes for embedding original inputs into inputs for model
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat


class Embedding(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        """
        Generic class for encoding 
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialize_layers()
        
    def initialize_layers(self):
        self.layers = nn.Identity()
        
    def forward(self, x):
        return self.layers(x)
    
    
class LinearEmbedding(Embedding):
    """
    Fully-connected linear layer encoder
    - Assumes input is size B x ... x input_dim
    - input_dim should be patch_len, output_dim should be patch_len
    """
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 identity_init, 
                 identity_val=1):
        self.identity_init = identity_init
        self.identity_val  = identity_val  # Testing 
        super().__init__(input_dim, output_dim)
        
    def initialize_layers(self):
        self.layers = nn.Linear(self.input_dim, self.output_dim, 
                                bias=True)
        
        if self.identity_init:
            assert self.input_dim == self.output_dim, print(f'input dim ({self.input_dim}) != output dim ({self.output_dim})')
            w = torch.diag(torch.ones(self.input_dim) * self.identity_val)
            b = torch.zeros(self.input_dim)
            with torch.no_grad():
                self.layers.weight.copy_(w)
                self.layers.bias.copy_(b)
                
    def forward(self, x):
        # Assume input is B x C x N x T x L
        return self.layers(x)
    
    
class PositionEmbedding(Embedding):
    """
    n_positions is abused to be self.input_dim
    """
    def __init__(self, n_positions, output_dim):
        self.n_positions = n_positions
        super().__init__(n_positions, output_dim)
        
    def initialize_layers(self):
        # self.num_positions should be number of patches
        # self.output_dim should be unrolled patch len
        self.layers = nn.Embedding(self.n_positions, self.output_dim)
        
    def forward(self, x):
        # Assume input is shape B x C x N x T x L
        b, c, n, t, l = x.shape
        assert n == self.n_positions
        x = repeat(torch.arange(n), 'n -> b c n t', b=b, c=c, t=t)
        return self.layers(x)
    
    
class PatchEmbedding(nn.Module): 
    """
    Convert images  of shape B x C x H x W x T 
    ---> to patches of shape B x C x N x T x L
    where N is number of patches, L is unrolled patch length
    """
    def __init__(self, 
                 patch_size: int, 
                 dilation: int=1, 
                 padding: int=0,
                 stride: int=1):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size,
                                dilation=dilation,
                                padding=padding,
                                stride=stride)
        self.T = False
        
    def forward(self, x):
        # nn.Unfold requires B x C x H x W
        if len(x.shape) == 5:  # shape: B x C x H x W x T
            self.T = True
            b, c, h, w, t = x.shape
            x = rearrange(x, 'b c h w t -> (b t) c h w')
        # Patchifies and unrolls
        x = self.unfold(x)  
        # Take care of dims
        if self.T:
            x = rearrange(x, '(b t) (c l) n -> b c n t l', c=c, t=t)
            b, c, n, t, l = x.shape
        else:
            b, c, n, l = x.shape
        # Helpful for positional embeddings    
        self.n_patches = n  
        self.patch_len = l
        return x
    
    
class ModelEmbedding(nn.Module):
    """
    Model encoder should take care of mapping to different number of heads
    """
    def __init__(self,
                 patch_kwargs:    dict,
                 linear_kwargs:   dict=None,
                 position_kwargs: dict=None,
                 bidirectional:   bool=False,
                 output_dim:      str='bld'):  # HACK: testing
        super().__init__()
        self.patch_kwargs    = patch_kwargs
        self.linear_kwargs   = linear_kwargs
        self.position_kwargs = position_kwargs
        self.bidirectional   = bidirectional
        
        self.patch_embedding = PatchEmbedding(**patch_kwargs)
        
        if linear_kwargs is not None:
            self.linear_embedding = LinearEmbedding(**linear_kwargs)
        
        # Initialize position embedding object after first pass
        self.init_position = False
        
        self.output_dim = output_dim   # HACK: testing
            
    def position_embedding(self, x):
        if self.init_position is False:
            self._position_embedding = PositionEmbedding(
                self.patch_embedding.n_patches,
                self.patch_embedding.patch_len,
            )
            self.init_position = True
        return self._position_embedding(x)
            
    def forward(self, x):  # Assume x is B x C x H x W x T
        """
        Input  is size B x C x H x W x T
        Output is size B x L x D if self.output_dim == 'bld'
        """
        # Patchify images
        x = self.patch_embedding(x)  # B x C x N x T x L
        
        # Apply linear mapping
        if self.linear_kwargs is not None:
            x = self.linear_embedding(x)
        
        # Add positional embeddings
        if self.position_kwargs is not None:
            x += self.position_embedding(x)
            
        if self.bidirectional:
            x_r = torch.flip(x, [4])  # Reverse each unrolled patch
            x   = rearrange([x, x_r], 'r b c n t l -> b c (n r) t l')
            
        if self.output_dim == 'bld':   # HACK: testing
            # Should change to this one 
            # -> saves one less rearrange because encoder takes input B x L x D
            x = rearrange(x, 'b c n t l -> b (t l) (c n)')
            _, self.sample_len, self.sample_dim = x.shape
        else:
            x = rearrange(x, 'b c n t l -> b (c n) (t l)')
            _, self.sample_dim, self.sample_len = x.shape
        return x 
        
            