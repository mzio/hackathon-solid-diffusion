"""
Embedding class for the SSD network
"""
import torch.nn as nn
from einops import rearrange

from model.ssd.embeddings import LinearEmbedding, PatchEmbedding
from model.ssd.embeddings.position import get_position_embedding


class ModelEmbedding(nn.Module):
    """
    Model encoder should take care of mapping to different number of heads
    """
    def __init__(self,
                 patch_kwargs:    dict,
                 linear_kwargs:   dict=None,
                 position_kwargs: dict=None,
                 bidirectional:   bool=False,
                 output_shape:    str='bld'):  # HACK: testing
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
        
        self.output_shape = output_shape   # HACK: testing
            
    def get_position_embedding(self, x):
        if self.init_position is False:
            if self.position_kwargs['n_positions'] is None:
                n_positions = self.patch_embedding.n_patches
                embedding_dim = self.patch_embedding.patch_len
            else:
                n_positions = self.position_kwargs['n_positions']
                embedding_dim = self.position_kwargs['embedding_dim']
                
            kwargs = {'n_positions': n_positions,
                      'output_dim': embedding_dim}
            for k, v in self.position_kwargs['kwargs']:
                kwargs[k] = v
            self.position_embedding = get_position_embedding(self.position_kwargs['type'], **kwargs)
            # self.position_embedding = PositionEmbedding(n_positions, embedding_dim)
            self.init_position = True
        return self.position_embedding(x)
            
    def forward(self, x):  # Assume x is B x C x H x W x T
        """
        Input  is size B x C x H x W x T
        Output is size B x L x D if self.output_shape == 'bld'
        """
        # Patchify images
        x = self.patch_embedding(x)  # B x C x N x T x L
        
        # print('x.shape after patch embedding:', x.shape)
        
        # Apply linear mapping
        if self.linear_kwargs is not None:
            x = self.linear_embedding(x)
            if self.output_shape == 'bt(cnl)':  # Concatenate and use 1D positions
                x = rearrange(x, 'b c n t l d -> b c (n l) t 1 d')
                # print('x.shape after linear embedding:', x.shape)
        else:
            x = x.unsqueeze(-1)
        # x is now shape B x C x N x T x L x D, D := embedding dim
        
        # Add positional embeddings
        if self.position_kwargs is not None:
            # Only works if position embedding is same shape as linear embedding or pos embedding_dim is 1
            p = self.get_position_embedding(x)
            # print('p.shape:', p.shape)
            x += p
            
        x = rearrange(x, 'b c n t l d -> b c (n d) t l')  # Concatenate positional information together
            
        if self.bidirectional:
            x_r = torch.flip(x, [4])  # Reverse each unrolled patch
            x   = rearrange([x, x_r], 'r b c n t l -> b c (n r) t l')
            
        if self.output_shape == 'bt(cnl)':
            x = rearrange(x, 'b c n t l -> b t (c n l)')
            _, self.sample_len, self.sample_dim = x.shape
            
        elif self.output_shape == 'bld':   # HACK: testing
            # Should change to this one 
            # -> saves one less rearrange because encoder takes input B x L x D
            x = rearrange(x, 'b c n t l -> b (t l) (c n)')
            _, self.sample_len, self.sample_dim = x.shape
        else:
            x = rearrange(x, 'b c n t l -> b (c n) (t l)')
            _, self.sample_dim, self.sample_len = x.shape
        return x 
    
    def unpatch(self, x):
        # Assume x is shape: B x T x (C N L)
        b, t, d = x.shape
        # assert self.patch_embedding.fold is not None
        x = rearrange(x, 'b t (c n l) -> (b t) (c l) n',
                      c=self.patch_embedding.n_channels,
                      l=self.patch_embedding.patch_len,
                      n=self.patch_embedding.n_patches)
        x = self.patch_embedding.fold(x)  # 
        x = rearrange(x, '(b t) c h w -> b c h w t', b=b, t=t)
        return x
    
    def patch(self, x):
        # assume x is shape: B x C x H x W x T
        x = self.patch_embedding(x)  # B x C x N x T x L
        if self.output_shape == 'bt(cnl)':
            x = rearrange(x, 'b c n t l -> b t (c n l)')
        elif self.output_shape == 'bld':   
            x = rearrange(x, 'b c n t l -> b (t l) (c n)')
        else:
            x = rearrange(x, 'b c n t l -> b (c n) (t l)')
        return x