import torch.nn as nn
from einops import rearrange


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
        self.dilation   = dilation
        self.padding    = padding
        self.stride     = stride
        self.unfold     = nn.Unfold(kernel_size=patch_size,
                                    dilation=dilation,
                                    padding=padding,
                                    stride=stride)
        self.fold   = None  # Initialize in first forward pass
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
        # Helpful for positional embeddings and folding 
        self.n_channels = c
        self.patch_len  = l
        self.n_patches  = n  
        
        if self.fold is None:
            self.fold = nn.Fold(output_size=(h, w), 
                                kernel_size=self.patch_size,
                                dilation=self.dilation,
                                padding=self.padding,
                                stride=self.stride)
        return x