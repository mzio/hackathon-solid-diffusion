""" Utility nn components, in particular handling activations, initializations, and normalization layers """

# from functools import partial
import math
# from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract
    
    
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True):
        """ tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        For some reason tie=False is dog slow, prob something wrong with torch.distribution
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            return X * mask * (1.0/(1-self.p))
        return X    

    
def Activation(activation=None, size=None, dim=-1, inplace=False):
    if activation in [None, 'id', 'identity', 'linear' ]:
        return nn.Identity(inplace)
    elif activation == 'tanh':
        return nn.Tanh(inplace)
    elif activation == 'relu':
        return nn.ReLU(inplace)
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU(inplace)
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid(inplace)
    # elif activation == 'modrelu':
    #     return Modrelu(size)
    # elif activation == 'sqrelu':
    #     return SquaredReLU()
    # elif activation == 'ln':
    #     return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

        
def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer
    
    