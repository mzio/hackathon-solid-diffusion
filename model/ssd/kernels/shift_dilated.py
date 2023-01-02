import torch
import opt_einsum as oe

from einops import repeat, rearrange

from model.ssd.kernels.shift import ShiftKernel
from model.functional.krylov import krylov


class DilatedShiftKernel(ShiftKernel):
    def __init__(self, dilations: list, **kwargs):        
        assert len(dilations) == 1, print('Sorry only single dilation support now.')
        self.dilations = dilations  
        self.dilation = max(self.dilations)
        
        # Length of [1, 0 * dilation, 2, 0 * dilation, ..., d_kernel]
        self._fp_c = (kwargs['n_channels'], kwargs['n_heads'], kwargs['d_kernel'])
        super().__init__(**kwargs)
        
    def init_weights(self):
        self.d_kernel = self.d_kernel * (self.dilation + 1) - self.dilation
        self._fp = (self.n_channels, self.n_heads, self.d_kernel)
        
        # A matrix ix fixed
        a = repeat(torch.zeros(self.d_kernel).float(), 'd -> c h d',
                   c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("a", a, trainable=False, lr=None, wd=None)
        
        # B matrix is fixed
        b    =  torch.zeros(self.d_kernel).float()
        b[0] = 1.
        b    = repeat(b, 'd -> c h d', 
                      c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C, K, D matrices are trainable by default
        trainable = True if self.requires_grad is True else False
        
        c = torch.randn(*self._fp_c)
        z = torch.zeros(c.shape).type(c.type())
        # Expand out c
        c = rearrange([c] + self.dilation * [z], 
                      'r c h d -> c h (d r)')[..., :-self.dilation]
        self.register("c", c, trainable, lr=None, wd=None)
        
        if self.closed_loop:
            k = torch.randn(*self._fp_c)
            k = rearrange([k] + self.dilation * [z], 
                          'r c h d -> c h (d r)')[..., :-self.dilation]
            self.register("k", k, trainable, lr=None, wd=None)
        
        # D matrix (skip connection) is trainable
        d = torch.randn(self.n_channels, self.n_heads)
        self.register("d", d, trainable, lr=None, wd=None)
        
    def fft_conv(self, u, v):
        # Same as before, but handles convolved kernel lengths 
        # possibly being different than input length
        L   = u.shape[-1]
        Lv  = v.shape[-1]
        
        if Lv > L:
            v = v[:, :, :L]
        elif Lv == L:
            pass
        else:
            v = F.pad(v, pad=(0, L - Lv, 0, 0), value=0)
            
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        v_f = torch.fft.rfft(v, n=2*L) # (C H L)

        y_f = oe.contract('b h l, c h l -> b c h l', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)
        return y