import torch
import opt_einsum as oe

from einops import repeat, rearrange

from model.ssd.kernels.companion import CompanionKernel
from model.functional.krylov import krylov


class ShiftKernel(CompanionKernel):
    def __init__(self, n_hidden_state, **kwargs):
        self.n_hidden_state = n_hidden_state
        super().__init__(**kwargs)
        
    def init_weights(self):
        # A matrix ix fixed
        a = repeat(torch.zeros(self.d_kernel).float(), 'd -> c h d',
                   c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("a", a, trainable=False, lr=None, wd=None)
        self.A = None
        
        # B matrix is fixed
        b    =  torch.zeros(self.d_kernel).float()
        b[0] = 1.
        b    = repeat(b, 'd -> c h d', 
                      c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C, K, D matrices are trainable by default
        trainable = True if self.requires_grad is True else False
        
        c = torch.randn(*self._fp)
        self.register("c", c, trainable, lr=None, wd=None)
        
        if self.closed_loop:
            k = torch.randn(*self._fp)
            self.register("k", k, trainable, lr=None, wd=None)
        
        # D matrix (skip connection) is trainable
        d = torch.randn(self.n_channels, self.n_heads)
        self.register("d", d, trainable, lr=None, wd=None)
        
    def matrix_power(self, l, c, b, a):
        ch, h, d = b.shape 
        # Construct companion matrix if needed
        if self.A is none:
            A = self.shift_matrix.to(a.device) + oe.contract(
                'c h i, c h j -> c h j i',  self.a_padding.to(a.device), a
            )
            self.A = A
        # Use repeated squares
        g = krylov(l, self.A, b, c)  # Need to import
        return g
        
    def get_hidden_state(self, l, u):
        # u.shape is C x H x D
        # a = self.norm(self.a, ord=self.norm_ord)
        k = self.matrix_power(l, None, self.b, self.a) # C x H x D x L
        x = self.fft_conv_d(u, k)  # B x C x H x L x D
        return x
        
    def get_kernel(self, u, c=None, l=None):
        l = u.size(-1) if l is None else l
        c = self.c if c is None else c
        k = self.matrix_power(l, c, self.b, self.a).to(u.device)
        return k
    
    def convolve(self, u, A, l):
        f = krylov(l, A, self.b, self.c)
        y = self.fft_conv(u, f)
        y = rearrange(y, 'b c h l -> b (c h) l')
        return y
    
    def recurse_hidden_state(self, u, A, l, n_hidden_state):
        # Compute hidden state
        f = krylov(n_hidden_state, A, self.b, None)
        x  = self.fft_conv_d(u, f)
        BK = oe.contract('c h i, c h j -> c h i j', self.b, self.k)
        x  = krylov(L=l - n_hidden_state,  
                    A=A + BK, 
                    b=x[:, :, :, n_hidden_state - 1, :], 
                    c=None)
        return x
    
    def forward(self, u):
        ch, h, d = self.b.shape
        b , d, l = u.shape
        
        # Outputs
        y, y_, u_ = None, None, None
        
        # Construct companion matrix
        a = self.a
        A = self.shift_matrix.to(a.device) + oe.contract(
            'c h i, c h j -> c h j i',  self.a_padding.to(a.device), a
        )
        
        if self.closed_loop:  
            # Can also compute from just first hidden_state = B * e, e ~ N(0, I)
            # - Then also only initialize the A + BK:
            #   - Once for each sample during training
            #   - Once for all samples during test  (Faster)
            x = self.recurse_hidden_state(u, A, l, self.n_hidden_state)
            y = torch.einsum('...nl, ...n -> ...l', x, self.c).contiguous()
            y = rearrange(y, 'b c h l -> b (c h) l')
            
            # During training, compute closed-loop outputs and open-loop outputs
            if not self.inference_only:
                u_ = torch.einsum('...nl, ...n -> ...l', x, self.k).contiguous()
                u_ = rearrange(u_, 'b c h l -> b l (c h)')  # Save 1 extra rearrange
                # Compute open-loop
                y_ = self.convolve(u, A, l)
        else:
            y = self.convolve(u, A, l)
            
        # Return:
        # 1. Closed-loop output prediction (if closed-loop, else open-loop)
        # 2. 2nd-to-last layer prediction
        # 3. Open-loop output prediction
        return y, u_, y_
            