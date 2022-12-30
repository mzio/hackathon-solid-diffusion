"""
Parent kernel class
"""
from model.ssd import OurModule


class Kernel(OurModule):
    def __init__(self,
                 n_heads,
                 d_kernel,
                 n_channels,
                 skip_connection,
                 closed_loop,
                 train):
        super().__init__()
        
        self.n_heads    = n_heads
        self.d_kernel   = d_kernel
        self.n_channels = n_channels
        
        self.skip_connection = skip_connection
        self.closed_loop     = closed_loop  
        self.requires_grad   = not train
        
        self._fp = (self.n_channels, self.n_heads, self.d_kernel)
        
    def fft_conv(self, u, v):
        L   = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        v_f = torch.fft.rfft(v, n=2*L) # (C H L)

        y_f = oe.contract('bhl,chl->bchl', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)
        return y
    
    def init_weights(self):
        raise NotImplementedError
    
    def forward(self, u):
        raise NotImplementedError