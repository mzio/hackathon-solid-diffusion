import torch.nn as nn

from model.ssd.encoders import *
from model.ssd.kernels import *


"""
from model.ssd.kernels import *
from model.ssd.encoders import *
"""


class SSDLayer(nn.Module):
    def __init__(self, 
                 kernel: dict,
                 decoder: dict, 
                 skip_connection: bool,
                 closed_loop: bool):
        super().__init__()
        self.kernel_args     = kernel
        self.decoder_args    = decoder
        self.skip_connection = skip_connection
        self.closed_loop     = closed_loop
        
        self.kernel  = self.get_kernel()
        self.decoder = self.get_decoder()
        
        self.inference_only = False
    
    def get_kernel(self):
        if self.kernel_args['type'] == 'companion':
            kernel_class = CompanionKernel
        elif self.kernel_args['type'] == 'shift':
            kernel_class = ShiftKernel
        elif self.kernel_args['type'] in ['dilated_shift', 'shift_dilated']:
            kernel_class = DilatedShiftKernel
        # elif self.kernel_args['type'] in ['simple', 
        #                                   'diagonal_continuous']:
        #     kernel_class = ContinuousDiagonalKernel  
        # elif self.kernel_args['type'] in ['simple_discrete', 
        #                                   'diagonal_discrete']:
        #     kernel_class = DiscreteDiagonalKernel
        else:
            # TODO: implement more kernels
            raise NotImplementedError(f'Error: {kernel_name} not implemented')
        return kernel_class(**self.kernel_args['kwargs'])
    
    def get_decoder(self):
        return get_decoder(self.decoder_args)      
        
    def forward(self, u, y_=None, u_=None):  # 
        # Assume input shape is (B, L, H)
        y     = rearrange(u, 'b l h -> b h l')
        y, *z = self.kernel(y)  # could output multiple, so should modify this
        y     = rearrange(y, 'b h l -> b l h')
        y     = self.decoder(x)
        if self.skip_connection:
            y = y + u
            
        if self.closed_loop and not self.inference_only:  # if len(z) == 2 and z[1] is not None:
            # Hacky, output open-loop prediction
            # z[0] will be the closed-loop prediction for the next input
            y_ = rearrange(z[1], 'b h l -> b l h')  
        return y, y_, z[0] # u_
                 