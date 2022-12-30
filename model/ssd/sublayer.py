"""
Module that computes the kernel
"""

class SubLayer(nn.Module):
    def __init__(self,
                 d_kernel,
                 n_heads,
                 n_channels,
                 **kernel_args):
        super().__init__()
        
        self.d_kernel   = d_kernel
        self.n_heads    = n_heads
        self.n_channels = n_channels
        
        # Setup kernels
        self.n_kernels = {}
        # Add other kernel properties
        other_kernel_args = {'d_kernel': d_kernel,
                             'n_channels': n_channels} 
        for kernel_arg, arg_val in kernel_args.items():
            if kernel_arg[:2] == 'n_':  # Add number of kernels
                self.n_kernels[kernel_arg[2:]] = arg_val  # Hack to remove 'n_' prefix
            else:  
                other_kernel_args[kernel_arg] = arg_val

        self.kernels = self.get_kernels(other_kernel_args)
            
    def get_kernels(self, kernel_args):
        self.kernels = nn.ModuleDict()
        for kernel_type, n_heads in self.n_kernels.items():
            if n_heads > 0:
                kernel_class = get_kernel_class(kernel_type)
                kernel_args['n_heads'] = n_heads
                self.kernels.update({
                    kernel_type: kernel_class(**kernel_args)
                })  
        return self.kernels
        
    # def forward(self, x):
    #     # Can fuse these and make this faster?
    #     head_ix = 0
    #     y = []
    #     for _, kernel in self.kernels.items():
    #         y.append(kernel(x[:, head_ix:head_ix + kernel.n_heads, :]))
    #         head_ix += kernel.n_heads
    #     return torch.cat(y, dim=1)
    
        