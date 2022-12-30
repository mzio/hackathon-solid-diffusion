from .shift import *
from .scale_shift import *
from .companion import *
from .diagonal_discrete import *
from .diagonal_continuous import *


def get_kernel(kernel_name):
    if kernel_name == 'companion':
        kernel_class = CompanionKernel
    elif kernel_name == 'shift':
        kernel_class = ShiftKernel
    elif kernel_name == 'dilated_shift':
        kernel_class = DilatedShiftKernel
    elif kernel_name in ['simple', 'diagonal_continuous']:
        kernel_class = ContinuousDiagonalKernel
    elif kernel_name in ['simple_discrete', 'diagonal_discrete']:
        kernel_class = DiscreteDiagonalKernel
    else:
        print(f'Error: {kernel_name} not implemented')
        # TODO: implement more kernels
        raise NotImplementedError
        
    return kernel_class