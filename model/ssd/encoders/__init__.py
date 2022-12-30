"""
Generic class for mapping from input to output
"""
import torch.nn as nn

from .dense import DenseEncoder
from .repeat import RepeatEncoder
from .convolution import ConvEncoder


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        """
        Generic class for mapping from input to output
        - Input  shape: [B, ..., input_dim]
        - Output shape: [B, ..., output_dim]
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialize_layers()
        
    def initialize_layers(self):
        self.layers = nn.Identity()
        
    def forward(self, x):
        return self.layers(x)