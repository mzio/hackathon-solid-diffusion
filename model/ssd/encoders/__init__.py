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
    
    
def get_encoder(encoder_config):
    if encoder_config['type'] == 'repeat':
        encoder = RepeatEncoder
    elif encoder_config['type'] == 'dense':
        encoder = DenseEncoder
    elif encoder_config['type'] == 'convolution':
        encoder = ConvEncoder
    return encoder(**encoder_config['kwargs'])


def get_decoder(decoder_config):
    # Same classes for both
    return get_encoder(decoder_config)