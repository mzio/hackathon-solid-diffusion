"""

"""
import torch.nn as nn

from model.ssd.embeddings.model import ModelEmbedding
from model.ssd.encoders import *
from model.ssd.layer import SSDLayer


class StateSpaceDiffusion(nn.Module):
    def __init__(self,
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 ssd_layer_config: dict):
        super().__init__()
        self.embedding_config = embedding_config
        self.encoder_config   = encoder_config
        self.decoder_config   = decoder_config
        self.ssd_layer_config = ssd_layer_config  # hack
        
        self.input_embedding = self.init_embeddings()
        self.input_encoder   = self.init_encoder()
        self.input_decoder   = self.init_decoder()
        self.ssd_encoder, self.ssd_decoder = self.init_ssd_layers()
        
        self.inference_only = False
        
    def set_inference_only(self, mode: bool=False):
        try:
            for ix in range(len(self.ssd_encoder)):
                self.ssd_encoder[ix].inference_only        = mode
                self.ssd_encoder[ix].kernel.inference_only = mode
                # self.ssd_encoder[ix].kernel.requires_grad  = not mode
        except AttributeError:
            pass  # Will fail if ssd_encoder is identity  
        
        for ix in range(len(self.ssd_decoder)):
            self.ssd_decoder[ix].inference_only        = mode
            self.ssd_decoder[ix].kernel.inference_only = mode
            # self.ssd_decoder[ix].kernel.requires_grad  = not mode
        self.inference_only = mode
        self.requires_grad  = not mode
        
    def set_inference_length(self, length: int):
        for ix in range(len(self.ssd_decoder)):
            self.ssd_decoder[ix].kernel.target_length = length
        
    def init_embeddings(self):
        embed_kwargs = {}
        # Hardcoded - change?
        for embed_type in ['patch', 'linear', 'position']:
            if embed_type in self.embedding_config.keys():
                embed_kwargs[f'{embed_type}_kwargs'] = (
                    self.embedding_config[embed_type]
                )
            else:
                embed_kwargs[f'{embed_type}_kwargs'] = None
        # Hacky  
        for kwarg in ['bidirectional', 'output_shape']:
            embed_kwargs[kwarg] = self.embedding_config[kwarg]
        return ModelEmbedding(**embed_kwargs)
    
    def init_encoder(self):
        return self._init_encoder(self.encoder_config)
    
    def init_decoder(self):
        # Same classes for both
        return self._init_encoder(self.decoder_config)
    
    def _init_encoder(self, config: dict):
        if config['type'] == 'repeat':
            encoder = RepeatEncoder
        elif config['type'] in ['dense']:
            encoder = DenseEncoder
        elif config['type'] == 'convolution':
            encoder = ConvEncoder
        return encoder(**config['kwargs'])
        
    def init_ssd_layers(self):
        ssd_encoder = []  # First N - 1 Layers
        ssd_decoder = []  # Last  Nth layer
        
        # Can refactor this with YAML list
        if len(self.ssd_layer_config.layers.items()) == 1:
            # Init encoder
            ssd_encoder = [IdentitySSDLayer()]  # nn.Identity()
            # Init decoder
            for idx, config in self.ssd_layer_config.layers.items():
                try:
                    ssd_decoder = [SSDLayer(**config)]
                except Exception as e:
                    print(idx)
                    print(config)
                    raise e
        else:
            for idx, config in self.ssd_layer_config.layers.items():
                if int(idx) < len(self.ssd_layer_config.layers.items()) - 1:
                    ssd_encoder.append(SSDLayer(**config))
                else:
                    ssd_decoder.append(SSDLayer(**config))
        # Init encoder
        ssd_encoder = OurSequential(*ssd_encoder)
        # Init decoder
        ssd_decoder = OurSequential(*ssd_decoder)
        return ssd_encoder, ssd_decoder
    
    
    def forward(self, u):
        """
        If self.closed_loop is True, returns:
          - closed-loop output, open-loop output, closed-loop last-layer input, open-loop last-layer input
        else, returns:
          - open-loop output, None, None, open-loop last-layer input (ignored)
        """
        
        # u is shape B, C, H, W
        y, y_, u_ = None, None, None  # Initialize outputs
        
        u = self.input_embedding(u)  # B, C, H, W -> B, (LT), (NC * 2); i.e., (B, L, D)
        # print('after embedding', u.shape)
        u = self.input_encoder(u)    # B, L, D -> B, L, D
        # print('after encoder',u.shape)
        # Compute SSD outputs
        z, *z_ = self.ssd_encoder(u)
        # print('after ssd_encoder', z.shape)
        # if self.inference_only:
        #     y, *_ = self.ssd_decoder(z)
        #     # y, y_, z_pred = self.ssd_decoder(z)
        # else:
        #     y, y_, z_pred = self.ssd_decoder(z)
        y, y_, z_pred = self.ssd_decoder(z)
        y  = self.input_decoder(y)
        
        if not self.inference_only:
            y_ = self.input_decoder(y_)
        
        # if self.closed_loop is True, returns:
        # closed-loop output, open-loop output, 
        return y, y_, z_pred, z 
        
    def embed(self, u):
        return self.input_embedding(u)
    
    def encode(self, u):
        return self.input_encoder(self.embed(u))
        
        
class SSD(StateSpaceDiffusion):
    # Alternative reference
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
class IdentitySSDLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x, None, None


class OurSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


