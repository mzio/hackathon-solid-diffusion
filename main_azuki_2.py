"""
python main_azuki_2.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 16 --n_heads 6 --embedding_dim 1 --embedding_type none --n_positions 49 --beta_weight_loss --seed 42 --samples_per_class 32 --max_epochs 1000 --timesteps 800 --criterion_weights 1 1

python main_azuki_2.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 16 --n_heads 6 --embedding_dim 1 --embedding_type none --n_positions 49 --beta_weight_loss --seed 42 --samples_per_class 32 --max_epochs 1000 --timesteps 800 --criterion_weights 1 1 --batch_size 8

python main_azuki_2.py --batch_size 2 --d_kernel 32 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 4096 --beta_weight_loss --max_epochs 1000 --timesteps 1000 --criterion_weights 1 1 --seed 42 --no_wandb

python main_azuki.py --batch_size 2 --d_kernel 32 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 4096 --beta_weight_loss --max_epochs 1000 --timesteps 1000 --criterion_weights 1 1 --seed 42 --no_wandb
"""



# General Setup
import os
import sys
import copy
import time

import random

from os.path import join
from copy import deepcopy

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import opt_einsum as oe
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

from diffusion.schedulers import DDPMScheduler, get_equal_steps
from diffusion.utils.visualize import visualize_diffusions, get_display_image, display_image


# HuggingFace Imports
from datasets import load_dataset
from diffusers import AutoencoderKL  # , UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

auth_token = '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'


from omegaconf import OmegaConf
from utils.config import print_config

# SSD MODEL
from model.ssd.embeddings import PatchEmbedding, LinearEmbedding
from model.ssd.embeddings.position import LearnedPositionEmbedding, get_position_embedding
from model.ssd.encoders import *
from model.ssd.kernels.companion import CompanionKernel
from model.functional.krylov import krylov

from einops import rearrange, repeat


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def pil_to_tensor(pil_image):
    """
    Convert 4 channel PIL image to 3 channel tensor
    """
    to_tensor_transform = transforms.ToTensor()
    tensor_image = to_tensor_transform(pil_image)
    if (tensor_image.shape[0] == 4 and 
        (tensor_image[-1, ...] == torch.ones(tensor_image.shape[1:])).sum().item()
        == np.prod(tensor_image.shape[1:])):
        tensor_image = tensor_image[:-1, ...]
    return tensor_image


# TRAIN INITIAL
config = """
max_epochs: 100
data:
  image_size: 512
  train_batch_size: 32
  eval_batch_size: 32  # How many images to sample during eval
optimizer:
  name: Adam
  learning_rate: 1e-4
  weight_decay: 5e-5
logging:
  save_image_epochs: 10
  save_model_epochs: 20
seed: 42
no_gpu: false 
"""
config = OmegaConf.create(config)
config.device = ('cuda:0' 
                 if torch.cuda.is_available() and config.no_gpu is False
                 else 'cpu')

# DATA PREPROCESSING
config_autoencoder = """
autoencoder:
  mean: 0.5
  std: 0.5
"""
config_autoencoder = OmegaConf.create(config_autoencoder)
config = OmegaConf.merge(config, config_autoencoder)

image_transform = transforms.Compose(
    [
        transforms.Resize((config.data.image_size, 
                           config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([config.autoencoder.mean],
                             [config.autoencoder.std])
    ]
)

def get_transform(samples, image_transform, 
                  tokenizer, text_encoder,
                  guided=True):
    # Process images
    images = [image_transform(image.convert("RGB")) for image in samples["image"]]

     # Process text captions
    with torch.no_grad():
        text_embeddings = []
        for text in samples['text']:
            text_input = tokenizer(text, padding='max_length', max_length=tokenizer.model_max_length,
                                   truncation=True, return_tensors='pt')
            text_embedding = text_encoder(text_input.input_ids)[0]

            if guided:
                max_length = text_input.input_ids.shape[-1]
                unconditional_input = tokenizer(
                    [""] * 1,  # config.train_batch_size, 
                    padding='max_length', 
                    max_length=max_length, return_tensors='pt'
                )
                unconditional_embedding = text_encoder(unconditional_input.input_ids)[0]
                text_embedding = torch.cat([unconditional_embedding, text_embedding])

            # print(text_embedding.shape) 
            # [D x clip_token_size x clip_embed_dim]
            # D is 2 if guided else 1
            text_embeddings.append(text_embedding)

    return {"image": images, 'text': samples['text'], 'text_embedding': text_embeddings}


# AUTOENCODER
def image_encode(images, autoencoder, return_stat, device):
    autoencoder.to(device)
    with torch.no_grad():
        latents = autoencoder.encode(images.to(device))
    autoencoder.cpu()
    images = images.cpu()
    
    # Autoencoder is a VAE
    if return_stat == 'mode':
        try:
            return 0.18215 * latents.mode() # or .mean or .sample
        except:
            return 0.18215 * latents.latent_dist.mode()
    elif return_stat == 'mean':
        try:
            return 0.18215 * latents.mean() # or .mean or .sample
        except:
            return 0.18215 * latents.latent_dist.mean()
    elif return_stat == 'sample':
        try:
            return 0.18215 * latents.sample() # or .mean or .sample
        except:
            return 0.18215 * latents.latent_dist.sample()
        
def tensor_to_image(tensors, image_mean, image_std):
    images = (tensors * image_std + image_mean).clamp(0, 1)
    images = images.detach().cpu() * 255
    return images

def image_decode(latents, autoencoder, image_mean, image_std, device):
    autoencoder.to(device)
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        latents = latents.to(device)
        images = autoencoder.decode(latents).sample
        latents = latents.cpu()
    autoencoder.cpu()
    
    images = (images * image_std + image_mean).clamp(0, 1)
    images = images.detach().cpu() * 255 
    return images

def image_decode_to_pil(latents, **kwargs):
    images = image_decode(latents, **kwargs)
    images = images.permute(0, 2, 3, 1).numpy().round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def image_to_pil(images):
    if images.shape[1] == 1:  # Expand channels to 3
        images = repeat(images, 'b c h w -> b (r c) h w', r=3)
    images = images.permute(0, 2, 3, 1).numpy().round().astype('uint8')
    # return images    
        # image.permute(1, 2, 0) * std + mean).clamp(0, 1).numpy()
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# DIFFUSION
config_diffusion = """
diffusion_scheduler:
  name: DDPM
  kwargs:
    beta_start: 1e-5
    beta_end: 1e-2
    T: 750
"""  # T" 1000
config_diffusion = OmegaConf.create(config_diffusion)



# MODEL
# Unpatch input
def unpatch(x, patch_embedding, patch_pattern='b (t l) (c n)'): 
    b, _, d = x.shape 
    x = rearrange(x, f'{patch_pattern} -> b t c n l',
                  b=b, 
                  c=patch_embedding.n_channels,
                  n=patch_embedding.n_patches, 
                  l=patch_embedding.patch_len)
    t = x.shape[1]
    x = rearrange(x, 'b t c n l -> (b t) (c l) n')
    x = patch_embedding.fold(x)
    x = rearrange(x, '(b t) c h w -> b c h w t', b=b, t=t)
    return x


def patch_input(noise, patch_embedding, bidirectional, 
                    rearrange_pattern='b c n t l -> b (t l) (c n)'):
    # assume noise is B x C x H x W x T
    noise = patch_embedding(noise)
    if bidirectional:
        noise = rearrange([noise, torch.flip(noise, [4])],
                          'r b c n t l -> b c (n r) t l')
    return rearrange(noise, rearrange_pattern)


class ShiftKernel(CompanionKernel):
    """
    
    """
    def __init__(self, n_hidden_state, **kwargs):
        self.n_hidden_state = n_hidden_state
        self.target_length = None
        super().__init__(**kwargs)
        
    def init_weights(self):
        # A matrix ix fixed shift matrix
        A = torch.zeros(self.n_channels, self.n_heads, 
                        self.d_kernel, self.d_kernel)
        A[:, :, 1:, :-1] = torch.eye(self.d_kernel - 1)
        self.register("A", A, trainable=False, lr=None, wd=None)
        
        # B matrix is fixed
        b    =  torch.zeros(self.d_kernel).float()
        b[0] = 1.
        b    = repeat(b, 'd -> c h d', 
                      c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C, K, D matrices are trainable by default
        trainable = True if self.requires_grad is True else False
        
        # Hack for now
        if self.closed_loop:
            k = torch.randn(*self._fp)
            self.register("k", k, trainable, lr=None, wd=None)
        else:
            c = torch.randn(*self._fp)
            self.register("c", c, trainable, lr=None, wd=None)
        
        # D matrix (skip connection) is trainable
        d = torch.randn(self.n_channels, self.n_heads)
        self.register("d", d, trainable, lr=None, wd=None)
    
    def convolve(self, u, A, l, b, c):
        """
        if self.closed_loop:
        - compute: \sum_{i=0}^{t - 1} c(A + bc)^{t - 1 - i}b * u[i]
        else:
        - compute: \sum_{i=0}^{t - 1} cA^{t - 1 - i}b * u[i]
        """
        f = krylov(l, A, b, c)
        y = self.fft_conv(u, f)
        y = rearrange(y, 'b c h l -> b (c h) l')
        return y
    
    def norm(self, x, ord=1):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=2, keepdim=True)
        x = x / x_norm
        return x
    
    def forward(self, u):
        ch, h, d = self.b.shape
        b,  d, l = u.shape  # assume u is B x D x L
        
        # Inference
        if self.target_length is not None: l = self.target_length
        
        if self.closed_loop:  # Assume u already has the noise terms coded in
            # Hacks for stabilization right now
            k = self.norm(self.k.clamp(1 / 16, 1), ord=1)  # Not sure where 16 is coming from
            BK = oe.contract('c h i, c h j -> c h i j', self.b, k)
            
            # 1st computes cb, c(A + BK)b, ..., c(A + BK)^(l-1)b
            # -> f = krylov(l, self.norm(self.A + BK), b=self.b, c=self.k) 
            # Then compute convolution: 
            # -> y = f * u
            y = self.convolve(u, self.norm(self.A + BK), l, self.b, self.k)
        else:
            y = self.convolve(u, self.A, l, self.b, self.c)
            
        return y, None
    
    
# Update get_encoder method
def get_encoder(encoder_config):
    if encoder_config['type'] == 'repeat':
        encoder = RepeatEncoder
    elif encoder_config['type'] == 'dense':
        encoder = DenseEncoder
    elif encoder_config['type'] == 'convolution':
        encoder = ConvEncoder
    elif encoder_config['type'] == 'identity':
        encoder = Encoder  # identity
    else:
        print(encoder_config['type'])
        raise NotImplementedError
    return encoder(**encoder_config['kwargs'])


def get_decoder(decoder_config):
    # Same classes for both
    return get_encoder(decoder_config)



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
        
        # Bug with shape matching
        assert self.skip_connection is False
    
    def get_kernel(self):
        if self.kernel_args['type'] == 'companion':
            kernel_class = CompanionKernel
        elif self.kernel_args['type'] == 'shift':
            kernel_class = ShiftKernel
        else:
            # TODO: implement more kernels
            raise NotImplementedError(f'Error: {kernel_name} not implemented')
        return kernel_class(**self.kernel_args['kwargs'])
    
    def get_decoder(self):
        return get_decoder(self.decoder_args)      
        
    def forward(self, u):  # 
        # Assume input shape is (B, L, H)
        u = rearrange(u, 'b l h -> b h l')
        y, _ = self.kernel(u)  # could output multiple, so should modify this
        y = rearrange(y, 'b h l -> b l h')
        y = self.decoder(y)
        return y
    
    
class OurSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
    
class SSD(nn.Module):
    def __init__(self, config, noise_schedule, noise_stride=1):
        super().__init__()
        self.encoder_config = config.encoder
        self.decoder_config = config.decoder
        self.rollout_config = config.rollout
        
        self.noise_schedule = noise_schedule  # e.g., betas
        self.noise_stride   = noise_stride    # How many to time-steps to apply same beta
        
        self.encoder, self.decoder, self.rollout = self.init_layers()
        
        self.inference_only = False
        
    def init_layers(self):
        encoder = []
        decoder = []
        rollout = []
        
        for ix, layer_config in enumerate(self.encoder_config):
            encoder.append(SSDLayer(**layer_config.layer))
        for ix, layer_config in enumerate(self.decoder_config):
            decoder.append(SSDLayer(**layer_config.layer))
        for ix, layer_config in enumerate(self.rollout_config):
            rollout.append(SSDLayer(**layer_config.layer))
            
        encoder = nn.Sequential(*encoder)
        decoder = nn.Sequential(*decoder)
        rollout = nn.Sequential(*rollout)
        
        return encoder, decoder, rollout
        
    def set_inference_length(self, length: int):
        """
        Use during evaluation to rollout up to specified length
        """
        for ix in range(len(self.ssd_decoder)):
            self.rollout[ix].kernel.target_length = length
            
    def set_inference_only(self, mode: bool=False):
        """
        Use during evaluation to only go through rollout branch
        """
        self.inference_only = mode
        self.requires_grad  = not mode  # Not implemented
        
    def sample_noise(self, z, noise_schedule):
        # z is shape B x L x D
        noise = torch.randn_like(z)  # .to(z.device)
        var   = repeat(noise_schedule, 'l -> (l r)', r=self.noise_stride)
        noise = oe.contract('b l d, l -> b l d', noise.to(z.device), var.to(z.device))
        return noise
        
    def compute_rollout(self, z):
        # z is shape B x L x D
        noise = self.sample_noise(z, self.noise_schedule)
        
        # Replace first noise_stride terms with first noise_stride terms of z
        mask = torch.zeros(z.shape).type(z.type())
        mask[:, :self.noise_stride, :] = 1.
        z = (z * mask.to(z.device)) + (noise.to(z.device) * (1. - mask))
        
        # Compute rollout with closed-loop SSM
        z = self.rollout(z)
        
        # Sample outputs as mu + sigma * noise
        # z += self.sample_noise(z, self.noise_schedule[1:], 0)
        z[:, :-1, :] += noise[:, 1:, :]
        
        noise = noise.cpu()
        return z
        
    def forward(self, u):
        # u is shape B x L x D
        z = self.encoder(u)
        # Compute closed-loop rollout
        z_rollout = self.compute_rollout(z)
        # rollout is a prediction for future samples, so keep first input sample
        z_rollout = torch.cat([z[:, :1, :], z_rollout[:, :-1, :]], dim=1)
        y_rollout = self.decoder(z_rollout)
        
        if not self.inference_only:
            # During training, can also compute outputs from available inputs
            y = self.decoder(z)
        else:
            y = None
            
        return y_rollout, y
    
    
class ModelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Patch
        self.patch_embed = PatchEmbedding(**config.patch)
        # Linear
        if config.linear is not None:
            self.linear_embed = LinearEmbedding(**config.linear)
        else:
            self.linear_embed = lambda x: x.unsqueeze(-1)  # Hack
        # Positional embedding initialized on first forward pass
        self.position_kwargs = config.position
        self.init_position = False
        
        self.bidirectional = config.bidirectional
        self.output_shape = config.output_shape  # bt(cnl)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.linear_embed(x)  # B x C x B x T x L x D
        if self.output_shape == 'bt(cnl)':
            x = rearrange(x, 'b c n t l d -> b c (n l) t 1 d')
        x += self.get_position_embedding(x)
        # Concatenate positional information together
        x = rearrange(x, 'b c n t l d -> b c (n d) t l')  
        if self.bidirectional:
            x_r = torch.flip(x, [4])
            x = rearrange([x, x_r], 'r b c n t l -> b c (n r) t l')
            
        if self.output_shape == 'bt(cnl)':
            x = rearrange(x, 'b c n t l -> b t (c n l)')
            _, self.sample_len, self.sample_dim = x.shape
        elif self.output_shape == 'b(tl)(cn)':
            x = rearrange(x, 'b c n t l -> b (t l) (c n)')
            _, self.sample_len, self.sample_dim = x.shape
        return x
    
    def unpatch(self, x):
        # Assume x is shape: B x T x (C N L)
        b, t, d = x.shape
        # assert self.patch_embedding.fold is not None
        x = rearrange(x, 'b t (c n l) -> (b t) (c l) n',
                      c=self.patch_embed.n_channels,
                      l=self.patch_embed.patch_len,
                      n=self.patch_embed.n_patches)
        x = self.patch_embed.fold(x)  # 
        x = rearrange(x, '(b t) c h w -> b c h w t', b=b, t=t)
        return x
    
    def patch(self, x):
        # assume x is shape: B x C x H x W x T
        x = self.patch_embed(x)  # B x C x N x T x L
        if self.output_shape == 'bt(cnl)':
            x = rearrange(x, 'b c n t l -> b t (c n l)')
        elif self.output_shape == 'bld':   
            x = rearrange(x, 'b c n t l -> b (t l) (c n)')
        else:
            x = rearrange(x, 'b c n t l -> b (c n) (t l)')
        return x
    
    def get_position_embedding(self, x):
        # Implement option for none
        if self.init_position is False:
            if self.position_kwargs['n_positions'] is None:
                n_positions = self.patch_embed.n_patches
                embedding_dim = self.patch_embed.patch_len
            else:
                n_positions = self.position_kwargs['n_positions']
                embedding_dim = self.position_kwargs['embedding_dim']
                
            kwargs = {'n_positions': n_positions,
                      'output_dim': embedding_dim}
            try:
                for k, v in self.position_kwargs['kwargs']:
                    kwargs[k] = v
            except:
                pass
            try:
                self.position_embedding = get_position_embedding(self.position_kwargs.type, **kwargs)
            except Exception as e:
                print(self.position_kwargs.type)
                raise e
                
            
            # self.position_embedding = PositionEmbedding(n_positions, embedding_dim)
            self.init_position = True
            
        return self.position_embedding(x)
    
    
class StateSpaceDiffusionModel(nn.Module):
    def __init__(self,
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 ssd_config: dict,
                 noise_schedule,
                 noise_stride: int):
        super().__init__()

        self.embedding_config = embedding_config
        self.encoder_config   = encoder_config
        self.decoder_config   = decoder_config
        self.ssd_config       = ssd_config
        self.noise_schedule   = noise_schedule
        self.noise_stride     = noise_stride
        
        # Hard-coded for now
        self.input_embed  = ModelEmbedding(self.embedding_config)
        # print('a')
        self.input_encode = get_encoder(self.encoder_config)
        # print('b')
        self.input_decode = get_decoder(self.decoder_config)
        # print('c')
        self.ssd          = SSD(ssd_config, noise_schedule, noise_stride)
        # print('d')
        
        
    def forward(self, x, x_=None):
        # print('input shape', x.shape)
        x = self.input_embed(x)
        # print('shape after input_embed', x.shape)
        x = self.input_encode(x)
        # First is closed-loop, second is open-loop 
        # print('shape after input_encode', x.shape)
        x, x_ = self.ssd(x)
        # print('shape after ssd (CL)', x.shape)
        # print('shape after ssd (OL)', x_.shape)
        x = self.input_decode(x)
        # print('shape after input_decode (CL)', x.shape)
        if not self.ssd.inference_only:
            x_ = self.input_decode(x_)
            # print('shape after input_decode (OL)', x_.shape)
            
        # print('shape after unpatch', self.input_embed.unpatch(x).shape)
        return x, x_

        
class SSDModel(StateSpaceDiffusionModel):
    # Alternative reference
    def __init__(self, 
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 ssd_config: dict,
                 noise_schedule,
                 noise_stride: int):
        super().__init__(embedding_config,
                         encoder_config,
                         decoder_config,
                         ssd_config,
                         noise_schedule,
                         noise_stride)
        
        
# TRAINING
def train(model, autoencoder, epoch, **kwargs):
    model.train()
    model.ssd.set_inference_only(mode=False)
    return run_epoch(model, autoencoder, True, epoch, **kwargs)
    
    
def evaluate(model, autoencoder, epoch, **kwargs):
    model.eval()
    model.ssd.set_inference_only(mode=True)
    
    with torch.no_grad():
        return run_epoch(model, autoencoder, False, epoch, **kwargs)
    
    
def run_epoch(model, autoencoder, train, epoch, dataloader, scheduler, 
              optimizer, criterion, beta_weight_loss, device, wandb):
    
    model.zero_grad()
    pbar = tqdm(dataloader, leave=False)
    
    torch.cuda.empty_cache()
    
    # breakpoint()
    
    T = scheduler.T
    
    # print(len(dataloader))
    for batch_ix, data in enumerate(pbar):
        # Encode inputs
        encoded_inputs = image_encode(data['image'], autoencoder, return_stat='mode', device=device)
        encoded_inputs = encoded_inputs.cpu()

        # Run forward diffusion on latent inputs
        diffusion_start = time.time()
        noisy, means, noise, noise_var = scheduler.get_forward_samples(encoded_inputs, 
                                                                       np.arange(T))  # hard code   
        diffusion_end = time.time()
        # print(f"-- Generating {T * len(noisy)} samples took {diffusion_end - diffusion_start:3f} seconds --")

        # Reverse diffusion ordering
        u_input = torch.flip(noisy, [-1])
        
        if not train:  # Sample noise
            u_input = torch.randn_like(u_input)
        
        u_noise = torch.flip(noise, [-1])
        u_beta  = torch.flip(noise_var, [-1])
        v_beta  = 1. / u_beta
        y_means = torch.flip(means, [-1])
        
        model.to(device)
        u_input = u_input.to(device)
        v_beta = v_beta.to(device)
        
        # # patch_embeding hack
        # try:
        #     model.input_embed.n_channels
        # except:
        #     _u = model.input_embed.patch_embed(u_input)
        
        # c is closed loop, o is open loop
        start = time.time()
        y_c, y_o = model(u_input)
        end = time.time()
        
        # if train is False:
        #     print(f'Time to generate {len(y_c)} samples: {end - start:.5f}s')
        
        # u_input = u_input.cpu()
        # v_beta = v_beta.cpu()
        # model.cpu()
        
        # y_c = model.input_embedding.unpatch(y_c)
        # y_c = unpatch(y_c, model.patch_embedding) # fix this
        y_c = model.input_embed.unpatch(y_c)
        loss_c = criterion(y_c[..., model.d_kernel:], 
                           y_means[..., model.d_kernel:].to(device))
        
        if train and not model.ssd.inference_only:
            # y_o = unpatch(y_o, model.patch_embedding) # fix this
            y_o = model.input_embed.unpatch(y_o)
            loss_o = criterion(y_o[..., model.d_kernel:], 
                               y_means[..., model.d_kernel:].to(device))
        else:
            loss_o = torch.zeros(loss_c.shape).to(loss_c.device)
            
        # all_losses = [loss_c, loss_o]
        # loss_names = ['closed-loop', 'open-loop']
        # loss = loss_c + loss_o
        if beta_weight_loss:
            loss_c = oe.contract('b c h w t, t -> b c h w t',
                                 loss_c, v_beta[..., model.d_kernel:]).mean()
            loss_o = oe.contract('b c h w t, t -> b c h w t',
                                 loss_o, v_beta[..., model.d_kernel:]).mean()
        else:
            loss_c = loss_c.mean()
            loss_o = loss_o.mean()
            
        
        loss = loss_c + loss_o
            
        if train:
            # model.to(device)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
        model.cpu()
        u_input = u_input.cpu()
        v_beta = v_beta.cpu()
        loss = loss.cpu()
        y_c = y_c.cpu()
        try:
            y_o = y_o.cpu()
        except:
            pass
        loss_c = loss_c.cpu()
        loss_o = loss_o.cpu()
        
        all_losses = [loss_c, loss_o]
        loss_names = ['closed-loop', 'open-loop']
        
            
        fwd_diff_desc = f'fwd diff: {diffusion_end - diffusion_start:.3f}s'
        rev_diff_time = end - start
        rev_diff_desc = f'rev diff: {rev_diff_time:.3f}s ({rev_diff_time / len(noisy):.3f}s/sample)'
        loss_desc = ' | '.join([f'{loss_names[ix]}: {all_losses[ix]:.3f}'
                                for ix in range(len(all_losses))])
        
        pbar_desc = f'Batch: {batch_ix}/{len(dataloader)} | {fwd_diff_desc} | {rev_diff_desc} | {loss_desc}'
        pbar.set_description(pbar_desc)
        
        # print(model.device)
        # for p in model.parameters():
        #     print(p.device)
        # print(loss.device)
        # print([l.device for l in all_losses])
        # print(v_beta.device)
        # print(y_c.device)
        # print(y_o.device)
        # print(u_input.device)
        # print(loss_c.device)
        # print(loss_o.device)
        # Free memory
        loss = loss.detach().cpu().item()
        loss_c = loss_c.detach().cpu().item()
        loss_o = loss_o.detach().cpu().item()
        all_losses = [loss_c, loss_o]
        del v_beta
        # del y_c
        del y_o
        del u_input
        
        torch.cuda.empty_cache()
        # breakpoint()
        
        if not train and batch_ix == 0:
        # if train and batch_ix == 0:
            all_decoded_images = []
            steps = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
            steps = [s for s in steps if s < y_c.shape[-1]]
            n_samples = y_c.shape[0]
            n_steps = len(steps)
            for step in tqdm(steps, leave=False, desc='Running diffusion eval'):
                # if step < y_c.shape[-1]:
                decoded_inputs = image_decode(y_c[..., step], autoencoder,
                                              config.autoencoder.mean,
                                              config.autoencoder.std, 
                                              device)
                decoded_inputs = decoded_inputs.cpu()
                # decoded_inputs = image_decode(y_c[..., step], 0., 1., device)
                decoded_images = image_to_pil(decoded_inputs)
                all_decoded_images.append(decoded_images)
               
            
            # print(f'Sample at step {step}, beta_{step}: {u_beta[step]}')
            # fig, axis = plt.subplots(10, 4, figsize=(16, 40))
            fig, axis = plt.subplots(n_steps, n_samples, 
                                     figsize=(n_samples * 4, n_steps * 4))
            
            # axis[plot_ix, sample_ix]
            # for sample_ix in range(num_samples):
            #     for plot_ix, step_ix in enumerate(range(0, total_steps, total_steps // num_steps)):
            for i in range(n_samples):
                for step_ix, step in enumerate(steps):
                    try:
                        axis[step_ix, i].imshow(
                            all_decoded_images[step_ix][i]
                        )
                    except Exception as e:
                        print(step_ix, i)
                        print(all_decoded_images[step_ix][i])
                        raise e
                    axis[step_ix, i].set_axis_off()
                    
            # plt.show()
            # fig.savefig(f'azuki_gen-epoch={epoch:03d}.png')
            save_name = f'{model.save_name}-epoch={epoch:03d}.png'
            fig.savefig(save_name)
            
            if wandb is not None:
                wandb.log({save_name: fig})
               
            del y_c
            del all_decoded_images
            torch.cuda.empty_cache()
                
            # model.cpu()
            # loss = loss.cpu()
            # # for ix in range(len(all_losses)):
            # #     all_losses[ix] = all_losses[ix].cpu()
            # u = u.cpu()
            # y = y.cpu()
            # y_c = y_c.cpu()
            # y_o = y_o.cpu()
            # z_c = z_c.cpu()
            # z_o = z_o.cpu()
                          
    return model, loss, all_losses, loss_names
    

from setup import initialize_args, initialize_network_config
from utils.config import print_config


def get_resampled_dataloader(dataset, sample_indices, **kwargs):
    dataset_rs = deepcopy(dataset)
    dataset_rs.data = dataset_rs.data[sample_indices]
    dataset_rs.targets = dataset_rs.targets[sample_indices]
    return DataLoader(dataset_rs, **kwargs)
    
    
def init_wandb(args):
    if not args.no_wandb:
        import wandb
        wandb.init(config={'lr': args.lr},
                   entity=args.wandb_entity,
                   name=args.run_name,
                   project=args.project_name,
                   dir=args.log_dir)
        wandb.config.update(args)
    else:
        wandb = None
    return wandb
    

def main():
    args = initialize_args()
    seed_everything(args.seed)
    criterion_weights = '_'.join(args.criterion_weights)
    diffusion_args = f'dsc={args.diffusion_scheduler}-beta=({args.beta_start}_{args.beta_end})-ts={args.timesteps}-bwl={int(args.beta_weight_loss)}-pds={int(args.predict_sample)}'
    args.criterion_weights = [float(w) for w in args.criterion_weights]
    args.project_name = f'azuki_v2'
    args.run_name     = f'dk={args.d_kernel}-nh={args.n_heads}-ps={args.patch_size}-ed={args.embedding_dim}-et={args.embedding_type}-op={args.optimizer}-lr={args.lr}-wd={args.weight_decay}-{diffusion_args}-sd={args.seed}'
    
    save_name = f'{args.project_name}-{args.run_name}'
    
    # Initialize wandb run
    wandb = init_wandb(args)
    print(f'WandB: {wandb}')
    
    # LOAD PRETRAINED MODELS
    
    # Stable Diffusion
    checkpoint_path_sd = '/dfs/scratch1/common/pretrained_models/stable-diffusion-v1-4'
    model_id_diffusion_autoencoder = 'CompVis/stable-diffusion-v1-4'

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(model_id_diffusion_autoencoder, subfolder="vae", use_auth_token=auth_token,
                                        cache_dir=join(checkpoint_path_sd, 'vae'))
    vae.save_pretrained(join(checkpoint_path_sd, 'vae'))
    
    # CLIP
    checkpoint_path_clip = '/dfs/scratch1/common/pretrained_models/clip'
    model_id_clip = 'openai/clip-vit-large-patch14'

    # Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(model_id_clip, use_auth_token=auth_token,
                                              cache_dir=checkpoint_path_clip)
    text_encoder = CLIPTextModel.from_pretrained(model_id_clip, use_auth_token=auth_token,
                                                 cache_dir=checkpoint_path_clip)
    
    # LOAD DATA
    config.dataset_name = '/dfs/scratch1/mzhang/projects/solid-diffusion/data/azuki/image'
    config.dataset_dir = '/dfs/scratch1/common/public-datasets/azuki'
    dataset = load_dataset(config.dataset_name, split="train", 
                           cache_dir=config.dataset_dir)
    
    default_transform = lambda x: get_transform(x, image_transform, 
                                                tokenizer, text_encoder, 
                                                guided=True) 
    dataset.set_transform(default_transform)
    
    config.data.train_batch_size = args.batch_size
    config.data.eval_batch_size = args.batch_size
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.data.train_batch_size, 
                                               shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.data.eval_batch_size, 
                                              shuffle=False)
    
    # DIFFUSION
    if args.diffusion_scheduler == 'ddpm':    
        scheduler = DDPMScheduler(args.beta_start, args.beta_end, args.timesteps)
    else:
        raise NotImplementedError
    T = args.timesteps
    
    
    # MODEL
    embedding_config = f"""
    patch:
      patch_size: 4
      dilation: 1
      padding: 0
      stride: 4
    linear:
      input_dim: 16
      output_dim:
        - 16
        - {args.embedding_dim}
      identity_init: true
      identity_val: 1
    position:
      type: learn_1d
      n_positions: 4096
      embedding_dim: {args.embedding_dim}
    bidirectional: false
    output_shape: bt(cnl)
    """
    embedding_config = OmegaConf.create(embedding_config)
    
    # patch_embedding = PatchEmbedding(**embedding_config.patch.kwargs)

    input_encoder_config = f"""
    type: repeat
    kwargs:
      input_dim: {16384 * embedding_config.position.embedding_dim}
      output_dim: {16384 * args.n_heads}
      input_shape: bld
    """
    input_encoder_config = OmegaConf.create(input_encoder_config)

    input_decoder_config = f"""
    type: dense
    kwargs:
      input_dim: {int(16384 * args.n_heads / 16)} 
      output_dim: {16384}
      activation: gelu
      n_layers: 2
      n_activations: 1
      input_shape: bld
    """
    input_decoder_config = OmegaConf.create(input_decoder_config)

    ssd_encoder_config = f"""
    encoder:
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: {args.d_kernel}
            n_heads: {16384 * args.n_heads}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: dense
          kwargs:
            input_dim: {16384 * args.n_heads}
            output_dim: {int(16384 * args.n_heads / 16)}
            activation: gelu
            n_layers: 2
            n_activations: 1
        skip_connection: false
        closed_loop: false
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: 16
            n_heads: {int(16384 * args.n_heads / 16)}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: dense
          kwargs:
            input_dim: {int(16384 * args.n_heads / 16)}
            output_dim: {int(16384 * args.n_heads / 128)}
            activation: gelu
            n_layers: 2
            n_activations: 1
        skip_connection: false
        closed_loop: false
    """
    ssd_encoder_config = OmegaConf.create(ssd_encoder_config)

    ssd_decoder_config = f"""
    decoder:
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: {args.d_kernel}
            n_heads: {int(16384 * args.n_heads / 128)}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: dense
          kwargs:
            input_dim: {int(16384 * args.n_heads / 128)}
            output_dim: {int(16384 * args.n_heads / 16)}
            activation: gelu
            n_layers: 2
            n_activations: 1
        skip_connection: false
        closed_loop: false
    """
    ssd_decoder_config = OmegaConf.create(ssd_decoder_config)

    ssd_rollout_config = f"""
    rollout:
    - layer:
        kernel:
          type: shift
          kwargs:
            d_kernel: {args.d_kernel}
            n_heads: {int(16384 * args.n_heads / 128)}
            n_channels: 1
            n_hidden_state: 1
            skip_connection: false
            closed_loop: true
            train: true
        decoder:
          type: identity
          kwargs:
            input_dim: {int(16384 * args.n_heads / 128)}
            output_dim: {int(16384 * args.n_heads / 128)}
        skip_connection: false
        closed_loop: True
    """
    ssd_rollout_config = OmegaConf.create(ssd_rollout_config)
    
    ssd_config = OmegaConf.merge(ssd_encoder_config, ssd_decoder_config, ssd_rollout_config)
    
    seed_everything(config.seed)
    criterion = nn.MSELoss(reduction='none')

    u_beta = torch.flip(scheduler.beta_t, [0])

    model = SSDModel(embedding_config,
                     input_encoder_config,
                     input_decoder_config,
                     ssd_config,
                     u_beta,  
                     noise_stride=1)  # 16
    model.d_kernel = args.d_kernel  # HACKS
    print(model)
    model.save_name = save_name  # HACKS
    optim_config = {
        'lr': 1e-4,   
        'weight_decay': 5e-5
    }
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), **optim_config)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), **optim_config)

    config_diffusion.diffusion_scheduler.kwargs.T = args.timesteps
    T = config_diffusion.diffusion_scheduler.kwargs.T

    train_config = {
        'dataloader': train_loader,  # Fill-in / update below 
        'scheduler': scheduler,
        'optimizer': optimizer, 
        'criterion': torch.nn.MSELoss(reduction='none'), 
        # 'criterion_weights': args.criterion_weights,  # [1., 1., 10., 10.], 
        'beta_weight_loss': True,
        'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        'wandb': wandb
    }

    # Training
    epoch_pbar = tqdm(range(args.max_epochs))
    # Can do gradient accumulation
    for epoch in epoch_pbar:
        # data_indices = np.concatenate(
        #     [np.random.choice(c, size=args.samples_per_class, replace=False)
        #      for c in train_class_indices])
        # _train_loader = get_resampled_dataloader(train_dataset, data_indices,
        #                                          batch_size=args.batch_size,
        #                                          shuffle=True)
        # train_config['dataloader'] = _train_loader
        

        if (epoch + 1) % 1 == 0:
            model, loss, all_losses, loss_names = train(model, vae, epoch, 
                                                        **train_config)
            evaluate(model, vae, epoch, **train_config)
        else:
            model, loss, all_losses, loss_names = train(model, vae, epoch, 
                                                        **train_config)

        loss_desc = ' | '.join([f'{loss_names[ix]}: {all_losses[ix]:.3f}'
                                for ix in range(len(all_losses))])
        
        pbar_desc = f'Epoch: {epoch}/{len(epoch_pbar)} | {loss_desc}'
        epoch_pbar.set_description(pbar_desc)
        
        if wandb is not None:
            log_metrics = {'loss': 0}
            for lix, loss in enumerate(loss_names):
                log_metrics[loss] = all_losses[lix].item()
                log_metrics['loss'] += all_losses[lix].item()
            wandb.log(log_metrics, step=epoch)
            
            
if __name__ == '__main__':
    main()
